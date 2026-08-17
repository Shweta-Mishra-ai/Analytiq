"""
services/user_store.py — client accounts.

Each client gets a username/password created by the admin (see
POST /api/admin/users). Passwords are salted + PBKDF2-hashed, never
stored or returned in plaintext. Deliberately dependency-free (stdlib
hashlib) to match the rest of the project's minimal-dependency style.
"""
from __future__ import annotations
import logging

import hashlib
import hmac
import os
import pickle
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.config import config

logger = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 200_000


@dataclass
class User:
    username: str
    salt: bytes
    password_hash: bytes
    created_at: float
    is_admin: bool = False


@dataclass
class PublicUser:
    """Safe-to-return view of a User — never includes the password hash."""
    username: str
    created_at: float
    is_admin: bool = False


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)


class UserStore:
    """Thread-safe, disk-backed store of client accounts."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(config.data_dir, "users.pkl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.RLock()
        # Set by _load() when the account file exists but cannot be read.
        # Nothing may be served in that state — see `unreadable`.
        self._load_error: Optional[str] = None
        self._users: dict[str, User] = self._load()

    def _load(self) -> dict[str, User]:
        """Load the accounts, distinguishing "none yet" from "cannot read".

        These two states used to be the same empty dict, and the
        difference is the whole security model. No accounts *and* no
        admin key is single-user open mode: auth is bypassed and every
        request is treated as an administrator, which is correct for a
        fresh local install. An account file that exists but cannot be
        read is the opposite situation — accounts were created, and the
        store has just lost sight of them.

        A corrupt file, a permission change, or a pickle that a newer
        Python cannot load therefore turned a multi-client deployment
        into an open one, silently, with `/api/admin/*` included. The
        failure now stays visible: the accounts are not empty, they are
        unknown, and requests are refused until an operator looks at it.
        """
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "rb") as f:
                loaded = pickle.load(f)
        except Exception as exc:
            self._load_error = "{}: {}".format(type(exc).__name__, exc)
            logger.error(
                "Account file %s exists but could not be read (%s). Refusing "
                "every request until this is resolved — restoring the file or "
                "removing it (which resets to single-user open mode) are the "
                "two ways out.", self.path, self._load_error)
            return {}
        if not isinstance(loaded, dict):
            self._load_error = "account file does not contain an account map"
            logger.error("Account file %s is not an account map; refusing "
                         "every request until it is restored.", self.path)
            return {}
        return loaded

    @property
    def unreadable(self) -> Optional[str]:
        """Why the accounts could not be read, or None if they were.

        Consulted by the auth middleware before anything else: an
        unreadable account file is a 503, never an open door.
        """
        return self._load_error

    def _save(self) -> None:
        if self._load_error:
            # Writing now would replace a file that still holds the real
            # accounts with one holding whatever is in memory, turning a
            # recoverable read failure into permanent data loss.
            raise RuntimeError(
                "Refusing to write accounts: the existing account file could "
                "not be read ({}). Restore or remove it first.".format(
                    self._load_error))
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._users, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._users) == 0

    def exists(self, username: str) -> bool:
        with self._lock:
            return username in self._users

    def create(self, username: str, password: str,
               is_admin: bool = False) -> PublicUser:
        username = username.strip().lower()
        if not username or not password:
            raise ValueError("Username and password are required")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        with self._lock:
            if username in self._users:
                raise ValueError(f"User '{username}' already exists")
            salt = secrets.token_bytes(16)
            user = User(
                username=username,
                salt=salt,
                password_hash=_hash_password(password, salt),
                created_at=time.time(),
                is_admin=is_admin,
            )
            self._users[username] = user
            self._save()
        return PublicUser(username, user.created_at, user.is_admin)

    def verify(self, username: str, password: str) -> Optional[PublicUser]:
        username = username.strip().lower()
        with self._lock:
            user = self._users.get(username)
        if not user:
            return None
        candidate = _hash_password(password, user.salt)
        if not hmac.compare_digest(candidate, user.password_hash):
            return None
        return PublicUser(user.username, user.created_at, user.is_admin)

    def get(self, username: str) -> Optional[PublicUser]:
        with self._lock:
            user = self._users.get(username.strip().lower())
        return PublicUser(user.username, user.created_at, user.is_admin) if user else None

    def list(self) -> list[PublicUser]:
        with self._lock:
            out = [PublicUser(u.username, u.created_at, u.is_admin)
                   for u in self._users.values()]
        out.sort(key=lambda u: u.created_at)
        return out

    def delete(self, username: str) -> bool:
        username = username.strip().lower()
        with self._lock:
            if username in self._users:
                del self._users[username]
                self._save()
                return True
        return False


user_store = UserStore()
