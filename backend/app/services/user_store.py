"""
services/user_store.py — client accounts.

Each client gets a username/password created by the admin (see
POST /api/admin/users). Passwords are salted + PBKDF2-hashed, never
stored or returned in plaintext. Deliberately dependency-free (stdlib
hashlib) to match the rest of the project's minimal-dependency style.
"""
from __future__ import annotations
import logging

import base64
import hashlib
import hmac
import json
import os
import pickle
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from app.config import config

logger = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 200_000


class UserStoreLoadError(RuntimeError):
    """Raised when an existing account store cannot be read. Deliberately
    fatal: silently treating an unreadable store as empty would drop the
    whole app into single-user open mode — auth disabled — which is the
    opposite of what a corrupted-or-tampered accounts file should do."""


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
        # Accounts are stored as JSON, not pickle: the file holds only
        # usernames, timestamps and base64 salt/hash, so it needs no code
        # execution to read, and unpickling an accounts file an attacker
        # could influence is a needless RCE surface. A legacy users.pkl is
        # migrated once, on first load.
        self.path = path or os.path.join(config.data_dir, "users.json")
        self._legacy_path = os.path.join(os.path.dirname(self.path), "users.pkl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.RLock()
        self._users: dict[str, User] = self._load()

    def _load(self) -> dict[str, User]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return {u: User(
                    username=u,
                    salt=base64.b64decode(d["salt"]),
                    password_hash=base64.b64decode(d["password_hash"]),
                    created_at=float(d["created_at"]),
                    is_admin=bool(d.get("is_admin", False)),
                ) for u, d in raw.items()}
            except Exception as e:
                # Fail closed — do NOT return {} (that would disable auth).
                raise UserStoreLoadError(
                    f"Account store at {self.path} exists but could not be "
                    f"read ({e}). Refusing to start with auth disabled. "
                    f"Restore the file from backup or remove it deliberately "
                    f"to reset to open mode.") from e
        # One-time migration of a legacy pickle store.
        if os.path.exists(self._legacy_path):
            try:
                with open(self._legacy_path, "rb") as f:
                    users = pickle.load(f)
                logger.info("Migrating legacy users.pkl → users.json")
                self._users = users
                self._save()
                return users
            except Exception as e:
                raise UserStoreLoadError(
                    f"Legacy account store {self._legacy_path} exists but "
                    f"could not be read ({e}). Refusing to start with auth "
                    f"disabled.") from e
        return {}

    def _save(self) -> None:
        payload = {u: {
            "salt": base64.b64encode(usr.salt).decode("ascii"),
            "password_hash": base64.b64encode(usr.password_hash).decode("ascii"),
            "created_at": usr.created_at,
            "is_admin": usr.is_admin,
        } for u, usr in self._users.items()}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            logger.debug("could not chmod %s", self.path, exc_info=True)

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
