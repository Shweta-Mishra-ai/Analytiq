"""
services/user_store.py — client accounts.

Each client gets a username/password created by the admin (see
POST /api/admin/users). Passwords are salted + PBKDF2-hashed, never
stored or returned in plaintext. Deliberately dependency-free (stdlib
hashlib) to match the rest of the project's minimal-dependency style.

Accounts persist as JSON rather than a pickle: this is the one file
whose contents are the login system, and unpickling it would run
whatever it happened to contain.
"""
from __future__ import annotations
import logging

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
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
        self.path = path or os.path.join(config.data_dir, "users.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.RLock()
        self._users: dict[str, User] = self._load()

    def _load(self) -> dict[str, User]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {
                name: User(
                    username=rec["username"],
                    salt=base64.b64decode(rec["salt"]),
                    password_hash=base64.b64decode(rec["password_hash"]),
                    created_at=float(rec["created_at"]),
                    is_admin=bool(rec.get("is_admin", False)),
                )
                for name, rec in raw.items()
            }
        except Exception:
            # An unreadable account file must not silently hand out an
            # empty store, because an empty store means first-run setup.
            logger.error("could not read the account file at %s", self.path,
                         exc_info=True)
            raise

    def _save(self) -> None:
        # JSON, not pickle: this file holds the credentials, and loading
        # a pickle executes whatever it contains. Salt and hash are raw
        # bytes, so they travel base64-encoded.
        payload = {
            name: {
                "username": u.username,
                "salt": base64.b64encode(u.salt).decode(),
                "password_hash": base64.b64encode(u.password_hash).decode(),
                "created_at": u.created_at,
                "is_admin": u.is_admin,
            }
            for name, u in self._users.items()
        }
        tmp = self.path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
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
