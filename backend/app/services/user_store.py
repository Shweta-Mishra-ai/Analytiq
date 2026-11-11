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
import os
import pickle
import re
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
    # When the password was last set. Tokens issued before it stop working,
    # so changing a password signs out every other device.
    password_changed_at: float = 0.0
    # SHA-256 of the key this client's Claude connector signs in with (the
    # key itself is shown once and never stored), and when it was made.
    connector_key_hash: bytes = b""
    connector_key_at: float = 0.0


@dataclass
class PublicUser:
    """Safe-to-return view of a User — never includes the password hash."""
    username: str
    created_at: float
    is_admin: bool = False
    password_changed_at: float = 0.0
    connector_key_at: float = 0.0      # 0 when there is no connector key


CONNECTOR_KEY_PREFIX = "aqk-"


# The passwords tried first in any guessing attack. "password" was accepted.
_COMMON_PASSWORDS = {
    "password", "password1", "password12", "password123", "passw0rd", "12345678",
    "123456789", "1234567890", "qwertyui", "qwerty123", "qwertyuiop", "11111111",
    "00000000", "iloveyou", "admin123", "administrator", "letmein1", "welcome1",
    "welcome123", "abc12345", "abcd1234", "1q2w3e4r", "changeme", "analytiq",
    "analytiq123", "football", "baseball", "sunshine", "princess", "trustno1",
}


def check_password(password: str, username: str = "") -> None:
    """Raise ValueError saying what is wrong with a new password."""
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters")
    if len(password) > 200:
        raise ValueError("Password must be at most 200 characters")
    low = password.lower()
    if low in _COMMON_PASSWORDS or len(set(low)) < 4:
        raise ValueError("That password is too easy to guess; choose a less common one")
    if username and low == username.lower():
        raise ValueError("Password must not be the username")


def _public(user: "User") -> PublicUser:
    return PublicUser(user.username, user.created_at, user.is_admin,
                      float(getattr(user, "password_changed_at", 0.0) or 0.0),
                      float(getattr(user, "connector_key_at", 0.0) or 0.0)
                      if getattr(user, "connector_key_hash", b"") else 0.0)


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)


# A username becomes a directory name under DATA_DIR (datasets/{owner}/,
# rag/{owner}/), so it is held to a filesystem-safe alphabet. Without this,
# "../x" or a name with a NUL byte was accepted at creation and then made
# every request that account sent fail with a 500.
_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
# "admin" is the principal an admin-key request runs as, and "local" owns
# every dataset uploaded in open dev mode — an account by either name would
# inherit an identity that is not its own.
_RESERVED_USERNAMES = {"admin", "local"}

# Verifying a password for an unknown user must cost the same as for a
# known one, or response time tells an attacker which usernames exist.
_DUMMY_SALT = secrets.token_bytes(16)
_DUMMY_HASH = _hash_password("not-a-real-password", _DUMMY_SALT)


class UserStoreCorrupt(RuntimeError):
    """users.pkl exists but cannot be read."""


class UserStore:
    """Thread-safe, disk-backed store of client accounts."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(config.data_dir, "users.pkl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.RLock()
        self._users: dict[str, User] = self._load()

    def _load(self) -> dict[str, User]:
        """The persisted accounts.

        A file that exists but cannot be read must stop the app, not load
        as empty: an empty store with no admin key is open dev mode, so a
        truncated or corrupted users.pkl would silently switch off
        authentication and expose every client's data.
        """
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "rb") as f:
                # A file this server wrote to its own data dir; uploads are never unpickled.
                users = pickle.load(f)  # nosec B301
        except Exception as e:
            raise UserStoreCorrupt(
                f"Cannot read the account store at {self.path} ({e}). Refusing "
                f"to start with no accounts, which would disable "
                f"authentication. Restore the file from backup, or delete it "
                f"deliberately to start over.") from e
        if not isinstance(users, dict):
            raise UserStoreCorrupt(
                f"The account store at {self.path} does not hold accounts.")
        return users

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._users, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._users) == 0

    def exists(self, username: str) -> bool:
        with self._lock:
            return username.strip().lower() in self._users

    def create(self, username: str, password: str,
               is_admin: bool = False) -> PublicUser:
        username = username.strip().lower()
        if not username or not password:
            raise ValueError("Username and password are required")
        if not _USERNAME_RE.match(username):
            raise ValueError(
                "Username must be 1-64 characters of a-z, 0-9, '.', '_' or "
                "'-', starting with a letter or digit")
        if username in _RESERVED_USERNAMES:
            raise ValueError(f"'{username}' is reserved")
        check_password(password, username)
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
        return _public(user)

    def set_password(self, username: str, new_password: str) -> PublicUser:
        """Set a new password (admin reset, or after the old one was
        verified). Every token issued before now stops working."""
        username = username.strip().lower()
        check_password(new_password, username)
        with self._lock:
            user = self._users.get(username)
            if user is None:
                raise ValueError("No such user")
            user.salt = secrets.token_bytes(16)
            user.password_hash = _hash_password(new_password, user.salt)
            user.password_changed_at = time.time()
            self._save()
            return _public(user)

    def new_connector_key(self, username: str) -> str:
        """A new key for this client's Claude connector, replacing any
        earlier one. Unlike a sign-in token it does not expire, so the
        connector keeps working until the key is replaced or removed; it
        opens the analysis tools only, not the web app or the account.
        Returned once: only its hash is kept."""
        username = username.strip().lower()
        with self._lock:
            user = self._users.get(username)
            if user is None:
                raise ValueError("No such user")
            name = base64.urlsafe_b64encode(username.encode()).rstrip(b"=").decode()
            key = f"{CONNECTOR_KEY_PREFIX}{name}-{secrets.token_urlsafe(32)}"
            user.connector_key_hash = hashlib.sha256(key.encode()).digest()
            user.connector_key_at = time.time()
            self._save()
        return key

    def revoke_connector_key(self, username: str) -> bool:
        username = username.strip().lower()
        with self._lock:
            user = self._users.get(username)
            if user is None or not getattr(user, "connector_key_hash", b""):
                return False
            user.connector_key_hash = b""
            user.connector_key_at = 0.0
            self._save()
        return True

    def user_for_connector_key(self, key: str) -> Optional[PublicUser]:
        """The account a connector key belongs to, or None. The key comes
        straight from a request header, so anything can be in it."""
        try:
            if not key.startswith(CONNECTOR_KEY_PREFIX) or len(key) > 400:
                return None
            name = key[len(CONNECTOR_KEY_PREFIX):].split("-", 1)[0]
            username = base64.urlsafe_b64decode(name + "=" * (-len(name) % 4)).decode()
            digest = hashlib.sha256(key.encode()).digest()
        except (ValueError, UnicodeError):
            return None
        with self._lock:
            user = self._users.get(username)
        stored = getattr(user, "connector_key_hash", b"") if user else b""
        if not stored or not hmac.compare_digest(stored, digest):
            return None
        return _public(user)

    def verify(self, username: str, password: str) -> Optional[PublicUser]:
        username = username.strip().lower()
        with self._lock:
            user = self._users.get(username)
        if not user:
            hmac.compare_digest(_hash_password(password, _DUMMY_SALT), _DUMMY_HASH)
            return None
        candidate = _hash_password(password, user.salt)
        if not hmac.compare_digest(candidate, user.password_hash):
            return None
        return _public(user)

    def get(self, username: str) -> Optional[PublicUser]:
        with self._lock:
            user = self._users.get(username.strip().lower())
        return _public(user) if user else None

    def list(self) -> list[PublicUser]:
        with self._lock:
            out = [_public(u) for u in self._users.values()]
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
