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
from dataclasses import dataclass
from typing import Optional

from app.config import config

logger = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 200_000

# Short enough not to drive people to a sticky note, long
# enough that the PBKDF2 work above is not the only defence.
MIN_PASSWORD_LENGTH = 8


@dataclass
class User:
    username: str
    salt: bytes
    password_hash: bytes
    created_at: float
    is_admin: bool = False
    # Added when accounts became self-service. Every field below has a
    # default because accounts created before them exist on disk, and a
    # deploy that cannot read its own users.json locks everyone out.
    email: str = ""
    plan: str = ""
    plan_since: float = 0.0
    # Set by the billing layer so a webhook can find the account again.
    billing_customer_id: str = ""
    billing_subscription_id: str = ""


@dataclass
class PublicUser:
    """Safe-to-return view of a User — never includes the password hash."""
    username: str
    created_at: float
    is_admin: bool = False
    email: str = ""
    plan: str = ""
    plan_since: float = 0.0
    billing_customer_id: str = ""


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
                    # .get with a default throughout: an account file
                    # written before plans existed is still a valid
                    # account file, and a KeyError here would lock every
                    # existing client out on the deploy that added them.
                    email=str(rec.get("email", "") or ""),
                    plan=str(rec.get("plan", "") or ""),
                    plan_since=float(rec.get("plan_since", 0.0) or 0.0),
                    billing_customer_id=str(
                        rec.get("billing_customer_id", "") or ""),
                    billing_subscription_id=str(
                        rec.get("billing_subscription_id", "") or ""),
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
                "email": u.email,
                "plan": u.plan,
                "plan_since": u.plan_since,
                "billing_customer_id": u.billing_customer_id,
                "billing_subscription_id": u.billing_subscription_id,
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
               is_admin: bool = False, email: str = "",
               plan: str = "") -> PublicUser:
        username = username.strip().lower()
        if not username or not password:
            raise ValueError("Username and password are required")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError("Password must be at least {} characters"
                             .format(MIN_PASSWORD_LENGTH))
        email = str(email or "").strip().lower()
        with self._lock:
            if username in self._users:
                raise ValueError(f"User '{username}' already exists")
            # One account per address, so a password reset addresses one
            # account and "you already have an account" is answerable.
            if email and any(u.email == email for u in self._users.values()):
                raise ValueError("That email address already has an account")
            salt = secrets.token_bytes(16)
            now = time.time()
            user = User(
                username=username,
                salt=salt,
                password_hash=_hash_password(password, salt),
                created_at=now,
                is_admin=is_admin,
                email=email,
                plan=plan or "",
                plan_since=now if plan else 0.0,
            )
            self._users[username] = user
            self._save()
        return self._public(user)

    def _public(self, user: User) -> PublicUser:
        return PublicUser(
            username=user.username, created_at=user.created_at,
            is_admin=user.is_admin, email=user.email, plan=user.plan,
            plan_since=user.plan_since,
            billing_customer_id=user.billing_customer_id,
        )

    def set_plan(self, username: str, plan: str, *,
                 customer_id: str = "", subscription_id: str = ""
                 ) -> Optional[PublicUser]:
        """Move an account onto a plan. Used by billing and by an admin."""
        username = str(username or "").strip().lower()
        with self._lock:
            user = self._users.get(username)
            if user is None:
                return None
            user.plan = str(plan or "").strip().lower()
            user.plan_since = time.time()
            if customer_id:
                user.billing_customer_id = customer_id
            # An empty subscription id is how a cancellation clears it,
            # so this one is assigned rather than guarded.
            user.billing_subscription_id = subscription_id or ""
            self._save()
            return self._public(user)

    def by_email(self, email: str) -> Optional[PublicUser]:
        email = str(email or "").strip().lower()
        if not email:
            return None
        with self._lock:
            for user in self._users.values():
                if user.email == email:
                    return self._public(user)
        return None

    def by_customer_id(self, customer_id: str) -> Optional[PublicUser]:
        """The account a billing webhook is talking about."""
        customer_id = str(customer_id or "").strip()
        if not customer_id:
            return None
        with self._lock:
            for user in self._users.values():
                if user.billing_customer_id == customer_id:
                    return self._public(user)
        return None

    def set_password(self, username: str, password: str) -> bool:
        username = str(username or "").strip().lower()
        if len(password or "") < MIN_PASSWORD_LENGTH:
            raise ValueError("Password must be at least {} characters"
                             .format(MIN_PASSWORD_LENGTH))
        with self._lock:
            user = self._users.get(username)
            if user is None:
                return False
            # A new salt as well as a new hash: reusing the old one lets
            # anyone holding the previous file tell that the password
            # changed but the salt did not.
            user.salt = secrets.token_bytes(16)
            user.password_hash = _hash_password(password, user.salt)
            self._save()
            return True

    def verify(self, username: str, password: str) -> Optional[PublicUser]:
        username = username.strip().lower()
        with self._lock:
            user = self._users.get(username)
        if not user:
            return None
        candidate = _hash_password(password, user.salt)
        if not hmac.compare_digest(candidate, user.password_hash):
            return None
        return self._public(user)

    def get(self, username: str) -> Optional[PublicUser]:
        with self._lock:
            user = self._users.get(str(username or "").strip().lower())
        return self._public(user) if user else None

    def list(self) -> list[PublicUser]:
        with self._lock:
            out = [self._public(u) for u in self._users.values()]
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
