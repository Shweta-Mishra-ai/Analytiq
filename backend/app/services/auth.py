"""
services/auth.py — multi-tenant auth.

Each client gets their own account (POST /api/admin/users, admin-key
protected) and logs in with username/password (POST /api/auth/login)
to receive a token scoped to only their own data — see
`request.state.username`, which every dataset/RAG store call is keyed
on for isolation.

Account management itself (/api/admin/*) is gated by a separate admin
key (APP_ADMIN_KEY, or the legacy APP_PASSWORD name) — a master key
held only by whoever runs the deployment, never by a client.

Unset admin key *and* zero accounts created → single-user open dev
mode (unchanged from the original single-workspace behavior), so local
development and the test suite need zero setup. The moment either an
admin key is set or an account exists, auth is enforced.
"""
from __future__ import annotations
import logging

import hmac
import threading
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import config
from app.services.tokens import verify_token
from app.services.user_store import user_store

logger = logging.getLogger(__name__)

# endpoints that must stay reachable without a token
PUBLIC_PATHS = {"/api/health", "/api/auth/login"}

# implicit owner used only in single-user open dev mode
LOCAL_OWNER = "local"


def _open_mode() -> bool:
    return not config.effective_admin_key and user_store.is_empty()


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (not path.startswith("/api")
                or path in PUBLIC_PATHS
                or request.method == "OPTIONS"):
            return await call_next(request)

        if _open_mode():
            request.state.username = LOCAL_OWNER
            request.state.is_admin = True
            return await call_next(request)

        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""

        if path.startswith("/api/admin"):
            if token and check_admin_key(token):
                request.state.username = "admin"
                request.state.is_admin = True
                return await call_next(request)
            # also allow a regular admin-flagged user account
            username = verify_token(token) if token else None
            user = user_store.get(username) if username else None
            if user and user.is_admin:
                request.state.username = user.username
                request.state.is_admin = True
                return await call_next(request)
            return JSONResponse(
                {"detail": "Admin key required"}, status_code=401)

        username = verify_token(token) if token else None
        if not username or not user_store.exists(username):
            return JSONResponse(
                {"detail": "Not authenticated"}, status_code=401)
        request.state.username = username
        request.state.is_admin = False
        return await call_next(request)


class LoginThrottle:
    """Refuses further attempts on a username after repeated failures.

    Without it nothing slows online password guessing: every attempt is
    answered as fast as PBKDF2 allows, indefinitely. Keyed on the username
    rather than the client address because behind a proxy every caller
    shares one address. In memory and per process — this deploys as a
    single uvicorn worker — so a restart clears it, which is acceptable
    for a throttle whose job is to make guessing slow, not impossible.
    """

    def __init__(self, max_failures: int = 10, window_sec: float = 900.0):
        self.max_failures = max_failures
        self.window_sec = window_sec
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> list[float]:
        recent = [t for t in self._failures.get(key, []) if now - t < self.window_sec]
        if recent:
            self._failures[key] = recent
        else:
            self._failures.pop(key, None)
        return recent

    def retry_after(self, username: str) -> int:
        """Seconds until `username` may try again; 0 when it may now."""
        key = username.strip().lower()
        now = time.time()
        with self._lock:
            recent = self._recent(key, now)
            if len(recent) < self.max_failures:
                return 0
            return max(1, int(self.window_sec - (now - recent[0])))

    def record_failure(self, username: str) -> None:
        key = username.strip().lower()
        now = time.time()
        with self._lock:
            self._recent(key, now)
            self._failures.setdefault(key, []).append(now)

    def record_success(self, username: str) -> None:
        with self._lock:
            self._failures.pop(username.strip().lower(), None)


login_throttle = LoginThrottle()


def check_admin_key(candidate: str) -> bool:
    # Compared as bytes: compare_digest on str raises TypeError for any
    # non-ASCII character, and the candidate is an arbitrary header value.
    key = config.effective_admin_key
    return bool(key) and hmac.compare_digest(
        candidate.encode("utf-8", "surrogateescape"),
        key.encode("utf-8", "surrogateescape"))


def current_owner(request: Request) -> str:
    """FastAPI dependency: the authenticated client's username, used to
    scope every dataset/RAG lookup. Falls back to LOCAL_OWNER only because
    the middleware guarantees this is set on every non-public /api request
    (open mode included) before a route handler ever runs."""
    return getattr(request.state, "username", LOCAL_OWNER)
