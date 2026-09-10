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

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import config
from app.services.tokens import verify_token
from app.services.user_store import user_store

logger = logging.getLogger(__name__)

# endpoints that must stay reachable without a token
PUBLIC_PATHS = {
    "/api/health",
    "/api/auth/login",
    # Reachable without a token by definition: the caller is asking to
    # become one of the accounts this middleware checks against.
    "/api/auth/signup",
    "/api/auth/signup-status",
    "/api/auth/password-reset",
    # The pricing table is public — it is what a signed-out visitor is
    # deciding on.
    "/api/billing/plans",
    # Stripe cannot hold a bearer token. This one authenticates by
    # verifying the signature on the payload instead, and refuses
    # outright when no webhook secret is configured — see
    # api/billing.py. Without this entry the middleware answered 401 and
    # no subscription would ever have been applied.
    "/api/billing/webhook",
}

# Prefixes that answer without a session. Exactly one so far: a share
# link, which carries its own signed grant for a single artifact. It is
# public by design — the recipient is the CFO who does not have an
# account, which is the entire point — and the token is the credential,
# checked by the route itself rather than here.
PUBLIC_PREFIXES = ("/api/shared/",)

# implicit owner used only in single-user open dev mode
LOCAL_OWNER = "local"


def _open_mode() -> bool:
    return not config.effective_admin_key and user_store.is_empty()


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (not path.startswith("/api")
                or path in PUBLIC_PATHS
                or path.startswith(PUBLIC_PREFIXES)
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
        # So every log line under this request says who it was for.
        from app.services.request_context import set_account
        set_account(username)
        return await call_next(request)


def check_admin_key(candidate: str) -> bool:
    key = config.effective_admin_key
    return bool(key) and hmac.compare_digest(candidate, key)


def current_owner(request: Request) -> str:
    """FastAPI dependency: the authenticated client's username, used to
    scope every dataset/RAG lookup. Falls back to LOCAL_OWNER only because
    the middleware guarantees this is set on every non-public /api request
    (open mode included) before a route handler ever runs."""
    return getattr(request.state, "username", LOCAL_OWNER)
