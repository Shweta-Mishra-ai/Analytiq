"""
services/auth.py — single-workspace bearer-token auth.

Set APP_PASSWORD in the environment and every /api request must carry
`Authorization: Bearer <APP_PASSWORD>`. Unset → open mode (local dev).

Honest scope: this is workspace protection (one shared secret), not
multi-user accounts. It keeps a deployed instance private to its owner.
Uses constant-time comparison to avoid timing attacks.
"""
from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import config

# endpoints that must stay reachable without a token
PUBLIC_PATHS = {"/api/health", "/api/auth/login"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        password = config.app_password
        path = request.url.path
        if (not password
                or not path.startswith("/api")
                or path in PUBLIC_PATHS
                or request.method == "OPTIONS"):
            return await call_next(request)

        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        if not token or not hmac.compare_digest(token, password):
            return JSONResponse(
                {"detail": "Not authenticated"}, status_code=401)
        return await call_next(request)


def check_password(candidate: str) -> bool:
    return bool(config.app_password) and hmac.compare_digest(
        candidate, config.app_password)
