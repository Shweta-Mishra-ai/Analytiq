"""
services/tokens.py — lightweight signed bearer tokens for per-user auth.

Deliberately not JWT (no extra dependency): a small HMAC-SHA256-signed
payload is all this needs — one claim (username) and an expiry. Anyone
holding APP_SECRET can forge tokens, so it's generated and persisted to
disk the same way a real secret would be, and never logged or returned
by any endpoint.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

from app.config import config

_secret_cache: Optional[bytes] = None


def _get_secret() -> bytes:
    """APP_SECRET env var if set; otherwise a persisted random secret in
    DATA_DIR so tokens survive restarts without requiring config."""
    global _secret_cache
    if config.app_secret:
        return config.app_secret.encode("utf-8")
    if _secret_cache is not None:
        return _secret_cache
    path = os.path.join(config.data_dir, ".secret_key")
    os.makedirs(config.data_dir, exist_ok=True)
    if os.path.exists(path):
        with open(path, "rb") as f:
            _secret_cache = f.read()
    else:
        _secret_cache = os.urandom(32)
        with open(path, "wb") as f:
            f.write(_secret_cache)
        os.chmod(path, 0o600)
    return _secret_cache


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def issue_token(username: str, ttl_days: Optional[int] = None) -> str:
    ttl = ttl_days if ttl_days is not None else config.token_ttl_days
    payload = json.dumps({
        "u": username,
        "exp": time.time() + ttl * 86400,
    }, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url(payload)
    sig = hmac.new(_get_secret(), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_token(token: str) -> Optional[str]:
    """Returns the username if the token is valid and unexpired, else None."""
    if not token or "." not in token:
        return None
    payload_b64, _, sig = token.partition(".")
    expected = hmac.new(_get_secret(), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    username = payload.get("u")
    return username if isinstance(username, str) and username else None
