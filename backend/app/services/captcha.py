"""
services/captcha.py — Cloudflare Turnstile on the sign-up form.

Free accounts are worth something (free daily credits), so a script that
opens them by the hundred is worth stopping. Turnstile checks the browser
without a puzzle for most people. On with TURNSTILE_SITE_KEY and
TURNSTILE_SECRET; off otherwise.
"""
from __future__ import annotations

import logging

import httpx
from fastapi import HTTPException

from app.config import config

logger = logging.getLogger(__name__)

VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def enabled() -> bool:
    return bool(config.turnstile_site_key and config.turnstile_secret)


def _post(data: dict) -> dict:
    """Ask Cloudflare (the tests replace this)."""
    return httpx.post(VERIFY_URL, data=data, timeout=10).json()


def check(token: str, ip: str) -> None:
    """Raise 400 unless the sign-up form's Turnstile token is valid."""
    if not enabled():
        return
    if not token:
        raise HTTPException(400, "Please complete the check that you are not a robot.")
    try:
        result = _post({"secret": config.turnstile_secret, "response": token, "remoteip": ip})
    except (httpx.HTTPError, ValueError) as e:
        # Failing open would let a script through whenever Cloudflare is
        # slow; failing closed costs a person one retry.
        logger.warning("Turnstile unreachable: %s", e)
        raise HTTPException(503, "The robot check could not be completed. Please try again.")
    if not result.get("success"):
        raise HTTPException(400, "The robot check failed. Please try again.")
