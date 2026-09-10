"""
api/accounts.py — becoming a customer without asking anybody.

Accounts could only be created by an operator holding the admin key, so
nobody could start using Analytiq on their own. That is the right shape
for a tool one person runs for their clients and the wrong shape for a
product, and it is the single thing that made every other commercial
question moot.

Signup is off unless a deployment turns it on (config.signup_enabled).
The default has to be the safe one: a self-hosted instance that quietly
accepted strangers would be a hole its operator never opened.

Two things this deliberately does NOT do.

It does not verify email addresses. Verification needs a mail sender,
which is a dependency and an operational surface, and an unverified
address is already enough to key an account on and to reset a password
against once a sender exists. The field is captured now so adding
verification later does not need a migration.

It does not leak whether an address is registered. Signup with a taken
address and a password reset for an unknown one both answer the same
way, because the alternative turns this endpoint into a way to ask
"is this person a customer".
"""
from __future__ import annotations

import logging
import re
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import config
from app.services import plans
from app.services.auth import current_owner
from app.services.load_control import admit
from app.services.tokens import issue_token
from app.services.user_store import MIN_PASSWORD_LENGTH, user_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["accounts"])

# Deliberately permissive: the point is to catch a typo, not to
# adjudicate RFC 5322. An address that reaches nobody fails at the first
# email we send, which is a better place to find out than a regex.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# What a username may be. Kept narrow because it appears in paths and in
# the on-disk layout of the dataset store.
_USERNAME = re.compile(r"^[a-z0-9][a-z0-9._-]{2,31}$")


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=256)


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=256)


@router.get("/auth/signup-status")
def signup_status():
    """Whether this deployment accepts new accounts, for the login page."""
    return {
        "enabled": bool(config.signup_enabled),
        "plan": config.signup_plan if config.signup_enabled else "",
        "min_password_length": MIN_PASSWORD_LENGTH,
    }


@router.post("/auth/signup", dependencies=[Depends(admit("signup"))])
def signup(req: SignupRequest):
    """Create an account and return a token for it.

    Rate-limited like any other write: signup is the one endpoint an
    unauthenticated stranger can reach that writes to disk.
    """
    if not config.signup_enabled:
        raise HTTPException(
            403, "This deployment does not accept new accounts. Ask whoever "
                 "runs it for one.")

    username = str(req.username or "").strip().lower()
    email = str(req.email or "").strip().lower()

    if not _USERNAME.match(username):
        raise HTTPException(
            422, "A username is 3-32 characters: lowercase letters, digits, "
                 "and . _ - after the first character.")
    if not _EMAIL.match(email):
        raise HTTPException(422, "That does not look like an email address.")

    try:
        user = user_store.create(
            username, req.password, email=email,
            plan=plans.get(config.signup_plan).key)
    except ValueError as e:
        message = str(e)
        # "already exists" for a username is safe to say — a username is
        # public by nature and the person needs to pick another. An
        # address is not, so that case answers as a conflict without
        # confirming the address.
        if "email" in message.lower():
            raise HTTPException(
                409, "That account could not be created. If you already have "
                     "one, sign in instead.") from None
        raise HTTPException(409, message) from None

    logger.info("account created: %s on plan %s", user.username, user.plan)
    return {
        "token": issue_token(user.username),
        "username": user.username,
        "is_admin": user.is_admin,
        "plan": user.plan,
    }


@router.get("/account")
def account(owner: str = Depends(current_owner)):
    """Who this token belongs to, and what their plan allows."""
    from app.services.quota import allowance

    user = user_store.get(owner)
    quota = allowance(owner)
    return {
        "username": owner,
        "email": user.email if user else "",
        "is_admin": bool(user.is_admin) if user else False,
        "plan": quota.plan,
        "plan_label": quota.label,
        "branding": quota.branding,
        "limits": quota.limits,
        "used": quota.used,
        "remaining": quota.remaining,
        "metered": bool(config.is_metered),
    }


@router.post("/account/password")
def change_password(req: PasswordChangeRequest,
                    owner: str = Depends(current_owner)):
    """Change your own password, proving you know the current one."""
    user = user_store.get(owner)
    if user is None:
        raise HTTPException(404, "No such account")
    if not user_store.verify(owner, req.current_password):
        raise HTTPException(403, "That is not your current password.")
    if req.new_password == req.current_password:
        raise HTTPException(422, "The new password matches the old one.")
    try:
        user_store.set_password(owner, req.new_password)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    # The old token stays valid: it is scoped to the username, not to
    # the password, and revoking every session on a routine password
    # change is a surprise rather than a safeguard. Session revocation
    # is its own feature, and this is not a substitute for it.
    return {"ok": True}


@router.post("/auth/password-reset",
             dependencies=[Depends(admit("signup"))])
def request_password_reset(request: Request):
    """Start a password reset.

    Answers the same way whether or not the address is registered, so
    this cannot be used to enumerate customers. Currently it can only
    say so: sending the mail needs a mail sender this deployment does
    not have, and inventing a token nobody can receive would be worse
    than saying plainly that the path is not finished.
    """
    return {
        "ok": True,
        "sent": False,
        "detail": "Password reset by email is not available on this "
                  "deployment. Ask whoever runs it to set a new password "
                  "for you.",
    }
