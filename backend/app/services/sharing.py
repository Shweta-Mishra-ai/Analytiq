"""
services/sharing.py — sending someone a report without giving them an
account.

The product could produce a report and could not deliver one. A client
who wanted to show a finding to their CFO had to download a file and
attach it to an email, which means the CFO gets a PDF with no provenance
and the client's own audit trail has a hole in it where the most
important step happened.

A share link is a genuine widening of access, so every choice here is
about keeping it narrow:

**Signed, not guessed.** The link carries an HMAC over exactly what it
grants — one artifact, one owner, one expiry. Nothing in it is trusted;
a link that does not verify is refused without a database lookup, so a
flood of forged links costs no I/O.

**It expires, and the expiry is inside the signature.** A recipient
cannot extend their own access by editing a query parameter, because the
expiry is part of what was signed.

**One artifact, never a bearer token.** The link grants exactly the one
file it names. It is not a credential: it cannot list, cannot read a
dataset, cannot be traded for a session. If it leaks, the loss is that
report — which is the smallest a leak of a shareable link can be.

**Revocable.** A link can be withdrawn before it expires, because "we
sent it to the wrong address" happens and "wait three days" is not an
answer. Revocation is by artifact, recorded on the artifact itself.

**Recorded.** Creating a link and following one are both audit events.
Who a report was shared with is exactly the kind of question a client's
compliance team asks, and "we don't log that" is the wrong answer.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# A share link is a wider grant than a session, so it lives a shorter
# life by default. Seven days covers "look at this on Monday" without
# covering "still works next quarter".
DEFAULT_TTL_DAYS = 7
MAX_TTL_DAYS = 90


class ShareError(Exception):
    """Refusal written for whoever followed the link."""


def _secret() -> bytes:
    # The same persisted secret the session tokens use: one secret to
    # protect, and rotating it invalidates both, which is the correct
    # blast radius for "the signing key leaked".
    from app.services.tokens import _get_secret
    return _get_secret()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def mint(owner: str, artifact_id: str, ttl_days: int = DEFAULT_TTL_DAYS) -> Tuple[str, float]:
    """A share token for one artifact. Returns (token, expires_at)."""
    ttl = max(1, min(int(ttl_days or DEFAULT_TTL_DAYS), MAX_TTL_DAYS))
    expires = time.time() + ttl * 86400
    payload = json.dumps({"o": owner, "a": artifact_id, "exp": expires},
                         separators=(",", ":")).encode("utf-8")
    body = _b64(payload)
    sig = hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).hexdigest()
    return "{}.{}".format(body, sig), expires


def read(token: str) -> Tuple[str, str]:
    """(owner, artifact_id) for a valid token, or raise ShareError.

    The signature is checked before anything is parsed, so a malformed
    or forged token costs one HMAC and no disk access.
    """
    if not token or "." not in token:
        raise ShareError("That link is not valid.")
    body, _, sig = token.partition(".")
    # The token arrives as a URL path segment, so it can be any text at
    # all. A mint always produces base64url; anything that is not even
    # ASCII is refused here rather than raising out of .encode() and
    # reaching whoever followed the link as a 500 on a public route.
    try:
        signed = body.encode("ascii")
    except UnicodeEncodeError:
        raise ShareError("That link is not valid.") from None
    expected = hmac.new(_secret(), signed, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        raise ShareError("That link is not valid.")

    try:
        payload = json.loads(_unb64(body))
    except Exception:
        raise ShareError("That link is not valid.") from None

    if payload.get("exp", 0) < time.time():
        raise ShareError(
            "That link has expired. Ask whoever sent it for a new one — "
            "links are short-lived on purpose, so a report cannot keep "
            "circulating long after the data behind it moved on.")

    owner, artifact_id = payload.get("o"), payload.get("a")
    if not owner or not artifact_id:
        raise ShareError("That link is not valid.")
    return str(owner), str(artifact_id)


def revoke(owner: str, artifact_id: str) -> bool:
    """Withdraw every link to an artifact before it expires."""
    from app.services.artifacts import store as artifacts

    art = artifacts.get(owner, artifact_id)
    if art is None:
        return False
    art.meta["shares_revoked_at"] = time.time()
    _persist(art)
    logger.info("share links for artifact %s revoked by %s",
                artifact_id, owner)
    return True


def is_revoked(owner: str, artifact_id: str) -> bool:
    from app.services.artifacts import store as artifacts
    art = artifacts.get(owner, artifact_id)
    return bool(art and art.meta.get("shares_revoked_at"))


def record_share(owner: str, artifact_id: str, expires_at: float,
                 note: str = "") -> None:
    """Note on the artifact that a link exists, and when it dies."""
    from app.services.artifacts import store as artifacts

    art = artifacts.get(owner, artifact_id)
    if art is None:
        return
    shares = art.meta.setdefault("shares", [])
    shares.append({"created_at": time.time(), "expires_at": expires_at,
                   "note": note[:120]})
    # Minting a new link is also the answer to "we revoked it by
    # mistake" — the newest link wins over an older revocation.
    art.meta.pop("shares_revoked_at", None)
    _persist(art)


def record_access(owner: str, artifact_id: str) -> None:
    """Count a follow. Who saw the report is a question clients ask."""
    from app.services.artifacts import store as artifacts

    art = artifacts.get(owner, artifact_id)
    if art is None:
        return
    art.meta["share_views"] = int(art.meta.get("share_views", 0)) + 1
    art.meta["last_viewed_at"] = time.time()
    _persist(art)


def _persist(art) -> None:
    from dataclasses import asdict

    from app.services.artifacts import _atomic_write
    from app.services.artifacts import store as artifacts

    _atomic_write(artifacts._meta_path(art.owner, art.artifact_id),
                  json.dumps(asdict(art)).encode("utf-8"))
