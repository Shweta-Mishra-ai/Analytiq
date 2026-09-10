"""
services/load_control.py — refusing work is better than dying under it.

Two limits, for two different failures that both look like "the app is
down" from outside.

**Concurrency.** Building a report is 3.0 seconds of CPU, measured on a
1,500-row dataset, and it runs on the request thread. Nothing bounded
how many of those could be in flight at once, so ten concurrent report
requests meant thirty CPU-seconds of work queued ahead of everything
else — including the platform's health check, which times out, and the
container is restarted mid-report for every user on it. The failure is
not that the eleventh report is slow; it is that the first ten took the
service with them.

A bounded number run at a time and the rest are told to come back, with
a Retry-After. A queue that grows without limit is not politeness, it is
the same crash with a delay in front of it.

**Rate.** A single account looping the report endpoint costs the same as
a hundred clients using it normally. The bucket here is per account and
per operation, so a client running reports cannot starve their own
uploads, and one account cannot starve another's.

Both limits live in this process. That is honest rather than ideal: on a
single-container deployment it is exactly right, and across several
replicas each holds its own share, so the effective limit is the
configured one times the replica count. Making it exact needs shared
state (Redis), which is a dependency this does not have and should not
grow until a deployment actually runs more than one replica.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Tuple

# Imported at module level, not inside admit(): this module uses
# `from __future__ import annotations`, so FastAPI resolves the
# `request: Request` hint as a STRING against this module's globals. A
# local import leaves it unresolvable, and FastAPI then treats it as a
# query parameter — every gated endpoint answered 422 "Field required:
# query.request" instead of running.
from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# How many pieces of expensive work run at once. Two is deliberate on a
# small container: the work is CPU-bound, so more in flight than cores
# makes every one of them slower without finishing any sooner.
DEFAULT_CONCURRENCY = 2

# How long a caller waits for a slot before being turned away. Long
# enough to ride out a burst, short enough that the client gets an
# answer rather than a socket that eventually times out.
SLOT_WAIT_SECONDS = 20.0


class Busy(Exception):
    """Raised when the server is at capacity. Carries a retry hint."""

    def __init__(self, retry_after: int = 30):
        self.retry_after = retry_after
        super().__init__(
            "The server is at capacity building other reports. Try again "
            "in about {} seconds — the work is queued behind a small "
            "number of slots on purpose, so that a busy moment slows "
            "requests down instead of taking the service offline."
            .format(retry_after))


class RateLimited(Exception):
    """Raised when one account has asked for too much, too quickly."""

    def __init__(self, retry_after: int, limit: int, window_s: int, what: str):
        self.retry_after = retry_after
        super().__init__(
            "That is more than {} {} requests in {} seconds for this "
            "account. Wait about {} seconds and try again."
            .format(limit, what, window_s, retry_after))


class _Gate:
    """A counting semaphore that refuses rather than queues forever."""

    def __init__(self, slots: int = DEFAULT_CONCURRENCY):
        self._sem = threading.BoundedSemaphore(slots)
        self.slots = slots

    def __enter__(self):
        if not self._sem.acquire(timeout=SLOT_WAIT_SECONDS):
            logger.warning("load shed: all %d slots busy for %.0fs",
                           self.slots, SLOT_WAIT_SECONDS)
            raise Busy()
        return self

    def __exit__(self, *exc):
        self._sem.release()
        return False


# One gate for the whole process. Report building, deck building and
# model training all draw on it, because they compete for the same CPU.
heavy = _Gate()


class _RateLimiter:
    """A sliding window per (account, operation)."""

    def __init__(self):
        self._hits: Dict[Tuple[str, str], Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, account: str, what: str, limit: int, window_s: int) -> None:
        now = time.monotonic()
        key = (account or "anonymous", what)
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > window_s:
                hits.popleft()
            if len(hits) >= limit:
                retry = int(window_s - (now - hits[0])) + 1
                logger.info("rate limit: %s hit %d %s requests in %ds",
                            account, limit, what, window_s)
                raise RateLimited(retry, limit, window_s, what)
            hits.append(now)
            # A caller who stopped calling should not be remembered
            # forever; the deque empties itself above, and an empty one
            # is dropped so the map does not grow with every account
            # that ever used the service.
            if not hits:
                self._hits.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = _RateLimiter()


# What each operation costs, and therefore how often one account may ask
# for it. Reports are seconds of CPU; a listing is a dictionary lookup.
LIMITS = {
    "report":  (10, 60),     # 10 per minute
    "train":   (10, 60),
    "upload":  (30, 60),
    "analyse": (120, 60),
}


def guard(account: str, what: str) -> None:
    """Rate-limit one operation for one account. Raises RateLimited."""
    limit, window = LIMITS.get(what, LIMITS["analyse"])
    limiter.check(account, what, limit, window)


# ── FastAPI wiring ────────────────────────────────────────

def admit(what: str, metered: str = ""):
    """A dependency that rate-limits, meters, and holds a heavy-work slot.

    Written as a generator dependency so the slot is released when the
    response is finished — including when the handler raises, which a
    manual acquire/release around a long function body gets wrong the
    first time somebody adds an early return.

        @router.post("/{ds_id}/pdf",
                     dependencies=[Depends(admit("report", "reports"))])

    `metered` names the monthly counter this endpoint spends, if any.
    The check happens before the work and the count only after it
    succeeded — billing a customer for a report that then failed to
    build is the kind of thing they remember. The generator learns
    whether the handler raised by watching for the exception FastAPI
    throws back in, which is the only signal available here.

    Rate limit and plan ceiling answer different questions and both
    belong: the limiter stops one account monopolising the box in the
    next sixty seconds, the ceiling is what the account bought.
    """
    def dependency(request: Request):
        from app.services.auth import current_owner

        owner = current_owner(request)
        try:
            guard(owner, what)
        except RateLimited as e:
            raise HTTPException(
                429, str(e),
                headers={"Retry-After": str(e.retry_after)}) from None

        if metered:
            from app.services.quota import QuotaExceeded, check_event
            try:
                check_event(owner, metered)
            except QuotaExceeded as e:
                # 402: the caller can act on this, and the action is a
                # payment. 429 would tell them to wait, which is wrong —
                # waiting a minute changes nothing about a monthly cap.
                raise HTTPException(402, e.as_detail()) from None

        try:
            heavy.__enter__()
        except Busy as e:
            raise HTTPException(
                503, str(e),
                headers={"Retry-After": str(e.retry_after)}) from None
        # The release stays in a `finally`. Narrowing it to `except
        # Exception` looks equivalent and is not: a client that
        # disconnects mid-report raises CancelledError, which derives
        # from BaseException, so the slot would never come back. Two of
        # those and every caller gets 503 until the process restarts —
        # which is the exact failure this module exists to prevent.
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            heavy.__exit__(None, None, None)

        # Only reached when the handler returned normally, so a report
        # that failed to build is never charged for.
        if metered and succeeded:
            from app.services.quota import record
            record(owner, metered)

    return dependency
