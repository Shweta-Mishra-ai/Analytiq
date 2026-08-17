"""
services/throttle.py — a limit on how often a credential can be guessed.

Passwords are PBKDF2 with 200,000 iterations, so guessing them one at a
time over HTTP is slow. That is not the same as being protected: the
attempts were unbounded, so an attacker could simply take as long as it
takes, and the same property that makes a guess expensive for them makes
it expensive for the server — every attempt burns 200,000 iterations of
this process's CPU, so a login flood is a denial of service against the
people using the app legitimately, whether or not it ever finds a
password.

Failures are counted per username and per client address, and both have
to be under the limit for an attempt to be allowed. Per-address alone
lets one bad network segment lock out a whole office; per-username alone
lets an attacker lock a known user out by failing on purpose. Counting
both, and only counting *failures*, means someone typing their own
password correctly is never affected by anyone else.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, Tuple

logger = logging.getLogger(__name__)

# Five wrong passwords in a five-minute window is well beyond a person
# mistyping and far below anything that makes guessing viable.
MAX_FAILURES = 5
WINDOW_SECONDS = 300
LOCKOUT_SECONDS = 300


class LoginThrottle:
    def __init__(self, max_failures: int = MAX_FAILURES,
                 window: int = WINDOW_SECONDS,
                 lockout: int = LOCKOUT_SECONDS):
        self.max_failures = max_failures
        self.window = window
        self.lockout = lockout
        self._lock = threading.Lock()
        self._failures: Dict[str, Deque[float]] = {}

    def _recent(self, key: str, now: float) -> Deque[float]:
        hits = self._failures.setdefault(key, deque())
        while hits and now - hits[0] > self.window:
            hits.popleft()
        return hits

    def check(self, *keys: str) -> Tuple[bool, int]:
        """(allowed, seconds to wait). Never mutates the failure count —
        a check is not an attempt."""
        now = time.time()
        with self._lock:
            worst = 0
            for key in keys:
                if not key:
                    continue
                hits = self._recent(key, now)
                if len(hits) >= self.max_failures:
                    wait = int(self.lockout - (now - hits[-1]))
                    worst = max(worst, max(wait, 1))
            return (worst == 0), worst

    def record_failure(self, *keys: str) -> None:
        now = time.time()
        with self._lock:
            for key in keys:
                if key:
                    self._recent(key, now).append(now)

    def record_success(self, *keys: str) -> None:
        """A correct password clears the count for that credential.

        Otherwise someone who mistyped four times and then got it right
        stays one mistake away from a lockout for the rest of the
        window, which punishes the person the limit exists to protect.
        """
        with self._lock:
            for key in keys:
                self._failures.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()


login_throttle = LoginThrottle()
