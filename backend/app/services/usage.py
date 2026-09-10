"""
services/usage.py — what an account has actually used.

Ceilings are only real if something counts. This is that counter: a
small, disk-backed tally per account, keyed by calendar month for the
work-shaped limits and read live from the dataset store for the
storage-shaped ones.

Two deliberate choices.

Storage counts are *not* stored here. How many datasets an account has
is already a fact the dataset store knows, and a second copy of it would
drift the first time a cleanup ran or an upload half-failed. A number
that can disagree with itself is worse than a slower one.

Work counts are stored, because they are events rather than state: a
report built last Tuesday leaves nothing behind to count. They are kept
per calendar month, which is what a subscription renews on, and old
months are dropped on write so the file cannot grow forever.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from app.config import config

logger = logging.getLogger(__name__)

# The counters an account accrues over a month.
METERED_EVENTS = ("reports", "models")

# How many months of history to keep. Two: the current one, and the
# previous one so a renewal boundary can be reasoned about.
KEEP_MONTHS = 2


def current_period() -> str:
    """The billing period a request falls in, as YYYY-MM (UTC).

    UTC rather than local time so the boundary is the same for every
    account, and the same in a test as on a server in another zone.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m")


class UsageStore:
    """Thread-safe, disk-backed monthly tallies."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(config.data_dir, "usage.json")
        self._lock = threading.RLock()
        self._data: Dict[str, Dict[str, Dict[str, int]]] = self._load()

    # ── persistence ───────────────────────────────────────

    def _load(self) -> Dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            # Unlike the account file, losing this one is survivable: it
            # means an account's month resets, which is generous rather
            # than dangerous. Refusing to start over a counter would be
            # the worse trade.
            logger.warning("could not read usage at %s — starting empty",
                           self.path, exc_info=True)
            return {}

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)
        except Exception:
            logger.warning("could not write usage to %s", self.path,
                           exc_info=True)

    def _prune(self) -> None:
        """Drop months nobody will ask about again."""
        keep = sorted({p for acct in self._data.values() for p in acct},
                      reverse=True)[:KEEP_MONTHS]
        if not keep:
            return
        for account in list(self._data):
            self._data[account] = {p: v for p, v in self._data[account].items()
                                   if p in keep}
            if not self._data[account]:
                del self._data[account]

    # ── the API ───────────────────────────────────────────

    def record(self, owner: str, event: str, n: int = 1) -> int:
        """Count `n` of `event` against this account. Returns the new total."""
        owner = str(owner or "").strip().lower()
        if not owner or event not in METERED_EVENTS:
            return 0
        period = current_period()
        with self._lock:
            months = self._data.setdefault(owner, {})
            counts = months.setdefault(period, {})
            counts[event] = int(counts.get(event, 0)) + int(n)
            total = counts[event]
            self._prune()
            self._save()
        return total

    def used(self, owner: str, event: str) -> int:
        """How many of `event` this account has used this period."""
        owner = str(owner or "").strip().lower()
        with self._lock:
            return int(self._data.get(owner, {})
                       .get(current_period(), {}).get(event, 0))

    def snapshot(self, owner: str) -> Dict[str, int]:
        """Every metered count for this account, this period."""
        owner = str(owner or "").strip().lower()
        with self._lock:
            counts = self._data.get(owner, {}).get(current_period(), {})
            return {event: int(counts.get(event, 0))
                    for event in METERED_EVENTS}

    def reset(self, owner: str = "") -> None:
        """Clear one account's counters, or every account's.

        Exists for the tests and for an operator correcting a miscount;
        nothing in a request path calls it.
        """
        with self._lock:
            if owner:
                self._data.pop(str(owner).strip().lower(), None)
            else:
                self._data = {}
            self._save()


usage_store = UsageStore()
