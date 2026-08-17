"""
services/load_guard.py — how many heavy analyses may run at once.

Measured on this machine, one ML pipeline takes about ten seconds and
250MB; a full EDA takes three. Both are dispatched to Starlette's
threadpool, which is unbounded from the application's point of view — so
ten people pressing "train" together produce ten concurrent trainings,
about 2.5GB of resident memory and a hundred seconds of CPU. On a small
container that is an out-of-memory kill, and an OOM kill takes down
every request in flight, not just the ones that caused it.

A queue that admits a bounded number and turns the rest away with a
clear message is the honest behaviour: the person who is refused knows
to try again in a moment, and the people already running finish. The
alternative — accept everything and fall over — fails the requests that
were already nearly done.

Deliberately a semaphore per process rather than a distributed one. This
is a guard against a single container exhausting itself, not a
cluster-wide scheduler; a bigger deployment wants a real queue, and this
should not pretend to be one.
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

from app.config import config

logger = logging.getLogger(__name__)


class Busy(RuntimeError):
    """Raised when the process is already running its limit of heavy work."""


class _Guard:
    """A counting gate with no queueing — admit or refuse, immediately.

    Blocking the caller until a slot frees would hold an HTTP connection
    open for however long the running jobs take, which reads to the user
    as the app having hung. Refusing straight away with a wait time is
    something they can act on.
    """

    def __init__(self, limit: int, label: str):
        self.limit = max(1, int(limit))
        self.label = label
        self._lock = threading.Lock()
        self._running = 0

    @property
    def running(self) -> int:
        with self._lock:
            return self._running

    @contextmanager
    def slot(self):
        with self._lock:
            if self._running >= self.limit:
                logger.info("%s at capacity (%d running)", self.label,
                            self._running)
                raise Busy(
                    "The server is running its limit of {} concurrent {} "
                    "jobs. Nothing has been lost — try again in a minute."
                    .format(self.limit, self.label))
            self._running += 1
        try:
            yield
        finally:
            with self._lock:
                self._running -= 1


# Model training is the expensive one: several estimators, cross
# validation, and a copy of the feature matrix per fit.
TRAINING = _Guard(getattr(config, "max_concurrent_training", 2), "model training")
# Profiling, EDA, BI and report rendering — cheaper each, but a report
# build holds a full matplotlib figure set in memory.
ANALYSIS = _Guard(getattr(config, "max_concurrent_analysis", 6), "analysis")


@contextmanager
def http_slot(guard: _Guard, retry_after: int = 30):
    """A slot, with a refusal expressed the way HTTP expresses one.

    Every route that guards heavy work needs the same four lines, and
    the first version of them was written out by hand in two places and
    forgotten in five others — profiling, EDA, BI, the story engine and
    the advanced analytics all ran unbounded while report rendering was
    carefully limited. One helper is harder to forget than a pattern.
    """
    from fastapi import HTTPException

    try:
        with guard.slot():
            yield
    except Busy as exc:
        raise HTTPException(503, str(exc),
                            headers={"Retry-After": str(retry_after)})


def snapshot() -> dict:
    """What is running now — for the health endpoint."""
    return {
        "training": {"running": TRAINING.running, "limit": TRAINING.limit},
        "analysis": {"running": ANALYSIS.running, "limit": ANALYSIS.limit},
    }
