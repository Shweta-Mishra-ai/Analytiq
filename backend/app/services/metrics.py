"""
services/metrics.py — what the server is actually doing, in numbers.

`/api/health` answers "is the process up", which is the one question that
is never in doubt when someone reports that reports are slow or that a
section has gone missing. This records the three things that go wrong in
practice and that nothing else can reconstruct after the fact:

  * how long a report actually took to build, and which format
  * which engine failed, how often, and with what error
  * how much of the LLM spend the narrative cache is avoiding

Everything is in-process and bounded — a fixed number of timing samples
per operation, a fixed number of distinct failure keys. There is no
external metrics backend to configure and nothing to scrape; a restart
starts the counters over, which is the right trade for a single-box
deployment. If this ever runs multi-process, these numbers become
per-worker and want a real collector behind them.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict

logger = logging.getLogger(__name__)

# Enough samples for a stable median and p95 without the memory of a
# full history: 200 durations is a few kilobytes per operation.
_MAX_SAMPLES = 200
_MAX_KEYS = 200


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: Dict[str, Deque[float]] = {}
        self._counts: Dict[str, int] = {}
        self._failures: Dict[str, int] = {}
        self._last_error: Dict[str, str] = {}
        self.started_at = time.time()

    # ── recording ────────────────────────────────────────
    def record_duration(self, operation: str, seconds: float) -> None:
        with self._lock:
            if operation not in self._durations and len(self._durations) >= _MAX_KEYS:
                return
            samples = self._durations.setdefault(operation, deque(maxlen=_MAX_SAMPLES))
            samples.append(float(seconds))
            self._counts[operation] = self._counts.get(operation, 0) + 1

    def record_failure(self, component: str, error: BaseException | str) -> None:
        """A named component failed. The message is kept so the endpoint
        can show what went wrong without a log dive; only the last one
        per component, so a repeated failure cannot grow without bound."""
        text = f"{type(error).__name__}: {error}" if isinstance(
            error, BaseException) else str(error)
        with self._lock:
            if component not in self._failures and len(self._failures) >= _MAX_KEYS:
                return
            self._failures[component] = self._failures.get(component, 0) + 1
            self._last_error[component] = text[:300]

    @contextmanager
    def timed(self, operation: str):
        """Time a block, and count a failure if it raises. The exception
        is re-raised untouched — this observes, it does not handle."""
        start = time.perf_counter()
        try:
            yield
        except BaseException as exc:
            self.record_failure(operation, exc)
            raise
        finally:
            self.record_duration(operation, time.perf_counter() - start)

    # ── reporting ────────────────────────────────────────
    @staticmethod
    def _percentile(sorted_samples: list[float], pct: float) -> float:
        if not sorted_samples:
            return 0.0
        # Nearest-rank. With 200 samples the difference from an
        # interpolated percentile is far below the noise in a wall clock.
        idx = min(len(sorted_samples) - 1,
                  max(0, int(round(pct / 100 * len(sorted_samples))) - 1))
        return sorted_samples[idx]

    def snapshot(self) -> dict:
        with self._lock:
            durations = {k: list(v) for k, v in self._durations.items()}
            counts = dict(self._counts)
            failures = dict(self._failures)
            last_error = dict(self._last_error)

        operations = {}
        for name, samples in durations.items():
            ordered = sorted(samples)
            operations[name] = {
                "count": counts.get(name, len(samples)),
                "median_sec": round(self._percentile(ordered, 50), 3),
                "p95_sec": round(self._percentile(ordered, 95), 3),
                "max_sec": round(ordered[-1], 3) if ordered else 0.0,
                "samples": len(ordered),
            }

        from app.services.llm_cache import llm_cache
        cache = llm_cache.stats()
        # A cached narrative is a call that was not billed. This is the
        # only number in here that maps to money.
        cache["calls_avoided"] = cache["hits"]

        return {
            "uptime_sec": round(time.time() - self.started_at, 1),
            "operations": operations,
            "failures": {
                name: {"count": n, "last_error": last_error.get(name, "")}
                for name, n in sorted(failures.items(),
                                      key=lambda kv: -kv[1])
            },
            "llm_cache": cache,
        }

    def reset(self) -> None:
        with self._lock:
            self._durations.clear()
            self._counts.clear()
            self._failures.clear()
            self._last_error.clear()
            self.started_at = time.time()


metrics = Metrics()
