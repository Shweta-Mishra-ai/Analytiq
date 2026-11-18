"""
services/limits.py — the limits that keep one client from taking the
service down for all of them.

Measured on this app: the process idles at about 240 MB; a few PDF reports
at once take it to 700 MB; a 59 MB upload peaks at 1 GB. None of that was
bounded, so a handful of clients pressing "Generate report" together, or
one large upload, could run a small server out of memory and kill every
session. Three limits:

  - heavy jobs (uploads, reports, model training) run a few at a time and
    the rest wait their turn, then get a clear "busy" answer;
  - each client has a storage quota and a dataset count, and uploads stop
    before the disk is full rather than after;
  - each caller has a request rate.

All are settings (see config.py) so they can follow the size of the server.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager

from fastapi import HTTPException

from app.config import config

logger = logging.getLogger(__name__)


# ── heavy jobs ───────────────────────────────────────────

class JobGate:
    def __init__(self, slots: int, wait_sec: float):
        self.slots = max(1, int(slots))
        self.wait_sec = wait_sec
        self._sem = threading.BoundedSemaphore(self.slots)
        self._lock = threading.Lock()
        self.running = 0
        self.waiting = 0

    @contextmanager
    def job(self, what: str = "request"):
        with self._lock:
            self.waiting += 1
        got = self._sem.acquire(timeout=self.wait_sec)
        with self._lock:
            self.waiting -= 1
            if got:
                self.running += 1
        if not got:
            logger.warning("heavy-job queue full: refused a %s after %ss", what, self.wait_sec)
            raise HTTPException(
                503, "The server is busy with other analyses. Your {} was not started; "
                     "please try again in a minute.".format(what),
                headers={"Retry-After": "60"})
        try:
            yield
        finally:
            with self._lock:
                self.running -= 1
            self._sem.release()

    def stats(self) -> dict:
        with self._lock:
            return {"slots": self.slots, "running": self.running, "waiting": self.waiting}


job_gate = JobGate(config.max_heavy_jobs, config.heavy_job_wait_sec)


def heavy_job(what: str = "request"):
    """`with heavy_job("report"):` around work that takes real memory."""
    return job_gate.job(what)


# ── storage ──────────────────────────────────────────────

def _dir_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def owner_usage(owner: str) -> dict:
    """What a client stores: bytes on disk and number of datasets."""
    from app.services.dataset_store import store
    ds_dir = os.path.join(store.base_dir, store._safe(owner))
    rag_dir = os.path.join(config.data_dir, "rag", store._safe(owner))
    used = (_dir_bytes(ds_dir) if os.path.isdir(ds_dir) else 0) + \
           (_dir_bytes(rag_dir) if os.path.isdir(rag_dir) else 0)
    n = len([d for d in os.listdir(ds_dir)]) if os.path.isdir(ds_dir) else 0
    return {"used_mb": round(used / 2**20, 1), "datasets": n,
            "quota_mb": config.max_storage_mb_per_owner,
            "max_datasets": config.max_datasets_per_owner}


def check_storage(owner: str, incoming_bytes: int) -> None:
    """Refuse an upload that would pass the client's quota or leave the
    disk nearly full. A stored dataset takes about twice its file size
    (the original and the working copy)."""
    usage = owner_usage(owner)
    # "local" is the single user of an install on their own machine (open
    # mode, or the local MCP server): their disk, no per-client quota.
    personal = owner == "local"
    if not personal and config.max_datasets_per_owner \
            and usage["datasets"] >= config.max_datasets_per_owner:
        raise HTTPException(
            413, "You have {} datasets, the most this account can hold. Delete one "
                 "you no longer need, then upload again.".format(usage["datasets"]))
    need_mb = incoming_bytes * 2 / 2**20
    if not personal and config.max_storage_mb_per_owner and \
            usage["used_mb"] + need_mb > config.max_storage_mb_per_owner:
        raise HTTPException(
            413, "This upload would pass your storage limit ({:.0f} MB used of {} MB). "
                 "Delete datasets you no longer need, then upload again.".format(
                     usage["used_mb"], config.max_storage_mb_per_owner))
    try:
        free_mb = shutil.disk_usage(config.data_dir).free / 2**20
    except OSError:
        return
    if free_mb - need_mb < config.min_free_disk_mb:
        logger.error("disk nearly full: %.0f MB free, upload of %.0f MB refused",
                     free_mb, need_mb)
        raise HTTPException(
            507, "The server is out of storage space, so uploads are paused. "
                 "Please contact support.")


# ── request rate ─────────────────────────────────────────

class RateLimiter:
    """Requests per caller in a sliding minute, in memory (one process)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> int:
        """0 when allowed, else seconds until the caller may try again."""
        if self.per_minute <= 0:
            return 0
        now = time.time()
        with self._lock:
            hits = [t for t in self._hits.get(key, ()) if now - t < 60]
            if len(hits) >= self.per_minute:
                self._hits[key] = hits
                return max(1, int(60 - (now - hits[0])))
            hits.append(now)
            self._hits[key] = hits
            if len(self._hits) > 20_000:
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > 60]:
                    del self._hits[k]
            return 0


rate_limiter = RateLimiter(config.rate_limit_per_minute)
