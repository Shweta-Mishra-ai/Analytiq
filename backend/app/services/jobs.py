"""
services/jobs.py — work that outlives the request that asked for it.

A report is 3.0 seconds of CPU, measured. Holding that on the request
thread caps throughput at roughly one report per worker per three
seconds, and it is why the load gate added earlier has to refuse people:
there is nowhere to put work except in front of a waiting socket.

More than throughput, though, it is the shape that blocks two features
the product needs. A schedule has nobody to hold a socket open for it.
A share link needs something that already exists to link to. Both want
the same thing: a run that is recorded, does its work away from the
request, and leaves a result behind.

What this is not: a distributed queue. It is a bounded thread pool and a
JSON record per run, in one process. That is the honest size for a
single-container deployment, and the seam is drawn so that swapping the
executor for a real broker later touches this file and nothing else —
callers submit and poll, and never learn how the work was run.

Failure is a first-class outcome. A job that raised is a job with a
status of "failed" and the reason on it, not a log line and a request
that never returns. The request id of whoever submitted it is carried
onto the run, so a failure in a background thread can still be traced
back to the person who asked for it.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

from app.config import config

logger = logging.getLogger(__name__)

# How many run at once. The work is CPU-bound, so more in flight than
# cores makes each slower without finishing any sooner — the same
# reasoning as the request-side gate, and deliberately the same number.
WORKERS = 2

# A run that has been going this long has hung. It is marked failed so a
# client polling it gets an answer rather than "running" forever.
MAX_RUNTIME_SECONDS = 900

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"

# Identifies this process. The pool is in memory, so any job still
# queued or running under a *different* boot id belongs to a process
# that is gone and will never finish it. That is an exact test, where a
# timeout is a guess — see reap_stalled.
BOOT_ID = uuid.uuid4().hex[:12]


@dataclass
class Job:
    job_id: str
    owner: str
    kind: str                    # "report" | "health-report" | "deck"
    dataset_id: str
    status: str = QUEUED
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    artifact_id: str = ""
    error: str = ""
    # The request that asked for this, so a background failure can be
    # traced to the person who triggered it.
    request_id: str = ""
    # Which run of the server accepted this job. See BOOT_ID.
    boot_id: str = ""
    params: Dict = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.status in (DONE, FAILED)

    @property
    def duration_s(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return round(end - self.started_at, 2)


class JobStore:
    """Job records, owned and on disk so a restart does not lose them."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(config.data_dir, "jobs")
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, owner: str, job_id: str) -> str:
        from app.services.artifacts import _safe
        directory = os.path.join(self.base_dir, _safe(owner))
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, _safe(job_id) + ".json")

    def save(self, job: Job) -> None:
        from app.services.artifacts import _atomic_write
        with self._lock:
            _atomic_write(self._path(job.owner, job.job_id),
                          json.dumps(asdict(job)).encode("utf-8"))

    def get(self, owner: str, job_id: str) -> Optional[Job]:
        path = self._path(owner, job_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return Job(**json.load(fh))
        except Exception:
            logger.warning("unreadable job record at %s", path, exc_info=True)
            return None

    def all_owners(self) -> List[str]:
        if not os.path.isdir(self.base_dir):
            return []
        return [d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))]

    def list(self, owner: str, dataset_id: str = "", limit: int = 30) -> List[Job]:
        from app.services.artifacts import _safe
        directory = os.path.join(self.base_dir, _safe(owner))
        if not os.path.isdir(directory):
            return []
        out: List[Job] = []
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            job = self.get(owner, name[:-5])
            if job is None:
                continue
            if dataset_id and job.dataset_id != dataset_id:
                continue
            out.append(job)
        out.sort(key=lambda j: j.created_at, reverse=True)
        return out[:limit]


store = JobStore()

# What each kind of job actually does. Registered rather than imported
# here so this module does not depend on the report builders — they
# depend on it, which is the direction that keeps the seam clean.
_HANDLERS: Dict[str, Callable] = {}


def register(kind: str, handler: Callable) -> None:
    """handler(job) -> (bytes, filename, fmt, title)."""
    _HANDLERS[kind] = handler


_pool = ThreadPoolExecutor(max_workers=WORKERS,
                           thread_name_prefix="analytiq-job")


def submit(owner: str, kind: str, dataset_id: str,
           params: Optional[Dict] = None) -> Job:
    """Record the run and hand it to the pool. Returns immediately."""
    if kind not in _HANDLERS:
        raise ValueError("There is no job of kind {!r}.".format(kind))

    from app.services.request_context import current_request_id

    job = Job(job_id=uuid.uuid4().hex[:16], owner=owner, kind=kind,
              dataset_id=dataset_id, created_at=time.time(),
              request_id=current_request_id(), boot_id=BOOT_ID,
              params=dict(params or {}))
    store.save(job)
    _pool.submit(_run, job)
    logger.info("job %s queued (%s) for %s", job.job_id, kind, owner)
    return job


def _run(job: Job) -> None:
    from app.services.artifacts import store as artifacts
    from app.services.request_context import bind_request_id

    # The submitting request's id follows the work into the worker
    # thread, so every line this run logs is attributable to the person
    # who asked for it rather than to "analytiq-job-1".
    with bind_request_id(job.request_id):
        job.status = RUNNING
        job.started_at = time.time()
        store.save(job)
        try:
            data, filename, fmt, title = _HANDLERS[job.kind](job)
            art = artifacts.put(
                owner=job.owner, dataset_id=job.dataset_id, kind=job.kind,
                fmt=fmt, data=data, filename=filename, title=title,
                job_id=job.job_id)
            job.artifact_id = art.artifact_id
            job.status = DONE

            # A schedule's whole point is that something arrives, so
            # delivery is part of finishing the job rather than a
            # separate thing that might not happen.
            schedule_id = job.params.get("_schedule_id")
            if schedule_id:
                from app.services.schedules import deliver
                try:
                    outcome = deliver(schedule_id, job.owner, art.artifact_id)
                    logger.info("schedule %s: %s", schedule_id, outcome)
                except Exception:
                    # The report exists; only delivery failed. That is
                    # worth an error and not worth failing the job over,
                    # because the client can still fetch it.
                    logger.error("schedule %s built but could not deliver",
                                 schedule_id, exc_info=True)
            logger.info("job %s finished in %.1fs -> artifact %s",
                        job.job_id, job.duration_s, art.artifact_id)
        except Exception as exc:                          # noqa: BLE001
            job.status = FAILED
            # The message a client sees, not a traceback: the traceback
            # goes to the log, where the request id ties it to this run.
            job.error = "{}: {}".format(type(exc).__name__, exc)[:400]
            logger.error("job %s failed after %.1fs: %s\n%s",
                         job.job_id, job.duration_s, job.error,
                         traceback.format_exc())
        finally:
            job.finished_at = time.time()
            store.save(job)


_ABANDONED = ("This run stopped without finishing — the server was "
              "probably restarted while it was working. Start it again.")


def reap_all() -> int:
    """Every account's stranded runs.

    The list endpoint reaps the account doing the listing, which covers
    anyone who comes back for their report. It does not cover a job a
    *schedule* started: nobody is polling that one, so without this it
    would sit at "queued" in the record for ever and the schedule would
    keep reporting that as its last outcome.
    """
    return sum(reap_stalled(owner) for owner in store.all_owners())


def reap_stalled(owner: str) -> int:
    """Mark as failed any run that nothing will ever finish.

    Two distinct cases, and only one of them is a timeout:

    *Left by a previous process.* The pool lives in memory, so a job
    still queued or running under an older boot id died with the process
    that accepted it. A job killed while it was merely queued never got
    a started_at, so a runtime timeout would never notice it and the
    client would poll "queued" forever. The boot id settles it exactly.

    *Hung in this process.* Still the current boot, but running far
    longer than any real report takes.

    Checked when jobs are listed, which is when it matters and costs
    nothing.
    """
    return sum(_reap(job) for job in store.list(owner, limit=100))


def reap(job: Job) -> Job:
    """The single-job form, for the poll route.

    Polling one job by id is how a client actually waits for a report,
    so the check has to happen there too — reaping only on the list
    endpoint would leave the one caller who cares reading "running"
    forever.
    """
    _reap(job)
    return job


def _reap(job: Job) -> bool:
    """Fail `job` in place if nothing will ever finish it. True if it
    was changed."""
    if job.terminal:
        return False
    now = time.time()
    stale_boot = job.boot_id != BOOT_ID
    hung = (job.status == RUNNING and job.started_at
            and now - job.started_at > MAX_RUNTIME_SECONDS)
    if not (stale_boot or hung):
        return False
    job.status = FAILED
    job.error = _ABANDONED
    job.finished_at = now
    store.save(job)
    return True
