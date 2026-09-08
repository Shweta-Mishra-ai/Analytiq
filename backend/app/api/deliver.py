"""
api/deliver.py — asking for a report, collecting it, and sending it on.

Three endpoints' worth of behaviour that the product was missing
entirely: a report you ask for and come back to, a place it lives once
it exists, and a link you can give to someone who does not have an
account.

The existing synchronous endpoints are untouched. They are the right
answer for "build this and show it to me now", and removing them to make
a point about architecture would break every current caller for no gain.
This is the other mode, for the report that takes long enough to walk
away from and for the one nobody is waiting on at all.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.artifacts import store as artifacts
from app.services.auth import current_owner
from app.services.dataset_store import store as datasets
from app.services import load_control
from app.services.load_control import admit

logger = logging.getLogger(__name__)

# How often one share link may be opened. Far above anyone reading a
# report they were sent, far below what it takes to make serving them
# everyone else's problem.
SHARE_OPENS_PER_MINUTE = 60

router = APIRouter(prefix="/api", tags=["deliver"])


class JobRequest(BaseModel):
    dataset_id: str
    kind: str = "report"
    params: dict = {}


# ── asking ───────────────────────────────────────────────

@router.post("/jobs", dependencies=[Depends(admit("report"))])
def create_job(req: JobRequest, owner: str = Depends(current_owner)):
    """Start a report and come back for it.

    Returns immediately with an id to poll. The alternative — holding
    the socket for the three seconds a report takes — is what caps
    throughput and what a schedule cannot use at all.
    """
    from app.services import report_jobs   # registers the handlers
    from app.services.jobs import submit

    if datasets.get_meta(owner, req.dataset_id) is None:
        raise HTTPException(404, "Dataset not found")

    try:
        job = submit(owner, req.kind, req.dataset_id, req.params)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return _job_json(job)


@router.get("/jobs")
def list_jobs(dataset_id: str = Query(""), owner: str = Depends(current_owner)):
    from app.services.jobs import reap_stalled, store as jobs
    reap_stalled(owner)
    return {"jobs": [_job_json(j) for j in jobs.list(owner, dataset_id)]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, owner: str = Depends(current_owner)):
    from app.services.jobs import reap, store as jobs

    job = jobs.get(owner, job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return _job_json(reap(job))


# ── collecting ───────────────────────────────────────────

@router.get("/artifacts")
def list_artifacts(dataset_id: str = Query(""),
                   owner: str = Depends(current_owner)):
    return {"artifacts": [_artifact_json(a)
                          for a in artifacts.list(owner, dataset_id)]}


@router.get("/artifacts/{artifact_id}")
def get_artifact(artifact_id: str, owner: str = Depends(current_owner)):
    art = artifacts.get(owner, artifact_id)
    if art is None:
        raise HTTPException(404, "That report is not available — it may "
                                 "have expired.")
    return _artifact_json(art)


@router.get("/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str, owner: str = Depends(current_owner)):
    art = artifacts.get(owner, artifact_id)
    data = artifacts.read(owner, artifact_id)
    if art is None or data is None:
        raise HTTPException(404, "That report is not available — it may "
                                 "have expired.")
    return _stream(art, data)


@router.delete("/artifacts/{artifact_id}")
def delete_artifact(artifact_id: str, owner: str = Depends(current_owner)):
    if not artifacts.delete(owner, artifact_id):
        raise HTTPException(404, "That report is not available.")
    return {"deleted": artifact_id}


# ── sending on ───────────────────────────────────────────

class ShareRequest(BaseModel):
    ttl_days: int = 7
    note: str = ""


@router.post("/artifacts/{artifact_id}/share")
def share_artifact(artifact_id: str, req: ShareRequest,
                   request: Request, owner: str = Depends(current_owner)):
    """A link that works without an account, for this report only."""
    from app.services import sharing

    art = artifacts.get(owner, artifact_id)
    if art is None:
        raise HTTPException(404, "That report is not available.")

    token, expires = sharing.mint(owner, artifact_id, req.ttl_days)
    sharing.record_share(owner, artifact_id, expires, req.note)
    datasets.record_event(owner, art.dataset_id, "share", {
        "artifact_id": artifact_id, "expires_at": expires,
        "note": req.note[:120],
    })
    return {"url": "{}/api/shared/{}".format(_public_base(request), token),
            "token": token, "expires_at": expires}


def _public_base(request: Request) -> str:
    """The address the recipient can actually reach.

    Behind a reverse proxy `request.base_url` is the internal one — the
    container's own host and port — so a link built from it works in
    development and is dead on arrival in production, which is the worst
    place to find out. PUBLIC_BASE_URL is the deployment saying what its
    outside address is; the scheduler's email already uses it, and the
    two must not disagree about the same link.
    """
    import os

    configured = (os.environ.get("PUBLIC_BASE_URL") or "").strip().rstrip("/")
    return configured or str(request.base_url).rstrip("/")


@router.delete("/artifacts/{artifact_id}/share")
def revoke_share(artifact_id: str, owner: str = Depends(current_owner)):
    from app.services import sharing

    if not sharing.revoke(owner, artifact_id):
        raise HTTPException(404, "That report is not available.")
    # The grant is in the audit trail; the withdrawal has to be there
    # too, or the trail says access is still open when it is not.
    art = artifacts.get(owner, artifact_id)
    if art is not None:
        datasets.record_event(owner, art.dataset_id, "revoke",
                              {"artifact_id": artifact_id})
    return {"revoked": artifact_id}


# The one route that answers without a session. It grants exactly the
# artifact its token names — it cannot list, cannot reach a dataset, and
# cannot be traded for anything else.
@router.get("/shared/{token}")
def open_shared(token: str):
    from app.services import sharing

    try:
        owner, artifact_id = sharing.read(token)
    except sharing.ShareError as e:
        raise HTTPException(404, str(e)) from None

    if sharing.is_revoked(owner, artifact_id):
        raise HTTPException(
            410, "That link has been withdrawn by whoever shared it.")

    # The only route that answers without a session, and every answer is
    # a multi-megabyte read off disk. It cannot go through admit(): that
    # keys on the account, and every anonymous caller collapses into one
    # bucket — so one recipient refreshing would lock out every other
    # client's CFO, which is worse than no limit at all.
    #
    # The token names exactly one grant, so that is the bucket. A flood
    # on one link cannot reach another. Checked after the signature, so
    # a forged link still costs one HMAC and no state, as designed.
    try:
        load_control.limiter.check(
            "share:{}:{}".format(owner, artifact_id), "shared",
            SHARE_OPENS_PER_MINUTE, 60)
    except load_control.RateLimited as e:
        raise HTTPException(
            429, "This link is being opened unusually often. Try again in "
                 "a moment.",
            headers={"Retry-After": str(e.retry_after)}) from None

    art = artifacts.get(owner, artifact_id)
    data = artifacts.read(owner, artifact_id)
    if art is None or data is None:
        raise HTTPException(
            404, "That report is no longer available — reports are kept "
                 "for a limited time and this one has expired.")

    sharing.record_access(owner, artifact_id)
    logger.info("shared artifact %s served (%d views)",
                artifact_id, art.meta.get("share_views", 1))
    return _stream(art, data)


# ── shaping ──────────────────────────────────────────────

def _stream(art, data: bytes) -> StreamingResponse:
    import io
    return StreamingResponse(
        io.BytesIO(data), media_type=art.media_type,
        headers={"Content-Disposition":
                 'attachment; filename="{}"'.format(_ascii(art.filename))})


def _ascii(name: str) -> str:
    """A filename safe for a header — no quotes, no newlines."""
    cleaned = "".join(c for c in str(name)
                      if c.isalnum() or c in "-_. ")
    return cleaned.strip() or "report"


def _job_json(job) -> dict:
    return {"job_id": job.job_id, "kind": job.kind, "status": job.status,
            "dataset_id": job.dataset_id, "created_at": job.created_at,
            "started_at": job.started_at, "finished_at": job.finished_at,
            "duration_s": job.duration_s, "artifact_id": job.artifact_id,
            "error": job.error}


def _artifact_json(art) -> dict:
    return {"artifact_id": art.artifact_id, "kind": art.kind,
            "format": art.fmt, "filename": art.filename, "title": art.title,
            "size_bytes": art.size_bytes, "created_at": art.created_at,
            "expires_at": art.expires_at, "dataset_id": art.dataset_id,
            "shared": bool(art.meta.get("shares")),
            "share_views": art.meta.get("share_views", 0),
            "shares_revoked": bool(art.meta.get("shares_revoked_at"))}


# ── arriving without being asked ─────────────────────────

class ScheduleRequest(BaseModel):
    dataset_id: str
    kind: str = "health-report"
    cadence: str = "weekly"
    hour: int = 7
    recipients: list = []
    params: dict = {}


@router.get("/schedules")
def list_schedules(owner: str = Depends(current_owner)):
    from app.services.schedules import next_run, store as schedules

    out = []
    for s in schedules.list(owner):
        out.append({
            "schedule_id": s.schedule_id, "dataset_id": s.dataset_id,
            "kind": s.kind, "cadence": s.cadence, "hour": s.hour,
            "recipients": s.recipients, "enabled": s.enabled,
            "last_run_at": s.last_run_at, "last_status": s.last_status,
            "last_artifact_id": s.last_artifact_id,
            "next_run_at": next_run(s).timestamp(),
        })
    return {"schedules": out}


@router.post("/schedules")
def create_schedule(req: ScheduleRequest, owner: str = Depends(current_owner)):
    from app.services.schedules import create

    if datasets.get_meta(owner, req.dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    try:
        s = create(owner, req.dataset_id, req.kind, req.cadence, req.hour,
                   req.recipients, req.params)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"schedule_id": s.schedule_id, "cadence": s.cadence}


@router.delete("/schedules/{schedule_id}")
def delete_schedule(schedule_id: str, owner: str = Depends(current_owner)):
    from app.services.schedules import store as schedules

    if not schedules.delete(owner, schedule_id):
        raise HTTPException(404, "No such schedule")
    return {"deleted": schedule_id}


@router.post("/schedules/tick")
def tick(owner: str = Depends(current_owner)):
    """Run whatever is due now, for this account.

    Exposed as well as timed: a deployment behind an external scheduler
    can drive it, and during setup a person can press it and see that
    the thing they configured actually produces a report — which is the
    only way anyone believes a schedule before its first firing.

    Scoped to the caller. The timed loop sweeps every account because
    that is its job, but a tick anyone signed in can press must not
    reach another tenant's schedules: it would spend their CPU and
    claim their window, so their real report would not fire.
    """
    from app.services.schedules import run_due
    return {"started": run_due(only_owner=owner)}
