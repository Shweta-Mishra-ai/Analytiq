"""
services/report_jobs.py — what each kind of background job actually does.

Kept apart from jobs.py deliberately. The runner knows how to record a
run, retry nothing, and store a result; it does not know what a report
is. That direction — report code depends on the runner, never the
reverse — is what would let the executor be swapped for a real broker
without touching anything that builds a document.

Each handler takes a Job and returns (bytes, filename, format, title).
Nothing here catches its own exceptions: a handler that fails should
fail, and the runner turns that into a job with a status of "failed" and
the reason on it. A handler that swallowed its own errors would produce
an empty PDF and call it success, which is the failure mode this whole
layer exists to avoid.
"""
from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def _dataset(job):
    from app.services.dataset_store import store

    df = store.get_df(job.owner, job.dataset_id)
    if df is None:
        raise LookupError(
            "That dataset is no longer available, so the report could not "
            "be built. It may have been deleted since this was scheduled.")
    return df


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def build_report(job):
    """The full analysis report."""
    from app.api.reports import PdfRequest, _generate_pdf

    req = PdfRequest(**{k: v for k, v in job.params.items()
                        if k in PdfRequest.model_fields})
    response = _generate_pdf(job.dataset_id, req, job.owner)
    data = _bytes_of(response)
    title = req.title or "Data Analysis Report"
    # The same request object drives the deck, so the stored artifact
    # has to follow it. Naming a deck .pdf gives it the wrong media
    # type as well as the wrong extension, and it arrives at whoever
    # opened the link as a file their machine refuses to open.
    fmt = "pptx" if getattr(req, "format", "pdf") == "pptx" else "pdf"
    return (data, "analytiq_report_{}.{}".format(_stamp(), fmt), fmt, title)


def build_health_report(job):
    """The client-facing data health and business insights report."""
    from app.engines.domains.registry import detect_domain
    from app.engines.health_engine import build_report_payload, compute_health
    from app.engines.health_pdf_builder import build_health_pdf
    from app.services.dataset_store import store

    df = _dataset(job)
    meta = store.get_meta(job.owner, job.dataset_id)
    # The synchronous endpoint records this before it builds, carrying
    # the digest of the data as it stands; a report built in the
    # background has to do the same or the integrity trail has a hole in
    # it exactly where the reports that circulate come from — a
    # scheduled PDF is the one most likely to be read months later.
    store.record_event(job.owner, job.dataset_id, "report",
                       {"format": "health-pdf", "rows": len(df),
                        "job_id": job.job_id})
    domain, _ = detect_domain(df)
    health = compute_health(df)
    payload = build_report_payload(df, domain)

    data = build_health_pdf(
        df, domain, health, payload["insights"],
        getattr(meta, "filename", "dataset.csv"),
        agency_name=job.params.get("agency_name", "Analytiq"),
        executive_summary=payload.get("executive_summary", ""),
        key_findings=payload.get("key_findings"),
        risks=payload.get("risks"),
        opportunities=payload.get("opportunities"),
        actions=payload.get("actions"))
    return (data, "analytiq_health_report_{}.pdf".format(_stamp()), "pdf",
            "Data Health & Business Insights")


def _bytes_of(response) -> bytes:
    """A handler may reuse an endpoint that returns a Response.

    Reusing the endpoint is the point — a scheduled report and a
    downloaded one must be the same document, and two code paths that
    are meant to agree eventually stop agreeing.

    The report endpoints hand back a StreamingResponse over a BytesIO,
    so the stream has to be drained here. It is already whole in memory;
    the streaming is how it reaches a socket, not how it is produced.
    """
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)
    body = getattr(response, "body", None)
    if body is not None:
        return bytes(body)

    iterator = getattr(response, "body_iterator", None)
    if iterator is None:
        raise TypeError("Unexpected report type: {}".format(type(response)))

    if hasattr(iterator, "__anext__"):
        import asyncio

        async def drain():
            chunks = []
            async for chunk in iterator:
                chunks.append(chunk)
            return chunks

        # A job runs on a worker thread with no event loop of its own,
        # which is exactly where asyncio.run belongs.
        parts = asyncio.run(drain())
    else:
        parts = list(iterator)
    return b"".join(bytes(part) for part in parts)


def register_all() -> None:
    from app.services.jobs import register

    register("report", build_report)
    register("health-report", build_health_report)


register_all()
