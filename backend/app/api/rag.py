"""
api/rag.py — knowledge bases: multimodal ingestion (docs, tables,
images, video), Q&A with citations, and report generation.
Every knowledge base is scoped to the authenticated client (`owner`).
"""
from __future__ import annotations
import logging

import io
import time

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import config
from app.rag import service
from app.rag.vector_store import rag_store
from app.services.auth import current_owner
from app.services.serialize import to_jsonable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


class KbCreate(BaseModel):
    name: str


class QueryRequest(BaseModel):
    question: str
    k: int = 6


class ReportRequest(BaseModel):
    title: str = "Knowledge Base Report"
    focus: str = ""


def _kb_or_404(owner: str, kb_id: str):
    kb = rag_store.get(owner, kb_id)
    if kb is None:
        raise HTTPException(404, "Knowledge base not found")
    return kb


@router.post("/kb")
def create_kb(body: KbCreate, owner: str = Depends(current_owner)):
    kb = rag_store.create(owner, body.name.strip() or "Untitled KB")
    return {"kb_id": kb.kb_id, "name": kb.name}


@router.get("/kb")
def list_kbs(owner: str = Depends(current_owner)):
    return {"knowledge_bases": to_jsonable(rag_store.list(owner))}


@router.get("/kb/{kb_id}")
def get_kb(kb_id: str, owner: str = Depends(current_owner)):
    kb = _kb_or_404(owner, kb_id)
    return {"kb_id": kb.kb_id, "name": kb.name, "embedder": kb.embedder,
            "chunks": len(kb.chunks), "files": to_jsonable(kb.files)}


@router.delete("/kb/{kb_id}")
def delete_kb(kb_id: str, owner: str = Depends(current_owner)):
    if not rag_store.delete(owner, kb_id):
        raise HTTPException(404, "Knowledge base not found")
    return {"deleted": kb_id}


def _process_kb_upload(kb, name: str, data: bytes) -> dict:
    """Heavy extraction/embedding — runs in the threadpool."""
    kind = service.file_kind(name)
    if kind in ("image", "video") and not config.gemini_api_key:
        raise HTTPException(
            503, "GEMINI_API_KEY must be configured to analyze images/video")
    try:
        result = service.ingest_file(kb, name, data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return to_jsonable(result)


@router.post("/kb/{kb_id}/files")
async def upload_file(kb_id: str, file: UploadFile = File(...),
                       owner: str = Depends(current_owner)):
    from starlette.concurrency import run_in_threadpool
    kb = _kb_or_404(owner, kb_id)
    data = await file.read()
    limit = config.max_media_mb * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds {config.max_media_mb} MB limit")
    name = file.filename or f"upload-{int(time.time())}"
    return await run_in_threadpool(_process_kb_upload, kb, name, data)


@router.post("/kb/{kb_id}/query")
def query(kb_id: str, req: QueryRequest, owner: str = Depends(current_owner)):
    kb = _kb_or_404(owner, kb_id)
    try:
        return to_jsonable(service.answer_question(kb, req.question, req.k))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@router.post("/kb/{kb_id}/report")
def report(kb_id: str, req: ReportRequest, owner: str = Depends(current_owner)):
    kb = _kb_or_404(owner, kb_id)
    try:
        return to_jsonable(service.generate_report(kb, req.title, req.focus))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@router.post("/kb/{kb_id}/report/pdf")
def report_as_pdf(kb_id: str, req: ReportRequest, owner: str = Depends(current_owner)):
    kb = _kb_or_404(owner, kb_id)
    try:
        result = service.generate_report(kb, req.title, req.focus)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    pdf = service.report_pdf(result["markdown"], req.title)
    return StreamingResponse(
        io.BytesIO(pdf), media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=rag_report.pdf"})
