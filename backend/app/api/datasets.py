"""
api/datasets.py — upload, preview, profile, clean, quality.
Mirrors pages 1 (Upload) and 2 (Data Quality) of the Streamlit app.
Every dataset is scoped to the authenticated client (`owner`).
"""
from __future__ import annotations
import logging

import io

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.config import config
from app.engines.data_cleaner import (DIALECTS, CleaningPolicy,
                                     auto_clean, get_cleaning_summary,
                                      table_name_from_filename)
from app.engines.data_loader import load_file
from app.engines.data_profiler import profile_dataset
from app.engines.readiness import assess_readiness, readiness_payload
from app.engines.data_validator import validate_dataframe, validate_file_size
from app.services.auth import current_owner
from app.services.dataset_store import store
from app.services.serialize import df_records, to_jsonable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class _UploadShim:
    """Adapts FastAPI UploadFile to the file-like object load_file expects
    (Streamlit's UploadedFile: .name, .size, .read, .seek)."""

    def __init__(self, filename: str, data: bytes):
        self.name = filename
        self.size = len(data)
        self._buf = io.BytesIO(data)

    def read(self, *a):
        return self._buf.read(*a)

    def seek(self, *a):
        return self._buf.seek(*a)

    def __getattr__(self, item):
        return getattr(self._buf, item)


def _process_upload(owner: str, filename: str, data: bytes, sheet: str) -> dict:
    """Heavy parsing/validation — runs in the threadpool, off the event loop."""
    ok, msg = validate_file_size(len(data))
    if not ok:
        raise HTTPException(413, msg)

    shim = _UploadShim(filename, data)
    sheet_arg = int(sheet) if sheet.isdigit() else sheet
    result = load_file(shim, sheet_name=sheet_arg)
    if not result.success or result.df is None:
        raise HTTPException(422, result.error or "Could not parse file.")

    df = result.df
    validation = validate_dataframe(df)
    if not validation.is_valid:
        raise HTTPException(422, "; ".join(validation.errors))

    meta = store.create(
        owner, df, filename,
        size_mb=len(data) / (1024 * 1024),
        sheet_names=getattr(result, "sheet_names", None) or [],
        warnings=(result.warnings or []) + (validation.warnings or []),
    )
    return {"meta": to_jsonable(meta), "preview": df_records(df, 100)}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), sheet: str = Query("0"),
                          owner: str = Depends(current_owner)):
    from starlette.concurrency import run_in_threadpool
    data = await file.read()
    return await run_in_threadpool(
        _process_upload, owner, file.filename or "upload.csv", data, sheet)


def _process_image_extract(owner: str, filename: str, data: bytes) -> dict:
    from app.services.table_extractor import (ExtractionError,
                                              extract_table_from_image)
    try:
        df, extraction_warnings = extract_table_from_image(filename, data)
    except ExtractionError as e:
        raise HTTPException(422 if "GEMINI" not in str(e) else 503, str(e))

    validation = validate_dataframe(df)
    if not validation.is_valid:
        raise HTTPException(422, "Extracted table is not usable: "
                            + "; ".join(validation.errors))

    meta = store.create(
        owner, df, f"{filename} (extracted)",
        size_mb=len(data) / (1024 * 1024),
        warnings=["Values were transcribed from an image by AI — "
                  "spot-check critical numbers against the original."]
        + extraction_warnings + (validation.warnings or []),
    )
    return {"meta": to_jsonable(meta), "preview": df_records(df, 100)}


@router.post("/extract-from-image")
async def extract_from_image(file: UploadFile = File(...),
                              owner: str = Depends(current_owner)):
    """Photo/screenshot of a table → real dataset (full pipeline works)."""
    from starlette.concurrency import run_in_threadpool
    data = await file.read()
    if len(data) > config.max_media_mb * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {config.max_media_mb} MB")
    return await run_in_threadpool(
        _process_image_extract, owner, file.filename or "image.png", data)


_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}


def _process_video_extract(owner: str, filename: str, data: bytes) -> dict:
    import os as _os
    from app.services.table_extractor import (ExtractionError,
                                              extract_table_from_video)
    ext = _os.path.splitext(filename.lower())[1]
    if ext not in _VIDEO_EXTS:
        raise HTTPException(422, f"Unsupported video type '{ext}'. "
                             f"Use one of: {', '.join(sorted(_VIDEO_EXTS))}")
    try:
        df, extraction_warnings = extract_table_from_video(filename, data, ext)
    except ExtractionError as e:
        raise HTTPException(422 if "GEMINI" not in str(e) else 503, str(e))

    validation = validate_dataframe(df)
    if not validation.is_valid:
        raise HTTPException(422, "Extracted table is not usable: "
                            + "; ".join(validation.errors))

    meta = store.create(
        owner, df, f"{filename} (extracted)",
        size_mb=len(data) / (1024 * 1024),
        warnings=["Values were transcribed from a video by AI — "
                  "spot-check critical numbers against the original."]
        + extraction_warnings + (validation.warnings or []),
    )
    return {"meta": to_jsonable(meta), "preview": df_records(df, 100)}


@router.post("/extract-from-video")
async def extract_from_video(file: UploadFile = File(...),
                              owner: str = Depends(current_owner)):
    """Video showing a table/spreadsheet/dashboard → real dataset.
    Slower than image extraction (Gemini File API processing + upload),
    so this route can take up to a few minutes for longer clips."""
    from starlette.concurrency import run_in_threadpool
    data = await file.read()
    if len(data) > config.max_media_mb * 1024 * 1024:
        raise HTTPException(413, f"Video exceeds {config.max_media_mb} MB")
    return await run_in_threadpool(
        _process_video_extract, owner, file.filename or "video.mp4", data)


@router.get("")
def list_datasets(owner: str = Depends(current_owner)):
    return {"datasets": to_jsonable(store.list_meta(owner))}


@router.get("/{ds_id}")
def get_dataset(ds_id: str, owner: str = Depends(current_owner)):
    meta = store.get_meta(owner, ds_id)
    if not meta:
        raise HTTPException(404, "Dataset not found")
    return {"meta": to_jsonable(meta)}


@router.delete("/{ds_id}")
def delete_dataset(ds_id: str, owner: str = Depends(current_owner)):
    if not store.delete(owner, ds_id):
        raise HTTPException(404, "Dataset not found")
    return {"deleted": ds_id}


@router.get("/{ds_id}/preview")
def preview(ds_id: str, rows: int = Query(100, le=1000),
            owner: str = Depends(current_owner)):
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return df_records(df, rows)


@router.get("/{ds_id}/profile")
def profile(ds_id: str, owner: str = Depends(current_owner)):
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    cached = store.cache_get(owner, ds_id, "profile")
    if cached is None:
        cached = profile_dataset(df)
        store.cache_set(owner, ds_id, "profile", cached)
    return to_jsonable(cached)


@router.get("/{ds_id}/governance")
def governance(ds_id: str, owner: str = Depends(current_owner)):
    """What the data is, where it came from, and who it could identify.

    The record a client's data owner asks for — usually after the
    analysis has been circulated. Includes k-anonymity over the
    quasi-identifiers, which is the number that decides whether the file
    can be shared: removing the name column does not anonymise a dataset
    where a postcode, an age and a job title single someone out.
    """
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    from app.engines.governance import build_governance
    record = build_governance(
        df, meta=store.get_meta(owner, ds_id),
        cleaning_summary=store.cache_get(owner, ds_id, "clean_report"),
        retention_days=config.data_ttl_days)
    return to_jsonable(record)


@router.get("/{ds_id}/readiness")
def readiness(ds_id: str, owner: str = Depends(current_owner)):
    """Is this dataset fit to analyse, and if not, what has to happen first."""
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return to_jsonable(readiness_payload(assess_readiness(df)))


@router.post("/{ds_id}/clean")
def clean(ds_id: str, aggressive: bool = False,
          owner: str = Depends(current_owner)):
    """Clean the dataset.

    Non-destructive by default: whitespace, types and ordinary missing
    values are corrected, while anything whose removal would be a
    judgement call — duplicate rows, very sparse columns, constant
    columns — is reported with the SQL to act on it, and kept. Blanket
    deduplication of a transactional table deletes real turnover, and a
    mostly-empty column can carry the strongest signal in the data.

    `aggressive=true` restores the older behaviour of removing them.
    """
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    policy = CleaningPolicy.aggressive() if aggressive else CleaningPolicy()
    cleaned, report = auto_clean(df, policy)
    store.update_active(owner, ds_id, cleaned)
    summary = get_cleaning_summary(report)
    store.cache_set(owner, ds_id, "clean_report", summary)

    # The equivalent SQL, so a client's data team can audit each step and
    # apply the same cleaning upstream rather than trusting the output.
    meta = store.get_meta(owner, ds_id)
    table = table_name_from_filename(meta.filename if meta else "")
    scripts = {d: report.sql_script(table, d) for d in DIALECTS}
    store.cache_set(owner, ds_id, "clean_sql", scripts["ansi"])
    store.cache_set(owner, ds_id, "clean_sql_all", scripts)

    return {"summary": to_jsonable(summary),
            "actions": to_jsonable(report.actions),
            "sql": scripts["ansi"],
            "sql_by_dialect": scripts,
            "dialects": list(DIALECTS),
            "sql_table": table,
            "preview": df_records(cleaned, 100)}


@router.get("/{ds_id}/clean/sql")
def clean_sql(ds_id: str, dialect: str = "ansi",
              owner: str = Depends(current_owner)):
    """The SQL for the last cleaning pass, in the requested dialect."""
    scripts = store.cache_get(owner, ds_id, "clean_sql_all")
    if scripts is None:
        sql = store.cache_get(owner, ds_id, "clean_sql")
        if sql is None:
            raise HTTPException(404, "Run cleaning first")
        return {"sql": sql, "dialect": "ansi", "dialects": list(DIALECTS)}
    key = str(dialect or "ansi").lower()
    if key not in scripts:
        raise HTTPException(
            400, "Unknown dialect {!r}. Available: {}".format(
                dialect, ", ".join(DIALECTS)))
    return {"sql": scripts[key], "dialect": key, "dialects": list(DIALECTS)}


@router.post("/{ds_id}/reset")
def reset(ds_id: str, owner: str = Depends(current_owner)):
    df = store.reset_active(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return {"preview": df_records(df, 100), "meta": to_jsonable(store.get_meta(owner, ds_id))}
