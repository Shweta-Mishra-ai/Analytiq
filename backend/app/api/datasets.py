"""
api/datasets.py — upload, preview, profile, clean, quality.
Mirrors pages 1 (Upload) and 2 (Data Quality) of the Streamlit app.
"""
from __future__ import annotations

import io

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.config import config
from app.engines.data_cleaner import auto_clean, get_cleaning_summary
from app.engines.data_loader import load_file
from app.engines.data_profiler import profile_dataset
from app.engines.data_validator import validate_dataframe, validate_file_size
from app.services.dataset_store import store
from app.services.serialize import df_records, to_jsonable

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


def _process_upload(filename: str, data: bytes, sheet: str) -> dict:
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
        df, filename,
        size_mb=len(data) / (1024 * 1024),
        sheet_names=getattr(result, "sheet_names", None) or [],
        warnings=(result.warnings or []) + (validation.warnings or []),
    )
    return {"meta": to_jsonable(meta), "preview": df_records(df, 100)}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), sheet: str = Query("0")):
    from starlette.concurrency import run_in_threadpool
    data = await file.read()
    return await run_in_threadpool(
        _process_upload, file.filename or "upload.csv", data, sheet)


def _process_image_extract(filename: str, data: bytes) -> dict:
    from app.services.table_extractor import (ExtractionError,
                                              extract_table_from_image)
    try:
        df = extract_table_from_image(filename, data)
    except ExtractionError as e:
        raise HTTPException(422 if "GEMINI" not in str(e) else 503, str(e))

    validation = validate_dataframe(df)
    if not validation.is_valid:
        raise HTTPException(422, "Extracted table is not usable: "
                            + "; ".join(validation.errors))

    meta = store.create(
        df, f"{filename} (extracted)",
        size_mb=len(data) / (1024 * 1024),
        warnings=["Values were transcribed from an image by AI — "
                  "spot-check critical numbers against the original."]
        + (validation.warnings or []),
    )
    return {"meta": to_jsonable(meta), "preview": df_records(df, 100)}


@router.post("/extract-from-image")
async def extract_from_image(file: UploadFile = File(...)):
    """Photo/screenshot of a table → real dataset (full pipeline works)."""
    from starlette.concurrency import run_in_threadpool
    data = await file.read()
    if len(data) > config.max_media_mb * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {config.max_media_mb} MB")
    return await run_in_threadpool(
        _process_image_extract, file.filename or "image.png", data)


@router.get("")
def list_datasets():
    return {"datasets": to_jsonable(store.list_meta())}


@router.get("/{ds_id}")
def get_dataset(ds_id: str):
    meta = store.get_meta(ds_id)
    if not meta:
        raise HTTPException(404, "Dataset not found")
    return {"meta": to_jsonable(meta)}


@router.delete("/{ds_id}")
def delete_dataset(ds_id: str):
    if not store.delete(ds_id):
        raise HTTPException(404, "Dataset not found")
    return {"deleted": ds_id}


@router.get("/{ds_id}/preview")
def preview(ds_id: str, rows: int = Query(100, le=1000)):
    df = store.get_df(ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return df_records(df, rows)


@router.get("/{ds_id}/profile")
def profile(ds_id: str):
    df = store.get_df(ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    cached = store.cache_get(ds_id, "profile")
    if cached is None:
        cached = profile_dataset(df)
        store.cache_set(ds_id, "profile", cached)
    return to_jsonable(cached)


@router.post("/{ds_id}/clean")
def clean(ds_id: str):
    df = store.get_df(ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    cleaned, report = auto_clean(df)
    store.update_active(ds_id, cleaned)
    summary = get_cleaning_summary(report)
    store.cache_set(ds_id, "clean_report", summary)
    return {"summary": to_jsonable(summary),
            "actions": to_jsonable(report.actions),
            "preview": df_records(cleaned, 100)}


@router.post("/{ds_id}/reset")
def reset(ds_id: str):
    df = store.reset_active(ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return {"preview": df_records(df, 100), "meta": to_jsonable(store.get_meta(ds_id))}
