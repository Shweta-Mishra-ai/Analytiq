"""
api/datasets.py — upload, preview, profile, clean, quality.
Mirrors pages 1 (Upload) and 2 (Data Quality) of the Streamlit app.
Every dataset is scoped to the authenticated client (`owner`).
"""
from __future__ import annotations
import logging

import io

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

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
from app.services.integrity import bytes_digest
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

    # Hashed from the bytes as received, before parsing — so the record
    # is of the file the client actually sent, not of our reading of it.
    meta = store.create(
        owner, df, filename,
        size_mb=len(data) / (1024 * 1024),
        sheet_names=getattr(result, "sheet_names", None) or [],
        warnings=(result.warnings or []) + (validation.warnings or []),
        source_bytes=len(data),
        source_sha256=bytes_digest(data),
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
        source_bytes=len(data),
        source_sha256=bytes_digest(data),
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
        source_bytes=len(data),
        source_sha256=bytes_digest(data),
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


@router.get("/{ds_id}/integrity")
def integrity_record(ds_id: str, owner: str = Depends(current_owner)):
    """Is this still the data that was uploaded, and can every change to
    it be accounted for?

    Governance says what the data is; this says whether it can be
    trusted. The verdict is recomputed from the stored frames on every
    call rather than read from a stored claim — a cached verdict is a
    verdict about the past, and an integrity check that trusts its own
    record is not a check.
    """
    result = store.integrity(owner, ds_id)
    if result is None:
        raise HTTPException(404, "Dataset not found")
    return to_jsonable(result)


@router.get("/{ds_id}/readiness")
def readiness(ds_id: str, owner: str = Depends(current_owner)):
    """Is this dataset fit to analyse, and if not, what has to happen first."""
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    cached = store.cache_get(owner, ds_id, "readiness")
    if cached is None:
        cached = to_jsonable(readiness_payload(assess_readiness(df)))
        store.cache_set(owner, ds_id, "readiness", cached)
    return cached


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
    summary = get_cleaning_summary(report)
    store.update_active(
        owner, ds_id, cleaned, event="clean",
        detail={"mode": "aggressive" if aggressive else "non-destructive",
                "steps": summary.get("actions", summary.get("steps", []))[:20]})
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


# ══════════════════════════════════════════════════════════
#  Warehouse sources
#
#  A CSV export is the weakest link in an analysis: a stale
#  point-in-time copy with no lineage and no types. These routes let a
#  dataset come straight from the client's database instead, and the
#  result lands in exactly the same store, with the same integrity
#  record and audit trail an upload gets — the trail records the query
#  it came from, so the figures trace back to a statement rather than to
#  a file someone emailed.
#
#  Credentials are never stored. The URL is supplied per request, used,
#  and redacted before anything is returned or written down.
# ══════════════════════════════════════════════════════════

class WarehouseConnection(BaseModel):
    url: str


class WarehouseQuery(BaseModel):
    url: str
    sql: str = ""
    table: str = ""
    schema_name: str = ""
    limit: int = 200_000
    name: str = ""


@router.get("/warehouse/backends")
def warehouse_backends():
    """Which databases this deployment can actually reach.

    Computed by importing each driver rather than declared, so the UI
    offers what will work and names the package to install for what
    will not.
    """
    from app.services import warehouse
    return {"sqlalchemy": warehouse.sqlalchemy_available(),
            "backends": warehouse.backends()}


@router.post("/warehouse/test")
def warehouse_test(body: WarehouseConnection,
                   owner: str = Depends(current_owner)):
    """Connect and disconnect, reporting what happened in terms the
    person typing the URL can act on.

    Authenticated like every other route here — this one makes an
    outbound connection to a host the caller names, so it is not
    something to leave open even behind a middleware that already gates
    the prefix.
    """
    from app.services import warehouse
    return warehouse.test_connection(body.url)


@router.post("/warehouse/tables")
def warehouse_tables(body: WarehouseConnection,
                     owner: str = Depends(current_owner)):
    from app.services import warehouse
    try:
        tables = warehouse.list_tables(body.url)
    except warehouse.WarehouseError as e:
        raise HTTPException(422, str(e))
    return {"source": warehouse.redact(body.url),
            "tables": [{"schema": t.schema, "name": t.name,
                        "qualified": t.qualified} for t in tables]}


@router.post("/warehouse/preview")
def warehouse_preview(body: WarehouseQuery,
                      owner: str = Depends(current_owner)):
    """Run the query but keep nothing — so a user can check they wrote
    what they meant before pulling two hundred thousand rows."""
    from app.services import warehouse
    try:
        result = (warehouse.preview_table(body.url, body.table,
                                          body.schema_name, limit=100)
                  if body.table else
                  warehouse.run_query(body.url, body.sql, limit=100))
    except warehouse.WarehouseError as e:
        raise HTTPException(422, str(e))
    return {"source": result.source, "sql": result.sql,
            "rows": len(result.df), "columns": list(result.df.columns),
            "preview": df_records(result.df, 100),
            "warnings": result.warnings}


@router.post("/warehouse/import")
def warehouse_import(body: WarehouseQuery, owner: str = Depends(current_owner)):
    """Pull the result of a query in as a dataset.

    Everything downstream — profiling, cleaning, EDA, ML, the report —
    then works exactly as it does for an upload, because this creates
    the same kind of dataset. The difference is what the audit trail
    says: the source is a query against a named database, not a file.
    """
    from app.services import warehouse
    try:
        result = (warehouse.preview_table(body.url, body.table,
                                          body.schema_name, limit=body.limit)
                  if body.table else
                  warehouse.run_query(body.url, body.sql, limit=body.limit))
    except warehouse.WarehouseError as e:
        raise HTTPException(422, str(e))

    validation = validate_dataframe(result.df)
    if not validation.is_valid:
        raise HTTPException(422, "; ".join(validation.errors))

    label = body.name.strip() or _warehouse_label(body, result)
    meta = store.create(
        owner, result.df, label,
        size_mb=float(result.df.memory_usage(deep=True).sum()) / (1024 * 1024),
        warnings=result.warnings + (validation.warnings or []),
        # There is no uploaded file to hash. The digest of the query text
        # takes its place: it identifies the statement that produced this
        # data, which is the closest thing a warehouse pull has to a
        # source file, and it is what someone re-running the analysis
        # would need to match.
        source_sha256=bytes_digest(result.sql.encode("utf-8")),
    )
    store.record_event(owner, meta.dataset_id, "ingest", {
        "source": result.source,
        "sql": result.sql[:500],
        "truncated": result.truncated,
        "row_limit": result.limit,
    })
    return {"meta": to_jsonable(meta),
            "preview": df_records(result.df, 100),
            "source": result.source,
            "sql": result.sql,
            "warnings": result.warnings}


def _warehouse_label(body: WarehouseQuery, result) -> str:
    """A dataset needs a name a person recognises in a list a week
    later. "query result" is not that."""
    if body.table:
        return f"{body.table} ({result.source})"
    return f"query on {result.source}"
