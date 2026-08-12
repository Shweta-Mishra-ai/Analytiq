"""
services/table_extractor.py — turn a photo/screenshot of a table, OR a
video that shows tabular/dashboard data, into a real DataFrame via
Gemini structured extraction.

This is what makes image/video analysis genuinely useful: instead of a
text description, the table becomes a normal dataset that flows through
the full pipeline (quality, dashboard, EDA, ML, reports).
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile

import pandas as pd

from app.config import config

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """You are a precise data-extraction engine.
The image contains one or more data tables. Extract the LARGEST / primary table.

Return ONLY valid JSON, exactly this shape:
{
  "found": true,
  "title": "short table title if visible, else empty string",
  "columns": ["col1", "col2", ...],
  "rows": [["cell", "cell", ...], ...]
}

Rules:
- Transcribe cell values EXACTLY as printed — never invent, estimate or
  fill in values that are not clearly visible.
- If a cell is unreadable or empty, use null.
- Keep numbers as printed (do not add/remove separators or currency signs).
- Every row must have exactly as many cells as there are columns.
- If the image contains NO data table, return {"found": false, "columns": [], "rows": []}.
"""




class ExtractionError(Exception):
    pass


def _parse_extraction_json(raw: str) -> dict:
    """Defensive JSON parsing. response_mime_type='application/json'
    should guarantee raw JSON, but real-world model output occasionally
    wraps it in a ```json fence anyway — don't let that turn into a hard
    error when the JSON itself is perfectly fine."""
    text = (raw or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    fenced = re.sub(r"\s*```$", "", fenced).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ExtractionError(
        "The model returned an unreadable response — try a clearer image/video")


def extract_table_from_image(filename: str, data: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Image bytes → (DataFrame, extraction_warnings). Raises
    ExtractionError with a clear, user-facing message on every failure mode."""
    if not config.gemini_api_key:
        raise ExtractionError(
            "GEMINI_API_KEY must be configured to extract tables from images")

    from app.ai import gemini_client
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise ExtractionError(f"'{filename}' is not a readable image")

    try:
        raw = gemini_client.generate_text(
            [EXTRACT_PROMPT, img], json_mode=True, temperature=0.0,
            timeout_sec=60)
    except Exception as e:
        logger.warning(f"Gemini extraction call failed: {e}")
        raise ExtractionError(f"Table extraction failed: {e}")

    payload = _parse_extraction_json(raw)
    return table_json_to_df(payload)


def extract_table_from_video(filename: str, data: bytes, ext: str) -> tuple[pd.DataFrame, list[str]]:
    """Video bytes → (DataFrame, extraction_warnings).

    Extracts a handful of visually-distinct frames locally via ffmpeg
    (free, no API cost — see services/video_frames.py) and runs each one
    through the same image-table-extraction path as a photo upload,
    merging the results into one dataset. This is cheaper, faster, and
    more reliable than uploading the whole clip to Gemini's video File
    API: fewer/smaller calls, no video-endpoint-specific timeouts, and
    it naturally handles a scrolled spreadsheet (each distinct scroll
    position becomes its own frame with its own rows, merged together)
    better than a single video-understanding pass would.

    Raises ExtractionError with a clear, user-facing message on every
    failure mode."""
    if not config.gemini_api_key:
        raise ExtractionError(
            "GEMINI_API_KEY must be configured to extract tables from video")

    from app.services.video_frames import FrameExtractionError, extract_table_frames

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name

    try:
        try:
            frames = extract_table_frames(path)
        except FrameExtractionError as e:
            raise ExtractionError(str(e))
    finally:
        os.unlink(path)

    results: list[tuple[pd.DataFrame, list[str]]] = []
    frame_errors: list[str] = []
    for i, frame_bytes in enumerate(frames):
        try:
            results.append(extract_table_from_image(f"{filename} (frame {i+1})", frame_bytes))
        except ExtractionError as e:
            frame_errors.append(str(e))

    if not results:
        # every frame failed — surface the most informative message rather
        # than a generic one, since this usually means no frame clearly
        # showed a table (e.g. a talking-head video with no data on screen)
        detail = frame_errors[0] if frame_errors else "no readable frames"
        raise ExtractionError(
            f"No table could be found in this video ({len(frames)} frame(s) "
            f"checked): {detail}")

    return _merge_frame_tables(results, len(frames))


def _merge_frame_tables(
        results: list[tuple[pd.DataFrame, list[str]]],
        frames_checked: int) -> tuple[pd.DataFrame, list[str]]:
    """Combines per-frame extractions into one dataset. Same column set
    across frames (e.g. a scrolled spreadsheet) -> concatenate rows,
    dropping exact duplicates from overlapping frames. Different column
    sets (the video showed more than one distinct table) -> use the
    largest and say so, rather than silently picking one."""
    def _colkey(df: pd.DataFrame) -> tuple:
        return tuple(sorted(str(c).lower().strip() for c in df.columns))

    groups: dict[tuple, list[pd.DataFrame]] = {}
    all_warnings: list[str] = []
    for df, warnings in results:
        groups.setdefault(_colkey(df), []).append(df)
        for w in warnings:
            if w not in all_warnings:
                all_warnings.append(w)

    if len(groups) == 1:
        (dfs,) = groups.values()
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates()
        merged = merged.reset_index(drop=True)
    else:
        # multiple distinct tables appeared across the video — use the
        # most complete one, but don't pretend the others didn't exist
        best_key = max(groups, key=lambda k: sum(len(d) for d in groups[k]))
        dfs = groups[best_key]
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
        other_tables = len(groups) - 1
        all_warnings.insert(0,
            f"This video appears to show {len(groups)} different tables "
            f"(different columns across frames) — the most complete one "
            f"({len(merged)} rows) was used. {other_tables} other table(s) "
            f"seen in the video were not included.")

    used = sum(len(d) for group in groups.values() for d in group)
    if frames_checked > used and len(results) < frames_checked:
        skipped = frames_checked - len(results)
        all_warnings.append(
            f"{skipped} of {frames_checked} checked video frame(s) didn't "
            f"clearly show the table and were skipped.")

    return merged, all_warnings


def table_json_to_df(payload: dict) -> tuple[pd.DataFrame, list[str]]:
    """Validated conversion of the extraction JSON into a DataFrame, plus
    a list of any data-integrity warnings the caller should surface to
    the user (rows dropped, cells truncated, etc.) — never silent.
    Pure function — unit-testable without an API key."""
    if not isinstance(payload, dict) or not payload.get("found", False):
        raise ExtractionError(
            "No data table was found in the file. Works best with clear "
            "screenshots/photos of tables or footage that clearly shows "
            "a spreadsheet, table, or dashboard grid.")

    columns = payload.get("columns") or []
    rows = payload.get("rows") or []
    if not columns or not rows:
        raise ExtractionError("The table found in the file appears to be empty")

    columns = [str(c).strip() or f"column_{i+1}" for i, c in enumerate(columns)]
    # de-duplicate column names
    seen: dict[str, int] = {}
    for i, c in enumerate(columns):
        if c in seen:
            seen[c] += 1
            columns[i] = f"{c}_{seen[c]}"
        else:
            seen[c] = 0

    width = len(columns)
    fixed_rows = []
    skipped_malformed = 0
    truncated_rows = 0
    for r in rows:
        if not isinstance(r, list):
            skipped_malformed += 1
            continue
        if len(r) > width:
            truncated_rows += 1
        r = list(r)[:width] + [None] * max(0, width - len(r))
        fixed_rows.append(r)
    if not fixed_rows:
        raise ExtractionError("No readable rows were extracted from the image")

    warnings: list[str] = []
    if skipped_malformed:
        warnings.append(
            f"{skipped_malformed} row(s) came back in an unexpected format "
            f"and were skipped — re-check the source for those entries.")
    if truncated_rows:
        warnings.append(
            f"{truncated_rows} row(s) had more cells than columns and were "
            f"truncated to fit — some values in those rows may be missing.")

    df = pd.DataFrame(fixed_rows, columns=columns)

    # strip common numeric decorations so dtype inference can work
    for col in df.columns:
        if (df[col].dtype == object
                or pd.api.types.is_string_dtype(df[col])):
            cleaned = (df[col].astype(str)
                       .str.strip()
                       .str.replace(r"^[\$₹€£]\s*", "", regex=True)
                       .str.replace(r"\s*[\$₹€£]$", "", regex=True)
                       .str.replace(",", "", regex=False)
                       .str.replace(r"%$", "", regex=True))
            # accounting-style negatives, e.g. "(500)" -> "-500"
            neg = cleaned.str.match(r"^\(.+\)$", na=False)
            cleaned = cleaned.where(~neg, "-" + cleaned.str.strip("()"))

            converted = pd.to_numeric(cleaned, errors="coerce")
            non_blank = df[col].notna() & (df[col].astype(str).str.strip() != "")
            good = converted.notna().sum() / max(non_blank.sum(), 1)
            if good > 0.8:
                lost = int((non_blank & converted.isna()).sum())
                if lost:
                    warnings.append(
                        f"Column '{col}' is mostly numeric — {lost} "
                        f"non-numeric value(s) in it were kept as text "
                        f"rather than converted.")
                # Keep the numeric value where conversion succeeded;
                # preserve the ORIGINAL cell — never a destroyed blank —
                # everywhere it didn't parse (e.g. a stray "Pending").
                df[col] = converted.where(converted.notna(), df[col])

    from app.engines.data_loader import _smart_dtype_inference
    df = _smart_dtype_inference(df)
    return df, warnings
