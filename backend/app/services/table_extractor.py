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

from app.services.dtypes import is_text_dtype

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """You are a precise data-extraction engine.
The image contains one or more data tables. Extract the LARGEST / primary table.

Return ONLY valid JSON, exactly this shape:
{
  "found": true,
  "title": "short table title if visible, else empty string",
  "columns": ["col1", "col2", ...],
  "rows": [["cell", "cell", ...], ...],
  "cut_off": {"top": false, "bottom": false, "left": false, "right": false},
  "other_tables": 0,
  "stated_total_rows": null,
  "unreadable_cells": 0
}

Rules:
- Transcribe cell values EXACTLY as printed — never invent, estimate or
  fill in values that are not clearly visible.
- If a cell is unreadable or empty, use null, and count it in
  "unreadable_cells".
- Keep numbers as printed (do not add/remove separators or currency signs).
- Every row must have exactly as many cells as there are columns.
- If the image contains NO data table, return {"found": false, "columns": [], "rows": []}.

Completeness — this matters as much as the values:
- "cut_off": set a side to true when the table visibly continues past that
  edge of the image (rows clipped at the bottom, columns clipped at the
  right, a scrollbar showing there is more, a partial row at an edge).
  This tells the reader that what you returned is a portion of the table,
  not all of it. Do not guess at the content beyond the edge.
- "other_tables": how many OTHER distinct tables are visible that you did
  not extract. 0 if this is the only one.
- "stated_total_rows": if the image states a total anywhere ("1-25 of
  480", "480 records"), put that number here. Otherwise null.
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
    from app.ai import multimodal
    from PIL import Image

    # Validate the image before spending a model call on it, and before
    # producing an error about models for what is really a broken file.
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise ExtractionError(f"'{filename}' is not a readable image")

    try:
        raw = multimodal.describe_image(
            image=data, prompt=EXTRACT_PROMPT, task="table_extraction",
            mime=Image.MIME.get(img.format or "", "image/png"),
            max_tokens=4096, json_mode=True, timeout_sec=60)
    except multimodal.NoCapableModel as e:
        # Names the missing capability rather than one vendor's key —
        # any model that can read an image can do this now, including a
        # local one.
        raise ExtractionError(str(e))
    except Exception as e:
        logger.warning(f"table extraction call failed: {e}")
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
    # No key check here: each frame goes through extract_table_from_image,
    # which asks the router and reports the real gap. Checking one
    # vendor's key up front would refuse a machine that has a perfectly
    # good local vision model.
    from app.services.video_frames import (FrameExtractionError,
                                           extract_table_frames_with_budget)

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name

    try:
        try:
            frames, unseen_views = extract_table_frames_with_budget(path)
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

    merged, warnings = _merge_frame_tables(results, len(frames))
    if unseen_views:
        warnings.insert(0,
            "This video contains more distinct views than can be read in one "
            "pass — {} were used and {} more were not looked at, so rows that "
            "appear only later in the clip are missing. For a long scroll, "
            "export the table instead of filming it.".format(
                len(frames), unseen_views))
    return merged, warnings


def _norm_header(col) -> str:
    """A column name reduced to what survives transcription noise.

    The same header read from two frames comes back as "Order Date",
    "order date" or "Order  Date" often enough that comparing them
    literally splits one table into two, and the second one is then
    discarded as "a different table". That is silent row loss caused by
    nothing but whitespace.
    """
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


def _norm_cell(value) -> str:
    """A cell reduced to what survives transcription noise.

    Used only to recognise the same row seen in two overlapping frames.
    "1,234", "1234" and " 1234 " are one value; without this the
    overlapping region of a scrolled table is appended twice and every
    total the report prints is inflated.
    """
    if value is None:
        return ""
    s = str(value).strip().lower()
    if s in ("nan", "none", "null"):
        return ""
    s = re.sub(r"[\s, ]", "", s)
    s = re.sub(r"^[\$₹€£]", "", s)
    try:
        f = float(s)
        return repr(round(f, 6))
    except ValueError:
        return s


def _row_keys(df: pd.DataFrame) -> list:
    return [tuple(_norm_cell(v) for v in row) for row in df.itertuples(index=False)]


def _group_frames_by_shape(frames: list) -> list:
    """Group frame tables that describe the same table.

    Grouped on normalised header names, then near-identical groups are
    folded together: two frames whose headers overlap by 70%+ are the
    same table with one header misread, not two tables.
    """
    groups: list[dict] = []
    for df in frames:
        keys = {_norm_header(c) for c in df.columns}
        placed = False
        for g in groups:
            overlap = len(keys & g["keys"]) / max(len(keys | g["keys"]), 1)
            if overlap >= 0.7:
                g["frames"].append(df)
                g["keys"] |= keys
                placed = True
                break
        if not placed:
            groups.append({"keys": keys, "frames": [df]})
    return groups


def _stack_without_double_counting(frames: list) -> tuple:
    """Concatenate frames of one table, counting an overlapping row once.

    `drop_duplicates()` over the concatenation was wrong in both
    directions. It kept the overlap when two frames transcribed the same
    row slightly differently ("1,234" vs "1234"), inflating every total;
    and it destroyed genuinely repeated rows — two orders with identical
    values are two orders, and collapsing them silently changes the
    count.

    Rows are matched on a normalised key, and a frame contributes only
    the occurrences it has *beyond* what previous frames already
    contributed. Two identical rows visible in one frame therefore both
    survive; the same row seen again in the next frame does not.
    """
    from collections import Counter

    merged_rows: list = []
    seen = Counter()
    overlap_rows = 0
    columns = list(frames[0].columns)

    for df in frames:
        df = df.reindex(columns=columns)
        keys = _row_keys(df)
        frame_counts = Counter()
        for key, (_idx, row) in zip(keys, df.iterrows()):
            frame_counts[key] += 1
            if frame_counts[key] <= seen[key]:
                overlap_rows += 1
                continue
            merged_rows.append(list(row.values))
        seen.update(frame_counts)

    merged = pd.DataFrame(merged_rows, columns=columns)
    return merged, overlap_rows


def _merge_frame_tables(
        results: list[tuple[pd.DataFrame, list[str]]],
        frames_checked: int) -> tuple[pd.DataFrame, list[str]]:
    """Combine per-frame extractions into one dataset without losing rows.

    A scrolled spreadsheet gives frames that overlap; a video of two
    dashboards gives frames of different tables. Both have to be handled
    without quietly dropping data — the merged dataset is what every
    figure in the report is then computed from.
    """
    all_warnings: list[str] = []
    for _df, warnings in results:
        for w in warnings:
            if w not in all_warnings:
                all_warnings.append(w)

    groups = _group_frames_by_shape([df for df, _w in results])
    groups.sort(key=lambda g: sum(len(d) for d in g["frames"]), reverse=True)

    merged, overlap_rows = _stack_without_double_counting(groups[0]["frames"])
    merged = merged.reset_index(drop=True)

    if overlap_rows:
        all_warnings.append(
            "{} row(s) appeared in more than one frame (an overlapping "
            "scroll) and were counted once.".format(overlap_rows))

    if len(groups) > 1:
        others = len(groups) - 1
        shapes = "; ".join(
            "{} column(s), {} row(s)".format(
                len(g["frames"][0].columns), sum(len(d) for d in g["frames"]))
            for g in groups[1:4])
        all_warnings.insert(0,
            "This video shows {} different tables (the columns change "
            "between frames). The largest — {} rows — was used. The other "
            "{} ({}) are NOT in this dataset; record them separately if you "
            "need them.".format(len(groups), len(merged), others, shapes))

    if len(results) < frames_checked:
        skipped = frames_checked - len(results)
        all_warnings.append(
            "{} of {} video frame(s) did not clearly show the table and were "
            "skipped — rows visible only in those frames are "
            "missing.".format(skipped, frames_checked))

    return merged, all_warnings


def _uses_comma_decimal(values: pd.Series) -> bool:
    """Is this column written 1.234,56 rather than 1,234.56?

    Decided on the whole column, never per cell: "1.234" alone is
    ambiguous, but a column where commas consistently appear last and
    with two digits after them is European notation.
    """
    sample = [s for s in values.dropna().astype(str)
              if re.fullmatch(r"-?[\d.,]+", s.strip()) and any(
                  ch in s for ch in ".,")]
    if len(sample) < 3:
        return False
    comma_decimal = 0
    dot_decimal = 0
    for s in sample:
        s = s.strip()
        last_comma, last_dot = s.rfind(","), s.rfind(".")
        if last_comma > last_dot and re.search(r",\d{1,2}$", s):
            comma_decimal += 1
        elif last_dot > last_comma and re.search(r"\.\d{1,2}$", s):
            dot_decimal += 1
    return comma_decimal > dot_decimal and comma_decimal >= 3


_SIDE_WORDS = {
    "bottom": "more rows below the visible area",
    "top": "rows above the visible area",
    "right": "more columns to the right",
    "left": "columns to the left",
}


def _completeness_warnings(payload: dict, n_rows: int) -> list[str]:
    """Turn the model's completeness signals into warnings the user sees.

    A screenshot of the first 25 rows of a 480-row table extracts
    perfectly and is still the wrong dataset to analyse. Without this the
    app reports a mean over 25 rows as though it were the mean of the
    table, and nothing anywhere says otherwise.
    """
    out: list[str] = []

    cut = payload.get("cut_off")
    if isinstance(cut, dict):
        sides = [_SIDE_WORDS[s] for s in ("bottom", "top", "right", "left")
                 if cut.get(s) and s in _SIDE_WORDS]
        if sides:
            out.append(
                "The table is cut off in the source image ({}), so this is a "
                "portion of it — {} row(s) were captured. Analysis will "
                "describe only what was visible.".format(
                    ", ".join(sides), n_rows))

    stated = payload.get("stated_total_rows")
    try:
        stated = int(stated) if stated is not None else None
    except (TypeError, ValueError):
        stated = None
    if stated and stated > n_rows:
        out.append(
            "The source states the full table has {:,} rows; {:,} were "
            "captured here ({:.0f}%). Export the remaining rows before "
            "relying on any total or average.".format(
                stated, n_rows, n_rows / stated * 100))

    try:
        others = int(payload.get("other_tables") or 0)
    except (TypeError, ValueError):
        others = 0
    if others > 0:
        out.append(
            "{} other table(s) were visible and not extracted — only the "
            "largest one was. Upload them separately if they are "
            "needed.".format(others))

    try:
        unreadable = int(payload.get("unreadable_cells") or 0)
    except (TypeError, ValueError):
        unreadable = 0
    if unreadable > 0:
        out.append(
            "{} cell(s) could not be read from the image and are blank in "
            "this dataset.".format(unreadable))

    return out


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

    completeness = _completeness_warnings(payload, len(rows))

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

    warnings: list[str] = list(completeness)
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
        if (is_text_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])):
            raw = (df[col].astype(str)
                   .str.strip()
                   .str.replace(r"^[\$₹€£]\s*", "", regex=True)
                   .str.replace(r"\s*[\$₹€£]$", "", regex=True))
            had_percent = raw.str.match(r"^-?[\d.,]+\s*%$", na=False)
            cleaned = raw.str.replace(r"\s*%$", "", regex=True)
            if _uses_comma_decimal(cleaned):
                # "1.234,56" is one thousand two hundred, not 1.23456.
                # Stripping commas the other way round moves the decimal
                # point three places and every figure derived from the
                # column is then wrong by a factor of a thousand.
                comma_decimal = True
                cleaned = (cleaned.str.replace(".", "", regex=False)
                                  .str.replace(",", ".", regex=False))
            else:
                comma_decimal = False
                cleaned = cleaned.str.replace(",", "", regex=False)
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
                if comma_decimal:
                    warnings.append(
                        f"Column '{col}' is written in the European format "
                        f"(1.234,56) and was read that way — check one value "
                        f"against the original to confirm.")
                if had_percent.any():
                    warnings.append(
                        f"Column '{col}' held percentages; the % sign was "
                        f"removed and the numbers kept as printed (45% is "
                        f"stored as 45, not 0.45).")
                # Keep the numeric value where conversion succeeded;
                # preserve the ORIGINAL cell — never a destroyed blank —
                # everywhere it didn't parse (e.g. a stray "Pending").
                df[col] = converted.where(converted.notna(), df[col])

    from app.engines.data_loader import _smart_dtype_inference
    df = _smart_dtype_inference(df)
    return df, warnings
