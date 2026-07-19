"""
services/table_extractor.py — turn a photo/screenshot of a table into a
real DataFrame via Gemini structured extraction.

This is what makes image analysis genuinely useful: instead of a text
description, the table in the image becomes a normal dataset that flows
through the full pipeline (quality, dashboard, EDA, ML, reports).
"""
from __future__ import annotations

import io
import json
import logging

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


def extract_table_from_image(filename: str, data: bytes) -> pd.DataFrame:
    """Image bytes → DataFrame. Raises ExtractionError with a clear,
    user-facing message on every failure mode."""
    if not config.gemini_api_key:
        raise ExtractionError(
            "GEMINI_API_KEY must be configured to extract tables from images")

    import google.generativeai as genai
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise ExtractionError(f"'{filename}' is not a readable image")

    genai.configure(api_key=config.gemini_api_key)
    model = genai.GenerativeModel(
        config.gemini_model,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    try:
        resp = model.generate_content([EXTRACT_PROMPT, img])
        raw = resp.text or ""
    except Exception as e:
        logger.warning(f"Gemini extraction call failed: {e}")
        raise ExtractionError(f"Table extraction failed: {e}")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise ExtractionError(
            "The model returned an unreadable response — try a clearer image")

    return table_json_to_df(payload)


def table_json_to_df(payload: dict) -> pd.DataFrame:
    """Validated conversion of the extraction JSON into a DataFrame.
    Pure function — unit-testable without an API key."""
    if not isinstance(payload, dict) or not payload.get("found", False):
        raise ExtractionError(
            "No data table was found in the image. Works best with clear "
            "screenshots or photos of tables/reports.")

    columns = payload.get("columns") or []
    rows = payload.get("rows") or []
    if not columns or not rows:
        raise ExtractionError("The table in the image appears to be empty")

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
    for r in rows:
        if not isinstance(r, list):
            continue
        r = list(r)[:width] + [None] * max(0, width - len(r))
        fixed_rows.append(r)
    if not fixed_rows:
        raise ExtractionError("No readable rows were extracted from the image")

    df = pd.DataFrame(fixed_rows, columns=columns)

    # strip common numeric decorations so dtype inference can work
    for col in df.columns:
        if (df[col].dtype == object
                or pd.api.types.is_string_dtype(df[col])):
            cleaned = (df[col].astype(str)
                       .str.strip()
                       .str.replace(r"^[\$₹€£]\s*", "", regex=True)
                       .str.replace(",", "", regex=False)
                       .str.replace(r"%$", "", regex=True))
            converted = pd.to_numeric(cleaned, errors="coerce")
            good = converted.notna().sum() / max(df[col].notna().sum(), 1)
            if good > 0.8:
                df[col] = converted

    from app.engines.data_loader import _smart_dtype_inference
    df = _smart_dtype_inference(df)
    return df
