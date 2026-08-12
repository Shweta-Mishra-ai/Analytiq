"""
rag/extractors.py — turn any supported file into text passages.

Text-ish files are parsed locally. Images and video go through Gemini
(vision + File API) and come back as rich analyst descriptions, so they
become searchable text like everything else.

Every extractor returns a list of passages:
    {"text": str, "source": filename, "locator": "page 3" / "0:45–1:30" / ...}
"""
from __future__ import annotations

import io
import logging
import os
from typing import List

from app.config import config

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
TEXT_EXTS = {".txt", ".md", ".log"}
TABLE_EXTS = {".csv", ".tsv"}

IMAGE_PROMPT = """You are a senior data analyst. Analyze this image thoroughly:
1. Describe what the image shows.
2. Transcribe ALL visible text, numbers, labels and table contents exactly.
3. If it contains a chart/graph: name the chart type, axes, series, and read
   out the approximate data values and the trend they show.
4. State 2-3 analytical takeaways.
Be factual — never invent values that are not visible."""

VIDEO_PROMPT = """You are a senior data analyst. Analyze this video:
1. Summarize what happens, segment by segment, with timestamps.
2. Transcribe any spoken narration or on-screen text/numbers.
3. If charts, dashboards or data appear: read out the metrics and values shown.
4. State the key analytical takeaways.
Be factual — never invent values that are not shown or said."""


def extract(filename: str, data: bytes) -> List[dict]:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return _pdf(filename, data)
    if ext == ".docx":
        return _docx(filename, data)
    if ext in TEXT_EXTS:
        return _plain(filename, data)
    if ext in TABLE_EXTS:
        return _table(filename, data)
    if ext in IMAGE_EXTS:
        return _image(filename, data)
    if ext in VIDEO_EXTS:
        return _video(filename, data, ext)
    raise ValueError(f"Unsupported file type: {ext}")


# ── documents ────────────────────────────────────────────

def _pdf(name: str, data: bytes) -> List[dict]:
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(data)
    out = []
    try:
        for i, page in enumerate(doc):
            text = page.get_textpage().get_text_bounded() or ""
            if text.strip():
                out.append({"text": text.strip(), "source": name,
                            "locator": f"page {i + 1}"})
    finally:
        doc.close()
    return out


def _docx(name: str, data: bytes) -> List[dict]:
    import docx
    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    text = "\n".join(parts)
    return [{"text": text, "source": name, "locator": "document"}] if text.strip() else []


def _plain(name: str, data: bytes) -> List[dict]:
    text = data.decode("utf-8", errors="replace")
    return [{"text": text, "source": name, "locator": "file"}] if text.strip() else []


def _table(name: str, data: bytes) -> List[dict]:
    import pandas as pd
    sep = "\t" if name.lower().endswith(".tsv") else ","
    df = pd.read_csv(io.BytesIO(data), sep=sep, on_bad_lines="skip")
    desc = [f"Table {name}: {len(df)} rows, columns: {', '.join(map(str, df.columns))}"]
    num = df.select_dtypes(include="number")
    if not num.empty:
        desc.append("Numeric summary:\n" + num.describe().round(2).to_string())
    # keep a readable sample of rows so RAG can quote actual data
    desc.append("Sample rows:\n" + df.head(50).to_string(index=False))
    return [{"text": "\n\n".join(desc), "source": name, "locator": "table"}]


# ── media via Gemini ─────────────────────────────────────

def _require_gemini():
    if not config.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required to analyze images and video")


def _image(name: str, data: bytes) -> List[dict]:
    _require_gemini()
    from app.ai import gemini_client
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    img.load()
    try:
        text = gemini_client.generate_text(
            [IMAGE_PROMPT, img], timeout_sec=60).strip()
    except Exception as e:
        raise RuntimeError(f"Image analysis failed: {e}")
    if not text:
        raise RuntimeError("Gemini returned no analysis for the image")
    return [{"text": f"[Image analysis of {name}]\n{text}",
             "source": name, "locator": "image"}]


def _video(name: str, data: bytes, ext: str) -> List[dict]:
    """Native video understanding via the Gemini File API
    (covers visuals, on-screen text and audio narration)."""
    _require_gemini()
    from app.ai import gemini_client
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        try:
            vfile = gemini_client.upload_file(path, timeout_sec=300)
        except Exception as e:
            raise RuntimeError(f"Could not process video for analysis: {e}")
        try:
            text = gemini_client.generate_text(
                [VIDEO_PROMPT, vfile], timeout_sec=120).strip()
        except Exception as e:
            raise RuntimeError(f"Video analysis failed: {e}")
        finally:
            gemini_client.delete_file(vfile.name)
        if not text:
            raise RuntimeError("Gemini returned no analysis for the video")
        return [{"text": f"[Video analysis of {name}]\n{text}",
                 "source": name, "locator": "video"}]
    finally:
        os.unlink(path)
