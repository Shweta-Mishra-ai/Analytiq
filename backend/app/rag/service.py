"""
rag/service.py — chunking, ingestion, Q&A and report generation
over a knowledge base of documents, tables, images and videos.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from app.config import config
from app.rag.extractors import IMAGE_EXTS, VIDEO_EXTS, extract
from app.rag.vector_store import KnowledgeBase

logger = logging.getLogger(__name__)

CHUNK_CHARS = 1100
CHUNK_OVERLAP = 150


def chunk_passages(passages: List[dict]) -> List[dict]:
    """Split extracted passages into overlapping chunks, keeping source refs."""
    chunks = []
    for p in passages:
        text = p["text"].strip()
        if not text:
            continue
        if len(text) <= CHUNK_CHARS:
            chunks.append({**p, "text": text})
            continue
        start = 0
        while start < len(text):
            end = min(start + CHUNK_CHARS, len(text))
            # try to break on a sentence/paragraph boundary
            if end < len(text):
                for sep in ("\n\n", "\n", ". "):
                    cut = text.rfind(sep, start + CHUNK_CHARS // 2, end)
                    if cut != -1:
                        end = cut + len(sep)
                        break
            chunks.append({**p, "text": text[start:end].strip()})
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c["text"]]


def file_kind(filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in (".csv", ".tsv"):
        return "table"
    return "document"


def ingest_file(kb: KnowledgeBase, filename: str, data: bytes) -> dict:
    passages = extract(filename, data)
    if not passages:
        raise ValueError(f"No text could be extracted from {filename}")
    chunks = chunk_passages(passages)
    kind = file_kind(filename)
    kb.add_chunks(chunks, filename, kind)
    preview = passages[0]["text"][:600]
    return {"filename": filename, "kind": kind,
            "chunks_added": len(chunks), "preview": preview}


# ── generation ───────────────────────────────────────────

def _gemini_generate(system: str, user: str, max_tokens: int = 2048) -> Optional[str]:
    if not config.gemini_api_key:
        return None
    try:
        from app.ai import gemini_client
        return gemini_client.generate_text(
            [user], system=system, max_output_tokens=max_tokens,
            temperature=0.2, timeout_sec=60).strip() or None
    except Exception as e:
        logger.warning(f"Gemini generation failed: {e}")
        return None


def _groq_generate(system: str, user: str, max_tokens: int = 2048) -> Optional[str]:
    if not config.groq_api_key:
        return None
    try:
        from app.ai.llm_client import get_client
        return get_client(config.groq_api_key).chat_task(
            system, user, task="default", max_tokens=max_tokens)
    except Exception as e:
        logger.warning(f"Groq generation failed: {e}")
        return None


def _generate(system: str, user: str, max_tokens: int = 2048) -> str:
    text = (_gemini_generate(system, user, max_tokens)
            or _groq_generate(system, user, max_tokens))
    if not text:
        raise RuntimeError(
            "No LLM available — set GEMINI_API_KEY or GROQ_API_KEY")
    return text


def _context_block(hits: List[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] ({h['source']}, {h['locator']})\n{h['text']}")
    return "\n\n---\n\n".join(lines)


QA_SYSTEM = """You are a senior data analyst answering questions from a \
knowledge base of documents, tables, images and videos.
Rules:
- Answer ONLY from the provided context. If the context is insufficient, say so.
- Cite sources inline like [1], [2] matching the numbered context blocks.
- Quote exact numbers from the context, never invent values.
- Be concise and analytical."""


def answer_question(kb: KnowledgeBase, question: str, k: int = 6) -> dict:
    hits = kb.search(question, k=k)
    if not hits:
        return {"answer": "The knowledge base is empty or nothing relevant "
                          "was found. Upload files first.", "sources": []}
    answer = _generate(
        QA_SYSTEM,
        f"CONTEXT:\n{_context_block(hits)}\n\nQUESTION: {question}",
        max_tokens=1024)
    return {
        "answer": answer,
        "sources": [{"ref": i + 1, "source": h["source"],
                     "locator": h["locator"], "score": round(h["score"], 3),
                     "excerpt": h["text"][:280]}
                    for i, h in enumerate(hits)],
    }


REPORT_SYSTEM = """You are a senior data analyst writing an executive report \
from a knowledge base of documents, tables, images and videos.
Write in clean Markdown with these sections:
# {title}
## Executive Summary  (3-5 sentences)
## Key Findings       (numbered, each with evidence and source citations [n])
## Data Highlights    (specific numbers/metrics found in the material)
## Risks & Gaps       (what the material warns about or fails to cover)
## Recommendations    (prioritized, actionable)
Rules: cite sources as [n] matching the context blocks; use ONLY facts from \
the context; quote exact numbers; plain English, no jargon."""


def generate_report(kb: KnowledgeBase, title: str, focus: str = "") -> dict:
    probes = [focus] if focus else []
    probes += ["key findings and metrics", "trends and performance",
               "risks problems issues", "recommendations and actions",
               "summary overview"]
    seen, hits = set(), []
    for q in probes:
        for h in kb.search(q, k=5):
            key = (h["source"], h["locator"], h["text"][:80])
            if key not in seen:
                seen.add(key)
                hits.append(h)
    hits = hits[:18]
    if not hits:
        raise ValueError("Knowledge base is empty — upload files first")

    user = (f"REPORT TITLE: {title}\n"
            + (f"FOCUS: {focus}\n" if focus else "")
            + f"\nCONTEXT:\n{_context_block(hits)}")
    markdown = _generate(REPORT_SYSTEM.replace("{title}", title), user,
                         max_tokens=3000)
    return {
        "markdown": markdown,
        "sources": [{"ref": i + 1, "source": h["source"],
                     "locator": h["locator"]} for i, h in enumerate(hits)],
    }


def report_pdf(markdown: str, title: str) -> bytes:
    """Simple, clean PDF rendering of the markdown report."""
    import io
    import re
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm,
                            rightMargin=18 * mm, topMargin=18 * mm,
                            bottomMargin=18 * mm, title=title)
    NAVY, INK = HexColor("#1a2744"), HexColor("#2b2f36")
    h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20,
                        textColor=NAVY, spaceAfter=8)
    h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=14,
                        textColor=NAVY, spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=10,
                          textColor=INK, leading=15, spaceAfter=4)

    def esc(s: str) -> str:
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)

    story = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if not line.strip():
            story.append(Spacer(1, 2 * mm))
        elif line.startswith("# "):
            story.append(Paragraph(esc(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(esc(line[3:]), h2))
        elif line.startswith(("### ", "#### ")):
            story.append(Paragraph(esc(line.lstrip("# ")), h2))
        elif line.lstrip().startswith(("- ", "* ")):
            story.append(Paragraph("• " + esc(line.lstrip()[2:]), body))
        elif re.match(r"^\s*\d+\.\s", line):
            story.append(Paragraph(esc(line.strip()), body))
        else:
            story.append(Paragraph(esc(line), body))
    doc.build(story)
    return buf.getvalue()
