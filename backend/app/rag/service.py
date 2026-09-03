"""
rag/service.py — chunking, ingestion, Q&A and report generation
over a knowledge base of documents, tables, images and videos.
"""
from __future__ import annotations

import logging
import os
from typing import List

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


class KbLimitError(ValueError):
    """A knowledge base limit was reached. Separate from other ValueErrors
    so the API can answer 413 rather than 422."""


def check_ingest_limits(kb: KnowledgeBase, new_chunks: int) -> None:
    """Refuse an ingest that would take a knowledge base past its limits.

    The whole KB is held in memory and rewritten on every ingest, so
    unbounded growth is not a slow degradation — it is one user's
    document library taking the process down for everyone. Refusing with
    a message that names the limit and the current count is far better
    than truncating the upload, which would leave the user with a KB that
    silently does not contain what they put in it.
    """
    if len(kb.files) >= config.rag_max_files_per_kb:
        raise KbLimitError(
            f"This knowledge base already holds the maximum of "
            f"{config.rag_max_files_per_kb} files. Delete a file, or create "
            f"a second knowledge base for the rest.")
    if len(kb.chunks) + new_chunks > config.rag_max_chunks_per_kb:
        room = max(0, config.rag_max_chunks_per_kb - len(kb.chunks))
        raise KbLimitError(
            f"This file adds {new_chunks:,} passages and the knowledge base "
            f"has room for {room:,} more (limit "
            f"{config.rag_max_chunks_per_kb:,}). Split the document, or "
            f"start a new knowledge base — nothing was added.")


def ingest_file(kb: KnowledgeBase, filename: str, data: bytes) -> dict:
    passages = extract(filename, data)
    if not passages:
        raise ValueError(f"No text could be extracted from {filename}")
    chunks = chunk_passages(passages)
    check_ingest_limits(kb, len(chunks))
    kind = file_kind(filename)
    kb.add_chunks(chunks, filename, kind)
    preview = passages[0]["text"][:600]
    return {"filename": filename, "kind": kind,
            "chunks_added": len(chunks), "preview": preview,
            "chunks_total": len(kb.chunks),
            "chunks_limit": config.rag_max_chunks_per_kb}


# ── generation ───────────────────────────────────────────

def _generate(system: str, user: str, task: str = "rag_answer",
              max_tokens: int = 2048) -> str:
    """Answer through the routed model for this task.

    This used to be a private ladder — local, then Gemini, then Groq —
    written before models were addressable. It meant OpenRouter,
    Cerebras and Together could never answer a knowledge-base question
    no matter how they were configured, and that the one genuinely
    important thing about it, the local-first privacy posture, was
    buried in the order of three function calls.

    That posture is now declared on the task itself
    (`prefers_local=True` in ai/tasks.py), so it is visible on the
    System page and cannot be undone by reordering an environment
    variable. A knowledge base holds the client's contracts and
    policies; where a model on their own hardware can answer, it does.
    """
    from app.ai import local_llm
    from app.ai.llm_client import get_client

    text = get_client().chat_task(system=system, user=user, task=task,
                                  max_tokens=max_tokens)
    if text:
        return text

    if local_llm.privacy_mode():
        raise RuntimeError(
            "LLM_PRIVACY_MODE is on and no local model answered. Set "
            "LOCAL_LLM_URL and LOCAL_LLM_MODEL (any OpenAI-compatible "
            "server: Ollama, llama.cpp, vLLM, LM Studio). No data was "
            "sent anywhere.")
    raise RuntimeError(
        "No model is configured that can answer from a knowledge base. "
        "Assign one on the System page, or set LOCAL_LLM_URL, "
        "GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, "
        "CEREBRAS_API_KEY or TOGETHER_API_KEY.")


def _context_block(hits: List[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] ({h['source']}, {h['locator']})\n{h['text']}")
    return "\n\n---\n\n".join(lines)


QA_SYSTEM = """You are a senior data analyst answering questions strictly \
from a client's own uploaded material.
Rules:
- Answer ONLY from the provided context. The context is the entire world;
  your own knowledge of the subject is not evidence and must not appear in
  the answer.
- If the context does not contain the answer, say exactly what is missing
  and stop. A short "the uploaded material does not cover X" is a correct
  and useful answer — a plausible answer assembled from adjacent material
  is not.
- Cite sources inline like [1], [2] matching the numbered context blocks.
  Every factual claim needs a citation.
- Quote exact numbers from the context, never invent, round or estimate
  values.
- Be concise and analytical."""


NOT_IN_KB = (
    "Nothing in your uploaded material covers this. The knowledge base was "
    "searched and no passage was close enough to the question to answer "
    "from — rather than assemble an answer out of unrelated text, this is "
    "left unanswered. Upload the document that covers it, or rephrase using "
    "the wording your documents use."
)


def _citation_count(answer: str) -> int:
    import re
    return len(set(re.findall(r"\[(\d+)\]", answer or "")))


def answer_question(kb: KnowledgeBase, question: str, k: int = 6) -> dict:
    """Answer from the client's documents, or say that they do not cover it.

    Returns `grounded` so the caller can distinguish "here is the answer"
    from "your documents do not contain this" — a distinction that
    matters more than the wording of either.
    """
    if not kb.chunks:
        return {"answer": "This knowledge base is empty — upload files first.",
                "sources": [], "grounded": False}

    hits = kb.search(question, k=k)
    if not hits:
        return {"answer": NOT_IN_KB, "sources": [], "grounded": False}

    answer = _generate(
        QA_SYSTEM,
        f"CONTEXT:\n{_context_block(hits)}\n\nQUESTION: {question}",
        task="rag_answer", max_tokens=1024)

    # An answer citing nothing was not written from the passages supplied,
    # whatever it says. Flagging it is honest; suppressing it would throw
    # away a legitimate "the material does not cover this" reply, which
    # also carries no citation.
    cited = _citation_count(answer)
    return {
        "answer": answer,
        "grounded": True,
        "cited_sources": cited,
        "uncited": cited == 0,
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
        if kb.chunks and focus:
            raise ValueError(
                f"Nothing in this knowledge base is about '{focus}'. The "
                f"material is there but none of it matches that focus — "
                f"remove the focus to report on what the documents actually "
                f"cover.")
        raise ValueError("Knowledge base is empty — upload files first")

    user = (f"REPORT TITLE: {title}\n"
            + (f"FOCUS: {focus}\n" if focus else "")
            + f"\nCONTEXT:\n{_context_block(hits)}")
    markdown = _generate(REPORT_SYSTEM.replace("{title}", title), user,
                         task="rag_report", max_tokens=3000)
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
