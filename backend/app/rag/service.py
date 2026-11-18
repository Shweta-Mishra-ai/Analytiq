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
from app.rag.vector_store import KnowledgeBase, kb_lock

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


def check_ingest_limits(kb: KnowledgeBase, new_chunks: int, replacing: bool = False) -> None:
    """Refuse an ingest that would take a knowledge base past its limits.

    The whole KB is held in memory and rewritten on every ingest, so
    unbounded growth is not a slow degradation — it is one user's
    document library taking the process down for everyone. Refusing with
    a message that names the limit and the current count is far better
    than truncating the upload, which would leave the user with a KB that
    silently does not contain what they put in it.
    """
    if not replacing and len(kb.files) >= config.rag_max_files_per_kb:
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
    kind = file_kind(filename)
    # Extraction above is the slow part and runs unlocked; the check and the
    # append must happen together or concurrent uploads race (see kb_lock).
    with kb_lock(kb):
        replaced = sum(1 for c in kb.chunks if c.get("source") == filename)
        check_ingest_limits(kb, len(chunks) - replaced,
                            replacing=any(f.get("filename") == filename for f in kb.files))
        kb.add_chunks(chunks, filename, kind)
    preview = passages[0]["text"][:600]
    return {"filename": filename, "kind": kind, "replaced": bool(replaced),
            "chunks_added": len(chunks), "preview": preview,
            "chunks_total": len(kb.chunks),
            "chunks_limit": config.rag_max_chunks_per_kb}


# ── generation ───────────────────────────────────────────

def _gemini_generate(system: str, user: str, max_tokens: int = 2048) -> Optional[str]:
    if not config.gemini_api_key:
        return None
    try:
        from app.ai import gemini_client
        return gemini_client.generate_text(
            [user], system=system, max_output_tokens=max_tokens,
            timeout_sec=60).strip() or None
    except Exception as e:
        logger.warning(f"Gemini generation failed: {e}")
        return None


def _groq_generate(system: str, user: str, max_tokens: int = 2048) -> Optional[str]:
    if not config.groq_api_key:
        return None
    try:
        from app.ai.llm_client import get_client
        # force: Claude was already tried by _generate; don't call it twice.
        return get_client(config.groq_api_key).chat_task(
            system, user, task="default", max_tokens=max_tokens, force="groq")
    except Exception as e:
        logger.warning(f"Groq generation failed: {e}")
        return None


def _local_generate(system: str, user: str, max_tokens: int = 2048) -> Optional[str]:
    from app.ai import local_llm
    return local_llm.generate(system, user, max_tokens=max_tokens)


class NoModelAvailable(RuntimeError):
    """No language model could be reached; nothing was sent anywhere."""


def _generate(system: str, user: str, max_tokens: int = 2048) -> str:
    """Answer with a local model where one is available or required.

    A knowledge base is the most sensitive thing in this app — it holds
    the client's own contracts, policies and reports — so privacy mode
    routes it to a model on their hardware and never falls back to a
    cloud provider.
    """
    from app.ai import local_llm

    if local_llm.privacy_mode():
        text = _local_generate(system, user, max_tokens)
        if not text:
            logger.warning("knowledge base: LLM_PRIVACY_MODE is on and no local model "
                           "answered; set LOCAL_LLM_URL and LOCAL_LLM_MODEL")
            raise NoModelAvailable(
                "This workspace keeps documents on its own server, and its AI model "
                "did not answer. No data was sent anywhere.")
        return text

    from app.ai import claude_client
    text = (_local_generate(system, user, max_tokens)
            or claude_client.generate(system, user, max_tokens, effort="medium")
            or _gemini_generate(system, user, max_tokens)
            or _groq_generate(system, user, max_tokens))
    if not text:
        # The client reads the message; the operator reads the log.
        logger.info("knowledge base: no model available (set ANTHROPIC_API_KEY, "
                    "GEMINI_API_KEY, GROQ_API_KEY or LOCAL_LLM_URL to write answers)")
        raise NoModelAvailable("No AI model is switched on for this workspace.")
    return text


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


_SENTENCE_END = r"(?<=[.!?])\s+|\n+"


def _extractive_answer(question: str, hits: List[dict], why: str,
                       max_quotes: int = 4) -> str:
    """Quote the sentences of the retrieved passages that bear on the question.

    Used when no model can write an answer. The search has already found
    the right material; returning only an error threw that away. Quoting
    is the honest form of it: the words are the document's own, each with
    its citation, and nothing is composed or inferred.
    """
    import re
    from app.rag.vector_store import _tokens

    q = set(_tokens(question))
    scored = []
    for ref, h in enumerate(hits, 1):
        for order, sent in enumerate(re.split(_SENTENCE_END, h["text"])):
            sent = " ".join(sent.split())
            if len(sent) < 12:
                continue
            overlap = len(q & set(_tokens(sent)))
            if overlap:
                # Earlier passages ranked higher; within one, keep reading order.
                scored.append((-overlap, ref, order, sent[:400]))
    scored.sort()
    # A sentence sharing one word with the question ("days" in a line about
    # annual leave) is not evidence; keep those close to the best match.
    if scored:
        need = -(-(-scored[0][0]) // 2)          # ceil(best overlap / 2)
        scored = [t for t in scored if -t[0] >= need]
    picked = sorted(scored[:max_quotes], key=lambda t: (t[1], t[2]))
    lines = [f"> {sent} [{ref}]" for _, ref, _, sent in picked]
    if not lines:      # every hit matched on words split across sentences
        lines = [f"> {' '.join(hits[0]['text'].split())[:400]} [1]"]
    return ("**No AI model is switched on, so this is not a written "
            "answer** — these are the passages in your documents that match "
            "the question, quoted word for word. Read them to judge whether "
            "they answer it.\n\n" + "\n>\n".join(lines)
            + f"\n\n_{why}_")


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

    extractive = False
    try:
        answer = _generate(
            QA_SYSTEM,
            f"CONTEXT:\n{_context_block(hits)}\n\nQUESTION: {question}",
            max_tokens=1024)
    except NoModelAvailable as e:
        answer, extractive = _extractive_answer(question, hits, str(e)), True

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
        "extractive": extractive,
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
    seen, hits, sections = set(), [], []
    for q in probes:
        mine = []
        for h in kb.search(q, k=5):
            key = (h["source"], h["locator"], h["text"][:80])
            if key not in seen and len(hits) < 18:
                seen.add(key)
                hits.append(h)
                mine.append(len(hits))
        sections.append((q, mine))
    if not hits and kb.chunks and not focus:
        # None of the general probes matched closely enough (a refund
        # policy shares no words with "key findings and metrics"), which was
        # reported as an empty knowledge base. A report on the whole
        # material reads the material: passages spread across it, in order.
        step = max(1, len(kb.chunks) // 18)
        seen_text = set()
        for c in kb.chunks[::step]:
            if c["text"] not in seen_text:
                seen_text.add(c["text"])
                hits.append(c)
        hits = hits[:18]
        sections = [("", list(range(1, len(hits) + 1)))]
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
    try:
        markdown, extractive = _generate(REPORT_SYSTEM.replace("{title}", title), user,
                                         max_tokens=3000), False
    except NoModelAvailable as e:
        markdown, extractive = _extractive_report(title, focus, hits, sections, str(e)), True
    # The [n] citations name the passages here; without the list the reader
    # had numbers that pointed nowhere, on the page and in the PDF.
    markdown = markdown.rstrip() + "\n\n## Sources\n\n" + "\n".join(
        f"{i}. {h['source']} — {h['locator']}" for i, h in enumerate(hits, 1))
    return {
        "markdown": markdown,
        "extractive": extractive,
        "sources": [{"ref": i + 1, "source": h["source"],
                     "locator": h["locator"]} for i, h in enumerate(hits)],
    }


_SECTION_NAMES = {"key findings and metrics": "Key findings and figures",
                  "trends and performance": "Trends and performance",
                  "risks problems issues": "Risks and issues",
                  "recommendations and actions": "Recommendations and actions",
                  "summary overview": "Overview", "": "What the documents say"}


def _extractive_report(title: str, focus: str, hits: List[dict], sections, why: str) -> str:
    """The report's sections filled with the documents' own sentences, each
    cited, when no model can write one. A report request that only failed
    threw away passages the search had already found."""
    import re
    out = [f"# {title}",
           f"**{why} This report quotes your documents rather than summarising "
           "them**: each section lists the passages that match it, word for word; "
           "the number after each is its source, listed at the end."]
    for probe, refs in sections:
        quotes = []
        for ref in refs[:3]:
            sentences = [" ".join(x.split()) for x in re.split(_SENTENCE_END, hits[ref - 1]["text"])]
            text = " ".join(x for x in sentences if x)[:400]
            if text:
                quotes.append(f"> {text} [{ref}]")
        if quotes:
            name = f"On '{focus}'" if focus and probe == focus else _SECTION_NAMES.get(probe, probe.capitalize())
            out.append(f"## {name}\n\n" + "\n>\n".join(quotes))
    return "\n\n".join(out)


def report_pdf(markdown: str, title: str) -> bytes:
    """Simple, clean PDF rendering of the markdown report."""
    import io
    import re
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer)
    from app.engines.pdf_fonts import FONT_BOLD, FONT_REG

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm,
                            rightMargin=18 * mm, topMargin=18 * mm,
                            bottomMargin=18 * mm, title=title)
    NAVY, INK = HexColor("#1a2744"), HexColor("#2b2f36")
    h1 = ParagraphStyle("h1", fontName=FONT_BOLD, fontSize=20,
                        textColor=NAVY, spaceAfter=8)
    h2 = ParagraphStyle("h2", fontName=FONT_BOLD, fontSize=14,
                        textColor=NAVY, spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("body", fontName=FONT_REG, fontSize=10,
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
