"""
ai/tasks.py — the jobs this app gives a model, and what each one needs.

One row per job the code actually does. That constraint is the point:
the previous routing table declared eight tasks and the application only
ever passed three of them, so four rows sat in the System page looking
like settings while changing nothing. A test at the bottom of the suite
asserts every task here is reachable from a real call site, so that
cannot come back.

Three fields carry most of the weight:

`requires` — the capabilities a model must have to be given this job at
all. This gates the *fallback chain*, not just the first choice. A
vision task falls back only to another vision model; there is no
arrangement of configuration that sends a photograph to a text model.

`min_context` — a number, not a capability. "Long context" as a boolean
cannot say that the deep-dive report needs 64k and a chart caption needs
four. Models below the line are filtered out.

`degrades_to` — one plain sentence about what happens when no capable
model is configured. Every routing table should be able to answer that
question, and a table that cannot is decoration. In this app the answer
is usually good news: the analysis engines compute every figure
themselves and write their own findings, so a missing model costs
polish, not correctness. Where that is *not* true — RAG has no
deterministic answer to fall back on — the sentence says so plainly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.ai.capabilities import Capability as C

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str
    description: str
    requires: frozenset
    #: Smallest context window that can hold this job's prompt.
    min_context: int = 0
    #: What ships in the report when nothing capable is configured.
    degrades_to: str = ""
    #: The model to use when nobody has said otherwise. Empty means the
    #: task is off unless someone turns it on — used for cover art,
    #: which must not appear in a deliverable by accident.
    default_model: str = ""
    #: Ordering preference among equally capable models.
    prefers: str = "balanced"          # fast | balanced | deep
    #: Off until someone assigns a model. Distinct from "has no default":
    #: an ordinary task with no default still falls through to any
    #: capable model, which is what makes the app work out of the box.
    #: An opt-in task does not — nothing happens unless it was asked
    #: for. Used where a model appearing by accident would be a defect
    #: rather than a convenience: generated artwork in a client's
    #: deliverable, and a rewrite of the most scrutinised paragraph in
    #: the report.
    opt_in: bool = False
    #: Put models running on the client's own hardware first, even when
    #: a cloud model would be picked otherwise. This is a privacy
    #: posture, not a performance one, and it is declared rather than
    #: left to emerge from the provider order — a guarantee that depends
    #: on an environment variable being in the right order is not a
    #: guarantee.
    prefers_local: bool = False


TASKS: dict[str, TaskSpec] = {
    "chart_caption": TaskSpec(
        name="chart_caption",
        label="Chart captions",
        description=(
            "Turns the figures already computed for one chart into a "
            "sentence of prose. Runs about twenty times per report."),
        requires=frozenset({C.TEXT}),
        min_context=8_000,
        # Cheapest and fastest wins here: it is the highest-volume call
        # in the app and the shortest output, and the numbers are the
        # engine's either way.
        default_model="groq/llama-3.1-8b-instant",
        prefers="fast",
        degrades_to=(
            "Each chart carries the wording its own engine wrote — the same "
            "figures, phrased more plainly.")),

    "tool_call": TaskSpec(
        name="tool_call",
        label="Chat commands",
        description=(
            "Turns a question typed on the Chat page into a tool call the "
            "app can run. The reply is parsed, not read."),
        requires=frozenset({C.TEXT, C.JSON}),
        min_context=8_000,
        default_model="groq/llama-3.3-70b-versatile",
        prefers="fast",
        degrades_to=(
            "The Chat page cannot answer without a model; it says so rather "
            "than guessing.")),

    "executive_summary": TaskSpec(
        name="executive_summary",
        label="Executive summary polish",
        description=(
            "Rewrites the summary the analysis engine already computed, "
            "keeping every figure, improving only the prose. Off unless a "
            "model is assigned — the deterministic wording is the default "
            "for the most scrutinised paragraph in the deliverable."),
        requires=frozenset({C.TEXT, C.REASONING}),
        min_context=32_000,
        default_model="",
        opt_in=True,
        prefers="deep",
        degrades_to=(
            "The summary the analysis engine wrote is used as-is. This is "
            "the default and it is a sound report.")),

    "rag_answer": TaskSpec(
        name="rag_answer",
        label="Knowledge base answers",
        description=(
            "Answers a question strictly from retrieved passages, with "
            "citations back to the source document."),
        requires=frozenset({C.TEXT, C.REASONING}),
        min_context=32_000,
        default_model="gemini/gemini-3.6-flash",
        prefers="deep",
        # A knowledge base holds the client's contracts, policies and
        # internal reports — the most sensitive thing in this app. Where
        # a local model can answer, it should.
        prefers_local=True,
        degrades_to=(
            "Nothing — a knowledge base answer has no deterministic "
            "equivalent. The question is refused with a reason.")),

    "rag_report": TaskSpec(
        name="rag_report",
        label="Knowledge base report",
        description=(
            "Writes a multi-section report over up to eighteen retrieved "
            "passages."),
        requires=frozenset({C.TEXT, C.REASONING}),
        min_context=64_000,
        default_model="gemini/gemini-3.6-flash",
        prefers="deep",
        prefers_local=True,
        degrades_to=(
            "Nothing — this report exists only if a model writes it.")),

    "table_extraction": TaskSpec(
        name="table_extraction",
        label="Tables from photos and video",
        description=(
            "Reads a photograph or screenshot of a table and returns it as "
            "real, analysable data."),
        requires=frozenset({C.VISION, C.JSON}),
        default_model="gemini/gemini-3.6-flash",
        prefers="balanced",
        degrades_to=(
            "The upload is refused, naming the missing capability. There is "
            "no way to read an image without a model that can see.")),

    "image_understanding": TaskSpec(
        name="image_understanding",
        label="Images in the knowledge base",
        description=(
            "Describes an uploaded image so its content becomes searchable "
            "alongside the documents."),
        requires=frozenset({C.VISION, C.TEXT}),
        default_model="gemini/gemini-3.6-flash",
        degrades_to="The image is not indexed; the rest of the upload is."),

    "video_understanding": TaskSpec(
        name="video_understanding",
        label="Video in the knowledge base",
        description=(
            "Watches a whole clip — visuals, on-screen text and narration "
            "together — rather than sampling frames."),
        requires=frozenset({C.VIDEO}),
        default_model="gemini/gemini-3.6-flash",
        prefers="deep",
        degrades_to=(
            "The video is not indexed. Few models can do this; Gemini is "
            "the only one this catalogue knows.")),

    "embedding": TaskSpec(
        name="embedding",
        label="Knowledge base embeddings",
        description=(
            "Turns passages into vectors so a search can find them by "
            "meaning rather than by shared words."),
        requires=frozenset({C.EMBEDDING}),
        default_model="gemini/gemini-embedding-001",
        degrades_to=(
            "A local sentence model, and below that a keyword index. "
            "Retrieval still works; it matches wording more than meaning.")),

    "cover_art": TaskSpec(
        name="cover_art",
        label="Report cover artwork",
        description=(
            "Generates a decorative background for the report cover and the "
            "deck title slide. Never touches a chart, a table or any figure "
            "— it is built from the report title alone and is labelled as "
            "an illustration wherever it appears."),
        requires=frozenset({C.IMAGE_GEN}),
        default_model="",
        opt_in=True,
        degrades_to=(
            "The flat cover colour the report already uses. This is the "
            "default, and for most deliverables it is the better one.")),

    "default": TaskSpec(
        name="default",
        label="Anything else",
        description="Used when a caller names no task.",
        requires=frozenset({C.TEXT}),
        min_context=4_000,
        default_model="groq/llama-3.3-70b-versatile",
        degrades_to="The caller's own wording."),
}


#: Task names that used to exist, and where their configuration now
#: applies. Silently dropping a row from someone's LLM_ROUTING would
#: change behaviour with no signal — the exact failure this whole change
#: exists to remove — so a stale name keeps working and the System page
#: says it is stale.
DEPRECATED_ALIASES = {
    "chart_analysis": "chart_caption",
    "json_output": "tool_call",
    "narrative": "chart_caption",
    "insight": "chart_caption",
    "root_cause": "executive_summary",
    "story": "executive_summary",
}


def get(name: str) -> Optional[TaskSpec]:
    """A task by name, following a deprecated alias if that is what was
    given. Returns None for a name nothing knows."""
    key = (name or "").strip()
    if key in TASKS:
        return TASKS[key]
    aliased = DEPRECATED_ALIASES.get(key)
    if aliased:
        logger.debug("task %r is a stale name for %r", key, aliased)
        return TASKS.get(aliased)
    logger.debug("no task named %r; the caller will get the default", key)
    return None


def resolve_name(name: str) -> str:
    """The current name for a possibly-stale task name."""
    key = (name or "").strip()
    if key in TASKS:
        return key
    return DEPRECATED_ALIASES.get(key, "default")


def all_tasks() -> list[TaskSpec]:
    return list(TASKS.values())
