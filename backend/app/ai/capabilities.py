"""
ai/capabilities.py — what a model can do, and what a job needs.

The app used to route work to a *provider*. That is the wrong unit.
Groq serves both a fast 70B and a reasoning model; OpenRouter serves
hundreds. "Send the summary to Gemini" cannot express "send the
twenty chart captions to something cheap and the root-cause reasoning
to something that can actually reason", which is the choice that
decides both output quality and the bill.

Routing by capability makes that choice expressible, and — more
importantly — makes one class of failure impossible. There are two
ways a wrong model hurts:

  * **Loudly.** A text-only model handed an image fails outright. Bad,
    but at least visible.
  * **Quietly.** A small chat model handed a root-cause question
    answers confidently and worse. Nobody notices for months.

The second is why capability gates the *fallback chain* and not just
the first pick. A vision task falls back only to another vision model,
or to nothing at all — and nothing at all is a supported outcome here,
because every narrative in this app has a deterministic engine behind
it.

Each capability below earns its place by distinguishing a decision the
app actually makes. Adding one nothing routes on would be inventing
configuration.
"""
from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class Capability(str, Enum):
    """str-valued so these serialize straight to JSON for the API and
    the routing file, and compare equal to their own names in config."""

    #: Plain prose. The floor — any generative model has it.
    TEXT = "text"

    #: Structured output that will be *parsed*, not read. The chat
    #: dispatcher turns the reply into {tool, params}; table extraction
    #: turns it into a DataFrame. A model that merely tends to produce
    #: valid JSON is not the same as one asked for it at the API level.
    JSON = "json"

    #: Long context and multi-step argument: root cause, an executive
    #: summary, answering from eighteen retrieved passages with
    #: citations. Distinct from TEXT because the failure is invisible.
    REASONING = "reasoning"

    #: Reads an image. A photo of a table, a screenshot, a chart.
    VISION = "vision"

    #: Native video understanding — visuals, on-screen text and audio
    #: together, over a whole clip. Deliberately not folded into
    #: VISION: essentially only Gemini's Files API does this, and
    #: collapsing them would let the fallback pick a model that can
    #: read one frame and not a video.
    VIDEO = "video"

    #: Produces embedding vectors. Not a generative capability at all;
    #: it shares the registry only because it shares the providers.
    EMBEDDING = "embedding"

    #: Creates an image. Deliberately narrow in this app — see
    #: engines/pdf/theme.py for the only place output may land.
    IMAGE_GEN = "image_gen"


#: Everything a generative text model is assumed to do at minimum.
TEXT_ONLY = frozenset({Capability.TEXT})

#: Human-readable, for the UI and for error messages. The wording is
#: aimed at whoever is choosing a model, not at a developer.
DESCRIPTIONS = {
    Capability.TEXT: "Writes prose",
    Capability.JSON: "Returns structured JSON that gets parsed",
    Capability.REASONING: "Long context and multi-step reasoning",
    Capability.VISION: "Reads images",
    Capability.VIDEO: "Understands video natively",
    Capability.EMBEDDING: "Produces embedding vectors",
    Capability.IMAGE_GEN: "Generates images",
}


def parse(values) -> frozenset[Capability]:
    """Turn whatever config or a JSON body supplied into capabilities.

    Unknown names are dropped rather than raising: this parses a user's
    typed input and an env var, and one bad word should cost that word,
    not the whole deployment's routing.
    """
    out = set()
    for value in values or ():
        try:
            out.add(Capability(str(value).strip().lower()))
        except ValueError:
            # Dropped, not fatal — but not silent either: this is a
            # typed capability that will now be missing from a model,
            # and the effect (a task refusing to route to it) is
            # otherwise hard to trace back to a misspelling.
            logger.info("ignoring unknown capability %r; known ones are %s",
                        value, ", ".join(c.value for c in Capability))
            continue
    return frozenset(out)


def names(caps) -> list[str]:
    """Sorted capability names, for stable output in APIs and tests."""
    return sorted(c.value for c in caps)


def describe_gap(required, available) -> str:
    """Why a model cannot do a job, in the words of whoever has to fix it.

    Used at the moment a routing choice is *saved*, so a bad assignment
    is refused with a reason rather than accepted and silently skipped
    at the point of use — which would look like the model simply never
    being called.
    """
    missing = sorted(set(required) - set(available))
    if not missing:
        return ""
    wanted = ", ".join(DESCRIPTIONS[c].lower() for c in missing)
    return f"it cannot do what this task needs: {wanted}"
