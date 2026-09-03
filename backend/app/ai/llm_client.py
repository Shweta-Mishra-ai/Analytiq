"""
ai/llm_client.py — the one way this app asks a model for words.

What changed and why: this file used to be Groq-shaped by construction.
`from groq import Groq` sat at the top, the client object *was* a Groq
client, and every other provider had to be special-cased around it.
Gemini was bolted on as an `elif`; anything else — a local model, a free
OpenRouter slug, a client's own gateway — was not expressible without
editing this file.

Now the providers live in `ai/providers.py` as data, and this file only
does the two things that are genuinely about *this app*:

  1. **Routing.** Which provider gets first refusal on which kind of
     work, and the order everything else falls through. Both come from
     config, so a deployment re-routes without a code change.
  2. **Degrading well.** Nothing here is load-bearing. Every narrative
     in this app has a deterministic fallback written by the engine that
     computed the numbers, so "no provider configured" is a supported
     state, not an outage. chat_task returns None and the caller writes
     the sentence itself.

The public surface is unchanged — LLMClient(api_key), .chat(),
.chat_safe(), .chat_task(), .status(), get_client() — so no call site
needed touching.
"""

from __future__ import annotations

import logging
import os

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import config
from app.ai import providers, routing, tasks
from app.ai.capabilities import Capability as C

logger = logging.getLogger(__name__)


# ── Task → provider routing ───────────────────────────────
# The defaults reflect what each provider is actually good at here:
# Groq and the other OpenAI-dialect free tiers are fast, which suits the
# many short per-chart calls; Gemini holds a longer argument together,
# which suits the summary and root-cause work. Override any row with
# LLM_ROUTING="executive_summary=openrouter,narrative=local".
DEFAULT_TASK_ROUTING = {
    "chart_analysis":    "groq",
    "narrative":         "groq",
    "json_output":       "groq",
    "executive_summary": "gemini",
    "insight":           "gemini",
    "root_cause":        "gemini",
    "story":             "gemini",
    "default":           "groq",
}


def _parse_routing(raw: str) -> dict:
    """LLM_ROUTING="task=provider,task=provider" → dict.

    Unparseable entries are dropped with a warning rather than raising:
    a typo in one environment variable should not stop the service from
    starting, and the self-check reports the effective routing so the
    typo is still visible.
    """
    out = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            logger.warning("LLM_ROUTING: ignoring %r (expected task=provider)",
                           part)
            continue
        task, prov = part.split("=", 1)
        task, prov = task.strip(), prov.strip()
        if not task or not prov:
            continue
        out[task] = prov
    return out


def task_routing() -> dict:
    """Effective routing: the defaults, with LLM_ROUTING layered on top."""
    routing = dict(DEFAULT_TASK_ROUTING)
    routing.update(_parse_routing(config.llm_routing))
    return routing


def fallback_order() -> list[str]:
    names = [n.strip() for n in (config.llm_provider_order or "").split(",")]
    return [n for n in names if n]


def resolve_chain(task: str = "default", force: str = "") -> list[str]:
    """The providers this call will try, in order, already filtered to
    the ones that could actually answer.

    Privacy mode collapses the chain to the local model alone — not by
    dropping the cloud entries quietly, but because a cloud call in
    privacy mode raises at the provider layer. Keeping them out of the
    chain means the failure reads as "no local model configured", which
    is the true problem, rather than a stack of refusals.
    """
    from app.ai import local_llm

    if local_llm.privacy_mode():
        chain = ["local"]
    else:
        preferred = force or task_routing().get(task, "default")
        chain = [preferred] + [n for n in fallback_order() if n != preferred]

    seen, out = set(), []
    for name in chain:
        if name in seen:
            continue
        seen.add(name)
        p = providers.get(name)
        if p is None:
            logger.warning("unknown provider %r in routing/order", name)
            continue
        if p.is_configured():
            out.append(name)
    return out


class LLMClient:
    """Provider-agnostic. Holds no vendor client, only routing."""

    def __init__(self, api_key: str = ""):
        # Historic signature: callers passed the Groq key positionally.
        # Honour it by filling the config slot when it is empty, so an
        # explicitly-passed key still works, without making Groq special
        # anywhere else in this file.
        if api_key and not config.groq_api_key:
            config.groq_api_key = api_key.strip()
        self.model = config.llm_model
        self._gemini_model = config.gemini_model

    # ─────────────────────────────────────────────────────
    #  Conversation API — used by the chat page and the
    #  tool dispatcher, which both send a message history.
    # ─────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(self, messages: list, system: str = "") -> str:
        """One completion from the first provider that answers.

        Raises if none does — the retry above then gets a second and
        third go at the whole chain, since the common failure is a rate
        limit that clears in a second or two.
        """
        chain = routing.resolve_models("tool_call")
        if not chain:
            raise RuntimeError(
                "No model is configured that can return structured JSON. Set "
                "one of GROQ_API_KEY, OPENROUTER_API_KEY, CEREBRAS_API_KEY, "
                "TOGETHER_API_KEY or GEMINI_API_KEY, or point LOCAL_LLM_URL "
                "at a local model.")

        errors = []
        for spec in chain:
            provider = providers.get(spec.provider)
            if provider is None:
                continue
            try:
                text = provider.complete(
                    messages, system=system,
                    max_tokens=config.llm_max_tokens,
                    temperature=config.llm_temperature,
                    timeout_sec=config.llm_timeout_sec,
                    model=spec.model, json_mode=True)
            except Exception as e:                 # noqa: BLE001 — collected
                errors.append(f"{spec.id}: {e}")
                logger.warning("[%s] chat failed: %s", spec.id, e)
                continue
            if text and text.strip():
                # Deliberately not stored on self. This used to write
                # self.model, which the narrative cache keys on — so a
                # chat turn could silently change the key a later
                # chat_task wrote under, and two different prompts could
                # collide on one cache entry.
                self.last_model = spec.id
                return text.strip()
            errors.append(f"{spec.id}: empty response")

        raise RuntimeError("; ".join(errors) or "no model answered")

    def chat_safe(
        self,
        messages: list,
        system:   str = "",
        fallback: str = '{"tool":"none","params":{},"explanation":"Unable to process."}',
    ) -> str:
        """chat() but never raises — the dispatcher needs a parseable
        answer more than it needs the truth about why one failed."""
        try:
            return self.chat(messages, system)
        except Exception as e:                     # noqa: BLE001
            logger.error(f"LLM failed: {e}")
            return fallback

    # ─────────────────────────────────────────────────────
    #  Report/narrative API
    # ─────────────────────────────────────────────────────

    def chat_task(
        self,
        system:     str,
        user:       str,
        task:       str = "default",
        max_tokens: int = 400,
        force:      str = "",
    ) -> str | None:
        """Route by task, fall through the chain, return None if nothing
        answers — the caller then uses its own rule-based wording."""
        from app.services.llm_cache import llm_cache

        spec = tasks.get(task) or tasks.TASKS["default"]
        needs_json = C.JSON in spec.requires
        chain = routing.resolve_models(task, force=force)

        # Regenerating a report re-ran every call: the same summary for the
        # same dataset, billed again and adding thirty seconds to a rebuild
        # that changed nothing. Keyed on the prompt, so a cleaned dataset
        # produces a different prompt and correctly misses — and on the
        # model that will actually answer, so switching a task to a
        # different model does not serve the old model's wording.
        cache_model = chain[0].id if chain else ""
        cached = llm_cache.get(system, user, task, cache_model)
        if cached:
            logger.debug("llm cache hit for task=%s model=%s", task, cache_model)
            return cached

        for model_spec in chain:
            provider = providers.get(model_spec.provider)
            if provider is None:
                continue
            try:
                result = provider.generate(system, user, max_tokens=max_tokens,
                                           temperature=0.15,
                                           model=model_spec.model,
                                           json_mode=needs_json)
            except Exception as e:                 # noqa: BLE001
                logger.warning("[%s] task=%s failed: %s", model_spec.id, task, e)
                continue
            if result and result.strip():
                text = result.strip()
                # Keyed on the model that answered, which is not always
                # the one asked first.
                llm_cache.put(system, user, task, model_spec.id, text)
                if model_spec.id != cache_model:
                    llm_cache.put(system, user, task, cache_model, text)
                return text

        return None

    # ─────────────────────────────────────────────────────
    #  Introspection
    # ─────────────────────────────────────────────────────

    def status(self) -> dict:
        """What is configured, without calling anything.

        `status()` answers "what did this deployment set up"; the
        /api/admin/llm-check endpoint answers "does it work" — those are
        different questions and a key can pass the first and fail the
        second.
        """
        from app.ai import local_llm

        rows = {}
        for p in providers.all_providers():
            rows[p.name] = {
                # The name is repeated inside the row, not left implicit
                # in the dict key: callers iterate the values (the UI
                # renders a list of cards) and a row that cannot say
                # which provider it is has to be threaded back to its key
                # by hand — which the UI got wrong.
                "name": p.name,
                "label": p.label,
                "configured": p.is_configured(),
                "model": p.model,
                "free": p.free,
                "local": p.local,
                "missing": p.missing(),
            }
        return {
            # Kept for the existing callers that read these two keys.
            # `.get` rather than `[...]`: the registry is data, and a
            # deployment that drops a provider from it should lose a row,
            # not crash the status page.
            "groq":         rows.get("groq", {}).get("configured", False),
            "gemini":       rows.get("gemini", {}).get("configured", False),
            "groq_model":   self.model,
            "gemini_model": self._gemini_model,
            # The full picture.
            "providers":    rows,
            "configured":   providers.configured_names(),
            "routing":      task_routing(),
            "order":        fallback_order(),
            "privacy_mode": local_llm.privacy_mode(),
            "any_available": bool(resolve_chain()),
        }

# ─────────────────────────────────────────────────────────
#  SINGLETON
# ─────────────────────────────────────────────────────────

_instance: LLMClient | None = None


def get_client(api_key: str = "") -> LLMClient:
    global _instance
    if _instance is None:
        _instance = LLMClient(api_key=api_key or _load_groq_key())
    return _instance


def reset_client() -> None:
    """Drop the singleton so the next call re-reads config. Used by the
    tests, and by the self-check after an environment change."""
    global _instance
    _instance = None


def _load_groq_key() -> str:
    return os.environ.get("GROQ_API_KEY", "").strip()
