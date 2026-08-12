"""
ai/llm_client.py — Analytiq
MERGED v2: Existing Groq client preserved + Google Gemini added.

Existing behaviour (unchanged):
  - LLMClient(api_key) constructor
  - chat(messages, system) with tenacity retry
  - chat_safe(messages, system, fallback)
  - config.settings used for model/temperature/max_tokens/timeout

New additions:
  - chat_task(system, user, task) — routes chart→Groq, summary→Gemini
  - status() — provider availability
  - get_client() — module-level singleton helper
  - Gemini loaded from GEMINI_API_KEY secret (optional)
"""

import logging
import os

from groq import Groq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from app.config import config
from app.ai import gemini_client

logger = logging.getLogger(__name__)


# ── Task → provider routing ───────────────────────────────
TASK_ROUTING = {
    "chart_analysis":    "groq",
    "narrative":         "groq",
    "json_output":       "groq",
    "executive_summary": "gemini",
    "insight":           "gemini",
    "root_cause":        "gemini",
    "story":             "gemini",
    "default":           "groq",
}


class LLMClient:

    def __init__(self, api_key: str):
        # ── Existing Groq setup — unchanged ──────────────
        self._client = Groq(api_key=api_key)
        self.model   = config.llm_model
        self._gemini_model = config.gemini_model

    # ─────────────────────────────────────────────────────
    #  EXISTING METHODS — completely unchanged
    #  All other files continue working with no changes
    # ─────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(self, messages: list, system: str = "") -> str:
        """Existing Groq chat — unchanged."""
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        resp = self._client.chat.completions.create(
            messages    = full,
            model       = self.model,
            temperature = config.llm_temperature,
            max_tokens  = config.llm_max_tokens,
            timeout     = config.llm_timeout_sec,
        )
        return resp.choices[0].message.content

    def chat_safe(
        self,
        messages: list,
        system:   str = "",
        fallback: str = '{"tool":"none","params":{},"explanation":"Unable to process."}',
    ) -> str:
        """Existing safe wrapper — unchanged."""
        try:
            return self.chat(messages, system)
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            return fallback

    # ─────────────────────────────────────────────────────
    #  NEW METHODS — for report_narrator.py only
    #  Does NOT affect any existing code
    # ─────────────────────────────────────────────────────

    def chat_task(
        self,
        system:     str,
        user:       str,
        task:       str = "default",
        max_tokens: int = 400,
        force:      str = "",
    ) -> str | None:
        """
        Route request to best provider by task type.
        chart_analysis → Groq  (fast)
        executive_summary → Gemini (deep reasoning), Groq fallback
        Returns text or None — caller should use rule-based fallback on None.
        """
        provider = force or TASK_ROUTING.get(task, "groq")
        order    = (["gemini", "groq"] if provider == "gemini"
                    else ["groq", "gemini"])

        for prov in order:
            try:
                if prov == "groq":
                    result = self._groq_report(system, user, max_tokens)
                elif prov == "gemini" and gemini_client.is_configured():
                    result = self._gemini(system, user, max_tokens)
                else:
                    continue
                if result and result.strip():
                    return result.strip()
            except Exception as e:
                logger.warning(f"[{prov}] task={task} failed: {e}")
                continue

        return None

    def status(self) -> dict:
        return {
            "groq":         True,
            "gemini":       gemini_client.is_configured(),
            "groq_model":   self.model,
            "gemini_model": self._gemini_model,
        }

    # ─────────────────────────────────────────────────────
    #  PRIVATE HELPERS
    # ─────────────────────────────────────────────────────

    def _groq_report(self, system: str, user: str,
                     max_tokens: int) -> str | None:
        """
        Simple Groq call for report tasks.
        Uses low temperature to reduce hallucination.
        No tenacity retry — fast fail preferred for reports.
        """
        try:
            resp = self._client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                model       = self.model,
                temperature = 0.15,
                max_tokens  = max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq report call failed: {e}")
            return None

    def _gemini(self, system: str, user: str,
                max_tokens: int) -> str | None:
        if not gemini_client.is_configured():
            return None
        try:
            return gemini_client.generate_text(
                [user], system=system, max_output_tokens=max_tokens,
                temperature=0.2, timeout_sec=60)
        except Exception as e:
            logger.warning(f"Gemini call failed: {e}")
            return None

    @staticmethod
    def _load_secret(key: str) -> str:
        return os.environ.get(key, "").strip()


# ─────────────────────────────────────────────────────────
#  SINGLETON — used by report_narrator.py
# ─────────────────────────────────────────────────────────

_instance: LLMClient | None = None


def get_client(api_key: str = "") -> LLMClient:
    """Get or create singleton. Used by report_narrator.py."""
    global _instance
    if _instance is None:
        key = api_key or _load_groq_key()
        _instance = LLMClient(api_key=key)
    return _instance


def _load_groq_key() -> str:
    return os.environ.get("GROQ_API_KEY", "").strip()
