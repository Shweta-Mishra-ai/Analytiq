"""
ai/claude_client.py — Anthropic Claude as the app's language model.

When ANTHROPIC_API_KEY is set, Claude is tried first for chat answers,
report narratives and knowledge-base answers; Groq, Gemini and a local
model remain the fallbacks. Privacy mode refuses it like any cloud
provider: the prompts carry the client's data.

On a provider failure every function returns None, so a caller can move
on to the next provider; the reason is logged. The one exception raised is
the privacy-mode refusal (local_llm.LocalLLMError), which is a decision,
not a fault.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.config import config

logger = logging.getLogger(__name__)

_client = None


def is_configured() -> bool:
    return bool(config.anthropic_api_key)


def _get():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(
            api_key=config.anthropic_api_key,
            timeout=float(config.claude_timeout_sec),
            max_retries=2,          # 408/409/429/5xx and connection errors
        )
    return _client


def _alternating(messages: List[dict]) -> List[dict]:
    """User/assistant turns as the Messages API requires them: starting
    with the user, roles alternating (consecutive turns of one role are
    joined), only text content."""
    out: List[dict] = []
    for m in messages:
        role = "assistant" if m.get("role") == "assistant" else "user"
        text = str(m.get("content") or "").strip()
        if not text:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + text
        else:
            out.append({"role": role, "content": text})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def chat(system: str, messages: List[dict], max_tokens: int = 2048,
         effort: str = "low") -> Optional[str]:
    """Claude's reply to a conversation, or None if it could not answer."""
    if not is_configured():
        return None
    from app.ai.local_llm import assert_cloud_allowed
    assert_cloud_allowed("Claude")
    turns = _alternating(messages)
    if not turns:
        return None
    import anthropic
    try:
        response = _get().beta.messages.create(
            model=config.claude_model,
            max_tokens=max_tokens,
            system=system or anthropic.NOT_GIVEN,
            messages=turns,
            # Effort is the depth/cost lever on current models; "low" suits
            # chat and short narratives, "medium" longer analysis.
            output_config={"effort": effort},
            # If a safety classifier declines, the API re-runs the request
            # on Anthropic's recommended fallback model instead of failing.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError:
        logger.error("Claude rejected ANTHROPIC_API_KEY (401)")
        return None
    except anthropic.RateLimitError as e:
        logger.warning("Claude rate limit after retries: %s", e)
        return None
    except anthropic.APIStatusError as e:
        logger.warning("Claude returned HTTP %s: %s", e.status_code, e)
        return None
    except anthropic.APIConnectionError as e:
        logger.warning("Claude could not be reached: %s", e)
        return None
    if response.stop_reason == "refusal":
        logger.warning("Claude declined the request (%s)",
                       getattr(response.stop_details, "category", None))
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason == "max_tokens":
        logger.info("Claude reply hit max_tokens=%s", max_tokens)
    return text or None


def generate(system: str, user: str, max_tokens: int = 1024,
             effort: str = "low") -> Optional[str]:
    """A single-prompt completion (narratives, knowledge-base answers)."""
    return chat(system, [{"role": "user", "content": user}],
                max_tokens=max_tokens, effort=effort)


def status() -> dict:
    return {"claude": is_configured(), "claude_model": config.claude_model}
