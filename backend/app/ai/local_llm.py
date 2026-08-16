"""
ai/local_llm.py — an open-source model running on the client's own
hardware, for work that must not leave it.

Two jobs:

1. **Privacy.** Every narrative in this app was written by sending the
   client's column names, sample values and computed figures to Groq or
   Google. For a lot of consulting work that is simply not allowed —
   employee records, patient data, anything under an NDA that names a
   third-party processor. Setting LLM_PRIVACY_MODE=1 makes cloud calls
   impossible rather than merely unlikely: they are refused at the client
   layer, not skipped by convention.

2. **Fallback.** When the cloud providers are down, out of quota, or the
   keys are missing, a local model keeps the app working instead of
   returning "No LLM available".

Talks to any OpenAI-compatible /v1/chat/completions endpoint, which is
what Ollama, llama.cpp's server, vLLM, LM Studio and text-generation-webui
all expose. Nothing here is specific to one of them, and no new dependency
is added — it is a plain HTTP POST.

The app is designed so that no LLM is ever load-bearing: every domain
engine writes its findings deterministically, and the LLM only adds
prose. Privacy mode with no local model configured is therefore a
supported configuration, not a broken one — the reports still build, they
just carry the engine's own wording.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

from app.config import config

logger = logging.getLogger(__name__)


class LocalLLMError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(config.local_llm_url and config.local_llm_model)


def privacy_mode() -> bool:
    """True when no client data may be sent to a third-party API."""
    return bool(config.llm_privacy_mode)


def assert_cloud_allowed(provider: str) -> None:
    """Raise if a cloud provider is about to be called in privacy mode.

    Deliberately an exception rather than a silent skip. A silent skip
    means a misconfiguration shows up as slightly worse prose; an
    exception means it shows up immediately, which is the correct
    trade-off when the alternative is client data on someone else's
    server.
    """
    if privacy_mode():
        raise LocalLLMError(
            f"LLM_PRIVACY_MODE is on: {provider} was not called and no data "
            f"left this machine. Configure LOCAL_LLM_URL/LOCAL_LLM_MODEL to "
            f"generate narratives locally, or turn privacy mode off if this "
            f"dataset may be sent to a third-party API.")


def generate(system: str, user: str, max_tokens: int = 1024,
             temperature: float = 0.2,
             timeout_sec: Optional[float] = None) -> Optional[str]:
    """One completion from the local model, or None if it is unavailable.

    Returns None rather than raising for the ordinary "not configured /
    not running" cases, so callers can fall through to the next provider
    without special-casing.
    """
    if not is_configured():
        return None

    url = config.local_llm_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": config.local_llm_model,
        "messages": (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": user}]),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")

    timeout = timeout_sec or config.local_llm_timeout_sec
    try:
        # A local model is on localhost or the client's own network, so
        # no proxy: sending it through one would defeat the point of
        # privacy mode.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        logger.warning("local model at %s unreachable: %s",
                       config.local_llm_url, e)
        return None
    except (TimeoutError, OSError) as e:
        logger.warning("local model call failed: %s", e)
        return None
    except json.JSONDecodeError:
        logger.warning("local model returned a non-JSON response")
        return None

    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.warning("local model response had an unexpected shape: %s",
                       str(body)[:200])
        return None
    return (text or "").strip() or None


def status() -> dict:
    """What the app will actually do with a narrative request."""
    return {
        "privacy_mode": privacy_mode(),
        "local_configured": is_configured(),
        "local_url": config.local_llm_url if is_configured() else "",
        "local_model": config.local_llm_model if is_configured() else "",
        "cloud_allowed": not privacy_mode(),
    }
