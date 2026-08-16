"""
ai/gemini_client.py — single shared wrapper around the `google-genai`
SDK, used by every Gemini call site in this app (llm_client, RAG
extractors/service, table_extractor, embeddings).

Why this exists instead of each call site importing google.genai
directly:

1. One model name, not five hardcoded ones. Before this module existed,
   `config.gemini_model` was itself set to a model that had already been
   shut down by Google, and `ai/llm_client.py` didn't even read it — it
   hardcoded a *different*, also-dead model name of its own. Centralizing
   means there's exactly one place to update when Google deprecates a
   model again (they do this every few months).

2. A HARD wall-clock timeout that actually works. `google-genai`'s own
   `http_options={"timeout": ...}` is not reliable — there are multiple
   open upstream issues where the SDK silently passes timeout=None to
   the underlying httpx client on some code paths, so a stalled
   connection hangs forever instead of raising:
     https://github.com/googleapis/python-genai/issues/911
     https://github.com/googleapis/python-genai/issues/1893
     https://github.com/pydantic/pydantic-ai/issues/4031
   We don't rely on the SDK to protect us from this. Every call here
   runs in a daemon thread with a deadline this module enforces itself
   — the same pattern proven out in scripts/check_api_keys.py. Without
   this, a single bad network moment could hang a threadpool worker
   (and the request it's serving) indefinitely.

3. Consistent error handling so callers get a clean, catchable
   Exception/None instead of raw SDK internals leaking into user-facing
   error messages.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Optional

from app.config import config

logger = logging.getLogger(__name__)

_client = None
_client_key: Optional[str] = None


def is_configured() -> bool:
    return bool(config.gemini_api_key)


def get_client():
    """Lazily creates (and caches) the genai.Client for the configured
    key. Returns None if no key is set — callers should treat that as
    'Gemini unavailable', not raise.

    Refuses outright in privacy mode: this is the single chokepoint every
    Gemini call in the app goes through, which makes it the right place
    to guarantee that no client data reaches Google.
    """
    global _client, _client_key
    from app.ai.local_llm import assert_cloud_allowed
    assert_cloud_allowed("Gemini")
    if not config.gemini_api_key:
        return None
    if _client is not None and _client_key == config.gemini_api_key:
        return _client
    from google import genai
    _client = genai.Client(api_key=config.gemini_api_key)
    _client_key = config.gemini_api_key
    return _client


def _run_with_hard_timeout(fn, timeout_sec: float):
    """Runs fn() in a daemon thread and gives up after timeout_sec no
    matter what the SDK itself is doing — see module docstring for why
    this is necessary rather than trusting the SDK's own timeout."""
    q: "queue.Queue" = queue.Queue(maxsize=1)

    def _run():
        try:
            q.put(("ok", fn()))
        except Exception as e:
            q.put(("error", e))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    try:
        status, value = q.get(timeout=timeout_sec)
    except queue.Empty:
        raise TimeoutError(
            f"Gemini call timed out after {timeout_sec}s (network issue, "
            f"or the request is larger than this deadline allows)")
    if status == "error":
        raise value
    return value


def generate_text(contents: list, system: Optional[str] = None,
                   json_mode: bool = False, max_output_tokens: int = 2048,
                   temperature: float = 0.2, timeout_sec: float = 60) -> str:
    """contents: list of str / PIL.Image / uploaded-file objects (the SDK
    auto-converts PIL images and accepts File objects from upload_file()
    directly in this list, same as the old SDK did). Raises on any
    failure — callers decide how to surface that."""
    client = get_client()
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    from google.genai import types
    cfg_kwargs: dict[str, Any] = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
    }
    if system:
        cfg_kwargs["system_instruction"] = system
    if json_mode:
        cfg_kwargs["response_mime_type"] = "application/json"

    def _call():
        resp = client.models.generate_content(
            model=config.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        return resp.text or ""

    return _run_with_hard_timeout(_call, timeout_sec)


def upload_file(path: str, timeout_sec: float = 300):
    """Uploads a file and blocks (with a hard deadline covering both the
    upload and Gemini's own processing) until it's ready to use in
    generate_text(). Returns the file object; pass it directly inside a
    `contents` list."""
    client = get_client()
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    def _call():
        f = client.files.upload(file=path)
        deadline = time.time() + timeout_sec
        while f.state.name == "PROCESSING":
            if time.time() > deadline:
                raise TimeoutError("File processing timed out")
            time.sleep(3)
            f = client.files.get(name=f.name)
        if f.state.name != "ACTIVE":
            raise RuntimeError(f"File processing failed: {f.state.name}")
        return f

    # small headroom over the internal polling deadline so the outer
    # hard-timeout isn't racing the inner one for the same budget
    return _run_with_hard_timeout(_call, timeout_sec + 20)


def delete_file(file_name: str) -> None:
    """Best-effort cleanup — never raises."""
    client = get_client()
    if client is None:
        return
    try:
        client.files.delete(name=file_name)
    except Exception:
        logger.debug("delete_file: suppressed exception", exc_info=True)


_TASK_TYPE_MAP = {
    "retrieval_document": "RETRIEVAL_DOCUMENT",
    "retrieval_query": "RETRIEVAL_QUERY",
}


def embed(texts: list[str], task: str = "retrieval_document",
          output_dimensionality: int = 768,
          timeout_sec: float = 30) -> Optional[list[list[float]]]:
    """Returns one embedding vector per input text, or None if Gemini
    isn't configured or the call fails (callers fall back to the local
    hashing embedder — see rag/vector_store.py)."""
    client = get_client()
    if client is None:
        return None

    from google.genai import types
    task_type = _TASK_TYPE_MAP.get(task, task.upper())

    def _call():
        resp = client.models.embed_content(
            model=config.gemini_embed_model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dimensionality,
            ),
        )
        return [e.values for e in resp.embeddings]

    try:
        return _run_with_hard_timeout(_call, timeout_sec)
    except Exception as e:
        logger.warning(f"Gemini embeddings failed: {e}")
        return None
