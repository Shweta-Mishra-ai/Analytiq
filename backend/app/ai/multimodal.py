"""
ai/multimodal.py — reading images and video through the router.

Before this module, every call that looked at a picture went straight to
`gemini_client`, and `rag/extractors._require_gemini()` refused outright
without a Google key. That made vision the one capability nobody could
choose a model for — the exact opposite of what routing is for, and on
the tasks where the choice matters most.

Two consequences of moving these under `resolve_models`:

  * **Vision stops being Gemini-only.** Any model declaring VISION can
    read a table photograph, including a local Gemma over Ollama — which
    means the whole pipeline, upload to report, can run on a machine
    with no API keys and no data leaving it.
  * **The failure names the missing capability.** "No model configured
    that can read images" points at a fixable thing. "GEMINI_API_KEY is
    required" pointed at one vendor, and was wrong as soon as there was
    a second.

Video is deliberately not the same capability. Gemini's Files API
watches a whole clip — visuals, on-screen text and narration together —
and almost nothing else does. Folding it into VISION would let the
fallback pick a model that can read one frame and call that watching a
video.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

from app.ai import providers, routing, tasks

logger = logging.getLogger(__name__)


class NoCapableModel(RuntimeError):
    """Nothing configured can do this job. The message says which job and
    what to do, because this is the error a user actually sees."""


def _require(task_name: str):
    """The models that can serve this task, or an error naming the gap."""
    chain = routing.resolve_models(task_name)
    if chain:
        return chain
    task = tasks.get(task_name) or tasks.TASKS["default"]
    from app.ai import local_llm
    if local_llm.privacy_mode():
        raise NoCapableModel(
            f"Privacy mode is on and no local model can {task.label.lower()}. "
            f"Point LOCAL_LLM_URL at a model that can, or turn privacy mode "
            f"off if this data may be sent to a third-party API. "
            f"No data has left this machine.")
    raise NoCapableModel(
        f"No model is configured that can handle {task.label.lower()}. "
        f"Assign one on the System page. {task.degrades_to}")


def describe_image(image: bytes, prompt: str, system: str = "",
                   task: str = "image_understanding",
                   mime: str = "image/png", max_tokens: int = 1024,
                   json_mode: bool = False,
                   timeout_sec: Optional[float] = 60) -> str:
    """Ask a vision model about an image, trying each capable model.

    Raises rather than returning None: every caller here is a user
    action that fails visibly (an upload, an extraction), so a silent
    empty answer would surface as an unexplained empty result later.
    """
    chain = _require(task)
    errors = []
    for spec in chain:
        provider = providers.get(spec.provider)
        if provider is None:
            continue
        try:
            text = provider.describe_image(
                system=system, user=prompt, image=image, mime=mime,
                max_tokens=max_tokens, model=spec.model,
                json_mode=json_mode, timeout_sec=timeout_sec)
        except Exception as e:                     # noqa: BLE001 — collected
            logger.warning("[%s] %s failed: %s", spec.id, task, e)
            errors.append(f"{spec.id}: {e}")
            continue
        if text and text.strip():
            return text.strip()
        errors.append(f"{spec.id}: empty response")
    raise NoCapableModel(
        "Every model that could read this image failed: " + "; ".join(errors))


def understand_video(data: bytes, ext: str, prompt: str,
                     max_tokens: int = 2048,
                     timeout_sec: float = 120) -> str:
    """Watch a clip whole, rather than sampling frames.

    Currently only the Gemini Files API can do this, and the task
    registry says so. The code is still written against the capability
    rather than the vendor, so the day a second provider offers it, this
    routes to it without a change here.
    """
    chain = _require("video_understanding")
    errors = []
    for spec in chain:
        provider = providers.get(spec.provider)
        if provider is None:
            continue
        handler = getattr(provider, "understand_video", None)
        if handler is None:
            errors.append(f"{spec.id}: this provider has no video path")
            continue
        try:
            text = handler(data=data, ext=ext, prompt=prompt,
                           max_tokens=max_tokens, model=spec.model,
                           timeout_sec=timeout_sec)
        except Exception as e:                     # noqa: BLE001
            logger.warning("[%s] video failed: %s", spec.id, e)
            errors.append(f"{spec.id}: {e}")
            continue
        if text and text.strip():
            return text.strip()
        errors.append(f"{spec.id}: empty response")
    raise NoCapableModel(
        "Every model that could watch this video failed: " + "; ".join(errors))


def gemini_video(data: bytes, ext: str, prompt: str, max_tokens: int = 2048,
                 model: str = "", timeout_sec: float = 120) -> str:
    """The Gemini Files API path, as a provider method would use it.

    Kept here rather than in gemini_client so the temp-file dance and
    the guaranteed cleanup live with the other multimodal code; the
    client stays a thin SDK wrapper.
    """
    from app.ai import gemini_client

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    uploaded = None
    try:
        uploaded = gemini_client.upload_file(path, timeout_sec=300)
        return gemini_client.generate_text(
            [prompt, uploaded], max_output_tokens=max_tokens,
            model=model or None, timeout_sec=timeout_sec) or ""
    finally:
        # Both cleanups are best-effort and must both run: a leaked temp
        # file fills the disk, and a leaked remote file is a client's
        # video sitting on Google's servers longer than it needed to.
        if uploaded is not None:
            try:
                gemini_client.delete_file(uploaded.name)
            except Exception:                      # noqa: BLE001
                logger.debug("could not delete the uploaded video",
                             exc_info=True)
        try:
            os.unlink(path)
        except OSError:
            logger.debug("could not remove the temporary video file")
