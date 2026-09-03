"""
ai/providers.py — one shape for every model this app can talk to.

Before this module, the LLM layer was Groq-shaped by construction: the
Groq SDK was imported at the top of llm_client.py and the client object
*was* a Groq client, so "add another provider" meant special-casing at
every call site. Anything that wasn't Groq or Gemini was not expressible.

Here a provider is data — a name, a base URL, a model, a key — plus one
method that turns a system/user pair into text. Adding a provider is
adding a row. Six are registered out of the box; five of them speak the
OpenAI /v1/chat/completions dialect, which is the de-facto standard that
Groq, OpenRouter, Together, Cerebras, Ollama, llama.cpp, vLLM and LM
Studio all expose, so they share a single adapter and differ only in
their row.

On cost and licensing, which is what the choice of provider is actually
about:

  * **groq** — free tier, no card. Serves openly-licensed weights
    (Llama, Gemma, Qwen) on their own hardware.
  * **cerebras** — free tier, same openly-licensed weights, very fast.
  * **openrouter** — the `:free` model slugs cost nothing and cover
    Llama 3.3 70B, DeepSeek and Qwen. One key, many models.
  * **together** — small free credit, then paid; openly-licensed weights.
  * **ollama** — the client's own hardware. No key, no cost, no data
    leaving the machine. The only provider permitted in privacy mode.
  * **gemini** — Google's own weights, not open. Free tier exists.

So "open-source and free" is the default path through this module, not a
fallback: with no keys at all, a machine running Ollama is fully
functional, and with only a Groq or OpenRouter free key the app is
fully functional without anyone entering card details.

Every provider is optional and every narrative in this app has a
deterministic fallback, so zero configured providers is a supported
state — reports still build, in the engines' own wording.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from app.config import config

logger = logging.getLogger(__name__)

# A check must not be able to hang the request that asked for it: the
# self-check endpoint calls every provider in turn, so a single stalled
# connection would otherwise hold the whole page.
CHECK_TIMEOUT_SEC = 12
_ERROR_BODY_CHARS = 400


@dataclass
class ProviderCheck:
    """What actually happened when we called this provider, in the terms
    a person debugging a deployment needs: is the key there, did the host
    answer, did the model produce words, and if not — exactly why."""
    name: str
    label: str
    configured: bool
    ok: bool = False
    latency_ms: int = 0
    model: str = ""
    reply: str = ""
    error: str = ""
    hint: str = ""
    free: bool = False
    local: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "configured": self.configured,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "reply": self.reply,
            "error": self.error,
            "hint": self.hint,
            "free": self.free,
            "local": self.local,
        }


class Provider:
    """Base class. A provider is configured or it isn't, and it either
    returns text or raises with a message a human can act on."""

    name = ""
    label = ""
    free = False
    local = False
    key_env = ""

    def __init__(self, name: str, label: str, model_getter: Callable[[], str],
                 key_getter: Callable[[], str] = lambda: "",
                 free: bool = False, local: bool = False, key_env: str = ""):
        self.name = name
        self.label = label
        self._model_getter = model_getter
        self._key_getter = key_getter
        self.free = free
        self.local = local
        self.key_env = key_env

    # ── identity ─────────────────────────────────────────
    @property
    def model(self) -> str:
        return self._model_getter() or ""

    @property
    def api_key(self) -> str:
        return (self._key_getter() or "").strip()

    def is_credentialed(self) -> bool:
        """Can this provider be *reached* — key present, endpoint known.

        Separate from is_configured() because model-level routing gets
        the model name from the catalogue, not from this provider's
        configured default. A provider holding a valid key but a blank
        *_MODEL is unusable under provider routing and perfectly usable
        under model routing; conflating the two would hide it.
        """
        return bool(self.api_key)

    def is_configured(self) -> bool:
        return self.is_credentialed() and bool(self.model)

    def missing(self) -> str:
        """Why this provider is unavailable, named so it can be fixed."""
        if not self.api_key:
            return f"{self.key_env} is not set"
        if not self.model:
            return "no model name configured"
        return ""

    # ── work ─────────────────────────────────────────────
    def complete(self, messages: list[dict], system: str = "",
                 max_tokens: int = 512, temperature: float = 0.2,
                 timeout_sec: Optional[float] = None,
                 model: str = "", json_mode: bool = False) -> Optional[str]:
        """The primitive: a conversation in, one completion out.

        Multi-turn matters here — the chat page and the tool dispatcher
        both send a history, and flattening it to one string is how a
        provider layer quietly makes the assistant forget what it just
        said.

        `model` overrides this provider's configured default for one
        call, which is what makes per-task model routing possible: the
        model name comes from the catalogue, not from the provider.
        Empty means "the configured default", so every existing caller
        is unaffected.

        `json_mode` asks the provider for structured output at the API
        level rather than in the prompt. The difference matters where
        the reply is *parsed* rather than read — a model that usually
        returns valid JSON is not the same as one that is required to.

        A message's `content` may be a plain string or a list of parts
        (`{"type": "text", ...}` / `{"type": "image", "data": bytes,
        "mime": ...}`), which is how an image reaches a model.
        """
        raise NotImplementedError

    def generate(self, system: str, user: str, max_tokens: int = 512,
                 temperature: float = 0.2,
                 timeout_sec: Optional[float] = None,
                 model: str = "", json_mode: bool = False) -> Optional[str]:
        """One-shot convenience over complete()."""
        return self.complete([{"role": "user", "content": user}],
                             system=system, max_tokens=max_tokens,
                             temperature=temperature, timeout_sec=timeout_sec,
                             model=model, json_mode=json_mode)

    def describe_image(self, system: str, user: str, image: bytes,
                       mime: str = "image/png", max_tokens: int = 1024,
                       model: str = "", json_mode: bool = False,
                       timeout_sec: Optional[float] = None) -> Optional[str]:
        """Ask about an image. Convenience over the parts form above."""
        return self.complete(
            [{"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image", "data": image, "mime": mime}]}],
            system=system, max_tokens=max_tokens, temperature=0.0,
            timeout_sec=timeout_sec, model=model, json_mode=json_mode)

    def check(self, timeout_sec: float = CHECK_TIMEOUT_SEC,
              model: str = "") -> ProviderCheck:
        """A real round trip, not a key-format check.

        A key can be present, well-formed, and rejected — expired, wrong
        project, out of quota, or blocked by the network the app is
        deployed on. Only a call finds that out, so this makes one, with
        a prompt whose correct answer is short enough to verify.
        """
        # With an explicit model, "configured" means credentialed: the
        # model name came from the caller, not from this provider's own
        # default, so a blank default must not read as unavailable.
        configured = (self.is_credentialed() if model else self.is_configured())
        chk = ProviderCheck(name=self.name, label=self.label,
                            configured=configured,
                            model=model or self.model, free=self.free,
                            local=self.local)
        if not chk.configured:
            chk.error = self.missing()
            chk.hint = self._hint_for_missing()
            return chk

        started = time.monotonic()
        try:
            reply = self.generate(
                system="You are a connectivity check. Reply with one word.",
                user="Reply with the single word: ready",
                max_tokens=16, temperature=0.0, timeout_sec=timeout_sec,
                model=model)
        except Exception as e:                     # noqa: BLE001 — reported
            chk.latency_ms = int((time.monotonic() - started) * 1000)
            chk.error = str(e)[:_ERROR_BODY_CHARS]
            chk.hint = _hint_for_error(chk.error, self)
            return chk

        chk.latency_ms = int((time.monotonic() - started) * 1000)
        if reply and reply.strip():
            chk.ok = True
            chk.reply = reply.strip()[:120]
        else:
            chk.error = "the provider answered but returned no text"
            chk.hint = ("Usually a model name that exists but is not served "
                        "to this account, or a safety filter on the reply.")
        return chk

    def _hint_for_missing(self) -> str:
        if self.local:
            return ("Set LOCAL_LLM_URL to your Ollama/llama.cpp/LM Studio "
                    "address (e.g. http://localhost:11434) and LOCAL_LLM_MODEL "
                    "to a tag you have pulled.")
        return (f"Add {self.key_env} to the environment. On Render this is "
                f"Settings → Environment; a GitHub Actions secret of the same "
                f"name is not visible to the running service unless the "
                f"workflow passes it through.")


class OpenAICompatibleProvider(Provider):
    """Anything exposing POST {base}/v1/chat/completions.

    Deliberately plain urllib rather than an SDK. Five providers share
    this one adapter, and each SDK would be a dependency, a version to
    pin, and its own timeout semantics to get wrong. The dialect is
    stable and small.
    """

    def __init__(self, *args, base_url_getter: Callable[[], str],
                 use_proxy: bool = True, extra_headers: Optional[dict] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._base_url_getter = base_url_getter
        self._use_proxy = use_proxy
        self._extra_headers = extra_headers or {}

    @property
    def base_url(self) -> str:
        return (self._base_url_getter() or "").rstrip("/")

    def is_credentialed(self) -> bool:
        if not self.base_url:
            return False
        # A local endpoint has no key and needs none; a hosted one does.
        return self.local or bool(self.api_key)

    def is_configured(self) -> bool:
        return self.is_credentialed() and bool(self.model)

    def missing(self) -> str:
        if not self.base_url:
            return "no endpoint URL configured"
        if not self.model:
            return "no model name configured"
        if not self.local and not self.api_key:
            return f"{self.key_env} is not set"
        return ""

    def complete(self, messages: list[dict], system: str = "",
                 max_tokens: int = 512, temperature: float = 0.2,
                 timeout_sec: Optional[float] = None,
                 model: str = "", json_mode: bool = False) -> Optional[str]:
        wire_model = model or self.model
        if not self.is_credentialed() or not wire_model:
            return None

        payload = {
            "model": wire_model,
            "messages": (([{"role": "system", "content": system}] if system else [])
                         + [_openai_message(m) for m in messages]),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self._extra_headers)

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST")

        timeout = timeout_sec or config.llm_timeout_sec
        # A local model is on localhost or the client's own network;
        # routing it through a proxy would defeat privacy mode and
        # usually just fails.
        opener = (urllib.request.build_opener()
                  if self._use_proxy
                  else urllib.request.build_opener(
                      urllib.request.ProxyHandler({})))
        try:
            with opener.open(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(_http_error_message(e)) from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"could not reach {self.base_url}: {e.reason}") from None
        except (TimeoutError, OSError) as e:
            raise RuntimeError(f"{self.base_url} did not respond: {e}") from None
        except json.JSONDecodeError:
            raise RuntimeError(
                f"{self.base_url} returned a non-JSON response") from None

        return _extract_openai_text(body)


    def generate_image(self, prompt: str, size: str = "1024x1024",
                       model: str = "",
                       timeout_sec: Optional[float] = 120) -> Optional[bytes]:
        """POST {base}/v1/images/generations — the dialect a local Stable
        Diffusion server speaks, as well as several hosted ones.

        Returns raw bytes. The response may carry base64 or a URL; only
        the base64 form is accepted, because fetching a URL would mean a
        second request to a host nobody vetted, for an image that is
        decoration.
        """
        wire_model = model or self.model
        if not self.is_credentialed() or not wire_model:
            return None

        payload = {"model": wire_model, "prompt": prompt, "size": size,
                   "n": 1, "response_format": "b64_json"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self._extra_headers)

        req = urllib.request.Request(
            f"{self.base_url}/v1/images/generations",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST")
        opener = (urllib.request.build_opener() if self._use_proxy
                  else urllib.request.build_opener(
                      urllib.request.ProxyHandler({})))
        try:
            with opener.open(req, timeout=timeout_sec or 120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(_http_error_message(e)) from None
        except Exception as e:                     # noqa: BLE001
            raise RuntimeError(f"{self.base_url}: {e}") from None

        try:
            encoded = body["data"][0]["b64_json"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                "the image endpoint returned no inline image data") from None
        return base64.b64decode(encoded)


class GeminiProvider(Provider):
    """Google, through the shared gemini_client wrapper — which is where
    the hard wall-clock timeout and the model name live."""

    def is_credentialed(self) -> bool:
        from app.ai import gemini_client
        return gemini_client.is_configured()

    def is_configured(self) -> bool:
        return self.is_credentialed() and bool(self.model)

    def generate_image(self, prompt: str, size: str = "1024x1024",
                       model: str = "",
                       timeout_sec: Optional[float] = 120) -> Optional[bytes]:
        from app.ai import gemini_client
        return gemini_client.generate_image(
            prompt, model=model or None, timeout_sec=timeout_sec or 120)

    def embed(self, texts: list[str], task: str = "retrieval_document",
              model: str = "") -> Optional[list]:
        from app.ai import gemini_client
        return gemini_client.embed(texts, task=task, model=model or None)

    def understand_video(self, data: bytes, ext: str, prompt: str,
                         max_tokens: int = 2048, model: str = "",
                         timeout_sec: float = 120) -> Optional[str]:
        """Native video understanding via the Files API. Declared here
        and nowhere else, which is why VIDEO is its own capability."""
        from app.ai import multimodal
        return multimodal.gemini_video(
            data=data, ext=ext, prompt=prompt, max_tokens=max_tokens,
            model=model, timeout_sec=timeout_sec)

    def complete(self, messages: list[dict], system: str = "",
                 max_tokens: int = 512, temperature: float = 0.2,
                 timeout_sec: Optional[float] = None,
                 model: str = "", json_mode: bool = False) -> Optional[str]:
        from app.ai import gemini_client
        if not gemini_client.is_configured():
            return None
        # Gemini takes a flat contents list rather than roled messages.
        # Prior assistant turns are labelled so the history still reads
        # as a conversation instead of one long user monologue. Image
        # parts go in as PIL images, which the SDK converts itself.
        contents: list = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            content = m.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        img = _as_pil(part.get("data"))
                        if img is not None:
                            contents.append(img)
                    elif part.get("text"):
                        contents.append(part["text"])
                continue
            if not content:
                continue
            contents.append(f"Assistant: {content}" if role == "assistant"
                            else content)
        if not contents:
            return None
        return gemini_client.generate_text(
            contents, system=system, max_output_tokens=max_tokens,
            temperature=temperature, json_mode=json_mode,
            model=model or None,
            timeout_sec=timeout_sec or config.llm_timeout_sec)


# ── helpers ──────────────────────────────────────────────

def _openai_message(message: dict) -> dict:
    """Render one message into the OpenAI wire dialect.

    A plain string passes through untouched — that is the overwhelming
    majority of calls, and rewriting them would be churn. A parts list
    becomes the vision dialect that OpenRouter, Together, vLLM and
    Ollama all speak, with the image inlined as a data URL rather than
    a link: the image is a client's data and must not be uploaded
    somewhere first to be fetched back.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts = []
    for part in content:
        if part.get("type") == "image":
            data = part.get("data")
            if not data:
                continue
            encoded = base64.b64encode(data).decode("ascii")
            mime = part.get("mime") or "image/png"
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        elif part.get("text"):
            parts.append({"type": "text", "text": part["text"]})
    return {"role": message.get("role", "user"), "content": parts}


def _as_pil(data):
    """Bytes → PIL image, or None. Pillow is already a dependency (the
    table extractor opens uploads with it), so this adds nothing."""
    if data is None:
        return None
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:                              # noqa: BLE001
        logger.warning("could not read an image part", exc_info=True)
        return None


def _extract_openai_text(body: dict) -> Optional[str]:
    """The dialect is standard but the error shape is not: some providers
    return HTTP 200 with an `error` object inside, which would otherwise
    read as an empty completion and get silently retried elsewhere."""
    if isinstance(body, dict) and body.get("error"):
        err = body["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise RuntimeError(str(msg)[:_ERROR_BODY_CHARS])
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"unexpected response shape: {str(body)[:200]}") from None
    return (text or "").strip() or None


def _http_error_message(e: urllib.error.HTTPError) -> str:
    """Carry the provider's own words through. "401" alone sends someone
    hunting; "401 — Invalid API Key" ends the hunt."""
    try:
        raw = e.read().decode("utf-8", "replace")
    except Exception:                              # noqa: BLE001
        raw = ""
    detail = ""
    try:
        parsed = json.loads(raw)
        err = parsed.get("error", parsed)
        detail = (err.get("message") if isinstance(err, dict) else str(err)) or ""
    except Exception:                              # noqa: BLE001
        detail = raw
    detail = " ".join(str(detail).split())[:_ERROR_BODY_CHARS]
    return f"HTTP {e.code}{' — ' + detail if detail else ''}"


def _hint_for_error(error: str, provider: Provider) -> str:
    """Turn the provider's error into the next thing to do."""
    low = error.lower()
    # Network faults are checked first and deliberately: a blocked
    # egress proxy answers with "Tunnel connection failed: 403", and
    # reading that 403 as a key problem sends someone to rotate a key
    # that was never the issue.
    if ("could not reach" in low or "did not respond" in low
            or "timed out" in low or "tunnel connection failed" in low
            or "connection refused" in low):
        return ("The host was unreachable from this machine — a network or "
                "egress rule, not the key. The same key may work fine from "
                "the deployment. Try another provider, or a local model, "
                "from this network.")
    if "401" in low or "invalid api key" in low or "unauthorized" in low:
        return (f"The key in {provider.key_env} was rejected. It is present, "
                f"so this is the wrong key, a revoked one, or one from a "
                f"different account — not a missing variable.")
    if "403" in low or "permission" in low:
        return ("The key is valid but not allowed this model or region. "
                "Check the model name and the account's access.")
    if "404" in low or "not found" in low or "does not exist" in low:
        return (f"The endpoint answered but has no model named "
                f"'{provider.model}'. Model names change; check the "
                f"provider's current list.")
    if "429" in low or "quota" in low or "rate limit" in low:
        return ("The key works — this is a rate or quota limit. Free tiers "
                "hit this; the app falls back to the next provider.")
    return ""


# ── the registry ─────────────────────────────────────────

def _providers() -> dict[str, Provider]:
    """Built fresh from config on each call rather than at import, so a
    key added to the environment takes effect without a code change and
    so tests can change config between cases."""
    return {
        "groq": OpenAICompatibleProvider(
            "groq", "Groq",
            model_getter=lambda: config.llm_model,
            key_getter=lambda: config.groq_api_key,
            free=True, key_env="GROQ_API_KEY",
            base_url_getter=lambda: config.groq_base_url),
        "gemini": GeminiProvider(
            "gemini", "Google Gemini",
            model_getter=lambda: config.gemini_model,
            key_getter=lambda: config.gemini_api_key,
            key_env="GEMINI_API_KEY"),
        "openrouter": OpenAICompatibleProvider(
            "openrouter", "OpenRouter",
            model_getter=lambda: config.openrouter_model,
            key_getter=lambda: config.openrouter_api_key,
            free=True, key_env="OPENROUTER_API_KEY",
            base_url_getter=lambda: config.openrouter_base_url,
            extra_headers={"X-Title": config.app_name}),
        "cerebras": OpenAICompatibleProvider(
            "cerebras", "Cerebras",
            model_getter=lambda: config.cerebras_model,
            key_getter=lambda: config.cerebras_api_key,
            free=True, key_env="CEREBRAS_API_KEY",
            base_url_getter=lambda: config.cerebras_base_url),
        "together": OpenAICompatibleProvider(
            "together", "Together AI",
            model_getter=lambda: config.together_model,
            key_getter=lambda: config.together_api_key,
            key_env="TOGETHER_API_KEY",
            base_url_getter=lambda: config.together_base_url),
        "local": OpenAICompatibleProvider(
            "local", "Local model",
            model_getter=lambda: config.local_llm_model,
            free=True, local=True, key_env="LOCAL_LLM_URL",
            base_url_getter=lambda: config.local_llm_url,
            use_proxy=False),
    }


def get(name: str) -> Optional[Provider]:
    return _providers().get(name)


def all_providers() -> list[Provider]:
    return list(_providers().values())


def configured_names() -> list[str]:
    return [p.name for p in all_providers() if p.is_configured()]


def check_all(only: Optional[list[str]] = None,
              timeout_sec: float = CHECK_TIMEOUT_SEC) -> list[ProviderCheck]:
    """Round-trip every provider (or the named ones) and report."""
    out = []
    for p in all_providers():
        if only and p.name not in only:
            continue
        out.append(p.check(timeout_sec=timeout_sec))
    return out
