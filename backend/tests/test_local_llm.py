"""
Running the narratives on the client's own hardware.

Two separate things are tested, and the first is the one that matters:

  - **Privacy mode blocks cloud calls.** Not "prefers not to make them" —
    blocks them. Every narrative in this app sends the client's column
    names, sample values and computed figures to a third party, which for
    employee or patient data is simply not permitted. A guarantee that
    holds only when the code paths happen to line up is not a guarantee,
    so the test asserts the refusal at the chokepoint every call goes
    through.
  - **The local model is a working fallback** when the cloud providers
    are missing or down.

The app never depends on any of them: the domain engines write their
findings deterministically and the LLM only adds prose, so privacy mode
with no local model is a supported configuration.
"""
from __future__ import annotations

import json

import pytest

from app.ai import local_llm
from app.config import config


@pytest.fixture()
def privacy_on(monkeypatch):
    monkeypatch.setattr(config, "llm_privacy_mode", True)


@pytest.fixture()
def local_configured(monkeypatch):
    monkeypatch.setattr(config, "local_llm_url", "http://localhost:11434")
    monkeypatch.setattr(config, "local_llm_model", "some-open-model")


# ══════════════════════════════════════════════════════════
#  Privacy mode
# ══════════════════════════════════════════════════════════

def test_gemini_is_refused_in_privacy_mode(privacy_on):
    """get_client() is the single chokepoint for every Gemini call in the
    app, so refusing there is what makes the guarantee hold."""
    from app.ai import gemini_client
    with pytest.raises(local_llm.LocalLLMError) as e:
        gemini_client.get_client()
    assert "no data left this machine" in str(e.value).lower()


def test_the_refusal_says_how_to_proceed(privacy_on):
    from app.ai import gemini_client
    with pytest.raises(local_llm.LocalLLMError) as e:
        gemini_client.get_client()
    msg = str(e.value)
    assert "LOCAL_LLM_URL" in msg
    assert "privacy mode off" in msg


def test_gemini_works_normally_when_privacy_mode_is_off(monkeypatch):
    from app.ai import gemini_client
    monkeypatch.setattr(config, "llm_privacy_mode", False)
    monkeypatch.setattr(config, "gemini_api_key", "")
    assert gemini_client.get_client() is None      # no key, but no refusal


def test_rag_refuses_to_reach_a_cloud_provider_in_privacy_mode(
        privacy_on, monkeypatch):
    """A knowledge base holds the client's contracts and policies — the
    most sensitive thing in the app. It must never silently fall back."""
    from app.rag import service

    def _boom(*a, **kw):
        raise AssertionError("a cloud provider was called in privacy mode")

    monkeypatch.setattr(service, "_gemini_generate", _boom)
    monkeypatch.setattr(service, "_groq_generate", _boom)
    monkeypatch.setattr(service, "_local_generate", lambda *a, **kw: None)

    with pytest.raises(RuntimeError) as e:
        service._generate("sys", "user")
    assert "no data was sent anywhere" in str(e.value).lower()


def test_rag_uses_the_local_model_in_privacy_mode(privacy_on, monkeypatch):
    from app.rag import service
    monkeypatch.setattr(service, "_local_generate",
                        lambda *a, **kw: "A local answer [1].")
    assert service._generate("sys", "user") == "A local answer [1]."


def test_privacy_mode_is_off_by_default():
    """It changes what the app can do; it has to be an explicit choice."""
    from app.config import AppConfig
    assert AppConfig().llm_privacy_mode is False


# ══════════════════════════════════════════════════════════
#  The local provider itself
# ══════════════════════════════════════════════════════════

def test_generate_returns_none_when_not_configured(monkeypatch):
    """Not configured is an ordinary state, not an error — callers fall
    through to the next provider."""
    monkeypatch.setattr(config, "local_llm_url", "")
    monkeypatch.setattr(config, "local_llm_model", "")
    assert local_llm.generate("s", "u") is None


def test_generate_posts_an_openai_shaped_request(local_configured, monkeypatch):
    """Ollama, llama.cpp, vLLM and LM Studio all speak this shape; being
    specific to none of them is the point."""
    seen = {}

    class _Resp:
        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "Local narrative."}}]
            }).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req, timeout=None):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data.decode())
            seen["timeout"] = timeout
            return _Resp()

    monkeypatch.setattr(local_llm.urllib.request, "build_opener",
                        lambda *a, **kw: _Opener())

    out = local_llm.generate("You are an analyst.", "Summarise.",
                             max_tokens=256)
    assert out == "Local narrative."
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["model"] == "some-open-model"
    assert seen["body"]["messages"][0]["role"] == "system"
    assert seen["body"]["messages"][1]["content"] == "Summarise."
    assert seen["body"]["stream"] is False


def test_an_unreachable_local_model_returns_none(local_configured, monkeypatch):
    """A model that is not running must not take the request down with
    it — the report still builds from the engines' own wording."""
    import urllib.error

    class _Opener:
        def open(self, req, timeout=None):
            raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(local_llm.urllib.request, "build_opener",
                        lambda *a, **kw: _Opener())
    assert local_llm.generate("s", "u") is None


def test_an_unexpected_response_shape_returns_none(local_configured, monkeypatch):
    class _Resp:
        def read(self):
            return json.dumps({"error": "model not found"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req, timeout=None):
            return _Resp()

    monkeypatch.setattr(local_llm.urllib.request, "build_opener",
                        lambda *a, **kw: _Opener())
    assert local_llm.generate("s", "u") is None


def test_the_local_call_bypasses_any_proxy(local_configured, monkeypatch):
    """Routing a localhost call through a proxy would defeat the point of
    privacy mode."""
    seen = {}

    def _build_opener(*handlers):
        seen["handlers"] = handlers

        class _O:
            def open(self, req, timeout=None):
                raise OSError("stop here")
        return _O()

    monkeypatch.setattr(local_llm.urllib.request, "build_opener", _build_opener)
    local_llm.generate("s", "u")
    assert any(isinstance(h, local_llm.urllib.request.ProxyHandler)
               for h in seen["handlers"])


# ══════════════════════════════════════════════════════════
#  Status
# ══════════════════════════════════════════════════════════

def test_status_reports_what_will_actually_happen(privacy_on, local_configured):
    st = local_llm.status()
    assert st["privacy_mode"] is True
    assert st["cloud_allowed"] is False
    assert st["local_configured"] is True
    assert st["local_model"] == "some-open-model"


def test_status_hides_local_details_when_not_configured(monkeypatch):
    monkeypatch.setattr(config, "local_llm_url", "")
    monkeypatch.setattr(config, "local_llm_model", "")
    st = local_llm.status()
    assert st["local_configured"] is False
    assert st["local_url"] == ""


def test_health_endpoint_exposes_the_llm_posture(monkeypatch):
    """A client running in privacy mode can confirm it from outside the
    app rather than taking the operator's word for it."""
    from fastapi.testclient import TestClient

    from app.main import app
    monkeypatch.setattr(config, "llm_privacy_mode", True)
    with TestClient(app) as c:
        body = c.get("/api/health").json()
    assert body["llm"]["privacy_mode"] is True
    assert body["llm"]["cloud_allowed"] is False
