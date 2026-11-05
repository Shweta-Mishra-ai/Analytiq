"""
Claude as the app's language model: tried first when ANTHROPIC_API_KEY is
set, falling through to the other providers on any failure, never called
in privacy mode. The SDK is replaced by a fake; no request leaves the test.
"""
from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.ai import claude_client
from app.config import config


class _FakeMessages:
    def __init__(self, reply="Revenue rose 5%.", stop="end_turn", error=None):
        self.reply, self.stop, self.error, self.calls = reply, stop, error, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return SimpleNamespace(
            stop_reason=self.stop,
            stop_details=SimpleNamespace(category="cyber") if self.stop == "refusal" else None,
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=self.reply)])


@pytest.fixture
def fake(monkeypatch):
    msgs = _FakeMessages()
    monkeypatch.setattr(config, "anthropic_api_key", "test-key")
    monkeypatch.setattr(config, "llm_privacy_mode", False)
    monkeypatch.setattr(claude_client, "_get",
                        lambda: SimpleNamespace(beta=SimpleNamespace(messages=msgs)))
    return msgs


def _status_error(cls, code):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(code, request=req), body=None)


def test_a_reply_is_the_text_blocks(fake):
    assert claude_client.generate("sys", "question") == "Revenue rose 5%."
    call = fake.calls[0]
    assert call["model"] == config.claude_model
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "low"}


def test_turns_are_made_valid_for_the_messages_api(fake):
    claude_client.chat("sys", [{"role": "assistant", "content": "hi"},
                               {"role": "user", "content": "a"},
                               {"role": "user", "content": "b"},
                               {"role": "assistant", "content": ""},
                               {"role": "system", "content": "c"}])
    assert fake.calls[0]["messages"] == [{"role": "user", "content": "a\n\nb\n\nc"}]


def test_a_refusal_returns_none(fake):
    fake.stop = "refusal"
    assert claude_client.generate("sys", "q") is None


@pytest.mark.parametrize("error", [
    _status_error(anthropic.AuthenticationError, 401),
    _status_error(anthropic.RateLimitError, 429),
    _status_error(anthropic.InternalServerError, 500),
    anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")),
])
def test_provider_failures_return_none(fake, error):
    fake.error = error
    assert claude_client.generate("sys", "q") is None


def test_not_called_without_a_key(monkeypatch):
    monkeypatch.setattr(config, "anthropic_api_key", "")
    assert claude_client.generate("sys", "q") is None


def test_privacy_mode_refuses_claude(fake, monkeypatch):
    from app.ai.local_llm import LocalLLMError
    monkeypatch.setattr(config, "llm_privacy_mode", True)
    with pytest.raises(LocalLLMError):
        claude_client.generate("sys", "q")
    assert fake.calls == []


def test_report_narratives_try_claude_first(fake, monkeypatch):
    from app.ai.llm_client import LLMClient
    client = LLMClient(api_key="")
    monkeypatch.setattr(client, "_groq_report", lambda *a: pytest.fail("Groq called"))
    assert client.chat_task("sys", "user", task="narrative") == "Revenue rose 5%."


def test_narratives_fall_through_when_claude_fails(fake, monkeypatch):
    from app.ai.llm_client import LLMClient
    fake.error = _status_error(anthropic.InternalServerError, 500)
    client = LLMClient(api_key="")
    monkeypatch.setattr(client, "_groq_report", lambda *a: "from groq")
    assert client.chat_task("sys", "user", task="narrative") == "from groq"


def test_chat_route_answers_with_claude(fake, client, uploaded_dataset_id, monkeypatch):
    monkeypatch.setattr(config, "groq_api_key", "")
    r = client.post(f"/api/chat/{uploaded_dataset_id}",
                    json={"message": "What is the average salary?"})
    assert r.status_code == 200, r.text
    assert fake.calls, "Claude was not called"
