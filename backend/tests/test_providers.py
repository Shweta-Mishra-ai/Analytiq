"""
Contract tests for the provider layer.

These run against a real HTTP server rather than a mocked urlopen. The
bugs this layer actually produces are wire-level — a provider that
returns its error inside a 200 body, a 401 whose message never reaches
the operator, a JSON shape that differs by one key — and a mock of
urlopen asserts only that we called the function we wrote.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.ai.providers import (
    CHECK_TIMEOUT_SEC,
    OpenAICompatibleProvider,
    _hint_for_error,
    check_all,
)


class _Handler(BaseHTTPRequestHandler):
    """Behaviour is set per-server by `scenario` on the server object."""

    def log_message(self, *a):        # silence the test output
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.received.append({
            "path": self.path,
            "auth": self.headers.get("Authorization", ""),
            "title": self.headers.get("X-Title", ""),
            "body": body,
        })
        status, payload = self.server.scenario(body)
        raw = json.dumps(payload).encode() if isinstance(payload, (dict, list)) \
            else payload
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    srv.received = []
    srv.scenario = lambda body: (200, {
        "choices": [{"message": {"content": "ready"}}]})
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _provider(server, key="test-key", model="test-model", **kwargs):
    return OpenAICompatibleProvider(
        "stub", "Stub", model_getter=lambda: model,
        key_getter=lambda: key, key_env="STUB_API_KEY",
        base_url_getter=lambda: f"http://127.0.0.1:{server.server_port}",
        use_proxy=False, **kwargs)


# ── the happy path ───────────────────────────────────────

def test_completion_round_trips(server):
    out = _provider(server).generate("be brief", "hello")
    assert out == "ready"
    sent = server.received[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["auth"] == "Bearer test-key"
    assert sent["body"]["model"] == "test-model"
    assert sent["body"]["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


def test_multi_turn_history_is_sent_intact(server):
    """Flattening a conversation to one string is how a provider layer
    quietly makes the assistant forget what it just said."""
    history = [
        {"role": "user", "content": "which column?"},
        {"role": "assistant", "content": "Overtime"},
        {"role": "user", "content": "and the effect size?"},
    ]
    _provider(server).complete(history, system="s")
    assert server.received[0]["body"]["messages"][1:] == history


def test_extra_headers_are_sent(server):
    _provider(server, extra_headers={"X-Title": "Analytiq"}).generate("", "hi")
    assert server.received[0]["title"] == "Analytiq"


# ── failure surfaces ─────────────────────────────────────

def test_http_error_carries_the_providers_own_words(server):
    """"401" alone sends someone hunting; "401 — Invalid API Key" ends
    the hunt."""
    server.scenario = lambda body: (
        401, {"error": {"message": "Invalid API Key"}})
    with pytest.raises(RuntimeError) as e:
        _provider(server).generate("", "hi")
    assert "401" in str(e.value)
    assert "Invalid API Key" in str(e.value)


def test_error_inside_a_200_body_is_raised_not_read_as_empty(server):
    """Some providers answer 200 with an error object. Treated as an
    empty completion, that becomes a silently missing narrative."""
    server.scenario = lambda body: (
        200, {"error": {"message": "model is overloaded"}})
    with pytest.raises(RuntimeError, match="overloaded"):
        _provider(server).generate("", "hi")


def test_unexpected_shape_is_reported_not_swallowed(server):
    server.scenario = lambda body: (200, {"result": "surprise"})
    with pytest.raises(RuntimeError, match="unexpected response shape"):
        _provider(server).generate("", "hi")


def test_non_json_body_is_reported(server):
    server.scenario = lambda body: (200, b"<html>gateway</html>")
    with pytest.raises(RuntimeError, match="non-JSON"):
        _provider(server).generate("", "hi")


def test_unreachable_host_names_the_url():
    p = OpenAICompatibleProvider(
        "stub", "Stub", model_getter=lambda: "m", key_getter=lambda: "k",
        key_env="STUB_API_KEY",
        base_url_getter=lambda: "http://127.0.0.1:1",
        use_proxy=False)
    with pytest.raises(RuntimeError, match="could not reach"):
        p.generate("", "hi")


# ── configuration ────────────────────────────────────────

def test_a_hosted_provider_without_a_key_is_not_configured(server):
    p = _provider(server, key="")
    assert not p.is_configured()
    assert p.missing() == "STUB_API_KEY is not set"
    assert p.generate("", "hi") is None


def test_a_local_provider_needs_no_key(server):
    p = OpenAICompatibleProvider(
        "local", "Local", model_getter=lambda: "gemma3:12b",
        local=True, key_env="LOCAL_LLM_URL",
        base_url_getter=lambda: f"http://127.0.0.1:{server.server_port}",
        use_proxy=False)
    assert p.is_configured()
    assert p.generate("", "hi") == "ready"


def test_a_local_provider_without_a_url_is_not_configured():
    p = OpenAICompatibleProvider(
        "local", "Local", model_getter=lambda: "gemma3:12b", local=True,
        key_env="LOCAL_LLM_URL", base_url_getter=lambda: "", use_proxy=False)
    assert not p.is_configured()
    assert p.missing() == "no endpoint URL configured"


# ── the check ────────────────────────────────────────────

def test_check_reports_a_working_provider(server):
    chk = _provider(server).check(timeout_sec=5)
    assert chk.ok
    assert chk.reply == "ready"
    assert chk.error == ""
    assert chk.latency_ms >= 0


def test_check_reports_a_rejected_key_with_a_hint(server):
    server.scenario = lambda body: (401, {"error": {"message": "Invalid API Key"}})
    chk = _provider(server).check(timeout_sec=5)
    assert not chk.ok
    assert "Invalid API Key" in chk.error
    assert "rotate" in chk.hint or "wrong key" in chk.hint or "rejected" in chk.hint


def test_check_distinguishes_an_unconfigured_provider_from_a_broken_one(server):
    chk = _provider(server, key="").check(timeout_sec=5)
    assert not chk.configured
    assert not chk.ok
    assert chk.error == "STUB_API_KEY is not set"
    assert not server.received, "an unconfigured provider must not be called"


def test_check_reports_an_empty_reply_as_a_failure(server):
    """A provider that answers with nothing has not passed the check —
    that is usually a model name it does not serve to this account."""
    server.scenario = lambda body: (200, {"choices": [{"message": {"content": ""}}]})
    chk = _provider(server).check(timeout_sec=5)
    assert not chk.ok
    assert "no text" in chk.error


# ── hints ────────────────────────────────────────────────

class _Dummy:
    key_env = "STUB_API_KEY"
    model = "some-model"


@pytest.mark.parametrize("error,expect", [
    ("HTTP 401 — Invalid API Key", "rejected"),
    ("HTTP 404 — model not found", "no model named"),
    ("HTTP 429 — rate limit exceeded", "rate or quota"),
    ("could not reach https://x: timed out", "network"),
])
def test_hints_point_at_the_right_fix(error, expect):
    assert expect in _hint_for_error(error, _Dummy())


def test_a_blocked_proxy_is_not_read_as_a_key_problem():
    """An egress proxy answers with "Tunnel connection failed: 403", and
    reading that 403 as a key problem sends someone to rotate a key that
    was never the issue."""
    hint = _hint_for_error(
        "could not reach https://api.groq.com/openai: "
        "Tunnel connection failed: 403 Forbidden", _Dummy())
    assert "network" in hint
    assert "rejected" not in hint


# ── the registry ─────────────────────────────────────────

def test_check_all_covers_every_registered_provider(monkeypatch):
    from app.config import config
    for field in ("groq_api_key", "gemini_api_key", "openrouter_api_key",
                  "cerebras_api_key", "together_api_key", "local_llm_url"):
        monkeypatch.setattr(config, field, "")
    results = check_all(timeout_sec=1)
    names = {c.name for c in results}
    assert {"groq", "gemini", "openrouter", "cerebras", "together", "local"} <= names
    assert all(not c.configured for c in results)
    assert all(c.error for c in results), "every row must say why it is unavailable"


def test_check_all_can_be_narrowed_to_one_provider():
    results = check_all(only=["groq"], timeout_sec=1)
    assert [c.name for c in results] == ["groq"]


def test_the_check_timeout_is_bounded():
    """A self-check that can hang holds the page it was called from."""
    assert 0 < CHECK_TIMEOUT_SEC <= 30


# ── the self-check endpoint ──────────────────────────────

def test_llm_check_endpoint_reports_every_provider(monkeypatch):
    """The endpoint exists because the only machine that can answer
    "does my key work" is the one holding the key — so it must run in
    the service, and it must report per provider rather than a single
    yes/no."""
    from fastapi.testclient import TestClient
    from app.config import config
    from app.main import app

    for field in ("groq_api_key", "gemini_api_key", "openrouter_api_key",
                  "cerebras_api_key", "together_api_key", "local_llm_url"):
        monkeypatch.setattr(config, field, "")

    body = TestClient(app).post("/api/admin/llm-check?timeout=1").json()
    assert body["any_working"] is False
    assert {r["name"] for r in body["providers"]} >= {"groq", "gemini", "local"}
    # The summary is the line a person reads first, so it has to say what
    # still works, not just what does not.
    assert "Reports still build" in body["summary"]
    assert "GROQ_API_KEY" in body["summary"]


def test_llm_check_reports_a_working_provider_against_a_real_server(
        server, monkeypatch):
    from fastapi.testclient import TestClient
    from app.config import config
    from app.main import app

    monkeypatch.setattr(config, "local_llm_url",
                        f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(config, "local_llm_model", "gemma3:12b")
    body = TestClient(app).post(
        "/api/admin/llm-check?providers=local&timeout=5").json()
    assert [r["name"] for r in body["providers"]] == ["local"]
    assert body["any_working"] is True
    assert body["working"] == ["local"]
    assert "Working: Local model" in body["summary"]


def test_llm_check_timeout_is_clamped(monkeypatch):
    """A caller must not be able to ask the endpoint to hang for an
    hour."""
    from fastapi.testclient import TestClient
    from app.main import app
    import app.ai.providers as registry

    seen = {}

    def _record(only, timeout):
        seen["t"] = timeout
        return []

    monkeypatch.setattr(registry, "check_all", _record)
    TestClient(app).post("/api/admin/llm-check?timeout=9999")
    assert seen["t"] <= 60


def test_llm_status_never_returns_a_key(monkeypatch):
    from fastapi.testclient import TestClient
    from app.config import config
    from app.ai.llm_client import reset_client
    from app.main import app

    monkeypatch.setattr(config, "groq_api_key", "gsk_super_secret_value")
    reset_client()
    raw = TestClient(app).get("/api/admin/llm-status").text
    reset_client()
    assert "gsk_super_secret_value" not in raw
    assert "GROQ_API_KEY" not in raw or "is not set" not in raw
