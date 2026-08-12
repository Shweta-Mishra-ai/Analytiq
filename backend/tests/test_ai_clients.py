"""
tests/test_ai_clients.py — coverage for ai/gemini_client.py and
ai/llm_client.py, neither of which had any tests.

These are the modules where the "dead model" bug lived: `llm_client`
hardcoded its own Gemini model name (a different, already-shut-down one)
instead of reading config, so every AI call 404'd against Google while
the app reported no error. The regression tests here lock in the two
properties that prevent a repeat:

  * model names come from config, never from a literal in the call site
  * a hung network call is bounded by our own wall-clock deadline rather
    than trusting the SDK's unreliable timeout

No network access and no API key required — the SDK boundary is faked.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from app.ai import gemini_client
from app.config import config


# ══════════════════════════════════════════════════════════
#  Unconfigured behaviour — must degrade, not explode
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def no_gemini_key(monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "", raising=False)
    monkeypatch.setattr(gemini_client, "_client", None, raising=False)
    monkeypatch.setattr(gemini_client, "_client_key", None, raising=False)


def test_is_configured_reflects_key(monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "", raising=False)
    assert gemini_client.is_configured() is False
    monkeypatch.setattr(config, "gemini_api_key", "fake-key", raising=False)
    assert gemini_client.is_configured() is True


def test_get_client_returns_none_without_key(no_gemini_key):
    assert gemini_client.get_client() is None


def test_generate_text_raises_without_key(no_gemini_key):
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini_client.generate_text(["hello"])


def test_upload_file_raises_without_key(no_gemini_key):
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini_client.upload_file("/tmp/whatever.mp4")


def test_embed_returns_none_without_key(no_gemini_key):
    """embed() must return None rather than raise — callers fall back to
    the local hashing embedder, so raising here would break RAG ingest
    for every user without a Gemini key."""
    assert gemini_client.embed(["some text"]) is None


def test_delete_file_never_raises_without_key(no_gemini_key):
    gemini_client.delete_file("files/abc123")  # must not raise


def test_embed_returns_none_when_underlying_call_fails(monkeypatch):
    monkeypatch.setattr(config, "gemini_api_key", "fake-key", raising=False)
    monkeypatch.setattr(gemini_client, "get_client", lambda: object())
    monkeypatch.setattr(gemini_client, "_run_with_hard_timeout",
                        lambda fn, t: (_ for _ in ()).throw(RuntimeError("boom")))
    assert gemini_client.embed(["text"]) is None


# ══════════════════════════════════════════════════════════
#  Hard timeout — the actual safety mechanism
# ══════════════════════════════════════════════════════════

def test_hard_timeout_returns_value_on_success():
    assert gemini_client._run_with_hard_timeout(lambda: "done", timeout_sec=5) == "done"


def test_hard_timeout_propagates_original_exception():
    def _boom():
        raise ValueError("inner failure")
    with pytest.raises(ValueError, match="inner failure"):
        gemini_client._run_with_hard_timeout(_boom, timeout_sec=5)


def test_hard_timeout_gives_up_on_a_hanging_call():
    """The core guarantee: a call that never returns must not hang the
    worker thread serving the request. Without this the SDK's own
    unreliable timeout could stall a request indefinitely."""
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        gemini_client._run_with_hard_timeout(lambda: time.sleep(30), timeout_sec=0.5)
    elapsed = time.monotonic() - started
    assert elapsed < 5, f"timeout was not enforced promptly (took {elapsed:.1f}s)"


# ══════════════════════════════════════════════════════════
#  Model names must come from config (the dead-model bug)
# ══════════════════════════════════════════════════════════

class _FakeModels:
    def __init__(self):
        self.model_used = None

    def generate_content(self, model, contents, config):  # noqa: A002
        self.model_used = model
        class _R:
            text = "fake response"
        return _R()


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def test_get_client_caches_the_same_instance(monkeypatch):
    """A new genai.Client per call would re-open connections on every
    request; the module caches one per key."""
    built = []

    class _Genai:
        @staticmethod
        def Client(api_key):
            built.append(api_key)
            return object()

    monkeypatch.setattr(config, "gemini_api_key", "key-one", raising=False)
    monkeypatch.setattr(gemini_client, "_client", None, raising=False)
    monkeypatch.setattr(gemini_client, "_client_key", None, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "google", type(
        "M", (), {"genai": _Genai})())

    first = gemini_client.get_client()
    second = gemini_client.get_client()
    assert first is second
    assert built == ["key-one"], "client should be constructed exactly once"


def test_get_client_rebuilds_when_the_key_changes(monkeypatch):
    """Regression: the cached client is keyed on the API key, so rotating
    GEMINI_API_KEY takes effect immediately. If this branch broke, a key
    rotation would silently keep using the old (possibly revoked) key."""
    built = []

    class _Genai:
        @staticmethod
        def Client(api_key):
            built.append(api_key)
            return object()

    monkeypatch.setattr(gemini_client, "_client", None, raising=False)
    monkeypatch.setattr(gemini_client, "_client_key", None, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "google", type(
        "M", (), {"genai": _Genai})())

    monkeypatch.setattr(config, "gemini_api_key", "key-one", raising=False)
    first = gemini_client.get_client()
    monkeypatch.setattr(config, "gemini_api_key", "key-two", raising=False)
    second = gemini_client.get_client()

    assert first is not second
    assert built == ["key-one", "key-two"]


def test_generate_text_uses_configured_model(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(config, "gemini_api_key", "fake-key", raising=False)
    monkeypatch.setattr(config, "gemini_model", "gemini-test-model", raising=False)
    monkeypatch.setattr(gemini_client, "get_client", lambda: fake)

    out = gemini_client.generate_text(["hi"], system="be brief")
    assert out == "fake response"
    assert fake.models.model_used == "gemini-test-model", (
        "generate_text must send config.gemini_model — a hardcoded model "
        "name is exactly the bug that made every Gemini call 404")


AI_DIR = Path(__file__).resolve().parent.parent / "app" / "ai"
_MODEL_LITERAL = re.compile(r'["\'](gemini-[\w.\-]+|text-embedding-\d+)["\']')


def test_no_hardcoded_model_names_in_ai_modules():
    """Regression: model identifiers must live in config.py only. A
    literal in a call site is how llm_client ended up pinned to a
    different, already-retired model than the configured one."""
    offenders = []
    for path in sorted(AI_DIR.glob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue  # commentary about retired models is fine
            if _MODEL_LITERAL.search(line):
                offenders.append(f"app/ai/{path.name}:{i}: {stripped}")
    assert not offenders, (
        "Hardcoded model name(s) outside config.py:\n" + "\n".join(offenders))


# ══════════════════════════════════════════════════════════
#  LLMClient routing / graceful degradation
# ══════════════════════════════════════════════════════════

def _make_llm_client(monkeypatch):
    """Builds an LLMClient without touching the real Groq constructor."""
    from app.ai import llm_client as mod
    monkeypatch.setattr(mod, "Groq", lambda api_key: object())
    return mod.LLMClient(api_key="fake"), mod


def test_chat_safe_returns_fallback_instead_of_raising(monkeypatch):
    client, _ = _make_llm_client(monkeypatch)
    monkeypatch.setattr(client, "chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    out = client.chat_safe([{"role": "user", "content": "hi"}], fallback="FALLBACK")
    assert out == "FALLBACK"


def test_chat_task_returns_none_when_all_providers_fail(monkeypatch):
    """None is the contract that tells callers to use their rule-based
    fallback. Returning "" or raising would surface an empty/blown-up
    section in the report instead."""
    client, mod = _make_llm_client(monkeypatch)
    monkeypatch.setattr(client, "_groq_report", lambda *a, **k: None)
    monkeypatch.setattr(mod.gemini_client, "is_configured", lambda: False)
    assert client.chat_task("sys", "user", task="narrative") is None


def test_chat_task_prefers_gemini_for_gemini_routed_tasks(monkeypatch):
    client, mod = _make_llm_client(monkeypatch)
    calls = []
    monkeypatch.setattr(mod.gemini_client, "is_configured", lambda: True)
    monkeypatch.setattr(client, "_gemini",
                        lambda *a, **k: calls.append("gemini") or "from gemini")
    monkeypatch.setattr(client, "_groq_report",
                        lambda *a, **k: calls.append("groq") or "from groq")
    out = client.chat_task("sys", "user", task="executive_summary")
    assert out == "from gemini"
    assert calls == ["gemini"], "gemini-routed task should not call Groq first"


def test_chat_task_falls_back_to_groq_when_gemini_unavailable(monkeypatch):
    client, mod = _make_llm_client(monkeypatch)
    monkeypatch.setattr(mod.gemini_client, "is_configured", lambda: False)
    monkeypatch.setattr(client, "_groq_report", lambda *a, **k: "from groq")
    assert client.chat_task("sys", "user", task="executive_summary") == "from groq"


def test_chat_task_skips_provider_that_raises(monkeypatch):
    client, mod = _make_llm_client(monkeypatch)
    monkeypatch.setattr(mod.gemini_client, "is_configured", lambda: True)
    monkeypatch.setattr(client, "_gemini",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gemini down")))
    monkeypatch.setattr(client, "_groq_report", lambda *a, **k: "from groq")
    assert client.chat_task("sys", "user", task="executive_summary") == "from groq"


def test_gemini_helper_returns_none_when_unconfigured(monkeypatch):
    client, mod = _make_llm_client(monkeypatch)
    monkeypatch.setattr(mod.gemini_client, "is_configured", lambda: False)
    assert client._gemini("sys", "user", 100) is None


def test_status_reports_provider_availability(monkeypatch):
    client, mod = _make_llm_client(monkeypatch)
    monkeypatch.setattr(mod.gemini_client, "is_configured", lambda: True)
    st = client.status()
    assert st["gemini"] is True
    assert st["groq_model"] == config.llm_model


def test_task_routing_table_targets_known_providers():
    from app.ai.llm_client import TASK_ROUTING
    assert set(TASK_ROUTING.values()) <= {"groq", "gemini"}
    assert "default" in TASK_ROUTING


# ══════════════════════════════════════════════════════════
#  report_narrator.clean_col — column-name humanisation
# ══════════════════════════════════════════════════════════

def test_clean_col_uses_known_mapping():
    from app.ai.report_narrator import clean_col
    assert clean_col("satisfaction_level") == "Employee Satisfaction Score"
    assert clean_col("  Satisfaction_Level  ") == "Employee Satisfaction Score"


def test_clean_col_falls_back_when_translator_unavailable(monkeypatch):
    """The translator import is optional; if it raises, clean_col must
    still return a readable label rather than propagating."""
    import app.ai.prompt_builder as pb
    monkeypatch.setattr(
        pb, "translate_column_name",
        lambda c: (_ for _ in ()).throw(RuntimeError("translator down")),
        raising=False)
    from app.ai.report_narrator import clean_col
    assert clean_col("total_order_value") == "Total Order Value"


def test_clean_col_repairs_known_source_typo(monkeypatch):
    """`average_montly_hours` (sic) is a real column spelling in the
    common HR dataset — the humanised label should not carry the typo."""
    import app.ai.prompt_builder as pb
    monkeypatch.setattr(
        pb, "translate_column_name",
        lambda c: (_ for _ in ()).throw(RuntimeError("translator down")),
        raising=False)
    from app.ai.report_narrator import clean_col
    assert "montly" not in clean_col("shift_montly_target").lower()
