"""
tests/test_ai_guards.py — the guards around what the language model may do,
checked by behaviour rather than by "it raised".

A privacy-mode refusal that is retried still raises in the end, so a test
that only expects the exception passes while each refused call sits through
two backoff waits and asks three times. And an aggregation whitelist tested
with one bad name says nothing about the six good ones still working.
"""
from __future__ import annotations

import pandas as pd
import pytest
from tenacity import wait_none

from app.ai import local_llm
from app.ai.llm_client import LLMClient
from app.ai.tool_dispatcher import dispatch
from app.config import config


class _Completions:
    def __init__(self, failures: int):
        self.calls, self.failures = 0, failures

    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary provider error")
        msg = type("M", (), {"content": "ok"})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def _client(monkeypatch, failures: int) -> tuple[LLMClient, _Completions]:
    monkeypatch.setattr(LLMClient.chat.retry, "wait", wait_none())
    c = LLMClient("gsk-test")
    comp = _Completions(failures)
    monkeypatch.setattr(c._client.chat, "completions", comp)
    return c, comp


# ── retry policy ──────────────────────────────────────────

def test_a_privacy_refusal_is_asked_once_and_not_retried(monkeypatch):
    asked = []

    def refuse(provider):
        asked.append(provider)
        raise local_llm.LocalLLMError("privacy mode")

    monkeypatch.setattr(local_llm, "assert_cloud_allowed", refuse)
    client, comp = _client(monkeypatch, failures=0)
    with pytest.raises(local_llm.LocalLLMError):
        client.chat([{"role": "user", "content": "x"}])
    assert asked == ["Groq"]
    assert comp.calls == 0, "the cloud provider was called in privacy mode"


def test_a_transient_provider_error_is_retried(monkeypatch):
    monkeypatch.setattr(config, "llm_privacy_mode", False)
    client, comp = _client(monkeypatch, failures=2)
    assert client.chat([{"role": "user", "content": "x"}]) == "ok"
    assert comp.calls == 3


def test_retries_stop_after_three_attempts(monkeypatch):
    monkeypatch.setattr(config, "llm_privacy_mode", False)
    client, comp = _client(monkeypatch, failures=10)
    with pytest.raises(RuntimeError, match="temporary provider error"):
        client.chat([{"role": "user", "content": "x"}])
    assert comp.calls == 3


# ── aggregation whitelist ─────────────────────────────────

_DF = pd.DataFrame({"g": ["a", "a", "a", "b", "b"], "v": [1.0, 2.0, 6.0, 10.0, 30.0]})


@pytest.mark.parametrize("func,expected", [
    ("sum", {"a": 9.0, "b": 40.0}),
    ("mean", {"a": 3.0, "b": 20.0}),
    ("median", {"a": 2.0, "b": 20.0}),
    ("count", {"a": 3, "b": 2}),
    ("min", {"a": 1.0, "b": 10.0}),
    ("max", {"a": 6.0, "b": 30.0}),
])
def test_each_offered_aggregation_computes_the_right_figure(func, expected):
    r = dispatch(_DF, "aggregate", {"group_col": "g", "value_col": "v", "agg_func": func})
    assert r.success, r.error
    assert dict(zip(r.dataframe["g"], r.dataframe["v"])) == expected


@pytest.mark.parametrize("func", [
    "__class__", "__init__", "apply", "pipe", "transform", "describe", "eval",
    "SUM", " sum", "",
])
def test_anything_else_is_refused_before_pandas_sees_it(func, monkeypatch):
    # pandas resolves the name with getattr; a refusal must come first.
    def fail(*a, **kw):
        raise AssertionError("pandas was asked to aggregate with '%s'" % func)

    monkeypatch.setattr(pd.core.groupby.generic.SeriesGroupBy, "agg", fail)
    r = dispatch(_DF, "aggregate", {"group_col": "g", "value_col": "v", "agg_func": func})
    assert not r.success
    assert "Unsupported aggregation" in r.error


def test_the_default_aggregation_is_sum():
    r = dispatch(_DF, "aggregate", {"group_col": "g", "value_col": "v"})
    assert r.success and dict(zip(r.dataframe["g"], r.dataframe["v"])) == {"a": 9.0, "b": 40.0}
