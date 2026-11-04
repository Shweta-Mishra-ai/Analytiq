"""
A chat reply opens with what the data says, not with the model's
announcement of what it was about to show.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.ai.prompt_builder import build_chat_system_prompt
from app.ai.tool_dispatcher import dispatch


@pytest.fixture()
def df():
    return pd.DataFrame({
        "department": ["Sales", "Sales", "HR", "HR", "Ops", "Ops"],
        "salary": [60000.0, 62000.0, 50000.0, 52000.0, 55000.0, 57000.0],
        "age": [30, 40, 35, 45, 50, 25],
    })


@pytest.mark.parametrize("agg, word, top, low", [
    ("sum", "total", "Sales", "HR"), ("mean", "average", "Sales", "HR"),
    ("max", "highest", "Sales", "HR"), ("min", "lowest", "Sales", "HR"),
])
def test_an_aggregate_answer_names_the_highest_and_lowest(df, agg, word, top, low):
    r = dispatch(df, "aggregate", {"group_col": "department", "value_col": "salary",
                                   "agg_func": agg}, "")
    assert r.success
    first = r.text_output.splitlines()[0]
    assert first.startswith(f"**{top}** has the highest {word} salary")
    assert f"**{low}** the lowest" in first and "across 3 department groups" in first
    assert "61,000" in r.text_output or agg != "mean"          # formatted, not 61000.00


def test_other_answers_are_sentences_from_the_data(df):
    assert dispatch(df, "filter", {"column": "salary", "operator": ">", "value": "56000"},
                    "").text_output.startswith("**3** of 6 rows (50%)")
    assert "A typical salary is **56,000**" in dispatch(
        df, "describe_column", {"column": "salary"}, "").text_output
    # r = -0.26 on six rows is chance (p = 0.62); it used to be reported as
    # "move in opposite directions slightly".
    corr = dispatch(df, "correlation", {"col_a": "age", "col_b": "salary"}, "").text_output
    assert corr.startswith("There is no reliable link between age and salary")
    assert "r = -0.26" in corr and "6 rows" in corr


def test_a_real_correlation_is_described_with_its_size_and_rows():
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    big = pd.DataFrame({"x": x, "y": -x + rng.normal(scale=0.5, size=200)})
    corr = dispatch(big, "correlation", {"col_a": "x", "col_b": "y"}, "").text_output
    assert corr.startswith("x and y move in opposite directions closely")
    assert "200 rows" in corr and "not proof that one causes the other" in corr


def test_a_constant_column_has_no_correlation(df):
    flat = df.assign(level=1)
    corr = dispatch(flat, "correlation", {"col_a": "level", "col_b": "salary"}, "").text_output
    assert "cannot be measured" in corr and "nan" not in corr


def test_a_ranking_within_chance_says_so():
    import numpy as np
    rng = np.random.default_rng(3)
    noise = pd.DataFrame({"region": rng.choice(list("NSEW"), 600),
                          "revenue": rng.normal(1000, 200, 600)})
    for agg in ("mean", "sum", "count"):
        text = dispatch(noise, "aggregate", {"group_col": "region", "value_col": "revenue",
                                             "agg_func": agg}, "").text_output
        assert "within what chance produces" in text, agg


def test_a_real_ranking_is_stated_plainly(df):
    import numpy as np
    rng = np.random.default_rng(4)
    big = pd.DataFrame({"team": np.repeat(["A", "B"], 100),
                        "score": np.r_[rng.normal(10, 1, 100), rng.normal(20, 1, 100)]})
    text = dispatch(big, "aggregate", {"group_col": "team", "value_col": "score",
                                       "agg_func": "mean"}, "").text_output
    assert text.startswith("**B** has the highest average score")
    assert "chance" not in text


def test_a_date_column_is_described_by_its_range():
    d = pd.DataFrame({"when": pd.date_range("2024-01-01", periods=30)})
    text = dispatch(d, "describe_column", {"column": "when"}, "").text_output
    assert text == "when runs from **2024-01-01** to **2024-01-30** (30 different dates)."


def test_a_missing_parameter_is_not_called_a_missing_column(df):
    r = dispatch(df, "filter", {"column": "salary", "operator": ">"}, "")
    assert not r.success and "missing 'value'" in r.error and "Column" not in r.error
    r = dispatch(df, "describe_column", {"column": "salry"}, "")
    assert not r.success and r.error.startswith("Column not found: 'salry'")


def test_the_prompt_asks_for_one_plain_line(df):
    prompt = build_chat_system_prompt(df)
    assert 'Do not start with "Here is"' in prompt and "no" in prompt.lower()


def test_an_unparseable_reply_gets_a_helpful_message(client, uploaded_dataset_id, monkeypatch):
    from app.api import chat as chat_api
    monkeypatch.setattr(chat_api, "_complete", lambda system, messages: "not json at all")
    r = client.post(f"/api/chat/{uploaded_dataset_id}", json={"message": "hello"})
    assert r.status_code == 200
    assert r.json()["text"].startswith("I couldn't work out an analysis for that")


def test_the_chat_reply_leads_with_the_computed_answer(client, uploaded_dataset_id, monkeypatch):
    from app.api import chat as chat_api
    monkeypatch.setattr(
        chat_api, "_complete", lambda system, messages:
        '{"tool": "aggregate", "params": {"group_col": "department", "value_col": "salary", '
        '"agg_func": "mean"}, "explanation": "Here is a chart that will help you understand."}')
    text = client.post(f"/api/chat/{uploaded_dataset_id}",
                       json={"message": "average salary by department"}).json()["text"]
    assert "has the highest average salary" in text and "Here is" not in text


def test_a_tool_call_followed_by_braces_is_still_read():
    from app.ai.response_parser import parse_tool_call
    raw = ('Sure! ```json\n{"tool": "describe_column", "params": {"column": "age"}}\n```'
           " Hope this helps {:)}")
    assert parse_tool_call(raw) == {"tool": "describe_column", "params": {"column": "age"},
                                    "explanation": ""}


def test_a_malformed_tool_call_is_refused_not_run():
    from app.ai.response_parser import parse_tool_call
    assert parse_tool_call('{"tool": "describe_column", "params": "age"}') is None
    assert parse_tool_call('{"note": {"tool": 1}}') is None
    assert parse_tool_call("The average is 50.") is None


def test_a_configured_model_that_is_down_is_not_called_unconfigured(
        client, uploaded_dataset_id, monkeypatch):
    from app.ai import local_llm
    from app.config import config
    monkeypatch.setattr(config, "groq_api_key", "")
    monkeypatch.setattr(config, "local_llm_url", "http://127.0.0.1:9")
    monkeypatch.setattr(config, "local_llm_model", "stub")
    monkeypatch.setattr(local_llm, "privacy_mode", lambda: False)
    monkeypatch.setattr(local_llm, "generate", lambda *a, **k: None)
    r = client.post(f"/api/chat/{uploaded_dataset_id}", json={"message": "hi"})
    assert r.status_code == 503
    assert "did not answer" in r.json()["detail"] and "not configured" not in r.json()["detail"]
