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
    corr = dispatch(df, "correlation", {"col_a": "age", "col_b": "salary"}, "").text_output
    assert corr.startswith("age and salary move in opposite directions slightly (r = -0.26)")
    assert "not proof that one causes the other" in corr


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
