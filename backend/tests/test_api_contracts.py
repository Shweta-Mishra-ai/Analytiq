"""
tests/test_api_contracts.py — request validation and error contracts.

Each case here was either a 500 from pandas internals, a silently ignored
input, or an answer to a different question than the one asked.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest


def _upload(client, df: pd.DataFrame, name="t.csv") -> str:
    r = client.post("/api/datasets/upload", files={
        "file": (name, io.BytesIO(df.to_csv(index=False).encode()), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["meta"]["dataset_id"]


# ── dashboard filters ─────────────────────────────────────

def test_contains_filter_is_a_literal_match(client):
    ds = _upload(client, pd.DataFrame({"note": ["a (b)", "c", "(x", "d"], "v": [1, 2, 3, 4]}))
    r = client.post(f"/api/charts/{ds}/kpis",
                    json={"filters": [{"column": "note", "op": "contains", "value": "("}]})
    assert r.status_code == 200, r.text
    rows = next(k for k in r.json()["kpis"] if k["label"] == "Rows")["value"]
    assert rows == 2


def test_a_filter_that_cannot_apply_is_refused_not_silently_dropped(client, uploaded_dataset_id):
    ds = uploaded_dataset_id
    for f in [{"column": "dropped_by_cleaning", "op": "eq", "value": "x"},
              {"column": "salary", "op": "gt", "value": "lots"},
              {"column": "salary", "op": "frobnicate", "value": 1}]:
        r = client.post(f"/api/charts/{ds}/kpis", json={"filters": [f]})
        assert r.status_code == 422, (f, r.text)


# ── chart builder ─────────────────────────────────────────

@pytest.mark.parametrize("body", [
    {"type": "bar", "x": "department", "y": "department"},
    {"type": "line", "x": "hire_date", "y": "hire_date"},
    {"type": "bar", "x": "department", "y": "hire_date", "agg": "sum"},
    {"type": "bar", "x": "department", "y": "attrition", "agg": "mean"},
    {"type": "bar", "x": "department", "y": "salary", "agg": "__class__"},
    {"type": "histogram", "x": "salary", "nbins": 10_000_000},
])
def test_invalid_chart_requests_are_422_not_500(client, uploaded_dataset_id, body):
    r = client.post(f"/api/charts/{uploaded_dataset_id}/build", json=body)
    assert r.status_code == 422, r.text


def test_count_of_a_text_column_still_builds(client, uploaded_dataset_id):
    r = client.post(f"/api/charts/{uploaded_dataset_id}/build",
                    json={"type": "bar", "x": "department", "y": "attrition", "agg": "count"})
    assert r.status_code == 200, r.text


# ── ML what-if ────────────────────────────────────────────

def test_what_if_never_answers_with_a_model_for_another_target(client, uploaded_dataset_id):
    ds = uploaded_dataset_id
    assert client.post(f"/api/ml/{ds}/train", json={"target": "salary"}).status_code == 200
    r = client.post(f"/api/ml/{ds}/what-if", json={"target": "attrition", "inputs": {}})
    assert r.status_code == 404
    assert "attrition" in r.json()["detail"]


# ── A/B test ──────────────────────────────────────────────

def test_ab_test_refuses_comparing_a_group_with_itself(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/ab-test", json={
        "group_col": "department", "metric_col": "salary",
        "group_a_value": "Sales", "group_b_value": "Sales"})
    assert r.status_code == 422


def test_ab_test_asks_which_value_is_a_success_when_it_cannot_tell(client):
    df = pd.DataFrame({"arm": ["a", "b"] * 50, "outcome": ["Kept", "Lapsed"] * 50})
    ds = _upload(client, df)
    r = client.post(f"/api/analytics/{ds}/ab-test", json={"group_col": "arm", "metric_col": "outcome"})
    assert r.status_code == 422
    assert "success_value" in r.json()["detail"]
    r = client.post(f"/api/analytics/{ds}/ab-test",
                    json={"group_col": "arm", "metric_col": "outcome", "success_value": "Kept"})
    assert r.status_code == 200, r.text


def test_ab_test_rejects_an_impossible_confidence_level(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/ab-test", json={
        "group_col": "department", "metric_col": "salary", "confidence_level": 1.5})
    assert r.status_code == 422


def test_ab_test_says_observational_groups_are_not_an_experiment(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/ab-test",
                    json={"group_col": "department", "metric_col": "salary"})
    assert r.status_code == 200
    assert any("randomly assigned" in w for w in r.json()["warnings"])


# ── uploads ───────────────────────────────────────────────

def test_an_oversized_upload_is_refused(client, monkeypatch):
    from app.config import config
    monkeypatch.setattr(config, "max_file_mb", 1)
    big = b"a,b\n" + b"1,2\n" * 400_000          # ~1.6 MB
    r = client.post("/api/datasets/upload", files={"file": ("big.csv", io.BytesIO(big), "text/csv")})
    assert r.status_code == 413


def test_rag_query_bounds_are_enforced(client):
    kb = client.post("/api/rag/kb", json={"name": "bounds"}).json()["kb_id"]
    try:
        assert client.post(f"/api/rag/kb/{kb}/query",
                           json={"question": "x", "k": 100_000}).status_code == 422
        assert client.post(f"/api/rag/kb/{kb}/query",
                           json={"question": ""}).status_code == 422
    finally:
        client.delete(f"/api/rag/kb/{kb}")


def test_survival_refuses_the_same_column_in_two_roles(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/survival", json={
        "duration_col": "tenure_years", "event_col": "attrition", "group_col": "tenure_years"})
    assert r.status_code == 422


def test_charts_group_on_a_column_mixing_numbers_and_text(client):
    df = pd.DataFrame({"revenue": [str(i) if i % 10 else "Pending" for i in range(60)],
                       "units": range(60)})
    ds = _upload(client, df)
    r = client.post(f"/api/charts/{ds}/build",
                    json={"type": "bar", "x": "revenue", "y": "units", "agg": "sum"})
    assert r.status_code == 200, r.text


# ── dashboard honesty ─────────────────────────────────────

def test_kpi_strip_does_not_sum_identifiers(client):
    df = pd.DataFrame({"customer_id": range(1, 61), "revenue": [10.0] * 60})
    ds = _upload(client, df)
    labels = [k["label"] for k in client.post(f"/api/charts/{ds}/kpis", json={"filters": []}).json()["kpis"]]
    assert "Σ revenue" in labels and "Σ customer_id" not in labels


def test_a_partial_final_month_is_not_drawn_as_a_collapse(client):
    # Steady daily revenue to 10 March: the March sum is a third of a month.
    dates = pd.date_range("2023-01-01", "2025-03-10", freq="D")
    ds = _upload(client, pd.DataFrame({"order_date": dates, "revenue": 100.0}))
    r = client.post(f"/api/charts/{ds}/build",
                    json={"type": "line", "x": "order_date", "y": "revenue", "agg": "sum"})
    assert r.status_code == 200, r.text
    fig = r.json()["figure"]
    ys = fig["data"][0]["y"]
    ys = ys if isinstance(ys, list) else None
    if ys is not None:      # plotly may binary-encode; check via the label otherwise
        assert min(ys) > 0.8 * max(ys)
    assert "partial month excluded" in fig["layout"]["xaxis"]["title"]["text"]


def test_means_over_time_keep_every_period(client):
    dates = pd.date_range("2023-01-01", "2025-03-10", freq="D")
    ds = _upload(client, pd.DataFrame({"order_date": dates, "revenue": 100.0}))
    r = client.post(f"/api/charts/{ds}/build",
                    json={"type": "line", "x": "order_date", "y": "revenue", "agg": "mean"})
    assert "excluded" not in str(r.json()["figure"]["layout"]["xaxis"].get("title", ""))


# ── pie charts are shares of the whole ────────────────────

def test_pie_keeps_the_largest_slice_and_groups_the_rest():
    from app.engines.chart_engine import make_pie
    df = pd.DataFrame({"cat": [f"c{i:02d}" for i in range(15)], "v": [1] * 14 + [1000]})
    pie = make_pie(df, "cat", "v").data[0]
    labels, values = list(pie.labels), [float(v) for v in pie.values]
    assert labels[0] == "c14"
    assert labels[-1].startswith("Other")
    assert sum(values) == df["v"].sum()          # nothing dropped from the whole


def test_dashboard_pie_uses_every_group_and_refuses_means(client):
    df = pd.DataFrame({"cat": [f"c{i:02d}" for i in range(30)] * 2, "v": range(60)})
    ds = _upload(client, df)
    r = client.post(f"/api/charts/{ds}/build",
                    json={"type": "pie", "x": "cat", "y": "v", "agg": "sum"})
    assert r.status_code == 200, r.text
    vals = r.json()["figure"]["data"][0]["values"]
    if isinstance(vals, list):
        assert sum(vals) == df["v"].sum()
    assert client.post(f"/api/charts/{ds}/build", json={
        "type": "pie", "x": "cat", "y": "v", "agg": "mean"}).status_code == 422


def test_chat_tools_reject_unknown_aggregations_and_match_literally():
    from app.ai.tool_dispatcher import dispatch
    df = pd.DataFrame({"g": ["a", "b"], "v": [1, 2], "note": ["x (1)", "y"]})
    assert not dispatch(df, "aggregate", {"group_col": "g", "value_col": "v",
                                          "agg_func": "__class__"}).success
    r = dispatch(df, "filter", {"column": "note", "operator": "contains", "value": "("})
    assert r.success and len(r.dataframe) == 1
