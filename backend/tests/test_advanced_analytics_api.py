"""API-level tests for api/advanced_analytics.py (RFM, A/B test, survival)."""
from __future__ import annotations

import io


def test_rfm_endpoints(client):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(11)
    n = 600
    df = pd.DataFrame({
        "customer_id": rng.integers(1, 90, n),
        "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
        "amount": rng.gamma(2.0, 30.0, n).round(2),
    })
    r = client.post("/api/datasets/upload",
                     files={"file": ("tx.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")})
    assert r.status_code == 200, r.text
    ds = r.json()["meta"]["dataset_id"]

    r = client.get(f"/api/analytics/{ds}/rfm/columns")
    assert r.status_code == 200
    assert r.json()["detected"]["customer_id"] == "customer_id"

    r = client.get(f"/api/analytics/{ds}/rfm")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_customers"] > 0
    assert len(body["segment_summary"]) > 0
    assert "customer_table" in body


def test_rfm_endpoint_422_when_data_cannot_support_it(client):
    r = client.post("/api/datasets/upload", files={
        "file": ("nope.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    r = client.get(f"/api/analytics/{ds}/rfm")
    assert r.status_code == 422


def test_ab_test_endpoint_continuous(client, uploaded_dataset_id):
    r = client.get(f"/api/analytics/{uploaded_dataset_id}/ab-test/fields")
    assert r.status_code == 200
    assert "department" in r.json()["group_columns"]

    r = client.post(f"/api/analytics/{uploaded_dataset_id}/ab-test", json={
        "group_col": "department", "metric_col": "salary"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["test_type"] == "continuous"
    assert "verdict" in body and "recommendation" in body


def test_ab_test_endpoint_conversion(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/ab-test", json={
        "group_col": "department", "metric_col": "attrition"})
    assert r.status_code == 200, r.text
    assert r.json()["test_type"] == "conversion"


def test_ab_test_endpoint_missing_column_422(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/ab-test", json={
        "group_col": "does_not_exist", "metric_col": "salary"})
    assert r.status_code == 422


def test_survival_endpoint(client, uploaded_dataset_id):
    r = client.get(f"/api/analytics/{uploaded_dataset_id}/survival/fields")
    assert r.status_code == 200
    assert "tenure_years" in r.json()["duration_columns"]

    r = client.post(f"/api/analytics/{uploaded_dataset_id}/survival", json={
        "duration_col": "tenure_years", "event_col": "attrition", "group_col": "department"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["overall_curve"]["n_total"] > 0
    assert len(body["group_curves"]) >= 2


def test_survival_endpoint_missing_column_422(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/survival", json={
        "duration_col": "tenure_years", "event_col": "does_not_exist"})
    assert r.status_code == 422


def test_advanced_analytics_404_for_unknown_dataset(client):
    r = client.get("/api/analytics/nonexistent-id/rfm")
    assert r.status_code == 404
    r = client.post("/api/analytics/nonexistent-id/survival", json={
        "duration_col": "a", "event_col": "b"})
    assert r.status_code == 404
    r = client.get("/api/analytics/nonexistent-id/benchmarks")
    assert r.status_code == 404
    r = client.get("/api/analytics/nonexistent-id/drivers")
    assert r.status_code == 404


# ── scenario / what-if ───────────────────────────────────────────────────

def test_scenario_endpoint(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/scenario", json={
        "driver_col": "satisfaction", "target_col": "salary", "change_pct": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "projected_target_mean" in body
    assert "reliable" in body
    assert "caveat" in body


def test_scenario_endpoint_422_for_non_numeric(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/scenario", json={
        "driver_col": "department", "target_col": "salary"})
    assert r.status_code == 422


def test_scenario_endpoint_422_for_unknown_column(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/scenario", json={
        "driver_col": "nope", "target_col": "salary"})
    assert r.status_code == 422


# ── benchmarks ───────────────────────────────────────────────────────────

def test_benchmarks_endpoint(client, uploaded_dataset_id):
    r = client.get(f"/api/analytics/{uploaded_dataset_id}/benchmarks")
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["benchmarks"], list)


def test_industry_benchmarks_endpoint(client, uploaded_dataset_id):
    r = client.get(f"/api/analytics/{uploaded_dataset_id}/industry-benchmarks")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "domain" in body
    assert isinstance(body["benchmarks"], list)


# ── predictive drivers ───────────────────────────────────────────────────

def test_drivers_endpoint_autodetects_target(client, uploaded_dataset_id):
    r = client.get(f"/api/analytics/{uploaded_dataset_id}/drivers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target"] == "attrition"
    assert body["top_drivers"]


def test_drivers_endpoint_422_when_no_binary_target(client):
    df_csv = b"a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n"
    r = client.post("/api/datasets/upload",
                     files={"file": ("plain.csv", io.BytesIO(df_csv), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    r = client.get(f"/api/analytics/{ds}/drivers")
    assert r.status_code == 422


# ── dataset comparison ───────────────────────────────────────────────────

def test_compare_endpoint(client, hr_csv_bytes):
    import pandas as pd
    r = client.post("/api/datasets/upload",
                     files={"file": ("q1.csv", io.BytesIO(hr_csv_bytes), "text/csv")})
    ds_a = r.json()["meta"]["dataset_id"]

    df_b = pd.read_csv(io.BytesIO(hr_csv_bytes))
    df_b["salary"] = df_b["salary"] * 1.2
    r = client.post("/api/datasets/upload", files={
        "file": ("q2.csv", io.BytesIO(df_b.to_csv(index=False).encode()), "text/csv")})
    ds_b = r.json()["meta"]["dataset_id"]

    r = client.post(f"/api/analytics/{ds_a}/compare", json={
        "other_dataset_id": ds_b, "label_a": "Q1", "label_b": "Q2"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label_a"] == "Q1"
    assert body["column_comparisons"]


def test_compare_endpoint_404_for_unknown_other_dataset(client, uploaded_dataset_id):
    r = client.post(f"/api/analytics/{uploaded_dataset_id}/compare", json={
        "other_dataset_id": "does-not-exist"})
    assert r.status_code == 404
