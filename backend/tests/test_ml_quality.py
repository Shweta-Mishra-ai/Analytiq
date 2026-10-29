"""
tests/test_ml_quality.py — a trained model's predictions, rankings and
verdicts have to match data whose answer is known.

Each case is a defect found by training every suggested target on real and
synthetic files: categories fed to linear models as ordinal codes, a blank
what-if category filled with the median code, importances that ranked a
row-stamp date first on a model with no skill, a boosted model saying "No"
with 99.7% certainty on noise, 53% against a 50% baseline called a signal,
and a 40-class target that took 37 seconds to score 2%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.ml_engine import predict_what_if, run_ml_pipeline, suggest_targets


def _regional(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(50, 10, n), rng.normal(20, 5, n)
    region = rng.choice(["North", "South", "East", "West"], n)
    y = 3 * x1 - 2 * x2 + np.where(region == "North", 500, 0) + rng.normal(0, 5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "region": region, "y_value": y})


def _leavers(n=1200, seed=1, effect=True):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 20, n)
    sat = rng.uniform(1, 5, n)
    dept = rng.choice(["A", "B", "C"], n)
    logit = (-1 + 1.5 * (sat < 2) + 1.2 * (dept == "C") - 0.1 * tenure) if effect \
        else np.full(n, -1.2)
    left = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return pd.DataFrame({"tenure": tenure, "satisfaction": sat, "dept": dept,
                         "noise": rng.normal(size=n), "left": left})


@pytest.fixture(scope="module")
def regional():
    df = _regional()
    return df, run_ml_pipeline(df, "y_value")


@pytest.fixture(scope="module")
def leavers():
    return run_ml_pipeline(_leavers(), "left")


def test_a_category_effect_is_recovered_exactly(regional):
    df, rep = regional
    typical = {f["name"]: f["default"] for f in rep.feature_inputs}
    north = predict_what_if(rep, dict(typical, region="North"))["prediction"]
    south = predict_what_if(rep, dict(typical, region="South"))["prediction"]
    assert north - south == pytest.approx(500, rel=0.05)
    assert rep.best_model.test_score > 0.99


def test_a_blank_category_means_the_typical_one_not_an_arbitrary_code(leavers):
    typical = {f["name"]: f["default"] for f in leavers.feature_inputs}
    full = predict_what_if(leavers, typical)["probabilities"]
    blank_dept = predict_what_if(leavers, {k: v for k, v in typical.items() if k != "dept"})
    assert "dept" in blank_dept["assumed_at_training_median"]
    assert blank_dept["probabilities"]["Yes"] == pytest.approx(full["Yes"], abs=0.02)


def test_a_real_effect_is_called_a_signal_with_the_right_direction(leavers):
    assert not leavers.no_signal
    imp = {f.feature: f for f in leavers.feature_importance}
    assert imp["satisfaction"].direction == "negative"     # low satisfaction → leave
    assert imp["tenure"].direction == "negative"
    assert imp["dept"].direction == "varies" and "'C'" in imp["dept"].explanation
    assert imp.get("noise") is None or imp["noise"].importance < 0.05


def test_noise_is_called_noise_and_not_ranked_as_an_effect():
    rep = run_ml_pipeline(_leavers(effect=False), "left")
    assert rep.no_signal
    assert all("not evidence of an effect" in f.explanation for f in rep.feature_importance)


def test_a_model_on_noise_is_not_confidently_wrong():
    rep = run_ml_pipeline(_leavers(effect=False), "left")
    typical = {f["name"]: f["default"] for f in rep.feature_inputs}
    out = predict_what_if(rep, typical)
    assert out["confidence"] < 95
    assert out["reliable"] is False


def test_regression_ranges_come_from_held_out_errors(regional):
    _, rep = regional
    out = predict_what_if(rep, {f["name"]: f["default"] for f in rep.feature_inputs})
    assert out["lower"] < out["prediction"] < out["upper"]
    assert "8 in 10" in out["confidence_note"]


def test_too_many_classes_is_refused_and_not_suggested():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=600),
                       "kind": rng.choice([f"type{i}" for i in range(40)], 600)})
    assert "kind" not in [t["column"] for t in suggest_targets(df)]
    with pytest.raises(ValueError, match="too many to predict"):
        run_ml_pipeline(df, "kind")


def test_outcomes_are_suggested_before_grouping_columns(hr_df):
    order = [t["column"] for t in suggest_targets(hr_df)]
    assert order.index("attrition") < order.index("department")


def test_a_column_only_known_after_the_outcome_is_left_out():
    df = _leavers()
    df["exit_reason"] = np.where(df["left"] == "Yes", "moved", "(none)")
    rep = run_ml_pipeline(df, "left")
    assert "exit_reason" not in rep.feature_cols
    assert any("exit_reason" in w for w in rep.warnings)


def test_training_api_refuses_a_many_class_target_with_422(client):
    import io
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"a": rng.normal(size=300),
                       "kind": rng.choice([f"t{i}" for i in range(30)], 300)})
    r = client.post("/api/datasets/upload", files={
        "file": ("k.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    out = client.post(f"/api/ml/{ds}/train", json={"target": "kind"})
    assert out.status_code == 422 and "too many" in out.json()["detail"]


# ══════════════════════════════════════════════════════════
#  Dashboard tiles and KPIs
# ══════════════════════════════════════════════════════════

def test_dashboard_tiles_follow_the_report_rules(hr_df):
    from app.api.charts import auto_tiles
    tiles = auto_tiles(hr_df)
    kinds = [(t["type"], t.get("x"), t.get("y"), t.get("agg")) for t in tiles]
    assert ("rate", "department", "attrition", None) in kinds
    assert not any(t["type"] == "bar" and t.get("y") == "salary" and t.get("agg") == "sum"
                   for t in tiles), kinds
    assert not any(t["type"] == "line" and t.get("x") == "hire_date" for t in tiles)
    assert not any("employee_id" in (t.get("x"), t.get("y")) for t in tiles)


def test_dashboard_kpis_average_what_should_be_averaged(client, uploaded_dataset_id):
    kpis = client.post(f"/api/charts/{uploaded_dataset_id}/kpis",
                       json={"filters": []}).json()["kpis"]
    labels = [k["label"] for k in kpis]
    assert "Average salary" in labels and "Attrition rate" in labels
    assert not any(label.startswith("Σ") or label == "Total age" for label in labels)


def test_rate_tile_builds_and_cross_filters(client, uploaded_dataset_id):
    ds = uploaded_dataset_id
    r = client.post(f"/api/charts/{ds}/build", json={
        "type": "rate", "x": "department", "y": "attrition", "filters": []})
    assert r.status_code == 200, r.text
    filtered = client.post(f"/api/charts/{ds}/build", json={
        "type": "rate", "x": "department", "y": "attrition",
        "filters": [{"column": "department", "op": "eq", "value": "Sales"}]})
    assert filtered.status_code == 200
    assert len(filtered.json()["figure"]["data"][0]["x"]) == 1
