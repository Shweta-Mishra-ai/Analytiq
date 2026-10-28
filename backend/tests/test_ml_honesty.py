"""
tests/test_ml_honesty.py — the ML page must say what a model is worth
against the free baseline, train on real features, and answer what-if
questions for the inputs a user actually types.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.ml_engine import predict_what_if, run_ml_pipeline


@pytest.fixture(scope="module")
def attrition_report():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import _make_hr_df
    return run_ml_pipeline(_make_hr_df(), "attrition")


def test_a_model_no_better_than_the_majority_class_is_called_that(attrition_report):
    # Synthetic attrition is random noise: 80% "No". Any honest verdict has
    # to say the model adds nothing over answering "No" every time.
    first = attrition_report.insights[0]
    assert "Good classifier" not in first
    assert "most common class" in first and "'No'" in first


def test_identifier_columns_are_not_features(attrition_report):
    assert "employee_id" not in attrition_report.feature_cols


def test_what_if_accepts_categories_and_dates_as_typed(attrition_report):
    out = predict_what_if(attrition_report, {
        "department": "Sales", "salary": "52000", "age": 30, "tenure_years": 2,
        "satisfaction": 2.5, "hire_date": "2019-06-01"})
    assert "error" not in out, out
    assert out["prediction_label"] in ("Yes", "No")
    assert set(out["probabilities"]) == {"Yes", "No"}


def test_what_if_names_the_features_it_had_to_assume(attrition_report):
    out = predict_what_if(attrition_report, {"salary": 52000})
    assert "error" not in out, out
    assert "department" in out["assumed_at_training_median"]


@pytest.mark.parametrize("inputs,needle", [
    ({"department": "Legal"}, "not a value"),
    ({"salary": "lots"}, "needs a number"),
    ({"not_a_feature": 1}, "Not features"),
])
def test_what_if_rejects_inputs_it_cannot_encode(attrition_report, inputs, needle):
    assert needle in predict_what_if(attrition_report, inputs)["error"]


def test_a_numeric_two_class_target_keeps_both_classes():
    # astype(int) turned 0.2 and 0.7 into a single class 0.
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=200),
                       "y": np.where(x > 0, 0.7, 0.2)})
    rep = run_ml_pipeline(df, "y")
    assert rep.best_model is not None and rep.best_model.model is not None
    assert len(rep.target_encoder.classes_) == 2
    assert rep.best_model.test_score > 0.9          # x separates the classes


def test_what_if_endpoint_turns_an_unencodable_input_into_422(client, uploaded_dataset_id):
    ds = uploaded_dataset_id
    assert client.post(f"/api/ml/{ds}/train", json={"target": "attrition"}).status_code == 200
    r = client.post(f"/api/ml/{ds}/what-if",
                    json={"target": "attrition", "inputs": {"department": "Legal"}})
    assert r.status_code == 422
    ok = client.post(f"/api/ml/{ds}/what-if",
                     json={"target": "attrition", "inputs": {"department": "Sales"}})
    assert ok.status_code == 200, ok.text


def test_dates_are_not_offered_or_accepted_as_targets(client):
    import io
    dates = pd.to_datetime(["2019-01-01", "2019-02-01", "2019-03-01"] * 20)
    df = pd.DataFrame({"hire_date": dates, "salary": np.arange(60.0), "team": ["a", "b", "c"] * 20})
    r = client.post("/api/datasets/upload",
                    files={"file": ("d.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    t = client.get(f"/api/ml/{ds}/targets")
    assert t.status_code == 200, t.text
    assert "hire_date" not in [x["column"] for x in t.json()["targets"]]
    assert client.post(f"/api/ml/{ds}/train", json={"target": "hire_date"}).status_code == 422
