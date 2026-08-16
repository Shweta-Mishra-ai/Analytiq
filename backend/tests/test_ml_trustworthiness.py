"""
Whether the prediction feature can be trusted.

Two defects made it untrustworthy in opposite directions.

It **used row identifiers as features**: `is_id_column` existed in the
codebase and the ML engine never called it, so a numeric `employee_id`
came out as the second most important predictor on a 500-row HR file. A
model keyed to a row identifier scores well in testing and predicts
nothing on data it has not seen.

And it **had no baseline at all**. "Best model: Gradient Boosting,
accuracy 0.65" on a 65/35 split reads as a working model; answering "no"
every time also scores 0.65. Without the comparison there is no way for
the reader — or the analyst delivering it — to tell the two apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.ml_engine import prepare_features, run_ml_pipeline


@pytest.fixture()
def hr_frame():
    rng = np.random.default_rng(300)
    n = 600
    df = pd.DataFrame({
        "employee_id": range(10_000, 10_000 + n),
        "staff_ref": [f"E{i:05d}" for i in range(n)],
        "department": rng.choice(["Sales", "Eng", "Ops"], n),
        "salary": rng.normal(60_000, 12_000, n).round(0),
        "tenure_years": rng.integers(0, 20, n),
    })
    # A real, learnable rule.
    df["attrition"] = (df["tenure_years"] < 5).astype(int)
    return df


# ══════════════════════════════════════════════════════════
#  Identifiers are not predictors
# ══════════════════════════════════════════════════════════

def test_numeric_identifiers_are_not_used_as_features(hr_frame):
    X, _y, _enc = prepare_features(hr_frame, "attrition")
    assert "employee_id" not in X.columns
    assert "staff_ref" not in X.columns


def test_real_predictors_survive(hr_frame):
    """Dropping identifiers must not take the actual features with it."""
    X, _y, _enc = prepare_features(hr_frame, "attrition")
    assert {"salary", "tenure_years", "department"} <= set(X.columns)


def test_an_identifier_never_appears_in_feature_importance(hr_frame):
    report = run_ml_pipeline(hr_frame, target_col="attrition")
    names = [f.feature for f in report.feature_importance]
    assert "employee_id" not in names, names


# ══════════════════════════════════════════════════════════
#  A model is measured against doing nothing
# ══════════════════════════════════════════════════════════

def test_a_baseline_is_always_computed(hr_frame):
    report = run_ml_pipeline(hr_frame, target_col="attrition")
    assert report.baseline_score is not None
    assert report.skill_over_baseline is not None


def test_a_real_signal_shows_skill_over_the_baseline(hr_frame):
    report = run_ml_pipeline(hr_frame, target_col="attrition")
    assert report.skill_over_baseline > 0.10


def test_a_noise_target_is_reported_as_unpredictable():
    """The honest answer is "these columns do not explain that", not a
    ranked list of models."""
    rng = np.random.default_rng(301)
    n = 600
    df = pd.DataFrame({
        "department": rng.choice(["A", "B", "C"], n),
        "salary": rng.normal(60_000, 12_000, n).round(0),
        "tenure_years": rng.integers(0, 20, n),
    })
    df["left"] = (rng.random(n) < 0.35).astype(int)      # pure noise
    report = run_ml_pipeline(df, target_col="left")
    joined = " ".join(report.warnings).lower()
    assert ("not support prediction" in joined
            or "weak signal" in joined), report.warnings
    assert "simplest possible rule" in joined or "baseline" in joined


def test_a_lucky_split_is_not_mistaken_for_skill():
    """A 20% holdout is noisy: a target of pure coin flips scored 0.68
    against a 0.57 baseline on one split. The verdict has to hold across
    folds, allowing for the variation between them."""
    rng = np.random.default_rng(301)
    n = 600
    df = pd.DataFrame({
        "department": rng.choice(["A", "B", "C"], n),
        "salary": rng.normal(60_000, 12_000, n).round(0),
        "tenure_years": rng.integers(0, 20, n),
    })
    df["left"] = (rng.random(n) < 0.35).astype(int)
    report = run_ml_pipeline(df, target_col="left")
    assert report.best_model.test_score > report.baseline_score, \
        "fixture no longer reproduces the lucky-split case"
    assert any("not support prediction" in w.lower()
               for w in report.warnings), report.warnings


def test_the_warning_names_both_numbers():
    """"Weak" with no figures is not something a reader can act on."""
    rng = np.random.default_rng(302)
    n = 500
    df = pd.DataFrame({
        "grp": rng.choice(["A", "B"], n),
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
    })
    df["target"] = (rng.random(n) < 0.30).astype(int)
    report = run_ml_pipeline(df, target_col="target")
    weak = [w for w in report.warnings if "baseline" in w.lower()]
    assert weak
    assert any(ch.isdigit() for ch in weak[0])


def test_a_regression_baseline_is_the_mean():
    rng = np.random.default_rng(303)
    n = 500
    df = pd.DataFrame({
        "size": rng.normal(100, 20, n),
        "grade": rng.choice(["a", "b", "c"], n),
    })
    df["price"] = rng.normal(500, 90, n)      # unrelated to the features
    report = run_ml_pipeline(df, target_col="price")
    assert report.task == "regression"
    assert report.baseline_score is not None
    # R² of predicting the mean is ~0 on held-out data.
    assert report.baseline_score < 0.05


# ══════════════════════════════════════════════════════════
#  The pipeline does not fall over
# ══════════════════════════════════════════════════════════

def test_a_frame_of_only_identifiers_fails_clearly():
    """No usable features is a message, not a traceback."""
    df = pd.DataFrame({
        "record_id": range(200),
        "ref_code": [f"R{i}" for i in range(200)],
        "target": np.random.default_rng(304).integers(0, 2, 200),
    })
    report = run_ml_pipeline(df, target_col="target")
    assert report.warnings
    assert any("no usable feature" in w.lower() for w in report.warnings)


def test_too_few_rows_is_refused_with_a_reason():
    rng = np.random.default_rng(305)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, 15),
        "grp": rng.choice(["a", "b"], 15),
        "target": rng.integers(0, 2, 15),
    })
    report = run_ml_pipeline(df, target_col="target")
    assert any("too few rows" in w.lower() for w in report.warnings)


def test_target_leakage_is_still_caught(hr_frame):
    """A column that copies the target produces a spectacular, useless
    model."""
    df = hr_frame.copy()
    df["already_left"] = df["attrition"]
    report = run_ml_pipeline(df, target_col="attrition")
    assert "already_left" not in report.feature_cols
    assert any("leakage" in w.lower() for w in report.warnings)


def test_missing_values_do_not_crash_the_pipeline(hr_frame):
    df = hr_frame.copy()
    df.loc[df.index[:100], "salary"] = np.nan
    df.loc[df.index[50:150], "department"] = None
    report = run_ml_pipeline(df, target_col="attrition")
    assert report.best_model is not None
