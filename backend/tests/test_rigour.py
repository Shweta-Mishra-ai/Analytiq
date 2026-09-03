"""
The gate between "a number was computed" and "this is a finding".

The defect this exists for: given a 91%-imbalanced target and features
that were pure random noise, the pipeline reported

    accuracy 0.9125   f1 0.0   roc_auc 0.4457
    "Excellent classifier: 91.2% accuracy on held-out data."
    "Most important predictor: 'f0' (31% contribution)."

An AUC of 0.4457 is worse than a coin flip and an F1 of 0.0 means the
model never predicted the minority class once. Nothing in the arithmetic
was wrong; the comparison against the obvious guess was simply missing.

The mirror-image mistake is just as important and is tested here too: a
useful risk model on imbalanced data often scores *below* the majority
baseline on accuracy, and must not be rejected for it.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.rigour import (
    MIN_USABLE_AUC, assess_classifier, assess_finding,
    assess_regressor, detect_leakage,
)


# ── the original defect ───────────────────────────────────

def test_a_model_that_never_beats_the_baseline_is_not_a_finding():
    rng = np.random.default_rng(3)
    n = 800
    y = rng.choice([0, 1], n, p=[.92, .08])
    always_majority = np.zeros(n, dtype=int)
    v = assess_classifier(y, always_majority, auc=0.4457)
    assert v.usable is False
    assert "No reliable predictive signal" in v.verdict


def test_the_verdict_names_the_baseline_it_was_measured_against():
    """'91% accurate' means nothing without '...against a 91% baseline'."""
    rng = np.random.default_rng(3)
    y = rng.choice([0, 1], 800, p=[.92, .08])
    v = assess_classifier(y, np.zeros(800, dtype=int), auc=0.45)
    assert "always predicting the majority class" in v.verdict
    assert v.baseline_strategy


def test_importances_are_withheld_when_there_is_no_signal():
    from app.engines.ml_engine import run_ml_pipeline
    rng = np.random.default_rng(3)
    n = 800
    df = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(6)})
    df["target"] = rng.choice([0, 1], n, p=[.92, .08])
    report = run_ml_pipeline(df, "target")
    assert report.verdict is not None and report.verdict.usable is False
    assert report.feature_importance == [], \
        "importances from a model with no signal describe the noise it fitted"


# ── the mirror-image mistake ──────────────────────────────

def test_a_useful_risk_model_is_not_rejected_for_low_accuracy():
    """An attrition model tuned to catch leavers scores below the
    majority baseline on accuracy by design. Judging it on accuracy would
    reject exactly the models worth having."""
    rng = np.random.default_rng(11)
    n = 900
    y = rng.choice([0, 1], n, p=[.81, .19])
    # Predicts the minority class often: low accuracy, useful ranking.
    pred = np.where(rng.random(n) < 0.35, 1, 0)
    v = assess_classifier(y, pred, auc=0.78)
    assert v.usable is True
    assert v.lift < 0, "fixture should score below the majority baseline"


def test_the_verdict_explains_a_below_baseline_accuracy():
    """A reader who spots 74% accuracy against an 81% baseline, with no
    explanation, stops trusting the rest of the page."""
    rng = np.random.default_rng(11)
    n = 900
    y = rng.choice([0, 1], n, p=[.81, .19])
    pred = np.where(rng.random(n) < 0.35, 1, 0)
    v = assess_classifier(y, pred, auc=0.78)
    assert "tuned to find the minority class" in v.verdict
    assert "threshold" in v.verdict


def test_a_genuinely_good_model_passes():
    rng = np.random.default_rng(5)
    n = 600
    sig = rng.normal(0, 1, n)
    y = (sig > 0).astype(int)
    pred = (sig + rng.normal(0, .3, n) > 0).astype(int)
    assert assess_classifier(y, pred, auc=0.93).usable is True


@pytest.mark.parametrize("auc", [0.30, 0.50, MIN_USABLE_AUC - 0.01])
def test_rankings_at_or_below_chance_are_rejected(auc):
    rng = np.random.default_rng(1)
    y = rng.choice([0, 1], 400, p=[.6, .4])
    assert assess_classifier(y, rng.choice([0, 1], 400), auc=auc).usable is False


def test_too_few_rows_is_not_a_finding():
    y = [0, 1] * 10
    assert assess_classifier(y, [0] * 20, auc=0.9).usable is False


# ── regression ────────────────────────────────────────────

def test_a_regressor_worse_than_the_mean_is_not_a_finding():
    rng = np.random.default_rng(6)
    v = assess_regressor(rng.normal(100, 15, 300), rng.normal(100, 15, 300))
    assert v.usable is False
    assert v.model_score < 0
    assert "predicting the mean" in v.verdict


def test_a_regressor_that_explains_variance_passes():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 400)
    assert assess_regressor(x * 3 + rng.normal(0, .3, 400), x * 3).usable is True


# ── leakage ───────────────────────────────────────────────

def test_a_column_recorded_after_the_outcome_is_flagged():
    rng = np.random.default_rng(3)
    n = 800
    churn = rng.choice([0, 1], n, p=[.7, .3])
    df = pd.DataFrame({
        "churn": churn,
        "settlement": np.where(churn == 1, rng.uniform(100, 900, n), 0),
        "tenure": rng.integers(1, 60, n),
    })
    assert "settlement" in [f.column for f in detect_leakage(df, "churn")]


def test_leakage_through_missingness_is_caught():
    """A churn_date is NULL for exactly the customers who did not churn.
    Its values look unremarkable; its presence gives the answer away."""
    rng = np.random.default_rng(3)
    n = 800
    churn = rng.choice([0, 1], n, p=[.7, .3])
    df = pd.DataFrame({
        "churn": churn,
        "churn_date_days": np.where(churn == 1, rng.uniform(1, 30, n), np.nan),
        "tenure": rng.integers(1, 60, n),
    })
    found = detect_leakage(df, "churn")
    assert "churn_date_days" in [f.column for f in found]
    assert "has not happened yet" in " ".join(f.reason for f in found)


def test_ordinary_predictors_are_not_called_leakage():
    rng = np.random.default_rng(3)
    n = 800
    tenure = rng.integers(1, 60, n)
    churn = (rng.random(n) < (0.5 - tenure / 200)).astype(int)
    df = pd.DataFrame({"churn": churn, "tenure": tenure,
                       "spend": rng.uniform(10, 500, n)})
    assert detect_leakage(df, "churn") == []


def test_leakage_is_removed_from_the_model_not_just_noted():
    from app.engines.ml_engine import run_ml_pipeline
    rng = np.random.default_rng(3)
    n = 700
    churn = rng.choice([0, 1], n, p=[.7, .3])
    df = pd.DataFrame({
        "churn": churn,
        "churn_date_days": np.where(churn == 1, rng.uniform(1, 30, n), np.nan),
        "tenure": rng.integers(1, 60, n),
        "spend": rng.uniform(10, 500, n),
    })
    report = run_ml_pipeline(df, "churn")
    assert "churn_date_days" not in report.feature_cols
    assert any("churn_date_days" in w for w in report.warnings)


# ── statistical findings ──────────────────────────────────

def test_significant_but_trivial_is_not_reportable():
    """On a large enough sample almost everything is significant."""
    v = assess_finding(p_value=0.001, effect_size=0.04, n=50_000)
    assert v.reportable is False
    assert "too small to act on" in v.reason


def test_large_effect_that_is_significant_is_reportable():
    assert assess_finding(p_value=0.02, effect_size=0.55, n=800).reportable


def test_insignificant_is_not_reportable():
    assert assess_finding(p_value=0.4, effect_size=0.8, n=300).reportable is False


def test_a_tiny_sample_is_not_reportable():
    assert assess_finding(p_value=0.01, effect_size=0.9, n=12).reportable is False


def test_a_test_that_did_not_converge_is_not_reportable():
    assert assess_finding(p_value=float("nan"), effect_size=0.9,
                          n=500).reportable is False
