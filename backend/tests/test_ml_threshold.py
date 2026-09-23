"""
Where to cut a classifier, and why the default is not a neutral answer.

The ML page reported every classification model at the 0.5 cut-off. On
an 80/20 attrition file that described the best model as F1 0.17,
catching 11% of the leavers — a model nobody could act on. The same
model, cut where it performs best, catches 64% of them.

The predictive engine behind the PDF had chosen its threshold from the
start and said so in the report. This page had not, so one product
shipped two standards and the weaker one was the one a user clicked on.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def imbalanced():
    """An outcome at roughly 20%, predictable but not trivially so."""
    rng = np.random.default_rng(11)
    n = 900
    overtime = rng.choice([0, 1], n, p=[0.7, 0.3])
    tenure = rng.integers(1, 20, n)
    score = 0.08 + 0.42 * overtime + 0.012 * (12 - np.clip(tenure, 0, 12))
    left = (rng.random(n) < np.clip(score, 0, 0.95)).astype(int)
    return pd.DataFrame({
        "overtime": overtime, "tenure": tenure,
        "salary": rng.normal(60000, 12000, n).round(0),
        "left": left,
    })


@pytest.fixture(scope="module")
def report(imbalanced):
    from app.engines.ml.runner import run_ml_pipeline
    return run_ml_pipeline(imbalanced, "left")


def test_a_classifier_reports_the_cut_it_performs_best_at(report):
    best = report.best_model
    assert best is not None
    assert best.threshold is not None, \
        "the model was reported only at the 0.5 default"
    assert 0.05 <= best.threshold <= 0.95


def test_the_chosen_cut_is_at_least_as_good_as_the_default(report):
    """It is chosen by maximising F1, so it cannot be worse."""
    best = report.best_model
    assert best.f1_at_threshold >= best.f1 - 1e-9


def test_precision_and_recall_come_with_it(report):
    """A threshold without them is a number the reader cannot weigh."""
    best = report.best_model
    assert best.precision_at_threshold is not None
    assert best.recall_at_threshold is not None
    assert 0.0 <= best.recall_at_threshold <= 1.0
    assert 0.0 <= best.precision_at_threshold <= 1.0


def test_the_reader_is_told_the_default_is_a_choice(report):
    """Silently reporting 0.5 lets somebody conclude the model is weak
    when what is weak is the cut-off."""
    best = report.best_model
    if best.f1_at_threshold <= best.f1 + 0.02:
        pytest.skip("this sample gains nothing from re-cutting")

    joined = " ".join(report.insights).lower()
    assert "cut at" in joined or "cut-off" in joined
    assert "0.5" in joined


def test_regression_is_not_given_a_threshold(imbalanced):
    """There is no cut-off on a continuous target, and inventing one
    would be noise in the table."""
    from app.engines.ml.runner import run_ml_pipeline

    rep = run_ml_pipeline(imbalanced, "salary")

    assert rep.task == "regression"
    assert rep.best_model.threshold is None


def test_a_multiclass_target_gets_no_single_cut():
    """One number cannot describe where to cut three classes."""
    from app.engines.ml.training import _at_operating_threshold

    y = np.array([0, 1, 2] * 30)
    proba = np.full((90, 3), 1 / 3)

    assert _at_operating_threshold(y, proba) == {}


def test_a_missing_probability_is_handled(imbalanced):
    from app.engines.ml.training import _at_operating_threshold

    assert _at_operating_threshold(np.array([0, 1]), None) == {}


# ══════════════════════════════════════════════════════════
#  The prose around it
# ══════════════════════════════════════════════════════════

def test_the_top_predictor_is_not_described_twice(report):
    """"'OverTime=No' (25% contribution). Important feature (25%
    contribution)." — the insight prefixed a percentage the explanation
    already carried."""
    line = next((i for i in report.insights
                 if i.startswith("Most important predictor")), "")
    if not line:
        pytest.skip("no importance ranking on this sample")

    import re
    assert len(re.findall(r"\d+% contribution", line)) <= 1, line


def test_a_direction_is_described_as_a_direction(report):
    """"Decreases effect on target" is not a sentence and does not say
    decreases what."""
    for imp in report.feature_importance[:5]:
        assert "effect on target" not in imp.explanation, imp.explanation
    joined = " ".join(i.explanation for i in report.feature_importance[:5])
    assert "prediction" in joined
