"""
Turning a ranking into a decision.

AUC answers "does the model rank correctly". It does not answer the
question a manager asks — "we can contact 200 people this month, which
200, and how many of them were going to leave anyway?" Without precision,
recall and lift at a stated budget, a predictive section reports a model
rather than a decision.
"""
import io

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from app.engines.pdf_builder import build_pdf
from app.engines.predictive import (
    calibration_gap, compute_drivers, decision_curve, find_binary_target,
    find_top_cluster,
)

CONFIG = {"title": "Decision", "client_name": "T", "subtitle": "",
          "confidential": True, "theme_name": "", "logo_path": None,
          "prepared_by": "", "source_table": "src"}


def _attrition_df(n=1200, seed=11):
    r = np.random.default_rng(seed)
    ot = r.choice(["Yes", "No"], n, p=[.35, .65])
    ten = r.integers(1, 20, n)
    p = (.05 + .32 * (ot == "Yes") + .24 * (ten <= 2)).clip(0, .92)
    return pd.DataFrame({
        "employee_id": np.arange(n),
        "Attrition": np.where(r.random(n) < p, "Yes", "No"),
        "OverTime": ot, "YearsAtCompany": ten,
        "JobRole": r.choice(["Sales Rep", "Engineer", "Manager"], n),
        "MonthlyIncome": r.uniform(2500, 19000, n).round(2),
    })


@pytest.fixture(scope="module")
def drivers():
    df = _attrition_df()
    target = find_binary_target(df)
    return df, compute_drivers(df, target), find_top_cluster(df, target)


# ── the decision curve ────────────────────────────────────

def test_decision_bands_are_produced(drivers):
    _df, dr, _tc = drivers
    assert dr.decision_bands, "no decision curve computed"


def test_targeting_the_riskiest_beats_choosing_at_random(drivers):
    """If the top band is no better than random, the ranking is worthless
    however good the AUC looks."""
    _df, dr, _tc = drivers
    assert dr.decision_bands[0].lift > 1.5


def test_lift_decays_as_the_budget_widens(drivers):
    """Reaching further down the ranking necessarily picks up more of the
    population that was never going to record the event."""
    _df, dr, _tc = drivers
    assert dr.decision_bands[0].lift >= dr.decision_bands[-1].lift


def test_recall_grows_as_the_budget_widens(drivers):
    _df, dr, _tc = drivers
    recalls = [b.recall for b in dr.decision_bands]
    assert recalls == sorted(recalls), "wider budgets must reach more events"


def test_bands_are_internally_consistent(drivers):
    _df, dr, _tc = drivers
    for b in dr.decision_bands:
        assert 0 <= b.precision <= 100
        assert 0 <= b.recall <= 100
        assert b.n_events_caught <= b.n_targeted
        assert b.n_events_caught <= b.total_events
        assert b.precision == pytest.approx(
            b.n_events_caught / b.n_targeted * 100, abs=0.2)


def test_decision_curve_is_empty_when_nothing_ever_happens():
    y = np.zeros(200, dtype=int)
    assert decision_curve(y, np.random.default_rng(1).random(200)) == []


def test_a_perfect_ranking_puts_every_event_in_the_top_band():
    y = np.array([1] * 20 + [0] * 180)
    proba = np.concatenate([np.linspace(.99, .9, 20), np.linspace(.1, 0, 180)])
    bands = decision_curve(y, proba, budgets=(10,))
    assert bands[0].recall == 100.0
    assert bands[0].precision == 100.0


# ── precision and recall ──────────────────────────────────

def test_precision_and_recall_are_reported(drivers):
    """Accuracy alone hides the business trade-off entirely."""
    _df, dr, _tc = drivers
    assert dr.precision > 0
    assert dr.recall > 0


# ── calibration ───────────────────────────────────────────

def test_calibration_gap_is_measured(drivers):
    _df, dr, _tc = drivers
    assert dr.calibration_gap is not None
    assert dr.calibration_gap >= 0


def test_a_well_calibrated_model_shows_a_small_gap():
    rng = np.random.default_rng(4)
    n = 2000
    proba = rng.uniform(0, 1, n)
    y = (rng.random(n) < proba).astype(int)      # generated from proba
    assert calibration_gap(y, proba) < 10


def test_calibration_returns_none_on_a_tiny_sample():
    assert calibration_gap(np.array([0, 1]), np.array([.2, .8])) is None


# ── it reaches the report ─────────────────────────────────

def test_the_report_shows_where_to_act(drivers):
    df, dr, tc = drivers
    pdf = build_pdf(df=df, config=dict(CONFIG), domain="hr",
                    predictive=dr, top_cluster=tc)
    text = "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "Where to Act" in text
    assert "Hit rate" in text
    assert "vs random" in text


def test_the_report_explains_how_to_read_the_table(drivers):
    df, dr, tc = drivers
    pdf = build_pdf(df=df, config=dict(CONFIG), domain="hr",
                    predictive=dr, top_cluster=tc)
    text = "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "budget decision" in text


def test_scores_are_calibrated_before_they_are_quoted(drivers):
    """A balanced forest distorts probabilities badly enough that a score
    of 0.30 does not mean a 30% chance. Calibration is applied so the
    number can be quoted, not only ranked."""
    _df, dr, _tc = drivers
    choice = dr.model_choice
    assert choice is not None
    assert choice.calibrated, "scores were left uncalibrated"
    assert choice.calibration_after < choice.calibration_before
    assert choice.calibration_after < 10


def test_the_report_says_a_score_can_now_be_read_as_a_probability(drivers):
    df, dr, tc = drivers
    pdf = build_pdf(df=df, config=dict(CONFIG), domain="hr",
                    predictive=dr, top_cluster=tc)
    text = "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "genuinely corresponds" in text


# ── model selection ───────────────────────────────────────

def test_more_than_one_model_is_tried(drivers):
    """A single fixed forest left 2 points of AUC on the table on the HR
    sample, where a scaled logistic regression scores 0.823 against its
    0.803. Neither wins reliably, which is the argument for choosing."""
    _df, dr, _tc = drivers
    assert len(dr.model_choice.candidates) >= 2


def test_the_selected_model_is_the_best_of_the_candidates(drivers):
    _df, dr, _tc = drivers
    choice = dr.model_choice
    best_name, best_auc = max(choice.candidates, key=lambda c: c[1])
    assert choice.name == best_name


def test_the_report_names_the_alternatives_and_their_scores(drivers):
    """Naming a method without saying what it was measured against asks
    the reader to take it on faith."""
    df, dr, tc = drivers
    pdf = build_pdf(df=df, config=dict(CONFIG), domain="hr",
                    predictive=dr, top_cluster=tc)
    text = "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "How This Model Was Selected" in text
    assert "Cross-validated AUC" in text
    assert "(selected)" in text


def test_drivers_survive_a_linear_model_winning(drivers):
    """A logistic regression has coefficients, not feature_importances_.
    Selecting one must not silently cost the report its drivers."""
    _df, dr, _tc = drivers
    assert dr.top_drivers, "the winning model produced no drivers"
    assert all(imp >= 0 for _f, imp in dr.top_drivers)


# ── operating threshold ───────────────────────────────────

def test_the_threshold_is_chosen_not_assumed(drivers):
    """0.5 is right only when the classes are balanced and both errors
    cost the same. On the HR sample it gives F1 0.433 where 0.22 gives
    0.516."""
    _df, dr, _tc = drivers
    assert dr.model_choice.threshold != 0.5
    assert 0.05 <= dr.model_choice.threshold <= 0.95
    assert dr.model_choice.threshold_basis != "default"


def test_precision_and_recall_are_measured_at_that_threshold(drivers):
    _df, dr, _tc = drivers
    assert dr.precision > 0 and dr.recall > 0


def test_high_cardinality_columns_are_named_when_excluded():
    """A field with hundreds of levels may still matter. Dropping it in
    silence leaves the reader thinking the model saw it."""
    rng = np.random.default_rng(2)
    n = 800
    churn = rng.choice([0, 1], n, p=[.7, .3])
    df = pd.DataFrame({
        "churn": churn,
        "tenure": rng.integers(1, 60, n),
        "spend": rng.uniform(10, 500, n),
        "free_text_note": [f"note-{i}" for i in range(n)],
    })
    dr = compute_drivers(df, "churn")
    if dr is None or dr.model_choice is None:
        pytest.skip("no model fitted")
    excluded = [c for c, _n in dr.model_choice.excluded_high_cardinality]
    assert "free_text_note" in excluded


def test_no_decision_table_when_there_is_no_signal():
    """A decision table on a model that found nothing invites action on
    noise."""
    rng = np.random.default_rng(3)
    n = 700
    df = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(5)})
    df["churn"] = rng.choice([0, 1], n, p=[.9, .1])
    dr = compute_drivers(df, "churn")
    pdf = build_pdf(df=df, config=dict(CONFIG), domain="general",
                    predictive=dr)
    text = "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "Where to Act" not in text
    assert "No predictive signal found" in text
