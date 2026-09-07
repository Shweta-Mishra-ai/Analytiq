"""
A trained model answers any question you ask it.

Type a salary of 5,000,000 into a model fitted on 30k–120k and it
returns a performance score of 1,937 — with a confidence interval, and
with nothing on the screen saying the number is the fitted line extended
a long way past anything the data contains. The model is not wrong; it
was asked a question it has no standing to answer, and it answered.

This is the same defect the scenario engine had, on the surface where a
user is most likely to walk into it: typing a big number is the obvious
thing to try. The prediction is still returned — the caller shows it
with the caveat rather than refusing — and the tolerance is the same one
the scenario engine uses, so two screens do not disagree about how far
is too far.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.ml.runner import run_ml_pipeline
from app.engines.ml.whatif import OUTSIDE_RANGE_TOLERANCE, predict_what_if


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(4)
    n = 900
    tenure = rng.uniform(0.5, 12, n)
    salary = rng.normal(72_000, 18_000, n).clip(30_000, 130_000)
    perf = salary / 1000 * 0.4 + tenure * 3 + rng.normal(0, 6, n)
    df = pd.DataFrame({
        "tenure_years": tenure.round(1),
        "salary":       salary.round(0),
        "performance":  perf.round(1),
    })
    return run_ml_pipeline(df, "performance")


def test_the_model_records_what_it_actually_saw(trained):
    assert "salary" in trained.feature_ranges
    lo = trained.feature_ranges["salary"]["min"]
    hi = trained.feature_ranges["salary"]["max"]
    assert 29_000 <= lo <= 31_000
    assert 129_000 <= hi <= 131_000


def test_an_ordinary_question_is_answered_without_a_caveat(trained):
    r = predict_what_if(trained, {"tenure_years": 5.0, "salary": 80_000})
    assert r["within_training_range"] is True
    assert r["out_of_range"] == []
    assert "range_note" not in r


def test_a_question_the_model_cannot_answer_says_so(trained):
    """The one that shipped: 5,000,000 in, 1,937 out, no caveat."""
    r = predict_what_if(trained, {"tenure_years": 5.0, "salary": 5_000_000})

    assert r["within_training_range"] is False
    assert r["out_of_range"], "the offending input must be named"

    note = r["range_note"]
    assert "Salary" in note
    assert "outside" in note
    # And it must say what the data does cover, not merely that
    # something is wrong.
    assert "30,000" in note and "130,000" in note


def test_the_prediction_is_still_returned(trained):
    """Refusing to answer is its own failure — the caller wants the
    number and the caveat, so it can show both."""
    r = predict_what_if(trained, {"tenure_years": 5.0, "salary": 5_000_000})
    assert isinstance(r.get("prediction"), float)


def test_a_step_past_the_edge_is_still_a_forecast(trained):
    """The guard marks invention, not every value above the maximum. A
    little beyond the edge is what a forecast is for."""
    hi = trained.feature_ranges["salary"]["max"]
    lo = trained.feature_ranges["salary"]["min"]
    just_past = hi + (hi - lo) * OUTSIDE_RANGE_TOLERANCE * 0.5

    r = predict_what_if(trained, {"tenure_years": 5.0, "salary": just_past})
    assert r["within_training_range"] is True


def test_a_report_without_ranges_does_not_crash_or_lie(trained):
    """Older reports carry no ranges. With no reference, the honest
    answer is to claim nothing rather than to claim it is fine."""
    import copy
    stale = copy.copy(trained)
    stale.feature_ranges = {}
    r = predict_what_if(stale, {"tenure_years": 5.0, "salary": 5_000_000})
    assert r["out_of_range"] == []
    assert "range_note" not in r
