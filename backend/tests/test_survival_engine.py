"""Unit tests for engines/survival_engine.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.survival_engine import run_survival_analysis


def test_survival_basic_kaplan_meier(hr_df):
    report = run_survival_analysis(hr_df, duration_col="tenure_years", event_col="attrition")
    assert report.overall_curve.n_total == len(hr_df)
    assert report.overall_curve.n_events > 0
    assert 0 <= report.overall_curve.n_events <= report.overall_curve.n_total
    for p in report.overall_curve.points:
        assert 0.0 <= p.survival_prob <= 1.0
        assert p.ci_lower <= p.survival_prob <= p.ci_upper
    # survival probability must be non-increasing over time
    probs = [p.survival_prob for p in report.overall_curve.points]
    assert all(probs[i] >= probs[i + 1] - 1e-9 for i in range(len(probs) - 1))


def test_survival_accepts_yes_no_and_numeric_events_equivalently():
    rng = np.random.default_rng(5)
    n = 200
    duration = rng.uniform(0, 10, n)
    event_num = rng.integers(0, 2, n)
    df_yesno = pd.DataFrame({
        "duration": duration,
        "event": np.where(event_num == 1, "Yes", "No"),
    })
    df_numeric = pd.DataFrame({"duration": duration, "event": event_num})

    r1 = run_survival_analysis(df_yesno, "duration", "event")
    r2 = run_survival_analysis(df_numeric, "duration", "event")
    assert r1.overall_curve.n_events == r2.overall_curve.n_events
    assert r1.overall_curve.median_survival == r2.overall_curve.median_survival


def test_survival_negative_durations_are_excluded_with_warning():
    df = pd.DataFrame({
        "duration": [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "event": [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    })
    report = run_survival_analysis(df, "duration", "event")
    assert report.overall_curve.n_total == 10
    assert any("negative" in w.lower() for w in report.warnings)


def test_survival_group_comparison_runs_logrank(hr_df):
    report = run_survival_analysis(
        hr_df, duration_col="tenure_years", event_col="attrition", group_col="department")
    assert len(report.group_curves) >= 2
    for cmp in report.pairwise_comparisons:
        assert 0.0 <= cmp.p_value <= 1.0
        assert isinstance(cmp.is_significant, bool)


def test_survival_requires_at_least_one_event():
    df = pd.DataFrame({
        "duration": list(range(1, 15)),
        "event": [0] * 14,
    })
    with pytest.raises(ValueError, match="No events"):
        run_survival_analysis(df, "duration", "event")


def test_survival_requires_minimum_rows():
    df = pd.DataFrame({"duration": [1, 2, 3], "event": [1, 0, 1]})
    with pytest.raises(ValueError, match="at least 10"):
        run_survival_analysis(df, "duration", "event")


def test_survival_missing_column_raises():
    df = pd.DataFrame({"a": range(20), "b": [1, 0] * 10})
    with pytest.raises(ValueError, match="not found"):
        run_survival_analysis(df, "nonexistent", "b")
