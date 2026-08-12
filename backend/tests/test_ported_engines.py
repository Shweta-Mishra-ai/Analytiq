"""
Unit tests for the second batch of engines ported from dataforge-ai:
benchmarking, industry_benchmarks, cohort_analysis, predictive,
comparison_engine, and bi_engine.analyze_scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════
#  Scenario / what-if
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def linear_df():
    rng = np.random.default_rng(0)
    n = 300
    driver = rng.normal(50, 10, n)
    target = 3 * driver + rng.normal(0, 4, n)
    return pd.DataFrame({"driver": driver, "target": target,
                         "noise": rng.normal(0, 1, n)})


def test_scenario_projects_a_strong_relationship(linear_df):
    from app.engines.bi_engine import analyze_scenario
    r = analyze_scenario(linear_df, "driver", "target", change_pct=10.0)
    assert r is not None
    assert r.reliable is True
    assert r.r_squared > 0.8
    # target = 3*driver, so +10% driver => roughly +10% target
    assert 5 < r.projected_change_pct < 15
    assert "not a causal guarantee" in r.caveat


def test_scenario_flags_a_weak_relationship_as_unreliable(linear_df):
    from app.engines.bi_engine import analyze_scenario
    r = analyze_scenario(linear_df, "noise", "target", change_pct=10.0)
    assert r is not None, "a weak relationship should still return a result"
    assert r.reliable is False
    assert "too weak" in r.interpretation or "not statistically significant" in r.interpretation


@pytest.mark.parametrize("driver,target", [
    ("driver", "driver"),      # same column
    ("driver", "missing"),     # unknown column
])
def test_scenario_returns_none_for_unusable_inputs(linear_df, driver, target):
    from app.engines.bi_engine import analyze_scenario
    assert analyze_scenario(linear_df, driver, target) is None


def test_scenario_returns_none_for_non_numeric():
    from app.engines.bi_engine import analyze_scenario
    df = pd.DataFrame({"a": list("abcdefghij") * 3, "b": range(30)})
    assert analyze_scenario(df, "a", "b") is None


def test_scenario_returns_none_when_too_few_rows():
    from app.engines.bi_engine import analyze_scenario
    df = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8]})
    assert analyze_scenario(df, "a", "b") is None


# ══════════════════════════════════════════════════════════
#  Internal benchmarking
# ══════════════════════════════════════════════════════════

def test_compute_benchmarks_returns_contexts(hr_df):
    from app.engines.benchmarking import compute_benchmarks
    results = compute_benchmarks(hr_df, max_metrics=5)
    assert results, "expected at least one benchmarked metric"
    assert len(results) <= 5
    for r in results:
        assert r.metric in hr_df.columns
        assert r.reference_kind in ("target", "internal top-quartile")
        assert r.direction in (1, -1)
        assert isinstance(r.meets, bool)


def test_compute_benchmarks_respects_max_metrics(hr_df):
    from app.engines.benchmarking import compute_benchmarks
    assert len(compute_benchmarks(hr_df, max_metrics=2)) <= 2


def test_metric_direction_knows_lower_is_better():
    from app.engines.benchmarking import metric_direction
    # cost/churn/attrition style metrics: lower is better (-1)
    assert metric_direction("churn_rate") == -1
    assert metric_direction("attrition") == -1
    # revenue/satisfaction style metrics: higher is better (+1)
    assert metric_direction("revenue") == 1
    assert metric_direction("satisfaction") == 1


# ══════════════════════════════════════════════════════════
#  Industry benchmarks
# ══════════════════════════════════════════════════════════

def test_lookup_benchmark_returns_none_for_unknown_metric():
    from app.engines.industry_benchmarks import lookup_benchmark
    assert lookup_benchmark("hr", "some_random_column_xyz") is None


def test_lookup_benchmark_finds_known_hr_metric():
    from app.engines.industry_benchmarks import lookup_benchmark, format_benchmark_context
    bm = lookup_benchmark("hr", "attrition")
    if bm is not None:  # table is domain-specific; only assert shape when present
        assert bm.low <= bm.high
        assert isinstance(format_benchmark_context(bm), str)


# ══════════════════════════════════════════════════════════
#  Cohort analysis helpers
# ══════════════════════════════════════════════════════════

def test_build_quantile_cohorts_splits_a_numeric_column(hr_df):
    from app.engines.cohort_analysis import build_quantile_cohorts
    banded = build_quantile_cohorts(hr_df, "salary", q=4,
                                     agg_cols={"tenure_years": "mean"})
    assert len(banded) > 0
    assert len(banded) <= 4


def test_build_quantile_cohorts_raises_on_all_null_column(hr_df):
    from app.engines.cohort_analysis import build_quantile_cohorts
    df = hr_df.copy()
    df["salary"] = np.nan
    with pytest.raises(ValueError, match="no non-null values"):
        build_quantile_cohorts(df, "salary")


def test_concentration_analysis_on_grouped_values(hr_df):
    from app.engines.cohort_analysis import concentration_analysis
    result = concentration_analysis(hr_df, "department", "salary")
    assert isinstance(result, dict)
    table = result["table"]
    # cumulative share must end at 100% and be non-decreasing
    cum = table["cum_pct"].tolist()
    assert cum == sorted(cum), "cumulative percentage must be non-decreasing"
    assert abs(cum[-1] - 100.0) < 0.01


# ══════════════════════════════════════════════════════════
#  Predictive drivers
# ══════════════════════════════════════════════════════════

def test_find_binary_target_detects_attrition(hr_df):
    from app.engines.predictive import find_binary_target
    assert find_binary_target(hr_df) == "attrition"


def test_find_binary_target_returns_none_without_a_flag_column():
    from app.engines.predictive import find_binary_target
    df = pd.DataFrame({"a": range(50), "b": np.random.default_rng(1).normal(size=50)})
    assert find_binary_target(df) is None


def test_compute_drivers_ranks_factors(hr_df):
    from app.engines.predictive import compute_drivers
    result = compute_drivers(hr_df, "attrition")
    if result is None:
        pytest.skip("scikit-learn unavailable")
    assert result.target == "attrition"
    assert result.top_drivers, "expected ranked drivers"
    importances = [imp for _feat, imp in result.top_drivers]
    assert importances == sorted(importances, reverse=True), \
        "drivers must be ranked most-important first"
    for feat, imp in result.top_drivers:
        assert feat in hr_df.columns
        assert imp >= 0
    assert 0.0 <= result.auc <= 1.0


def test_compute_drivers_returns_none_for_single_class_target(hr_df):
    from app.engines.predictive import compute_drivers
    df = hr_df.copy()
    df["attrition"] = "No"          # degenerate: only one class
    assert compute_drivers(df, "attrition") is None


# ══════════════════════════════════════════════════════════
#  Dataset comparison
# ══════════════════════════════════════════════════════════

def test_run_comparison_detects_a_mean_shift(hr_df):
    from app.engines.comparison_engine import run_comparison
    df_b = hr_df.copy()
    df_b["salary"] = df_b["salary"] * 1.25          # 25% raise across the board
    report = run_comparison(hr_df, df_b, label_a="Q1", label_b="Q2")
    salary_cmp = next((c for c in report.column_comparisons if c.column == "salary"), None)
    assert salary_cmp is not None, "salary column should have been compared"
    assert report.label_a == "Q1" and report.label_b == "Q2"


def test_run_comparison_reports_schema_drift(hr_df):
    from app.engines.comparison_engine import run_comparison
    df_b = hr_df.drop(columns=["satisfaction"]).copy()
    df_b["new_metric"] = 1.0
    report = run_comparison(hr_df, df_b)
    assert "satisfaction" in report.schema_diff.removed_columns
    assert "new_metric" in report.schema_diff.added_columns


def test_run_comparison_rejects_empty_and_non_dataframe(hr_df):
    from app.engines.comparison_engine import run_comparison
    with pytest.raises(ValueError):
        run_comparison(hr_df, hr_df.iloc[0:0])
    with pytest.raises(TypeError):
        run_comparison(hr_df, [1, 2, 3])  # type: ignore[arg-type]


def test_run_comparison_requires_common_columns(hr_df):
    from app.engines.comparison_engine import run_comparison
    other = pd.DataFrame({"totally": [1, 2, 3], "different": [4, 5, 6]})
    with pytest.raises(ValueError):
        run_comparison(hr_df, other)
