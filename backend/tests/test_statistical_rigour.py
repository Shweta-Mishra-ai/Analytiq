"""
Statistical assumption checks — the things a senior reviewer verifies
before accepting a finding.

Two failure modes are guarded here:

1. Chi-square applied to a table too sparse to support it. The p-value is
   then uninterpretable, but was previously reported as a confident
   "driver".
2. Scanning every column for drivers without multiple-comparison
   correction. Testing ~20 independent columns at alpha=0.05 yields about
   one significant result from chance alone — and it was presented as the
   headline root cause.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.stat_guards import (
    apply_fdr,
    bh_adjust,
    chi2_association,
    chi2_validity,
    cramers_v,
)


# ══════════════════════════════════════════════════════════
#  Chi-square assumptions
# ══════════════════════════════════════════════════════════

def test_sparse_table_is_rejected():
    """Expected counts below 5 make the normal approximation invalid."""
    ct = pd.crosstab(pd.Series(list("abcdefgh")), pd.Series(["x", "y"] * 4))
    assert chi2_association(ct) is None


def test_table_with_an_expected_count_below_one_is_rejected():
    exp = np.array([[0.4, 20.0], [30.0, 25.0]])
    valid, reason = chi2_validity(exp)
    assert valid is False
    assert "below 1" in reason


def test_table_with_over_20_percent_of_cells_below_five_is_rejected():
    exp = np.array([[2.0, 2.0], [40.0, 40.0]])   # 50% of cells < 5
    valid, reason = chi2_validity(exp)
    assert valid is False
    assert "below 5" in reason


def test_healthy_table_is_accepted_with_effect_size():
    rng = np.random.default_rng(0)
    n = 600
    g = rng.choice(["A", "B"], n)
    outcome = np.where(g == "A",
                       rng.choice(["yes", "no"], n, p=[.75, .25]),
                       rng.choice(["yes", "no"], n, p=[.25, .75]))
    res = chi2_association(pd.crosstab(pd.Series(g), pd.Series(outcome)))
    assert res is not None
    assert res["p"] < 0.01
    assert res["cramers_v"] > 0.3
    assert res["effect_label"] in ("moderate", "strong")
    assert res["n"] == n


def test_cramers_v_bounds_and_degenerate_inputs():
    assert cramers_v(0.0, 100, 2, 2) == 0.0
    assert 0.0 <= cramers_v(50.0, 500, 2, 2) <= 1.0
    assert cramers_v(10.0, 0, 2, 2) == 0.0     # no rows
    assert cramers_v(10.0, 100, 1, 1) == 0.0   # single-cell table


# ══════════════════════════════════════════════════════════
#  Multiple comparisons
# ══════════════════════════════════════════════════════════

def test_bh_adjust_is_monotone_and_never_below_raw_p():
    raw = [0.001, 0.01, 0.03, 0.2, 0.7]
    q = bh_adjust(raw)
    assert all(qi >= pi - 1e-12 for qi, pi in zip(q, raw)), \
        "an adjusted p-value came out below its raw p-value"
    assert q == sorted(q), "q-values are not monotone in p"


def test_apply_fdr_annotates_without_dropping():
    out = apply_fdr([{"p": 0.01}, {"p": 0.6}])
    assert len(out) == 2
    assert all("q" in f for f in out)


def test_apply_fdr_on_empty_family():
    assert apply_fdr([]) == []


# ══════════════════════════════════════════════════════════
#  End to end: noise must not produce findings
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def pure_noise_df():
    """20 independent random columns and a random target. There is no real
    relationship anywhere; a correct analysis reports no drivers."""
    rng = np.random.default_rng(4)
    n = 400
    data = {f"metric_{i}": rng.normal(0, 1, n) for i in range(14)}
    for i in range(6):
        data[f"segment_{i}"] = rng.choice(list("abcd"), n)
    data["outcome"] = rng.normal(50, 10, n)
    return pd.DataFrame(data)


def test_root_cause_reports_no_driver_on_pure_noise(pure_noise_df):
    """Without FDR correction this scan reported a confident 'top driver'
    for data containing no relationship at all."""
    from app.engines.bi_engine import analyze_root_cause
    result = analyze_root_cause(pure_noise_df, "outcome")
    assert result.drivers == [], (
        f"found {len(result.drivers)} driver(s) in pure noise: "
        f"{[d['factor'] for d in result.drivers]}")
    assert "No significant driver" in result.top_driver


def test_attrition_drivers_survive_correction_on_real_signal():
    """The correction must not be so aggressive that it suppresses a
    genuine effect."""
    from app.engines.domains.hr import _run_attrition
    rng = np.random.default_rng(5)
    n = 800
    dept = rng.choice(["Sales", "Eng", "HR"], n)
    # Sales attrites far more — a real, strong effect
    left = np.where(dept == "Sales",
                    rng.choice(["Yes", "No"], n, p=[.6, .4]),
                    rng.choice(["Yes", "No"], n, p=[.1, .9]))
    df = pd.DataFrame({
        "employee_id": range(n), "department": dept,
        "satisfaction": rng.uniform(1, 5, n).round(1),
        "attrition": left,
    })
    result = _run_attrition(df)
    assert result is not None
    factors = [d["factor"] for d in result.top_drivers]
    assert "department" in factors, \
        "a strong, real driver was suppressed by the correction"


def test_reported_drivers_carry_effect_size_and_n():
    """A p-value alone is not a finding; strength and sample size must
    travel with it so a reviewer can judge materiality."""
    from app.engines.bi_engine import analyze_root_cause
    rng = np.random.default_rng(6)
    n = 600
    grade = rng.choice(["low", "high"], n)
    score = np.where(grade == "low", rng.normal(30, 8, n), rng.normal(70, 8, n))
    df = pd.DataFrame({"grade": grade, "score": score,
                       "noise": rng.normal(0, 1, n)})
    result = analyze_root_cause(df, "score")
    cat_drivers = [d for d in result.drivers if d.get("dtype") == "categorical"]
    for d in cat_drivers:
        assert "cramers_v" in d and "n" in d
        assert "Cramér's V" in d["detail"] or "Cram" in d["detail"]


def test_root_cause_states_the_direction_the_numbers_show():
    # "56.3% lower absences (17.30 vs 7.56)" — the numbers said higher.
    from app.engines.bi_engine import analyze_root_cause
    rng = np.random.default_rng(0)
    n = 400
    absences = rng.integers(0, 20, n)
    score = 100 - 3 * absences + rng.normal(0, 3, n)
    df = pd.DataFrame({"emp_id": range(n), "score": score, "absences": absences})
    drivers = {d["factor"]: d for d in analyze_root_cause(df, "score").drivers}
    d = drivers["absences"]
    assert d["low_mean"] > d["high_mean"]
    assert "higher 'absences' than the rest" in d["detail"]
    assert "emp_id" not in drivers


# ── Business Intel page: measures, directions, real shares ─

def test_bi_never_analyses_an_identifier(hr_df):
    from app.engines.bi_engine import run_bi
    r = run_bi(hr_df)
    used = ([b.column for b in r.benchmarks] + [x.target_col for x in r.root_causes]
            + [c.metric_col for c in r.cohorts] + [p.value_col for p in r.pareto])
    assert "employee_id" not in used
    assert not any("employee_id" in i for i in r.key_insights)


def test_bi_reads_lower_is_better_metrics_the_right_way_round():
    from app.engines.bi_engine import analyze_root_cause, analyze_segment_health
    rng = np.random.default_rng(1)
    n = 600
    reg = rng.choice(list("ABCDEFG"), n)
    df = pd.DataFrame({"region": reg,
                       "complaints": rng.poisson(np.where(reg == "G", 6, 1)),
                       "cost": rng.gamma(2, 50, n)})
    rc = analyze_root_cause(df, "complaints")
    assert "highest 'complaints'" in rc.group_label        # the worst are the high ones
    health = {s.segment_name: s for s in analyze_segment_health(df, "region", ["complaints"])}
    assert min(health.values(), key=lambda s: s.health_score).segment_name == "G"


def test_bi_does_not_judge_metrics_with_no_known_direction():
    from app.engines.bi_engine import analyze_root_cause, analyze_segment_health
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"team": rng.choice(list("abc"), 300), "salary": rng.normal(5e4, 9e3, 300),
                       "age": rng.integers(20, 60, 300)})
    assert "performers" not in analyze_root_cause(df, "salary").group_label
    assert analyze_segment_health(df, "team", ["salary", "age"]) == []


def test_pareto_only_for_additive_non_negative_totals_with_enough_groups():
    from app.engines.bi_engine import run_bi
    rng = np.random.default_rng(3)
    n = 500
    reg = rng.choice(list("ABCDEFG"), n)
    df = pd.DataFrame({"region": reg, "revenue": rng.gamma(2, 100, n),
                       "satisfaction_score": rng.uniform(1, 5, n),
                       "profit": rng.normal(0, 50, n)})
    cols = {p.value_col: p for p in run_bi(df).pareto}
    assert "satisfaction_score" not in cols and "profit" not in cols
    p = cols["revenue"]
    assert f"top {p.top_20_pct_groups} of 7" in p.interpretation
    four = df[df["region"].isin(list("ABCD"))]
    assert run_bi(four).pareto == []


def test_cohort_absence_of_significance_is_not_called_absence_of_effect():
    from app.engines.bi_engine import analyze_cohort
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"g": rng.choice(list("ab"), 60), "v": rng.normal(0, 1, 60)})
    c = analyze_cohort(df, "g", "v")
    if not c.is_significant:
        assert "not the same as showing there is none" in " ".join(c.recommendations)
