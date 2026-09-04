"""Whether a bigger number is good news.

The generic opportunity detector printed the same sentence for every
numeric column: "bringing the bottom up to the median would be a N%
improvement". On a sales extract that came out as advice to raise the
discount rate by 107% and call it an opportunity. A metric's direction
cannot be read off its values — 40% is a good margin and a terrible
defect rate — so it has to come from the name, and where the name does
not say, neither should the report.
"""
import pandas as pd
import numpy as np
import pytest

from app.engines.domains.base import higher_is_better


@pytest.mark.parametrize("col", [
    "revenue", "Profit", "gross_margin", "MonthlyIncome", "csat",
    "satisfaction_score", "units", "conversion_rate", "uptime",
])
def test_columns_where_more_is_better(col):
    assert higher_is_better(col) is True


@pytest.mark.parametrize("col", [
    "discount_pct", "returned", "churn", "error_rate", "DaysLate",
    "cost", "complaints", "downtime", "attrition", "refunds",
])
def test_columns_where_less_is_better(col):
    assert higher_is_better(col) is False


@pytest.mark.parametrize("col", [
    "unit_price", "notes", "value", "amount_x", "reading",
    "turnover",     # revenue in finance, attrition in HR — says nothing
])
def test_columns_the_name_does_not_settle(col):
    assert higher_is_better(col) is None


def test_a_name_pulling_both_ways_is_treated_as_unknown():
    assert higher_is_better("revenue_loss") is None


# ══════════════════════════════════════════════════════════
#  WHAT THE REPORT SAYS ABOUT EACH KIND
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def spread_frame():
    """One column of each kind, each with a wide enough spread to trip
    the detector."""
    rng = np.random.default_rng(0)
    n = 3000
    heavy = rng.lognormal(3.2, 0.9, n).round(2)
    return pd.DataFrame({
        "revenue": heavy,
        "discount_pct": rng.lognormal(1.9, 0.9, n).round(2),
        "reading": heavy.copy(),
        "region": rng.choice(["North", "South"], n),
    })


def _sections(df):
    from app.engines.domains.general import _insights_general
    from app.engines.stats_engine import analyze
    result = analyze(df)
    stats = getattr(result, "column_stats", None)
    if stats is None:                       # dict form
        stats = result.get("column_stats", {})
    if stats and not isinstance(next(iter(stats.values())), dict):
        stats = {k: vars(v) for k, v in stats.items()}
    return _insights_general(df, stats, [])


def test_a_cost_metric_is_never_offered_as_an_uplift(spread_frame):
    out = _sections(spread_frame)
    joined = " ".join(out.get("opportunities") or [])
    assert "Discount" not in joined, (
        "raising the discount rate was offered as an opportunity")


def test_a_cost_metric_is_reported_as_something_to_reduce(spread_frame):
    risks = " ".join(_sections(spread_frame).get("risks") or [])
    assert "Discount" in risks
    assert "cut it by" in risks


def test_a_revenue_metric_is_still_offered_as_an_uplift(spread_frame):
    opps = " ".join(_sections(spread_frame).get("opportunities") or [])
    assert "Revenue" in opps and "gain" in opps


def test_an_unlabelled_metric_states_the_spread_without_taking_a_side(
        spread_frame):
    out = _sections(spread_frame)
    findings = " ".join(out.get("findings") or [])
    assert "Reading" in findings
    assert "depends on what this column measures" in findings
    for section in ("opportunities", "risks"):
        assert "Reading" not in " ".join(out.get(section) or [])


def test_the_wording_avoids_decile_and_quartile(spread_frame):
    """A business reader does not know which tenth a decile is — and the
    text called the 10th percentile "the bottom quartile", which is the
    wrong quarter of the wrong scale."""
    out = _sections(spread_frame)
    everything = " ".join(
        " ".join(out.get(k) or []) for k in
        ("opportunities", "risks", "findings")).lower()
    assert "decile" not in everything
    assert "quartile" not in everything
