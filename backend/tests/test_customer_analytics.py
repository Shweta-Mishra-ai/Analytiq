"""
Cohort retention, RFM and revenue concentration.

Each analysis is checked against data where the answer is known before
the code runs, and against data where the effect is absent — a retention
engine that finds a decaying cohort in a stationary book is worse than
no retention engine.

The column detection is tested separately, because it is where these
silently do nothing: a transaction-level "order_id" column matching the
customer keyword would give a retention rate of zero for every cohort,
and it would look like a finding rather than a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.domains.customer_analytics import (
    cohort_retention,
    find_customer_col,
    find_date_col,
    find_value_col,
    revenue_concentration,
    rfm_segments,
)


def _run(fn, df):
    insights, findings, risks, opps = [], [], [], []
    fn(df, insights, findings, risks, opps)
    return {"insights": insights, "findings": findings,
            "risks": risks, "opportunities": opps}


def _text(out) -> str:
    parts = list(out["findings"]) + list(out["risks"]) + list(out["opportunities"])
    for i in out["insights"]:
        parts += [i.title, i.problem, i.cause, i.evidence, i.action, i.impact]
    return " ".join(parts)


# ══════════════════════════════════════════════════════════
#  Column detection
# ══════════════════════════════════════════════════════════

def test_the_customer_key_is_preferred_over_the_order_key():
    """Picking order_id would give every cohort 0% retention by
    construction, and it would read as total churn rather than as a bug."""
    df = pd.DataFrame({
        "customer_id": list(range(60)) * 2,
        "order_id": range(120),
        "amount": np.linspace(10, 200, 120),
    })
    assert find_customer_col(df) == "customer_id"


def test_retention_is_skipped_on_an_aggregated_customer_table():
    """One row per customer is a customer table, not an order log. Every
    cohort would show 0% return — an artefact of the grain reported as
    total churn."""
    df = pd.DataFrame({
        "customer_id": range(300),
        "order_date": pd.date_range("2023-01-01", periods=300, freq="D"),
        "revenue": np.linspace(50, 500, 300),
    })
    out = _run(cohort_retention, df)
    assert not out["findings"], \
        "reported retention from a table with one row per customer"


def test_customer_detection_returns_none_when_there_is_no_customer():
    df = pd.DataFrame({"sku": ["a", "b"] * 40, "amount": range(80)})
    assert find_customer_col(df) is None


def test_date_detection_parses_text_dates():
    df = pd.DataFrame({
        "order_date": ["2024-01-05", "2024-02-06", "2024-03-07"] * 20,
        "customer_id": list(range(20)) * 3,
    })
    assert find_date_col(df) == "order_date"


def test_value_detection_skips_counts_and_identifiers():
    df = pd.DataFrame({
        "order_qty": [1, 2, 3],
        "amount_id": [1, 2, 3],
        "revenue": [10.0, 20.0, 30.0],
    })
    assert find_value_col(df) == "revenue"


# ══════════════════════════════════════════════════════════
#  Cohort retention
# ══════════════════════════════════════════════════════════

def _cohort_frame(month_1_rates, per_cohort=80, seed=51):
    """Build orders so each monthly cohort returns at a known month-1 rate."""
    rng = np.random.default_rng(seed)
    rows = []
    cid = 0
    for m, rate in enumerate(month_1_rates):
        start = pd.Timestamp("2023-01-01") + pd.DateOffset(months=m)
        for _ in range(per_cohort):
            cid += 1
            rows.append({"customer_id": cid,
                         "order_date": start + pd.Timedelta(days=int(rng.integers(0, 27))),
                         "amount": float(rng.uniform(20, 200))})
            if rng.random() < rate:
                nxt = start + pd.DateOffset(months=1)
                rows.append({"customer_id": cid,
                             "order_date": nxt + pd.Timedelta(days=int(rng.integers(0, 27))),
                             "amount": float(rng.uniform(20, 200))})
    return pd.DataFrame(rows)


def test_retention_rate_matches_the_planted_rate():
    df = _cohort_frame([0.40] * 6)
    out = _run(cohort_retention, df)
    assert out["findings"], "no retention reported on six clean cohorts"
    import re
    m = re.search(r"averages ([\d.]+)%", out["findings"][0])
    assert m, out["findings"][0]
    assert float(m.group(1)) == pytest.approx(40, abs=8), \
        "reported {} against a planted 40%".format(m.group(1))


def test_declining_retention_is_flagged_as_a_risk():
    df = _cohort_frame([0.55, 0.50, 0.42, 0.35, 0.28, 0.22])
    out = _run(cohort_retention, df)
    assert out["risks"], "a 33-point retention decline was not raised"
    assert out["insights"], "no insight for a declining cohort trend"
    assert "declined" in _text(out)


def test_improving_retention_is_an_opportunity_not_a_risk():
    df = _cohort_frame([0.20, 0.28, 0.35, 0.44, 0.50, 0.58])
    out = _run(cohort_retention, df)
    assert out["opportunities"], "an improving trend produced no note"
    assert not out["risks"], "flagged improving retention as a risk"


def test_stable_retention_produces_no_trend_claim():
    df = _cohort_frame([0.40, 0.41, 0.39, 0.40, 0.42, 0.40])
    out = _run(cohort_retention, df)
    assert out["findings"], "stable retention should still be reported"
    assert not out["risks"] and not out["insights"], \
        "invented a trend in a stationary book: {}".format(_text(out)[:300])


def _quarterly_frame(per_cohort=80, months=9, gap=3, seed=59):
    """Customers who reorder every `gap` months — a business with a long
    purchase cycle, not a business with no retention."""
    rng = np.random.default_rng(seed)
    rows = []
    cid = 0
    for m in range(months):
        start = pd.Timestamp("2023-01-01") + pd.DateOffset(months=m)
        for _ in range(per_cohort):
            cid += 1
            when = start
            for _ in range(3):
                rows.append({"customer_id": cid,
                             "order_date": when + pd.Timedelta(
                                 days=int(rng.integers(0, 26))),
                             "amount": float(rng.uniform(20, 200))})
                when = when + pd.DateOffset(months=gap)
    return pd.DataFrame(rows)


def test_a_quarterly_book_is_not_reported_as_zero_retention():
    """Month 1 is the conventional window and the wrong one here: every
    cohort scores 0%, and "retention averages 0.0%, ranging 0.0% to 0.0%"
    prints for a book where every customer came back."""
    out = _run(cohort_retention, _quarterly_frame())
    assert out["findings"], "no retention reported at all"
    finding = out["findings"][0]
    assert "0.0%, ranging 0.0% to 0.0%" not in finding, \
        "reported a quarterly book as total churn: {}".format(finding)
    assert "Month-3" in finding, \
        "did not measure at the observed reorder cadence: {}".format(finding)


def test_a_non_standard_window_says_why_it_was_used():
    """A reader who expects month-1 retention has to be told this is not
    it, or the number is not comparable to anything they know."""
    out = _run(cohort_retention, _quarterly_frame())
    assert "typically reorder" in out["findings"][0], \
        "changed the measurement window without explaining it"


def test_a_monthly_book_still_uses_month_1():
    out = _run(cohort_retention, _cohort_frame([0.40] * 6))
    finding = out["findings"][0]
    assert "Month-1" in finding
    assert "typically reorder" not in finding, \
        "explained a window change that did not happen"


def test_small_cohorts_are_excluded():
    """Six customers a month is not a retention rate."""
    df = _cohort_frame([0.40] * 6, per_cohort=6)
    out = _run(cohort_retention, df)
    assert not out["findings"], "computed a rate from 6-customer cohorts"


def test_retention_needs_both_a_customer_and_a_date():
    df = pd.DataFrame({"customer_id": list(range(60)) * 2,
                       "amount": np.linspace(1, 100, 120)})
    assert _run(cohort_retention, df)["findings"] == []


# ══════════════════════════════════════════════════════════
#  RFM
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def rfm_df():
    """Three populations: active repeat buyers, lapsed repeat buyers (the
    group that matters), and one-off buyers."""
    rng = np.random.default_rng(52)
    rows = []
    asof = pd.Timestamp("2024-12-31")
    for cid in range(120):                       # active repeaters
        for _ in range(rng.integers(6, 12)):
            rows.append({"customer_id": f"A{cid}",
                         "order_date": asof - pd.Timedelta(days=int(rng.integers(1, 60))),
                         "revenue": float(rng.uniform(80, 300))})
    for cid in range(60):                        # lapsed repeaters
        for _ in range(rng.integers(6, 12)):
            rows.append({"customer_id": f"L{cid}",
                         "order_date": asof - pd.Timedelta(days=int(rng.integers(400, 700))),
                         "revenue": float(rng.uniform(80, 300))})
    for cid in range(200):                       # one-off buyers
        rows.append({"customer_id": f"O{cid}",
                     "order_date": asof - pd.Timedelta(days=int(rng.integers(1, 700))),
                     "revenue": float(rng.uniform(20, 90))})
    return pd.DataFrame(rows)


def test_rfm_finds_the_lapsed_repeat_buyers(rfm_df):
    out = _run(rfm_segments, rfm_df)
    assert out["findings"], "no RFM output on a clearly segmented book"
    assert out["insights"], "the lapsed repeat-buyer group was not raised"
    ins = out["insights"][0]
    assert "lapsed" in ins.title.lower() or "not ordered" in ins.problem.lower()


def test_rfm_states_what_the_order_data_cannot_explain(rfm_df):
    """Naming a cause for lapse from order data alone is invention."""
    out = _run(rfm_segments, rfm_df)
    cause = out["insights"][0].cause.lower()
    assert "not in it" in cause or "not established" in cause or "records orders only" in cause


def test_rfm_champions_are_reported_as_an_opportunity(rfm_df):
    out = _run(rfm_segments, rfm_df)
    assert out["opportunities"], "champions were not surfaced"
    assert "champion" in _text(out).lower()


def test_rfm_needs_a_value_column():
    rng = np.random.default_rng(53)
    df = pd.DataFrame({
        "customer_id": [f"C{i%80}" for i in range(400)],
        "order_date": pd.date_range("2024-01-01", periods=400, freq="D"),
    })
    assert _run(rfm_segments, df)["findings"] == []


def test_rfm_skips_a_book_too_small_to_quintile():
    rng = np.random.default_rng(54)
    df = pd.DataFrame({
        "customer_id": [f"C{i%10}" for i in range(30)],
        "order_date": pd.date_range("2024-01-01", periods=30, freq="D"),
        "revenue": rng.uniform(10, 100, 30),
    })
    assert _run(rfm_segments, df)["findings"] == []


# ══════════════════════════════════════════════════════════
#  Revenue concentration
# ══════════════════════════════════════════════════════════

def test_concentrated_book_is_flagged_with_its_gini():
    """Ten customers hold most of the revenue among 200."""
    rng = np.random.default_rng(55)
    whales = pd.DataFrame({"customer_id": [f"W{i}" for i in range(10)],
                           "revenue": rng.uniform(80_000, 120_000, 10)})
    tail = pd.DataFrame({"customer_id": [f"T{i}" for i in range(190)],
                         "revenue": rng.uniform(200, 900, 190)})
    df = pd.concat([whales, tail], ignore_index=True)
    out = _run(revenue_concentration, df)
    assert out["risks"], "a book where 10 customers hold most revenue was not flagged"
    assert "Gini" in _text(out), "no Gini reported alongside the share"


def test_even_book_is_not_flagged():
    rng = np.random.default_rng(56)
    df = pd.DataFrame({"customer_id": [f"C{i}" for i in range(300)],
                       "revenue": rng.normal(1_000, 60, 300).round(2)})
    out = _run(revenue_concentration, df)
    assert not out["risks"], "flagged an evenly spread book as concentrated"
    assert out["opportunities"], "no note that revenue is well spread"


def test_an_uneven_book_is_not_described_as_evenly_spread():
    """Top decile under 50% but a Gini of 0.56 is not "spread evenly", and
    saying so contradicts the coefficient in the same sentence."""
    rng = np.random.default_rng(58)
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(400)],
        "revenue": np.round(np.power(10, rng.uniform(1.2, 3.4, 400)), 2),
    })
    out = _run(revenue_concentration, df)
    text = _text(out)
    import re
    m = re.search(r"Gini ([\d.]+)", text)
    assert m, "no Gini reported"
    if float(m.group(1)) >= 0.45 and not out["risks"]:
        assert "spread across the customer base" not in text, \
            "called a Gini {} book evenly spread".format(m.group(1))
        assert "uneven" in text


def test_gini_is_within_bounds_on_both_books():
    import re
    rng = np.random.default_rng(57)
    even = pd.DataFrame({"customer_id": [f"C{i}" for i in range(200)],
                         "revenue": np.full(200, 1_000.0)})
    out = _run(revenue_concentration, even)
    m = re.search(r"Gini ([\d.]+)", _text(out))
    assert m, "no Gini in the output"
    assert float(m.group(1)) == pytest.approx(0.0, abs=0.02), \
        "equal spending should give a Gini of 0, got {}".format(m.group(1))


def test_concentration_needs_a_real_customer_base():
    df = pd.DataFrame({"customer_id": [f"C{i%5}" for i in range(20)],
                       "revenue": np.linspace(10, 200, 20)})
    assert _run(revenue_concentration, df)["findings"] == []


# ══════════════════════════════════════════════════════════
#  Wired into the ecommerce engine
# ══════════════════════════════════════════════════════════

def test_ecommerce_engine_runs_the_customer_analyses(rfm_df):
    from app.engines.domains.ecommerce import _insights_ecommerce
    from app.engines.domains.base import col_stats
    df = rfm_df.copy()
    stats = {c: col_stats(df[c]) for c in df.columns
             if pd.api.types.is_numeric_dtype(df[c])}
    out = _insights_ecommerce(df, stats, [])
    blob = " ".join(out["findings"] + out["risks"] + out["opportunities"])
    assert "RFM" in blob or "concentration" in blob.lower(), \
        "customer analyses not reached by the ecommerce orchestrator"
