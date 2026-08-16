"""
Win rates by rep, and cycle length for won versus lost deals.

The multiplicity test is the important one here. A twenty-rep team all
converting at the same underlying rate will, scanned at p<0.05 without
correction, throw up a "standout performer" most of the time — and that
name goes into somebody's review. The null-data test below is the guard
against shipping that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.domains.sales_performance import (
    cycle_by_outcome,
    find_outcome_col,
    find_rep_col,
    rep_win_rates,
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


def _deals(rep_rates, n_per_rep=120, seed=61, include_open=False):
    """One row per deal, each rep converting at their given rate."""
    rng = np.random.default_rng(seed)
    rows = []
    for rep, rate in rep_rates.items():
        for _ in range(n_per_rep):
            won = rng.random() < rate
            rows.append({"sales_rep": rep,
                         "deal_status": "Won" if won else "Lost",
                         "amount": float(rng.uniform(1_000, 20_000))})
    if include_open:
        for i in range(len(rows) // 4):
            rows.append({"sales_rep": list(rep_rates)[i % len(rep_rates)],
                         "deal_status": "In Progress",
                         "amount": float(rng.uniform(1_000, 20_000))})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
#  Detection
# ══════════════════════════════════════════════════════════

def test_outcome_detection_reads_won_and_lost():
    df = _deals({"A": 0.5, "B": 0.5}, n_per_rep=40)
    found = find_outcome_col(df)
    assert found is not None
    col, outcome = found
    assert col == "deal_status"
    assert outcome.dropna().isin([True, False]).all()


def test_open_deals_are_excluded_not_counted_as_losses():
    """A pipeline that is 20% open would otherwise report a win rate that
    is really a snapshot of timing."""
    df = _deals({"A": 0.5, "B": 0.5}, n_per_rep=100, include_open=True)
    _col, outcome = find_outcome_col(df)
    assert outcome.isna().sum() == (df["deal_status"] == "In Progress").sum()


def test_outcome_detection_needs_both_outcomes_present():
    df = pd.DataFrame({"deal_status": ["Won"] * 80, "sales_rep": ["A"] * 80})
    assert find_outcome_col(df) is None


def test_rep_detection_skips_identifier_columns():
    df = pd.DataFrame({
        "rep_id": range(100),
        "sales_rep": [f"R{i % 5}" for i in range(100)],
        "deal_status": ["Won", "Lost"] * 50,
    })
    assert find_rep_col(df) == "sales_rep"


# ══════════════════════════════════════════════════════════
#  Rep win rates — the multiplicity guard
# ══════════════════════════════════════════════════════════

def test_no_rep_is_named_when_they_all_convert_the_same():
    """Twenty reps, one true rate. Any name reported here is a false
    positive produced by scanning, and it would land in a performance
    review."""
    df = _deals({f"R{i:02d}": 0.35 for i in range(20)}, n_per_rep=100, seed=62)
    out = _run(rep_win_rates, df)
    assert out["findings"], "no win-rate summary at all"
    assert not out["insights"], \
        "named a standout rep in pure noise: {}".format(_text(out)[:400])
    assert not out["risks"] and not out["opportunities"]
    assert "no rep's win rate differs" in _text(out).lower()


def test_a_genuinely_weak_rep_is_found():
    rates = {f"R{i:02d}": 0.40 for i in range(8)}
    rates["R99"] = 0.10
    df = _deals(rates, n_per_rep=120, seed=63)
    out = _run(rep_win_rates, df)
    assert out["risks"], "a rep converting at a quarter of the team rate was missed"
    assert "R99" in _text(out)


def test_a_genuinely_strong_rep_is_found():
    rates = {f"R{i:02d}": 0.30 for i in range(8)}
    rates["STAR"] = 0.75
    df = _deals(rates, n_per_rep=120, seed=64)
    out = _run(rep_win_rates, df)
    assert out["opportunities"], "an outperforming rep was not surfaced"
    assert "STAR" in _text(out)


def test_reps_with_too_few_deals_are_excluded():
    """3 of 4 outranks 60 of 100 on raw rate, and the table gets read as a
    league."""
    rates = {f"R{i:02d}": 0.40 for i in range(6)}
    df = _deals(rates, n_per_rep=100, seed=65)
    thin = pd.DataFrame({"sales_rep": ["NEWSTARTER"] * 4,
                         "deal_status": ["Won", "Won", "Won", "Lost"],
                         "amount": [1000.0] * 4})
    out = _run(rep_win_rates, pd.concat([df, thin], ignore_index=True))
    assert "NEWSTARTER" not in _text(out), \
        "a rep with 4 deals was ranked against reps with 100"


def test_evidence_names_the_correction_applied():
    rates = {f"R{i:02d}": 0.40 for i in range(8)}
    rates["R99"] = 0.10
    out = _run(rep_win_rates, _deals(rates, n_per_rep=120, seed=66))
    assert out["insights"]
    ev = out["insights"][0].evidence.lower()
    assert "binomial" in ev and ("benjamini" in ev or "q<" in ev), \
        "the evidence does not state the test or the correction: {!r}".format(ev)


def test_cause_does_not_blame_the_individual():
    """Territory and lead quality move a win rate, and none of it is held
    constant. Stating the rep is the cause is unfounded."""
    rates = {f"R{i:02d}": 0.40 for i in range(8)}
    rates["R99"] = 0.10
    out = _run(rep_win_rates, _deals(rates, n_per_rep=120, seed=67))
    cause = out["insights"][0].cause.lower()
    assert "territory" in cause or "lead quality" in cause
    assert "not by itself evidence" in cause or "not held constant" in cause


def test_headline_agrees_with_itself_grammatically():
    """"1 of 9 reps differ" reads as a typo, and a typo in the headline is
    what a reader notices before the finding."""
    rates = {f"R{i:02d}": 0.40 for i in range(8)}
    rates["R99"] = 0.10
    out = _run(rep_win_rates, _deals(rates, n_per_rep=120, seed=72))
    title = out["insights"][0].title
    if title.startswith("1 of "):
        assert "differs" in title, title


def test_a_team_too_small_to_compare_is_skipped():
    df = _deals({"A": 0.6, "B": 0.2}, n_per_rep=100, seed=68)
    out = _run(rep_win_rates, df)
    assert not out["insights"], "compared a two-person team as if it were a distribution"


# ══════════════════════════════════════════════════════════
#  Cycle length by outcome
# ══════════════════════════════════════════════════════════

def _cycle_frame(won_days, lost_days, n=300, seed=69):
    """Deals with lognormal cycle times, so the distribution is skewed the
    way real cycle data is."""
    rng = np.random.default_rng(seed)
    rows = []
    for won, scale in ((True, won_days), (False, lost_days)):
        days = rng.lognormal(np.log(scale), 0.5, n).round()
        opened = pd.Timestamp("2024-01-01") + pd.to_timedelta(
            rng.integers(0, 200, n), unit="D")
        for o, d in zip(opened, days):
            rows.append({"created_date": o,
                         "close_date": o + pd.Timedelta(days=int(min(d, 700))),
                         "deal_status": "Won" if won else "Lost"})
    return pd.DataFrame(rows)


def test_slower_lost_deals_are_reported_with_an_effect_size():
    df = _cycle_frame(won_days=30, lost_days=95)
    out = _run(cycle_by_outcome, df)
    assert out["findings"], "no comparison produced"
    text = _text(out)
    assert "rank-biserial" in text, "no effect size reported"
    assert out["insights"], "a three-fold cycle difference produced no insight"
    assert out["opportunities"], "no stall threshold offered"


def test_no_claim_when_cycles_are_the_same():
    df = _cycle_frame(won_days=45, lost_days=45)
    out = _run(cycle_by_outcome, df)
    assert not out["insights"], \
        "found a difference between identical distributions: {}".format(_text(out)[:300])
    assert "does not usefully separate" in _text(out)


def test_significant_but_trivial_difference_is_not_raised():
    """On thousands of deals a two-day difference is significant and
    worthless. The effect floor is what keeps it out of the report."""
    df = _cycle_frame(won_days=60, lost_days=62, n=3000, seed=70)
    out = _run(cycle_by_outcome, df)
    assert not out["insights"], \
        "raised a two-day median difference as a finding"


def test_medians_are_used_rather_than_means():
    """A handful of stalled deals move a mean by more than the difference
    being measured."""
    df = _cycle_frame(won_days=30, lost_days=95)
    out = _run(cycle_by_outcome, df)
    assert "median" in out["findings"][0].lower()


def test_cycle_comparison_needs_enough_of_both_outcomes():
    df = _cycle_frame(won_days=30, lost_days=95, n=300)
    few_lost = pd.concat([df[df["deal_status"] == "Won"],
                          df[df["deal_status"] == "Lost"].head(5)],
                         ignore_index=True)
    out = _run(cycle_by_outcome, few_lost)
    assert not out["findings"], "compared distributions with 5 lost deals"


def test_cycle_comparison_needs_date_columns():
    df = _deals({"A": 0.4, "B": 0.4, "C": 0.4}, n_per_rep=60)
    assert _run(cycle_by_outcome, df)["findings"] == []


# ══════════════════════════════════════════════════════════
#  Wired into the sales engine
# ══════════════════════════════════════════════════════════

def test_sales_engine_runs_the_outcome_analyses():
    from app.engines.domains.base import col_stats
    from app.engines.domains.sales import _insights_sales

    rates = {f"R{i:02d}": 0.40 for i in range(8)}
    rates["R99"] = 0.10
    df = _deals(rates, n_per_rep=120, seed=71)
    stats = {c: col_stats(df[c]) for c in df.columns
             if pd.api.types.is_numeric_dtype(df[c])}
    out = _insights_sales(df, stats, [])
    blob = " ".join(out["findings"] + out["risks"] + out["opportunities"])
    assert "Win rate" in blob, "rep analysis not reached by the sales orchestrator"
