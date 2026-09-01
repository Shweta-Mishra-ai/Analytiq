"""
The general engine — what runs when a dataset is not recognisably HR,
finance, sales or e-commerce, which is most files a freelancer receives.

It could previously describe distributions, list correlations and compare
two segments: a description of the data rather than an account of the
business in it. These are the analyses that need no domain vocabulary and
answer questions a client actually asks — is it moving, where do the
extremes sit, and which segments are too small to conclude from.

The false-positive tests are the point of the file. A trend engine that
finds a trend in noise, or a segment comparison that calls a 6%
difference "underperformance", produces confident wrong statements — the
thing that gets a report rejected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.domains.general_depth import (outlier_concentration,
                                               thin_segments,
                                               trend_over_time)


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
#  Trend
# ══════════════════════════════════════════════════════════

def _series(slope_per_day, noise, n=300, seed=210):
    rng = np.random.default_rng(seed)
    base = 100 + slope_per_day * np.arange(n)
    return pd.DataFrame({
        "observed_at": pd.date_range("2024-01-01", periods=n, freq="D"),
        "value": (base + rng.normal(0, noise, n)).round(2),
    })


def test_a_real_trend_is_found_with_its_fit():
    out = _run(trend_over_time, _series(0.5, noise=5))
    assert out["findings"], "a clean upward trend produced nothing"
    text = _text(out)
    assert "rising" in text
    assert "R²" in text, "no measure of how well the line fits"


def test_no_trend_is_claimed_in_pure_noise():
    """A slope through scatter is a true statement about nothing."""
    out = _run(trend_over_time, _series(0.0, noise=20))
    assert not out["insights"], "invented a trend in noise: {}".format(
        _text(out)[:200])
    assert "no trend" in _text(out).lower()


def test_a_significant_but_weak_trend_is_reported_as_no_trend():
    """On 300 points a tiny slope is significant and explains nothing. The
    direction is real; acting on it is not supported."""
    out = _run(trend_over_time, _series(0.01, noise=25))
    assert not out["insights"]
    assert "distinguishable from ordinary variation" in _text(out)


def test_a_falling_trend_is_raised_as_a_risk():
    out = _run(trend_over_time, _series(-0.5, noise=5))
    assert out["risks"], "a sustained decline was not raised"
    assert "falling" in _text(out)


def test_the_trend_projection_names_itself_as_one():
    """"Would move by X" is a projection; "will move by X" is a claim
    about a future this data cannot observe."""
    out = _run(trend_over_time, _series(0.5, noise=5))
    assert out["insights"]
    impact = out["insights"][0].impact.lower()
    assert "projection, not a forecast" in impact


def test_the_trend_cause_does_not_invent_a_driver():
    out = _run(trend_over_time, _series(0.5, noise=5))
    cause = out["insights"][0].cause.lower()
    assert "not identifiable" in cause or "not why" in cause


def test_trend_needs_a_date_column():
    rng = np.random.default_rng(211)
    df = pd.DataFrame({"value": rng.normal(0, 1, 200), "g": ["a", "b"] * 100})
    assert _run(trend_over_time, df)["findings"] == []


def test_trend_needs_enough_observations():
    assert _run(trend_over_time, _series(0.5, noise=5, n=12))["findings"] == []


# ══════════════════════════════════════════════════════════
#  Outlier concentration
# ══════════════════════════════════════════════════════════

def _with_outliers(concentrated: bool, n=400, seed=212):
    rng = np.random.default_rng(seed)
    seg = rng.choice(["Alpha", "Beta", "Gamma"], n)
    val = rng.normal(100, 10, n)
    if concentrated:
        hit = (seg == "Gamma") & (rng.random(n) < 0.25)
    else:
        hit = rng.random(n) < 0.08
    val = np.where(hit, val * 8, val)
    return pd.DataFrame({"branch": seg, "reading": val.round(2)})


def test_concentrated_outliers_are_found_and_tested():
    out = _run(outlier_concentration, _with_outliers(concentrated=True))
    assert out["insights"], "a clear concentration was not reported"
    text = _text(out)
    assert "Gamma" in text
    assert "Fisher" in text, "the concentration was asserted without a test"


def test_spread_outliers_are_reported_as_spread():
    """Ten outliers across forty groups is the tail of a distribution and
    means nothing — saying so is more useful than staying silent."""
    out = _run(outlier_concentration, _with_outliers(concentrated=False))
    assert not out["insights"]
    assert "spread across" in _text(out)


def test_the_outlier_cause_lists_possibilities_without_choosing():
    out = _run(outlier_concentration, _with_outliers(concentrated=True))
    cause = out["insights"][0].cause.lower()
    assert "not in this data" in cause


def test_too_few_outliers_produces_nothing():
    rng = np.random.default_rng(213)
    df = pd.DataFrame({
        "branch": rng.choice(["A", "B"], 300),
        "reading": rng.normal(100, 5, 300).round(2),
    })
    assert _run(outlier_concentration, df)["findings"] == []


# ══════════════════════════════════════════════════════════
#  Thin segments
# ══════════════════════════════════════════════════════════

def test_thin_segments_are_named_before_a_client_finds_them():
    """A 90% rate on nine records will be quoted back in the meeting."""
    rng = np.random.default_rng(214)
    seg = np.concatenate([
        rng.choice(["Main", "Second"], 280),
        np.array(["Tiny"] * 9 + ["Smaller"] * 6),
    ])
    df = pd.DataFrame({"branch": seg,
                       "value": rng.normal(50, 8, len(seg)).round(2)})
    out = _run(thin_segments, df)
    assert out["findings"], "small groups were not flagged"
    assert "indicative" in _text(out)


def test_no_thin_segment_note_when_every_group_is_large():
    rng = np.random.default_rng(215)
    df = pd.DataFrame({
        "branch": rng.choice(["A", "B", "C"], 400),
        "value": rng.normal(50, 8, 400).round(2),
    })
    assert _run(thin_segments, df)["findings"] == []


def test_no_note_when_every_group_is_thin():
    """If nothing is large enough, the dataset is the problem and the
    segment note adds nothing — the readiness check covers it."""
    rng = np.random.default_rng(216)
    df = pd.DataFrame({
        "branch": [f"G{i // 5}" for i in range(100)],
        "value": rng.normal(50, 8, 100).round(2),
    })
    assert _run(thin_segments, df)["findings"] == []


# ══════════════════════════════════════════════════════════
#  Segment comparison no longer fires on trivial gaps
# ══════════════════════════════════════════════════════════

def test_a_trivial_segment_gap_is_not_called_underperformance():
    """On a few hundred rows Kruskal-Wallis returns p<0.001 for a 6%
    median difference, often produced by a handful of outliers in the
    other group. The report then names a segment as underperforming over
    a gap nobody would act on."""
    from app.engines.domains.base import col_stats
    from app.engines.domains.general import _insights_general

    rng = np.random.default_rng(200)
    n = 400
    seg = rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n, p=[.4, .3, .25, .05])
    val = np.linspace(100, 160, n) + rng.normal(0, 8, n)
    val = np.where(seg == "Gamma", val * rng.choice([1, 9], n, p=[.85, .15]), val)
    df = pd.DataFrame({
        "observed_at": pd.date_range("2024-01-01", periods=n, freq="D"),
        "branch": seg, "reading": val.round(2)})
    stats = {c: col_stats(df[c]) for c in df.columns
             if pd.api.types.is_numeric_dtype(df[c])}
    out = _insights_general(df, stats, [])
    assert not any("sits below" in r for r in out["risks"]), out["risks"]


def test_a_substantial_segment_gap_is_still_reported():
    from app.engines.domains.base import col_stats
    from app.engines.domains.general import _insights_general

    rng = np.random.default_rng(217)
    n = 400
    seg = rng.choice(["Alpha", "Beta"], n)
    val = np.where(seg == "Alpha", rng.normal(60, 8, n), rng.normal(120, 8, n))
    df = pd.DataFrame({"branch": seg, "reading": val.round(2),
                       "other": rng.normal(0, 1, n)})
    stats = {c: col_stats(df[c]) for c in df.columns
             if pd.api.types.is_numeric_dtype(df[c])}
    out = _insights_general(df, stats, [])
    # The gap is reported as one segment sitting below another, naming
    # both — not as a quoted column identifier.
    assert any("sits below" in r for r in out["risks"]), \
        "a two-fold gap was suppressed"
    gap_risk = next(r for r in out["risks"] if "sits below" in r)
    assert "Alpha" in gap_risk and "Beta" in gap_risk, gap_risk
    assert "'" not in gap_risk, "column names are quoted like identifiers"
