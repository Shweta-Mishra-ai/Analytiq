"""
What changed and why, the next three months, and what to act on first —
each checked against data whose answer is known.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from _check_data import daily_sales, planted_df
from app.engines.change_drivers import explain_change
from app.engines.forecast import forecast, monthly_series
from app.engines.priorities import build_priorities


# ── why ──────────────────────────────────────────────────

def test_the_planted_cause_is_found_one_level_down():
    e = explain_change(planted_df())
    assert e.delta < 0
    assert e.where.column == "Region" and e.where.contributions[0].segment == "West"
    assert e.within_segment == "West"
    assert e.within.column == "Category" and e.within.contributions[0].segment == "Furniture"
    # Volume and value add up to the change exactly.
    assert abs(e.volume_effect + e.value_effect - e.delta) < 1e-6 * abs(e.prior)


def test_segments_moving_by_chance_are_not_named():
    named = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        d = pd.date_range("2024-07-01", "2025-06-30")
        n = len(d) * 15
        df = pd.DataFrame({"Order Date": np.repeat(d, 15),
                           "Region": rng.choice(list("NSEWC"), n),
                           "Category": rng.choice(list("abcdefgh"), n),
                           "Sales": rng.gamma(2, 60, n).round(2)})
        e = explain_change(df)
        named += bool(e and e.where and e.where.contributions)
    assert named <= 2, named


# ── forecast ─────────────────────────────────────────────

def test_a_seasonal_series_is_forecast_better_than_last_month():
    f = forecast(daily_sales(seed=1, per_day=20, season=0.4))
    assert f is not None and len(f.points) == 3
    assert f.error_pct < f.naive_error_pct
    assert all(p.low <= p.value <= p.high for p in f.points)


def test_the_range_holds_about_eight_in_ten_months_it_never_saw():
    inside = total = 0
    for seed in range(12):
        df = daily_sales(seed=seed, per_day=10, season=0.3, end="2025-03-31")
        cut = pd.Timestamp("2025-01-01")
        f = forecast(df[df["Order Date"] < cut])
        truth = monthly_series(df)[2].iloc[-3:].to_numpy()
        for p, t in zip(f.points, truth):
            total += 1
            inside += p.low <= t <= p.high
    assert 0.65 <= inside / total <= 0.97, inside / total


def test_too_little_history_gives_no_forecast():
    assert forecast(daily_sales(seed=1, start="2024-06-01", end="2025-03-31")) is None


# ── priorities ───────────────────────────────────────────

def test_actions_are_ranked_by_money_a_year_with_their_evidence():
    ps = build_priorities(planted_df())
    findings = " | ".join(p.finding for p in ps)
    assert "West" in findings and "discounted" in findings
    assert all(p.value_basis and p.evidence for p in ps)
    assert all("a year" in p.value_basis for p in ps)
    values = [p.value for p in ps]
    assert values == sorted(values, reverse=True)


# ── page and report ──────────────────────────────────────

def _upload(client, df):
    b = io.BytesIO()
    df.to_csv(b, index=False)
    r = client.post("/api/datasets/upload", files={"file": ("d.csv", b.getvalue(), "text/csv")})
    return r.json()["meta"]["dataset_id"]


def test_the_outlook_endpoint_returns_all_three(client):
    ds = _upload(client, planted_df())
    j = client.get(f"/api/analytics/{ds}/outlook").json()
    assert j["why"]["column"] == "Region"
    assert len(j["forecast"]["points"]) == 3
    assert j["priorities"] and j["priorities"][0]["rank"] == 1


def test_hindi_reports_with_the_outlook_are_fully_translated(client):
    from app import i18n
    for df in (planted_df(), daily_sales(seed=2, season=0.4, spike=(2024, 7))):
        ds = _upload(client, df)
        i18n.clear_misses()
        for kind in ("pdf", "health-pdf"):
            assert client.post(f"/api/reports/{ds}/{kind}", json={"language": "hi"}).status_code == 200
        assert i18n.misses() == {}
