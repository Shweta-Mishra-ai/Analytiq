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

def _orders(months, per_month, drop_last=0.0, seed=3):
    """Orders in four regions over whole months; South's last month loses
    `drop_last` of its orders."""
    rng = np.random.default_rng(seed)
    rows, oid = [], 0
    periods = pd.period_range("2024-11", periods=months, freq="M")
    for m in periods:
        for region in ("North", "South", "East", "West"):
            n = rng.poisson(per_month)
            if region == "South" and m == periods[-1]:
                n = int(n * (1 - drop_last))
            for _ in range(n):
                oid += 1
                day = m.start_time + pd.Timedelta(days=int(rng.integers(0, m.days_in_month)))
                rows.append((f"O{oid}", day, region, round(float(rng.gamma(2, 150)), 2)))
    return pd.DataFrame(rows, columns=["order_id", "order_date", "region", "sales"])


def test_a_fall_in_the_latest_month_is_explained_when_the_year_was_flat():
    # Two flat years, then South loses half its orders in the last month:
    # the 12-month comparison barely moves, and the month is what matters.
    from app.engines.change_drivers import explain_change_or_reason
    e, reason = explain_change_or_reason(_orders(24, 150, drop_last=0.5))
    assert reason is None and e.months == 1
    assert e.period.startswith("month of") and e.delta < 0
    assert e.where.column == "region" and e.where.contributions[0].segment == "South"
    assert any("a year earlier" in s for s in e.sentences)


def test_no_change_says_why_in_words():
    from app.engines.change_drivers import explain_change_or_reason
    flat = _orders(24, 150)
    e, reason = explain_change_or_reason(flat)
    if e is None:
        assert "changed" in reason and "chance" in reason
    no_money = flat.rename(columns={"sales": "visits"})
    no_money["visits"] = 1
    assert "no money measure" in explain_change_or_reason(no_money)[1]
    assert "no date column" in explain_change_or_reason(flat.drop(columns="order_date"))[1]


def test_a_cause_named_on_noise_stays_rare():
    # Both looks (the headline periods and the latest month) share one 5%
    # budget. 200 such files measured 2.5%; at 5%, 60 files name a cause in
    # 3 on average and more than 6 about 4% of the time, so 7 or more means
    # the bar has slipped.
    from app.engines.change_drivers import explain_change
    named = sum(1 for seed in range(60)
                if (e := explain_change(_orders(18, 60, seed=100 + seed))) is not None
                and e.where is not None)
    assert named <= 6, named


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
