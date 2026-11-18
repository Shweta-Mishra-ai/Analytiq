"""
The report and dashboard lead with the business: headline measures, their
change on the prior period, the messages that follow from them, numbered
exhibits with sources and a prioritised recommendations table.

Every figure is checked against pandas computed independently, and random
data must produce the figures but no claimed effect.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pypdf
import pytest

from app.engines.scorecard import build_scorecard, fmt_change, priority


def _sales(n=2400, seed=11, discount_effect=True):
    rng = np.random.default_rng(seed)
    region = rng.choice(["North", "South", "East", "West"], n)
    cat = rng.choice(["Electronics", "Furniture", "Office Supplies"], n)
    date = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    sales = rng.gamma(2, 250, n) * (1 + (date.year - 2023) * 0.12)
    disc = rng.choice([0, .1, .2, .3], n, p=[.5, .25, .15, .1])
    margin = (0.22 - 0.9 * disc) if discount_effect else 0.15
    profit = sales * margin + rng.normal(0, 30, n)
    return pd.DataFrame({"order_id": range(10001, 10001 + n), "order_date": date,
                         "region": region, "category": cat,
                         "customer_id": rng.integers(1, 400, n), "sales": sales.round(2),
                         "quantity": rng.integers(1, 10, n), "discount": disc,
                         "profit": profit.round(2)})


def _upload(client, df, name="sales.csv"):
    r = client.post("/api/datasets/upload",
                    files={"file": (name, io.BytesIO(df.to_csv(index=False).encode()),
                                    "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["meta"]["dataset_id"]


# ── figures ──────────────────────────────────────────────────────────────

def test_the_period_comparison_matches_calendar_year_totals():
    df = _sales()
    sc = build_scorecard(df)
    assert (sc.period, sc.prior_period) == ("12 months to Dec 2024", "12 months to Dec 2023")
    by_year = df.groupby(df.order_date.dt.year)[["sales", "profit"]].sum()
    m = {x.label: x for x in sc.metrics}
    assert m["Revenue"].value == pytest.approx(by_year.loc[2024, "sales"])
    assert m["Revenue"].prior == pytest.approx(by_year.loc[2023, "sales"])
    assert m["Profit margin"].value == pytest.approx(
        by_year.loc[2024, "profit"] / by_year.loc[2024, "sales"])
    assert m["Orders"].value == (df.order_date.dt.year == 2024).sum()
    assert fmt_change(m["Profit margin"]).endswith(" pp")


def test_a_partial_last_month_is_left_out_of_the_period():
    df = _sales()
    df = df[df.order_date < "2024-12-15"]          # December only half covered
    sc = build_scorecard(df)
    assert sc.period == "3 months to Nov 2024"      # not two full years any more
    assert sc.prior_period == "3 months to Aug 2024"


def test_no_revenue_measure_means_no_scorecard():
    rng = np.random.default_rng(0)
    hr = pd.DataFrame({"department": rng.choice(["A", "B"], 300),
                       "salary": rng.normal(5e4, 9e3, 300), "age": rng.integers(20, 60, 300)})
    sc = build_scorecard(hr)
    assert sc.metrics == [] and sc.messages == []


def test_growth_is_split_into_volume_and_value_and_names_only_real_movers():
    df = _sales()
    detail = next(m.detail for m in build_scorecard(df).messages if m.lead)
    # Revenue rose 14% here on value per order, every region with it.
    assert "Orders fell 0.2% and average order value rose 14.2%" in detail
    # No region's change passes the chance test once the 7 segments tested
    # are allowed for (North, the largest, is z=2.4 against a bar of 2.7):
    # the largest rise used to be named whatever its size against chance.
    assert "By 'region'" not in detail


# ── honesty ──────────────────────────────────────────────────────────────

def test_the_discount_effect_is_found_when_it_is_there():
    msgs = build_scorecard(_sales()).messages
    disc = [m for m in msgs if m.title.startswith("Orders discounted")]
    assert disc and "lose money" in disc[0].title
    assert "association" in disc[0].detail


def test_random_data_gets_figures_but_no_claimed_effect():
    sc = build_scorecard(_sales(discount_effect=False, seed=3))
    assert sc.metrics
    titles = [m.title for m in sc.messages]
    assert not any(t.startswith(("Orders discounted", "Margin ranges")) for t in titles), titles


def test_priority_follows_the_revenue_concerned():
    from app.engines.scorecard import Message
    assert priority(Message("", "", "", at_stake=30), 100) == "High"
    assert priority(Message("", "", "", at_stake=6), 100) == "Medium"
    assert priority(Message("", "", "", at_stake=1), 100) == "Low"


def test_revenue_leads_the_messages():
    assert build_scorecard(_sales()).messages[0].lead


# ── the PDF ──────────────────────────────────────────────────────────────

def _text(pdf: bytes) -> str:
    return " ".join(" ".join(p.extract_text() or "" for p in
                             pypdf.PdfReader(io.BytesIO(pdf)).pages).split())


def test_the_report_leads_with_key_messages_and_a_scorecard(client):
    ds = _upload(client, _sales())
    pdf = client.post(f"/api/reports/{ds}/pdf",
                      json={"prepared_by": "Northbridge Advisory"}).content
    text = _text(pdf)
    first = _text_of_page(pdf, 0) + _text_of_page(pdf, 2)
    assert "Northbridge Advisory" in _text_of_page(pdf, 0)
    assert "Analytiq" not in _text_of_page(pdf, 0)
    assert "Key messages" in first and "Performance scorecard" in first
    assert "Exhibit 1" in text and "Source: sales.csv; 2,400 records, Jan 2023 to Dec 2024" in text
    assert "Recommendation Basis Priority" in text


def _text_of_page(pdf: bytes, i: int) -> str:
    return " ".join((pypdf.PdfReader(io.BytesIO(pdf)).pages[i].extract_text() or "").split())


def test_a_report_without_revenue_keeps_its_own_summary(client, hr_csv_bytes):
    r = client.post("/api/datasets/upload",
                    files={"file": ("hr.csv", io.BytesIO(hr_csv_bytes), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    text = _text(client.post(f"/api/reports/{ds}/pdf", json={}).content)
    assert "Performance scorecard" not in text and "Executive Summary" in text


def test_the_hindi_report_of_sales_data_is_fully_translated(client):
    from app import i18n
    ds = _upload(client, _sales())
    i18n.clear_misses()
    for kind in ("pdf", "health-pdf"):
        r = client.post(f"/api/reports/{ds}/{kind}", json={"language": "hi"})
        assert r.status_code == 200, r.text
    assert i18n.misses() == {}


def test_a_product_price_gap_is_not_reported_as_a_risk():
    from app.engines.story_engine import generate_story
    story = generate_story(_sales())
    assert not any("Office Supplies" in r and "trails" in r for r in story.business_risks)


# ── the dashboard ────────────────────────────────────────────────────────

def test_dashboard_cards_carry_the_change_on_the_prior_period(client):
    ds = _upload(client, _sales())
    body = client.post(f"/api/charts/{ds}/kpis", json={"filters": []}).json()
    cards = {k["label"]: k for k in body["kpis"]}
    assert list(cards)[:3] == ["Revenue", "Profit", "Profit margin"]
    assert body["period"] == "12 months to Dec 2024 against the 12 months before"
    assert cards["Profit margin"]["delta_unit"] == "pp"
    # A rise in discounting is bad news; no change is neither.
    disc = cards["Average discount"]
    assert disc["delta_good"] is (None if round(disc["delta"], 1) == 0 else disc["delta"] < 0)
    assert cards["Revenue"]["sub"].startswith("vs ") and "prior 12 months" in cards["Revenue"]["sub"]


def test_dashboard_cards_follow_the_filters(client):
    df = _sales()
    ds = _upload(client, df)
    body = client.post(f"/api/charts/{ds}/kpis",
                       json={"filters": [{"column": "region", "op": "eq", "value": "West"}]}).json()
    rev = {k["label"]: k for k in body["kpis"]}["Revenue"]["value"]
    expected = df[(df.region == "West") & (df.order_date.dt.year == 2024)].sales.sum()
    assert rev == pytest.approx(expected)


def test_a_two_year_trend_is_drawn_by_month(client):
    ds = _upload(client, _sales())
    fig = client.post(f"/api/charts/{ds}/build",
                      json={"type": "line", "x": "order_date", "y": "sales", "agg": "sum"}).json()
    xs = pd.to_datetime(pd.Series(fig["figure"]["data"][0]["x"]))
    assert len(xs) == 24                              # Jan 2023 – Dec 2024, by month
    assert (xs.dt.is_month_end).all()


def test_the_trend_and_the_cards_agree_on_whether_the_last_month_is_complete(client):
    df = _sales()                     # runs to 30 December
    ds = _upload(client, df)
    fig = client.post(f"/api/charts/{ds}/build",
                      json={"type": "line", "x": "order_date", "y": "sales", "agg": "sum"}).json()
    assert "last partial month excluded" not in fig["figure"]["layout"]["xaxis"]["title"]["text"]


def test_bars_carry_value_labels_only_when_they_can_be_read():
    from app.engines.chart_engine import make_bar
    few = pd.DataFrame({"g": list("ABCDE"), "v": [5.0, 4, 3, 2, 1]})
    many = pd.DataFrame({"g": [f"g{i}" for i in range(20)], "v": range(20)})
    assert make_bar(few, "g", "v").data[0].texttemplate == "%{y:.3~s}"
    assert make_bar(many, "g", "v").data[0].texttemplate is None
