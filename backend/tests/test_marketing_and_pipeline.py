"""
Marketing campaigns and CRM pipelines, analysed the way a careful analyst
would: rates from totals, tested differences, risks dated from the data
itself, quota counted once per rep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _crm_data import crm_df
from _marketing_data import marketing_df
from app.engines.domains import marketing as mk
from app.engines.domains import pipeline as pl
from app.engines.story_engine import detect_domain, generate_story


# ── marketing ─────────────────────────────────────────────

def test_a_campaign_export_is_recognised_and_its_columns_found():
    df = marketing_df()
    assert detect_domain(df)[0] == "marketing"
    c = mk.detect_columns(df)
    assert (c.spend, c.impressions, c.clicks, c.conversions, c.revenue, c.channel) == (
        "Spend", "Impressions", "Clicks", "Conversions", "Revenue", "Channel")


def test_rates_come_from_totals_not_from_averaged_row_rates():
    df = pd.DataFrame({"Channel": ["A"] * 2 + ["B"] * 2, "Spend": [10, 1000, 50, 50],
                       "Impressions": [100, 1_000_000, 1000, 1000],
                       "Clicks": [50, 10_000, 20, 20], "Conversions": [5, 100, 2, 2]})
    rep = mk.analyse(df)
    a = next(r for r in rep.by_channel if r.name == "A")
    assert abs(a.ctr - 10_050 / 1_000_100) < 1e-12        # not mean(0.5, 0.01)
    assert abs(rep.total.cpa - 1110 / 109) < 1e-9


def test_a_ratio_column_in_the_file_is_not_taken_for_a_component():
    c = mk.detect_columns(marketing_df())
    assert "CTR" not in (c.clicks, c.impressions, c.spend)


def test_a_tiny_lucky_channel_is_not_ranked():
    df = marketing_df()
    lucky = pd.DataFrame([{"Date": df["Date"].max(), "Channel": "Tiny", "Campaign": "T",
                           "Impressions": 50, "Clicks": 5, "Spend": 1.0, "Conversions": 5,
                           "Revenue": 5000.0, "CTR": 0.1}])
    df = pd.concat([df, lucky], ignore_index=True)
    s = generate_story(df)
    assert not any("Tiny" in i.title for i in s.top_insights)


def test_the_money_losing_channel_is_named():
    s = generate_story(marketing_df())
    assert any("TikTok" in r and "ROAS below 1" in r for r in s.business_risks)
    assert any("Return on ad spend" in i.title for i in s.top_insights)


def test_an_expensive_channel_is_traced_to_its_funnel_stage():
    s = generate_story(marketing_df())
    worst = next(i for i in s.top_insights if "per conversion" in i.title)
    assert "click-through rate" in worst.cause or "convert" in worst.cause


# ── pipeline ──────────────────────────────────────────────

def test_a_crm_export_gets_a_pipeline_review():
    df = crm_df()
    c = pl.detect_columns(df)
    assert c.usable and c.quota == "Quota" and c.owner == "Owner"
    s = generate_story(df)
    text = " ".join(s.key_findings)
    assert "Open pipeline as of 30 Sep 2025" in text       # the data's own date
    assert "Win rate" in text and "Weighted" in text


def test_won_and_lost_are_read_from_the_stage():
    st = pl._status(pd.Series(["Closed Won", "Closed Lost", "Won", "Lost", "Proposal",
                               "Not won yet", None]))
    assert list(st) == ["won", "lost", "won", "lost", "open", "open", "open"]


def test_quota_repeated_on_every_row_is_counted_once_per_rep():
    s = generate_story(crm_df())
    # The old row-sum compared every deal's repeated quota: "13% achievement".
    assert not any("Target Gap" in i.title for i in s.top_insights)
    assert any("reps have reached quota" in f for f in s.key_findings)


def test_a_forecast_category_is_not_a_product_category():
    s = generate_story(crm_df())
    assert not any("'Closed'" in r for r in s.business_risks)


def test_today_is_not_taken_from_future_close_dates():
    df = crm_df()
    df.loc[df["Stage"].str.contains("Closed"), "Close Date"] = pd.Timestamp("2027-01-01")
    res = pl.run_pipeline(df, [], [], [], [])
    assert res["asof"] == pd.Timestamp("2025-09-30")


def test_a_file_without_stages_is_not_a_pipeline():
    df = pd.DataFrame({"region": np.repeat(["N", "S"], 50), "amount": np.arange(100.0)})
    assert pl.run_pipeline(df, [], [], [], []) is None


# ── Hindi reports ─────────────────────────────────────────

import io

import pytest


@pytest.mark.parametrize("make", [marketing_df, crm_df], ids=["marketing", "crm"])
def test_hindi_marketing_and_pipeline_reports_are_fully_translated(client, make):
    from app import i18n
    buf = io.BytesIO()
    make().to_csv(buf, index=False)
    r = client.post("/api/datasets/upload", files={"file": ("d.csv", buf.getvalue(), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    i18n.clear_misses()
    for kind in ("pdf", "health-pdf"):
        assert client.post(f"/api/reports/{ds}/{kind}", json={"language": "hi"}).status_code == 200
    assert i18n.misses() == {}


def test_a_plain_value_does_not_take_a_whole_list():
    from app import i18n
    with i18n.use_language("hi"):
        i18n.clear_misses()
        out = i18n.tr("Marketing totals: Spend 500 · 40 conversions · cost per conversion 12.5.")
        assert "conversions" not in out and not i18n.misses()


def test_a_template_spanning_sentences_matches_inside_a_paragraph():
    from app import i18n
    with i18n.use_language("hi"):
        i18n.clear_misses()
        i18n.tr("Top-quartile reps post a median of 9 vs 4 for the bottom quartile (2.2x). "
                "5 reps with >=3 records. Top 20% of reps generate 23% of revenue. "
                "Team median per record=21.")
        assert not i18n.misses()
