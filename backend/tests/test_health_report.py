"""
Tests for the Health Report — engines/health_engine.py,
engines/health_pdf_builder.py, and the /api/reports/{id}/health* routes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.health_engine import (
    build_full_insights,
    build_insights,
    compute_health,
)


@pytest.fixture()
def signal_sales_df():
    """Sales data with deliberate, findable signal: one region badly
    underperforming, and a slice of loss-making deals."""
    rng = np.random.default_rng(7)
    n = 600
    region = rng.choice(["North", "South", "East", "West"], n, p=[.3, .3, .25, .15])
    revenue = (np.where(region == "West", 28000, 52000)
               + rng.normal(0, 8000, n)).round(2)
    return pd.DataFrame({
        "deal_id": range(n),
        "region": region,
        "revenue": revenue,
        "quota": np.full(n, 55000.0),
        "profit": (revenue * rng.uniform(-0.05, 0.25, n)).round(2),
        "product": rng.choice(["Basic", "Pro", "Enterprise"], n),
        "sales_rep": rng.choice([f"Rep {i}" for i in range(8)], n),
    })

_INSIGHT_KEYS = {"tag", "title", "body", "action", "severity"}
_VALID_SEVERITIES = {"critical", "warning", "positive", "info"}


# ══════════════════════════════════════════════════════════
#  Health scoring
# ══════════════════════════════════════════════════════════

def test_compute_health_shape_and_bounds(hr_df):
    h = compute_health(hr_df)
    assert 0 <= h["score"] <= 100
    assert h["grade"] in {"A+", "A", "B+", "B", "C", "D"}
    assert h["rows"] == len(hr_df)
    assert h["cols"] == len(hr_df.columns)
    assert h["color"].startswith("#")
    for k in ("missing_pct", "dup_pct", "outlier_pct"):
        assert 0 <= h[k] <= 100


def test_compute_health_penalises_missing_and_duplicates(hr_df):
    clean = compute_health(hr_df)
    dirty_df = pd.concat([hr_df, hr_df], ignore_index=True)   # 50% duplicates
    dirty_df.loc[: len(dirty_df) // 2, "salary"] = np.nan     # heavy missing
    dirty = compute_health(dirty_df)
    assert dirty["dup_pct"] > clean["dup_pct"]
    assert dirty["missing_pct"] > clean["missing_pct"]
    assert dirty["score"] <= clean["score"]


def test_compute_health_matches_profiler_score(hr_df):
    """The headline score must come from the same profiler the main report
    uses — two different scores for one file is a client-visible
    contradiction when both PDFs are delivered together."""
    from app.engines.data_profiler import profile_dataset
    assert compute_health(hr_df)["score"] == max(
        int(round(float(profile_dataset(hr_df).overall_quality_score))), 0)


# ══════════════════════════════════════════════════════════
#  Insight cards
# ══════════════════════════════════════════════════════════

def test_insights_have_the_shape_the_pdf_renders(hr_df):
    insights = build_insights(hr_df, "hr")
    assert insights, "expected at least one insight for a rich HR dataset"
    for ins in insights:
        assert _INSIGHT_KEYS <= set(ins), f"missing keys: {_INSIGHT_KEYS - set(ins)}"
        assert ins["severity"] in _VALID_SEVERITIES
        for key in ("tag", "title", "body", "action"):
            assert isinstance(ins[key], str) and ins[key].strip()


def test_no_nan_or_none_leaks_into_client_facing_text(hr_df):
    """Regression: a text column typed as pandas 3's `str` dtype (not
    `object`) fell through to the numeric branch, so to_numeric("Yes")
    produced all-NaN and the PDF printed "'Engineering' has the highest
    attrition: nan%" to a paying client."""
    for niche in ("hr", "sales", "ecommerce", "finance", "general"):
        for ins in build_insights(hr_df, niche):
            blob = f"{ins['tag']} {ins['title']} {ins['body']} {ins['action']}".lower()
            for bad in ("nan", "none%", "inf%", "nan%"):
                assert bad not in blob, f"{niche}: '{bad}' leaked into: {ins['title']}"


def test_hr_insights_detect_attrition(hr_df):
    titles = " ".join(i["title"].lower() for i in build_insights(hr_df, "hr"))
    assert "attrition" in titles


def test_build_insights_survives_a_minimal_frame():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    for niche in ("hr", "sales", "ecommerce", "finance", "general"):
        assert isinstance(build_insights(df, niche), list)


def test_build_insights_survives_an_all_text_frame():
    df = pd.DataFrame({"name": list("abcdefghij"), "dept": ["x", "y"] * 5})
    assert isinstance(build_insights(df, "hr"), list)


# ══════════════════════════════════════════════════════════
#  Full insight set — domain engines + data-quality cards
# ══════════════════════════════════════════════════════════

def test_full_insights_are_deeper_than_the_quality_cards_alone(signal_sales_df):
    """Regression: the report used to render only this module's own niche
    cards, producing a SINGLE card for a sales dataset even when the domain
    engine had found several critical findings in the same file. A
    one-card report is not something a client will pay for."""
    shallow = build_insights(signal_sales_df, "sales")
    full = build_full_insights(signal_sales_df, "sales")
    assert len(full) > len(shallow), (
        f"full set ({len(full)}) is no deeper than the quality cards "
        f"({len(shallow)}) — the domain engines are not being used")
    assert len(full) >= 3


def test_full_insights_surface_planted_critical_findings(signal_sales_df):
    """The regional gap and the loss-making deals are both real and severe;
    they must appear, and must be ranked critical rather than demoted."""
    full = build_full_insights(signal_sales_df, "sales")
    blob = " ".join(f"{c['title']} {c['body']}" for c in full).lower()
    assert "region" in blob, "the planted regional performance gap was missed"
    assert any(c["severity"] == "critical" for c in full), \
        "no finding ranked critical despite planted severe signal"


def test_full_insights_keep_the_card_contract(signal_sales_df, hr_df):
    for df, niche in [(signal_sales_df, "sales"), (hr_df, "hr")]:
        for c in build_full_insights(df, niche):
            assert _INSIGHT_KEYS <= set(c)
            assert c["severity"] in _VALID_SEVERITIES, (
                f"severity {c['severity']!r} is outside the set the PDF "
                "renders — it would silently fall back to a neutral card")
            for key in ("title", "body", "action"):
                assert isinstance(c[key], str) and c[key].strip()


def test_full_insights_are_ranked_worst_first(signal_sales_df):
    order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
    ranks = [order[c["severity"]] for c in build_full_insights(signal_sales_df, "sales")]
    assert ranks == sorted(ranks), "cards are not ordered by severity"


def test_full_insights_deduplicate_titles(hr_df):
    titles = [c["title"].strip().lower() for c in build_full_insights(hr_df, "hr")]
    assert len(titles) == len(set(titles)), "duplicate insight titles in the report"


def test_full_insights_respect_the_cap(signal_sales_df):
    assert len(build_full_insights(signal_sales_df, "sales", max_cards=2)) <= 2


def test_full_insights_survive_a_degenerate_frame():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    assert isinstance(build_full_insights(df, "general"), list)


# ══════════════════════════════════════════════════════════
#  PDF generation
# ══════════════════════════════════════════════════════════

def test_build_health_pdf_produces_a_real_pdf(hr_df):
    from app.engines.health_pdf_builder import build_health_pdf
    health = compute_health(hr_df)
    insights = build_insights(hr_df, "hr")
    pdf = build_health_pdf(hr_df, "hr", health, insights, "sample.csv",
                            agency_name="Analytiq")
    assert isinstance(pdf, (bytes, bytearray))
    assert pdf[:4] == b"%PDF", "output is not a PDF"
    assert len(pdf) > 20_000, "PDF suspiciously small — likely a blank shell"


def test_build_health_pdf_handles_no_insights(hr_df):
    """An empty insight list must still render a valid report, not crash."""
    from app.engines.health_pdf_builder import build_health_pdf
    pdf = build_health_pdf(hr_df, "general", compute_health(hr_df), [],
                            "sample.csv")
    assert pdf[:4] == b"%PDF"


# ══════════════════════════════════════════════════════════
#  API
# ══════════════════════════════════════════════════════════

def test_health_summary_endpoint(client, uploaded_dataset_id):
    r = client.get(f"/api/reports/{uploaded_dataset_id}/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["health"]["score"] <= 100
    assert body["health"]["grade"]
    assert isinstance(body["insights"], list)
    for ins in body["insights"]:
        assert _INSIGHT_KEYS <= set(ins)


def test_health_pdf_endpoint(client, uploaded_dataset_id):
    r = client.post(f"/api/reports/{uploaded_dataset_id}/health-pdf",
                     json={"agency_name": "Test Agency"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"
    assert len(r.content) > 20_000


def test_health_endpoints_404_for_unknown_dataset(client):
    assert client.get("/api/reports/nope/health").status_code == 404
    assert client.post("/api/reports/nope/health-pdf", json={}).status_code == 404
