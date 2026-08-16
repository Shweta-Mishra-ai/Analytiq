"""
The numbers a reader in each function expects, and the page that shows
them together.

Three KPIs were missing that nobody in the relevant role would leave out.

**Margin over time.** The finance engine reported the margin *level* and
its volatility, and revenue over time — but never margin over time. That
is the question a finance director asks second, because a business can
grow the top line while the margin erodes underneath it and every figure
in the report stays true while the trend that matters goes unsaid.

**The funnel.** Sales had one win rate for the whole pipeline and never
looked at the stage column that produced it. "Win rate is 60%" does not
say whether deals die at qualification or at procurement, and those are
different problems.

**Tenure.** HR measured attrition, pay and engagement and never looked at
tenure as a distribution. A workforce whose mass sits under two years is
a different organisation from one whose mass sits over eight, and the
attrition rate can be identical in both.

The dashboard page is the other half: a reader holding the PDF and not
the app had no way to see the shape of the data at a glance.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════
#  Finance: the margin trend
# ══════════════════════════════════════════════════════════

def _eroding(margin_drift: float = 0.004, revenue_growth: float = 0.02):
    rng = np.random.default_rng(5)
    rows = []
    for i, m in enumerate(pd.date_range("2023-01-31", periods=30, freq="ME")):
        for cc in ("Retail", "Wholesale"):
            rev = rng.normal(9e5, 3e4) * (1 + i * revenue_growth)
            cogs = rev * (0.60 + i * margin_drift)
            rows.append({"period": m, "cost_centre": cc,
                         "revenue": round(rev, 2), "cogs": round(cogs, 2),
                         "gross_profit": round(rev - cogs, 2),
                         "opex": round(rev * .18, 2)})
    return pd.DataFrame(rows)


def _story(df):
    from app.engines.story_engine import generate_story

    return generate_story(df)


def test_margin_erosion_is_reported():
    findings = " ".join(_story(_eroding()).key_findings)
    assert "margin eroded" in findings.lower(), findings
    assert "points" in findings, findings


def test_a_flat_margin_is_not_called_erosion():
    """A margin that moved less than half a point is noise; calling it
    erosion sends someone looking for a cause that is not there."""
    findings = " ".join(_story(_eroding(margin_drift=0.0)).key_findings)
    assert "eroded" not in findings.lower(), findings
    assert "held at" in findings, findings


def test_growth_masking_erosion_is_named():
    """Both halves look healthy on their own — this is the case the
    trend exists to catch."""
    story = _story(_eroding())
    hit = [i for i in story.top_insights if "Margin Down" in i.title]
    assert hit, [i.title for i in story.top_insights]
    assert "thinner margin" in hit[0].evidence, hit[0].evidence


def test_the_level_does_not_contradict_the_trend():
    """"Gross Margin Healthy: 33.7%" beside "Gross Margin Down 8 Points"
    are both true and read as a contradiction."""
    story = _story(_eroding())
    titles = [i.title for i in story.top_insights]
    assert not ("Gross Margin Healthy" in " ".join(titles)
                and "Margin Down" in " ".join(titles)), titles


def test_the_report_leads_with_the_erosion():
    assert "Margin Down" in _story(_eroding()).headline


def test_the_impact_is_arithmetic_not_a_forecast():
    hit = next(i for i in _story(_eroding()).top_insights
               if "Margin Down" in i.title)
    assert any(ch.isdigit() for ch in hit.impact)
    assert "forecast" not in hit.impact.lower()


def test_too_few_periods_says_nothing():
    findings = " ".join(_story(_eroding().head(6)).key_findings)
    assert "eroded" not in findings.lower(), findings


# ══════════════════════════════════════════════════════════
#  Sales: the funnel
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def pipeline():
    rng = np.random.default_rng(3)
    n = 900
    return pd.DataFrame({
        "opportunity_id": range(n),
        "created_date": pd.to_datetime("2024-01-01")
                        + pd.to_timedelta(rng.integers(0, 540, n), "D"),
        "sales_rep": rng.choice(list("ABCDE"), n),
        "deal_stage": rng.choice(
            ["Prospect", "Qualified", "Proposal", "Closed Won", "Closed Lost"],
            n, p=[.25, .20, .15, .22, .18]),
        "deal_amount": rng.lognormal(9.5, .8, n).round(2),
        "quota": 250_000.0,
    })


def test_the_open_pipeline_is_counted_by_stage(pipeline):
    findings = " ".join(_story(pipeline).key_findings)
    assert "still open" in findings, findings
    assert "stages" in findings, findings


def test_closed_deals_are_not_counted_as_open(pipeline):
    """The pipeline is what is undecided; folding won and lost back in
    makes the coverage figure meaningless."""
    story = _story(pipeline)
    hit = next(i for i in story.top_insights if "Pipeline:" in i.title)
    decided = pipeline.deal_stage.isin(["Closed Won", "Closed Lost"]).sum()
    open_n = len(pipeline) - decided
    assert "{:,}".format(open_n) in hit.title, (hit.title, open_n)


def test_the_funnel_names_its_source_column(pipeline):
    hit = next(i for i in _story(pipeline).top_insights
               if "Pipeline:" in i.title)
    assert "deal_stage" in hit.evidence, hit.evidence


def test_the_funnel_does_not_claim_deals_are_stalling(pipeline):
    """A fat stage and a slow stage look identical without entry dates,
    and the remedy differs."""
    hit = next(i for i in _story(pipeline).top_insights
               if "Pipeline:" in i.title)
    assert "stage entry dates" in hit.cause or "entry dates" in hit.action


def test_a_file_with_no_stage_column_gets_no_funnel():
    rng = np.random.default_rng(4)
    n = 300
    df = pd.DataFrame({"sales_rep": rng.choice(list("ABC"), n),
                       "deal_amount": rng.lognormal(9, .5, n),
                       "quota": 100_000.0})
    assert "still open" not in " ".join(_story(df).key_findings)


# ══════════════════════════════════════════════════════════
#  HR: tenure as a distribution
# ══════════════════════════════════════════════════════════

def _workforce(shape: float, scale: float, seed: int = 3):
    rng = np.random.default_rng(seed)
    n = 600
    df = pd.DataFrame({
        "employee_id": range(n),
        "department": rng.choice(["Sales", "Eng", "Ops"], n),
        "salary": rng.normal(60_000, 12_000, n).round(),
        "tenure_years": rng.gamma(shape, scale, n).round(1),
        "manager_id": rng.integers(1, 40, n),
    })
    df["attrition"] = np.where(
        (df.tenure_years < 2) & (rng.random(n) < .5), "Yes", "No")
    return df


def test_tenure_is_reported_as_a_distribution():
    findings = " ".join(_story(_workforce(1.2, 1.4)).key_findings)
    assert "Median tenure" in findings, findings
    assert "under two years" in findings, findings


def test_a_young_workforce_is_flagged():
    story = _story(_workforce(1.2, 1.4))
    assert [i for i in story.top_insights if "Under Two Years" in i.title]


def test_growth_and_churn_are_not_conflated():
    """They produce the same tenure shape and need opposite responses."""
    hit = next(i for i in _story(_workforce(1.2, 1.4)).top_insights
               if "Under Two Years" in i.title)
    assert "grew quickly" in hit.cause and "retaining" in hit.cause


def test_a_long_tenured_workforce_is_a_different_finding():
    story = _story(_workforce(9.0, 1.4))
    joined = " ".join(story.business_risks)
    assert "not being refreshed" in joined or "succession" in joined, joined


def test_a_file_with_no_tenure_column_says_nothing():
    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"employee_id": range(n),
                       "department": rng.choice(["a", "b"], n),
                       "salary": rng.normal(50_000, 8_000, n)})
    assert "Median tenure" not in " ".join(_story(df).key_findings)


# ══════════════════════════════════════════════════════════
#  The dashboard page in the document
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def report():
    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf_builder import build_pdf

    df = _eroding()
    df["budget"] = (df.revenue * 1.06).round(2)
    story = _story(df)
    return df, build_pdf(
        df=df,
        config={"title": "R", "subtitle": "", "client_name": "Acme",
                "confidential": False, "theme_name": "", "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None, stats_report=None,
        bi_report=None, ml_report=None, chart_data=[],
        executive_summary=story.executive_summary,
        findings=story.key_findings, risks=story.business_risks,
        opportunities=story.opportunities,
        recommendations=story.recommended_actions,
        top_insights=story.top_insights, attrition=None, domain=story.domain)


def _page_text(pdf):
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(io.BytesIO(pdf))
    return [(doc[i].get_textpage().get_text_range() or "")
            for i in range(len(doc))]


def test_the_document_carries_a_dashboard_page(report):
    _df, pdf = report
    pages = _page_text(pdf)
    assert any("Dashboard" in p[:400] for p in pages),         [p[:60] for p in pages]


def test_the_dashboard_fits_on_one_page(report):
    """It was splitting across two, leaving each half empty."""
    _df, pdf = report
    pages = _page_text(pdf)
    with_tiles = [i for i, p in enumerate(pages)
                  if "the question it answers" in p]
    assert len(with_tiles) == 1, with_tiles


def test_every_tile_is_captioned_with_its_question(report):
    _df, pdf = report
    page = next(p for p in _page_text(pdf) if "Dashboard" in p[:400])
    assert page.count("?") >= 4, page.count("?")


def test_the_page_matches_the_interactive_dashboard(report):
    """Built from the same spec, so the page and the screen cannot show
    different views of the same file."""
    from app.engines.dashboard_spec import build_spec

    df, pdf = report
    page = next(p for p in _page_text(pdf) if "Dashboard" in p[:400])
    for tile in build_spec(df, "finance", max_tiles=6)[:3]:
        assert tile.question[:30] in page, tile.question


def test_a_comparison_tile_plots_both_measures(report):
    """The PDF path mapped a comparison onto the single-series bar
    builder, so a tile titled "revenue against budget" showed revenue
    alone."""
    from app.engines.chart_exporter import make_comparison_chart

    df, _pdf = report
    png = make_comparison_chart(df, "cost_centre", "revenue", "budget")
    assert png[:4] == b"\x89PNG"


def test_a_file_with_nothing_chartable_omits_the_page():
    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf_builder import build_pdf

    df = pd.DataFrame({"row_id": range(200),
                       "ref": [f"R{i}" for i in range(200)]})
    pdf = build_pdf(
        df=df,
        config={"title": "R", "subtitle": "", "client_name": "A",
                "confidential": False, "theme_name": "", "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None, stats_report=None,
        bi_report=None, ml_report=None, chart_data=[],
        executive_summary="s", findings=[], risks=[], opportunities=[],
        recommendations=[], top_insights=[], attrition=None, domain="general")
    assert not any("the question it answers" in p for p in _page_text(pdf))
