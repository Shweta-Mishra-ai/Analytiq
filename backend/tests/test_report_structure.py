"""
A report should be structured for the function that reads it.

Every report the app produced had one skeleton — executive summary, data
quality, a flat "Top Insights" list, statistics, charts, recommendations
— whatever the data was about. A finance director reads position, then
cost structure, then variance; an HR director reads workforce profile,
then attrition. One ordering cannot serve both, and the one that served
neither read as a tool's default output rather than as a piece of work.

These tests check the structure, not the prose: that the sections a
domain's reader expects appear, in that domain's order, with findings
under the right headings, and that an HR report does not cite finance
conventions or the reverse.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pypdf
import pytest

from app.engines.report_blueprints import (BLUEPRINTS, GENERAL,
                                           blueprint_for, group_insights)


class _Insight:
    def __init__(self, category, title="A finding worth 12%"):
        self.category = category
        self.title = title
        self.problem = "Something measurable happened."
        self.cause = "What is established, and what is not."
        self.evidence = "n=300, p=0.01"
        self.action = "1. Do the first thing  2. Then the second"
        self.impact = "What it is worth."
        self.severity = "high"
        self.confidence = "High"


# ══════════════════════════════════════════════════════════
#  Blueprints
# ══════════════════════════════════════════════════════════

def test_every_domain_has_a_blueprint():
    for domain in ("finance", "hr", "sales", "ecommerce", "general"):
        assert blueprint_for(domain).domain == domain


def test_an_unknown_domain_falls_back_to_general():
    assert blueprint_for("astrology") is GENERAL
    assert blueprint_for("") is GENERAL
    assert blueprint_for(None) is GENERAL


def test_domains_do_not_share_a_section_order():
    """If two domains produce the same headings, the blueprint is doing
    nothing."""
    orders = {
        name: tuple(s.title for s in bp.sections)
        for name, bp in BLUEPRINTS.items() if name != "general"
    }
    assert len(set(orders.values())) == len(orders), \
        "two domains produce an identical report structure"


def test_each_blueprint_opens_with_basis_and_summary():
    """Scope before findings is the convention every review deliverable
    follows, and it is what stops a figure being read out of context."""
    for name, bp in BLUEPRINTS.items():
        assert bp.sections[0].key == "basis", name
        assert bp.sections[1].key == "summary", name


def test_each_blueprint_closes_with_recommendations_then_method():
    for name, bp in BLUEPRINTS.items():
        assert bp.sections[-2].key == "recommendations", name
        assert bp.sections[-1].key == "method", name


def test_section_titles_use_the_domain_vocabulary():
    hr = [s.title for s in blueprint_for("hr").sections]
    finance = [s.title for s in blueprint_for("finance").sections]
    assert "Workforce Profile" in hr
    assert "Attrition & Retention" in hr
    assert "Financial Position" in finance
    assert "Cost Structure & Operating Leverage" in finance
    assert "Workforce Profile" not in finance


def test_every_section_states_its_purpose():
    """A heading with no statement of what it is for reads as a label."""
    for name, bp in BLUEPRINTS.items():
        for section in bp.sections:
            assert section.purpose.strip(), f"{name}/{section.key}"
            assert len(section.purpose) > 25, f"{name}/{section.key}"


# ══════════════════════════════════════════════════════════
#  Grouping findings
# ══════════════════════════════════════════════════════════

def test_findings_land_in_their_domain_section():
    bp = blueprint_for("finance")
    grouped = group_insights(bp, [
        _Insight("finance_budget"),
        _Insight("finance_margin"),
        _Insight("finance_structure"),
    ])
    titles = [section.title for section, _items in grouped]
    assert titles == ["Financial Position",
                      "Cost Structure & Operating Leverage",
                      "Budget Variance"], titles


def test_grouping_follows_the_blueprint_order_not_the_input_order():
    bp = blueprint_for("finance")
    grouped = group_insights(bp, [
        _Insight("finance_budget"),      # third section
        _Insight("finance_margin"),      # first section
    ])
    assert grouped[0][0].key == "position"


def test_an_unmatched_finding_is_never_dropped():
    """A finding the report does not print is worse than one under a
    slightly wrong heading."""
    bp = blueprint_for("finance")
    grouped = group_insights(bp, [
        _Insight("finance_margin"),
        _Insight("something_the_blueprint_never_heard_of"),
    ])
    kept = [i for _s, items in grouped for i in items]
    assert len(kept) == 2


def test_unmatched_findings_survive_even_with_no_matching_section():
    bp = blueprint_for("finance")
    grouped = group_insights(bp, [_Insight("totally_unknown")])
    kept = [i for _s, items in grouped for i in items]
    assert len(kept) == 1


def test_empty_sections_are_not_emitted():
    """A section with nothing to say is padding, and padding is what makes
    a report feel generated."""
    bp = blueprint_for("finance")
    grouped = group_insights(bp, [_Insight("finance_margin")])
    assert len(grouped) == 1


def test_no_finding_appears_twice():
    bp = blueprint_for("ecommerce")
    # "rating" is listed under two sections in the ecommerce blueprint
    grouped = group_insights(bp, [_Insight("rating")])
    kept = [i for _s, items in grouped for i in items]
    assert len(kept) == 1


def test_grouping_handles_no_findings():
    assert group_insights(blueprint_for("hr"), []) == []


# ══════════════════════════════════════════════════════════
#  It reaches the PDF
# ══════════════════════════════════════════════════════════

def _pdf_text(df, domain, insights):
    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf_builder import build_pdf
    from app.engines.story_engine import generate_story

    story = generate_story(df)
    pdf = build_pdf(
        df=df,
        config={"title": "Review", "subtitle": "", "client_name": "Acme",
                "confidential": True, "theme_name": "", "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None,
        stats_report=None, bi_report=None, ml_report=None, chart_data=[],
        executive_summary=story.executive_summary, findings=story.key_findings,
        risks=story.business_risks, opportunities=story.opportunities,
        recommendations=story.recommended_actions, top_insights=insights,
        attrition=None, domain=domain,
    )
    return "\n".join((p.extract_text() or "")
                     for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)


@pytest.fixture()
def small_df():
    rng = np.random.default_rng(90)
    n = 120
    return pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=n, freq="D"),
        "category": rng.choice(["A", "B", "C"], n),
        "revenue": rng.normal(10_000, 1_500, n).round(2),
        "cost": rng.normal(6_000, 900, n).round(2),
    })


def test_a_finance_report_uses_finance_headings(small_df):
    text = _pdf_text(small_df, "finance", [
        _Insight("finance_margin"), _Insight("finance_structure")])
    assert "Financial Performance Review" in text
    assert "Financial Position" in text
    assert "Cost Structure" in text
    assert "Workforce Profile" not in text


def test_an_hr_report_uses_hr_headings(small_df):
    text = _pdf_text(small_df, "hr", [
        _Insight("attrition"), _Insight("compensation")])
    assert "Workforce Analytics Review" in text
    assert "Attrition & Retention" in text
    assert "Compensation & Equity" in text
    assert "Financial Position" not in text


def test_a_finance_report_does_not_cite_hr_bodies(small_df):
    """SHRM and Gallup were in the appendix of every report regardless of
    what it was about."""
    text = _pdf_text(small_df, "finance", [_Insight("finance_margin")])
    for body in ("SHRM", "Gallup", "Mercer"):
        assert body not in text, body
    assert "IFRS" in text or "CFA" in text


def test_an_hr_report_does_not_cite_accounting_standards(small_df):
    text = _pdf_text(small_df, "hr", [_Insight("attrition")])
    assert "IFRS" not in text
    assert "SHRM" in text or "Gallup" in text


def test_the_references_say_what_they_are_worth(small_df):
    """A reader asks "against what?" and the honest answer for most of
    these is a convention, not a licensed dataset."""
    text = _pdf_text(small_df, "hr", [_Insight("attrition")])
    assert "sector" in text.lower()
    assert "never a target" in text.lower() or "indicative" in text.lower()


def test_a_report_with_no_findings_says_so_rather_than_going_quiet(small_df):
    text = _pdf_text(small_df, "general", [])
    assert "not an omission" in text or "found nothing it could support" in text
