"""
Tests for the executive narrative and the report's narrative sections.

The executive summary used to be a template sentence — "This N-row X
dataset analysis identified N critical issue(s) and N risk(s)…" — which
says nothing a reader can't get from the section headings and reads as
filler at the top of a paid deliverable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.story_engine import (
    _first_meaningful_corr,
    _is_tautological_pair,
    generate_story,
)


@pytest.fixture()
def finance_df():
    rng = np.random.default_rng(11)
    n = 480
    cat = rng.choice(["Ops", "Marketing", "R&D", "Admin"], n, p=[.4, .25, .2, .15])
    revenue = (np.linspace(100_000, 70_000, n) + rng.normal(0, 6000, n)).round(2)
    cost = (revenue * np.where(cat == "Marketing", 0.95, 0.62)
            + rng.normal(0, 3000, n)).round(2)
    return pd.DataFrame({
        "invoice_id": range(n),
        "period": pd.date_range("2024-01-01", periods=n),
        "category": cat,
        "revenue": revenue,
        "cost": cost,
        "profit": (revenue - cost).round(2),
        "budget": np.full(n, 85_000.0),
        "opex": (cost * 0.4).round(2),
    })


# ══════════════════════════════════════════════════════════
#  Executive summary quality
# ══════════════════════════════════════════════════════════

def test_exec_summary_is_not_the_old_boilerplate(hr_df, finance_df):
    for df in (hr_df, finance_df):
        summary = generate_story(df).executive_summary.lower()
        assert not (summary.startswith("this") and "dataset analysis" in summary), \
            f"executive summary fell back to the old template: {summary[:120]}"
        assert "critical issue(s)" not in summary


def test_exec_summary_leads_with_a_substantive_claim(hr_df):
    """It should open on the finding, not on a count of findings."""
    summary = generate_story(hr_df).executive_summary
    assert len(summary) > 60
    first_sentence = summary.split(". ")[0].lower()
    assert any(ch.isdigit() for ch in first_sentence), \
        f"opening sentence carries no concrete figure: {first_sentence!r}"


def test_exec_summary_headlines_attrition_when_it_dominates():
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame({
        "employee_id": range(n),
        "department": rng.choice(["Sales", "Engineering", "HR"], n),
        "salary": rng.normal(60_000, 15_000, n).round(0),
        "tenure_years": rng.integers(0, 20, n),
        "satisfaction": rng.uniform(1, 5, n).round(1),
        "attrition": rng.choice(["Yes", "No"], n, p=[.30, .70]),
    })
    summary = generate_story(df).executive_summary.lower()
    assert "attrition" in summary
    assert "%" in summary


def test_exec_summary_never_empty(hr_df):
    assert generate_story(hr_df).executive_summary.strip()


def test_exec_summary_survives_a_minimal_frame():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": ["x", "y", "x", "y"]})
    assert generate_story(df).executive_summary.strip()


# ══════════════════════════════════════════════════════════
#  Correlation quality guards
# ══════════════════════════════════════════════════════════

def test_near_perfect_correlation_is_never_reported():
    """|r| >= 0.99 means the same measurement stored twice (a duplicated
    export column, "actual" copied from "revenue"). Presenting it as a
    discovered relationship tells a client nothing and signals the
    analysis is mechanical."""
    corrs = [
        {"col_a": "revenue", "col_b": "actual", "r": 1.0, "strength": "strong"},
        {"col_a": "spend", "col_b": "signups", "r": 0.74, "strength": "strong"},
    ]
    picked = _first_meaningful_corr(corrs)
    assert picked is not None
    assert picked["col_a"] == "spend", "the duplicated-column pair was reported"


def test_negative_near_perfect_correlation_also_excluded():
    corrs = [{"col_a": "a", "col_b": "b", "r": -0.995, "strength": "strong"}]
    assert _first_meaningful_corr(corrs) is None


def test_tautological_pairs_are_rejected():
    assert _is_tautological_pair("JobLevel", "MonthlyIncome")
    assert _is_tautological_pair("years_at_company", "total_years_experience")
    assert _is_tautological_pair("price", "mrp")
    assert not _is_tautological_pair("marketing_spend", "signups")


def test_exec_summary_does_not_cite_a_perfect_correlation(finance_df):
    """End-to-end: a duplicated column must not reach the summary text."""
    df = finance_df.copy()
    df["actual"] = df["revenue"]          # exact duplicate
    summary = generate_story(df).executive_summary
    assert "r=+1.00" not in summary and "r=1.00" not in summary


# ══════════════════════════════════════════════════════════
#  Narrative sections reach the report payload
# ══════════════════════════════════════════════════════════

def test_report_payload_carries_the_narrative_sections(finance_df):
    """Several domain analyses emit findings/risks/opportunities but no
    insight card — rendering cards alone threw that analysis away."""
    from app.engines.health_engine import build_report_payload
    payload = build_report_payload(finance_df, "finance")

    assert payload["executive_summary"].strip()
    assert payload["key_findings"], "no key findings reached the report"
    assert len(payload["key_findings"]) > len(payload["insights"]) or payload["risks"], (
        "narrative sections add nothing beyond the cards — the finance "
        "engine's margin/budget/concentration analyses are being dropped")
    for key in ("key_findings", "risks", "opportunities", "actions"):
        assert isinstance(payload[key], list)
        assert all(isinstance(x, str) for x in payload[key])


def test_health_pdf_renders_with_narrative_sections(finance_df):
    from app.engines.health_engine import build_report_payload, compute_health
    from app.engines.health_pdf_builder import build_health_pdf
    payload = build_report_payload(finance_df, "finance")
    pdf = build_health_pdf(
        finance_df, "finance", compute_health(finance_df), payload["insights"],
        "finance.csv", agency_name="Analytiq",
        executive_summary=payload["executive_summary"],
        key_findings=payload["key_findings"], risks=payload["risks"],
        opportunities=payload["opportunities"], actions=payload["actions"])
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 30_000


def test_health_pdf_still_builds_without_any_narrative(hr_df):
    """The narrative arguments are optional; older call sites must work."""
    from app.engines.health_engine import build_full_insights, compute_health
    from app.engines.health_pdf_builder import build_health_pdf
    pdf = build_health_pdf(hr_df, "hr", compute_health(hr_df),
                            build_full_insights(hr_df, "hr"), "hr.csv")
    assert pdf[:4] == b"%PDF"


# ══════════════════════════════════════════════════════════
#  Report structure — cover, contents, numbered sections
# ══════════════════════════════════════════════════════════

def _pdf_pages_text(pdf_bytes: bytes):
    pypdf = pytest.importorskip("pypdf")
    import io
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    return [(p.extract_text() or "") for p in reader.pages]


def _build(df, agency="Shweta Analytics"):
    from app.engines.health_engine import build_report_payload, compute_health
    from app.engines.health_pdf_builder import build_health_pdf
    payload = build_report_payload(df, "finance")
    return build_health_pdf(
        df, "finance", compute_health(df), payload["insights"], "acme_q3.csv",
        agency_name=agency, executive_summary=payload["executive_summary"],
        key_findings=payload["key_findings"], risks=payload["risks"],
        opportunities=payload["opportunities"], actions=payload["actions"])


def test_report_opens_with_a_cover_then_contents(finance_df):
    pages = _pdf_pages_text(_build(finance_df))
    assert len(pages) >= 4
    cover = pages[0]
    assert "SHWETA ANALYTICS" in cover.upper()
    assert "Data Health" in cover
    assert "acme_q3.csv" in cover
    assert "Contents" in pages[1]


def test_cover_carries_no_running_header_or_page_number(finance_df):
    """A cover with a running header and a '1/10' badge reads as page one
    of a document, not as a cover."""
    cover = _pdf_pages_text(_build(finance_df))[0]
    assert "CONFIDENTIAL" in cover.upper()   # cover has its own footer line
    assert "1/" not in cover, "page-number badge leaked onto the cover"


def test_contents_lists_only_sections_actually_present(finance_df):
    pages = _pdf_pages_text(_build(finance_df))
    toc = pages[1]
    body = "\n".join(pages[2:])
    for entry in ("Data Health Overview", "Executive Summary",
                  "Descriptive Statistics", "Recommended Actions"):
        assert entry in toc, f"'{entry}' missing from contents"
        assert entry in body, f"contents lists '{entry}' but no such section exists"


def test_section_numbers_match_the_contents_order(finance_df):
    """The heading number is derived from the contents list, so the two
    can't drift apart."""
    pages = _pdf_pages_text(_build(finance_df))
    body = "\n".join(pages[2:])
    assert "01" in body and "Data Health Overview" in body
    assert "02" in body and "Executive Summary" in body


def test_agency_name_brands_the_cover(finance_df):
    cover = _pdf_pages_text(_build(finance_df, agency="Acme Data Co"))[0]
    assert "ACME DATA CO" in cover.upper()


def test_pdf_text_cleaner_escapes_markup_and_markdown():
    """Column names containing & or < would otherwise be parsed as markup
    by reportlab and silently swallow the rest of the paragraph."""
    from app.engines.health_pdf_builder import _clean_text
    out = _clean_text("**Profit** for R&D <segment> rose")
    assert "**" not in out
    assert "&amp;" in out and "&lt;" in out and "&gt;" in out
