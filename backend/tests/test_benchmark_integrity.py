"""
Benchmarks: attributed, domain-locked, and never invented.

The benchmark section used to be three hardcoded tables citing
"Salesforce 2024", "Gartner 2024", "Klaviyo 2024" against specific
figures — precise-looking attributions to reports whose contents cannot
be checked from inside this app. A reader who looks one up and cannot
find the number stops believing the whole document, which is a worse
outcome than having shown no benchmark at all.

Two rules are enforced here:

  1. Every range names a source, and the source is a real, checkable
     body or convention rather than a vendor-and-year label.
  2. A report cites its own domain and no other. An HR report showing a
     finance ratio, or a finance report citing SHRM, tells the reader the
     tool does not know what it is looking at.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pypdf
import pytest

from app.engines.industry_benchmarks import (DOMAIN_BENCHMARKS,
                                             format_benchmark_context,
                                             lookup_benchmark)


# ══════════════════════════════════════════════════════════
#  The library itself
# ══════════════════════════════════════════════════════════

def test_every_range_names_a_source():
    unattributed = [
        f"{domain}/{key}"
        for domain, table in DOMAIN_BENCHMARKS.items()
        for key, bm in table.items()
        if not bm.source.strip()
    ]
    assert not unattributed, unattributed


def test_no_range_is_attributed_to_a_vendor_and_year():
    """"Salesforce 2024" reads as a citation and is not one — there is no
    document behind it that a reader could open."""
    import re
    suspect = []
    for domain, table in DOMAIN_BENCHMARKS.items():
        for key, bm in table.items():
            if re.search(r"\b(19|20)\d{2}\b", bm.source):
                suspect.append(f"{domain}/{key}: {bm.source}")
    assert not suspect, (
        "sources carrying a year read as specific published editions:\n"
        + "\n".join(suspect))


def test_every_range_is_ordered_and_plausible():
    for domain, table in DOMAIN_BENCHMARKS.items():
        for key, bm in table.items():
            assert bm.low <= bm.high, f"{domain}/{key}"
            if bm.unit == "%":
                # Above 100% is legitimate and important: net revenue
                # retention over 100 means the existing base grows on its
                # own. The bound only has to catch nonsense.
                assert -100 <= bm.low <= 1000, f"{domain}/{key}"
                assert -100 <= bm.high <= 1000, f"{domain}/{key}"


def test_every_range_explains_itself():
    """A range with no note is a number with no meaning."""
    for domain, table in DOMAIN_BENCHMARKS.items():
        for key, bm in table.items():
            assert len(bm.note) > 20, f"{domain}/{key}: {bm.note!r}"


def test_the_library_covers_the_domains_the_app_detects():
    for domain in ("hr", "sales", "finance", "ecommerce"):
        assert domain in DOMAIN_BENCHMARKS
    # and the ones added for wider consulting work
    for domain in ("marketing", "operations", "saas", "healthcare"):
        assert domain in DOMAIN_BENCHMARKS
        assert len(DOMAIN_BENCHMARKS[domain]) >= 5


# ══════════════════════════════════════════════════════════
#  Domain lock
# ══════════════════════════════════════════════════════════

def test_a_lookup_never_crosses_domains():
    """"attrition" is an HR measure. Asking the finance table for it must
    return nothing rather than the nearest finance ratio."""
    assert lookup_benchmark("hr", "attrition_rate") is not None
    assert lookup_benchmark("finance", "attrition_rate") is None
    assert lookup_benchmark("finance", "gross_margin") is not None
    assert lookup_benchmark("hr", "gross_margin") is None


def test_an_unknown_domain_returns_nothing():
    assert lookup_benchmark("astrology", "attrition_rate") is None
    assert lookup_benchmark("", "gross_margin") is None


def test_a_column_with_no_match_returns_nothing():
    """Guessing is worse than staying quiet — a wrong benchmark is read
    as a real comparison."""
    assert lookup_benchmark("hr", "employee_favourite_colour") is None


def test_format_includes_the_range_and_the_caveat():
    bm = lookup_benchmark("hr", "attrition_rate")
    text = format_benchmark_context(bm)
    assert "10-15%" in text
    assert "varies" in text.lower()


# ══════════════════════════════════════════════════════════
#  What reaches the report
# ══════════════════════════════════════════════════════════

def _pdf_text(df, domain):
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
        recommendations=story.recommended_actions, top_insights=[],
        attrition=None, domain=domain,
    )
    return "\n".join((p.extract_text() or "")
                     for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)


@pytest.fixture()
def hr_df():
    rng = np.random.default_rng(101)
    n = 300
    return pd.DataFrame({
        "employee_id": range(n),
        "department": rng.choice(["Sales", "Eng", "Ops"], n),
        "attrition_rate": rng.uniform(0.10, 0.25, n).round(3),
        "satisfaction_score": rng.uniform(3.0, 4.6, n).round(2),
        "salary": rng.normal(60_000, 9_000, n).round(0),
    })


@pytest.fixture()
def finance_df():
    rng = np.random.default_rng(102)
    n = 300
    return pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=n, freq="D"),
        "gross_margin": rng.uniform(25, 55, n).round(2),
        "opex_ratio": rng.uniform(55, 85, n).round(2),
        "revenue": rng.normal(50_000, 8_000, n).round(2),
    })


def test_an_hr_report_shows_hr_ranges_only(hr_df):
    text = _pdf_text(hr_df, "hr")
    assert "Performance Against Published Ranges" in text
    assert "attrition_rate" in text
    for foreign in ("gross_margin", "opex_ratio", "cart_abandonment",
                    "on_time_delivery"):
        assert foreign not in text, foreign


def test_a_finance_report_shows_finance_ranges_only(finance_df):
    text = _pdf_text(finance_df, "finance")
    assert "gross_margin" in text
    for foreign in ("attrition", "satisfaction_score", "SHRM", "Gallup"):
        assert foreign not in text, foreign


def test_the_report_states_what_the_ranges_are_worth(hr_df):
    text = _pdf_text(hr_df, "hr").lower()
    assert "not a licensed benchmark set" in text
    assert "prompt to look, not a finding" in text


def test_sources_are_named_per_row(hr_df):
    text = _pdf_text(hr_df, "hr")
    assert "SHRM" in text
    assert "Source" in text


def test_no_vendor_year_citations_reach_the_report(hr_df, finance_df):
    """These were in the old hardcoded tables."""
    for df, domain in ((hr_df, "hr"), (finance_df, "finance")):
        text = _pdf_text(df, domain)
        for fake in ("Salesforce 2024", "Gartner 2024", "HubSpot 2024",
                     "Klaviyo 2024", "BigCommerce 2024", "Forrester 2024",
                     "Shopify 2024", "Amazon/G2 2024"):
            assert fake not in text, fake


def test_a_dataset_with_no_benchmarkable_column_omits_the_section():
    """An empty benchmark table is worse than none — it advertises a
    comparison the report cannot make."""
    rng = np.random.default_rng(103)
    df = pd.DataFrame({
        "employee_id": range(200),
        "favourite_colour": rng.choice(["red", "blue"], 200),
        "desk_number": rng.integers(1, 300, 200),
    })
    text = _pdf_text(df, "hr")
    assert "Performance Against Published Ranges" not in text


def test_the_position_against_the_range_is_stated(hr_df):
    """A number beside a range, with nothing saying which side it falls
    on, makes the reader do the arithmetic."""
    text = _pdf_text(hr_df, "hr")
    assert ("within the range" in text or "above the range" in text
            or "below the range" in text)
