"""
The report sections added when the PDF layer was brought up to dataforge-ai:
Report at a Glance, Predictive Risk Analysis, and per-domain deep pages.

The predictive section is the notable one. predictive.py has been in the
codebase throughout, but build_pdf had no parameter to receive its output,
so the engine ran and the result was discarded. These tests keep it wired.
"""
import io

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from app.engines.pdf_builder import build_pdf
from app.engines.predictive import (
    compute_drivers, find_binary_target, find_top_cluster,
)

CONFIG = {"title": "Section Check", "client_name": "Test", "subtitle": "",
          "confidential": True, "theme_name": "", "logo_path": None,
          "prepared_by": "", "source_table": "src"}


def _text(pdf_bytes):
    return "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf_bytes)).pages)


@pytest.fixture(scope="module")
def churn_df():
    """Attrition driven by two real factors, so the model has something to
    find rather than fitting noise."""
    r = np.random.default_rng(11)
    n = 900
    overtime = r.choice(["Yes", "No"], n, p=[.35, .65])
    tenure = r.integers(1, 20, n)
    role = r.choice(["Sales Rep", "Engineer", "Manager"], n)
    p = (.06 + .30 * (overtime == "Yes") + .22 * (tenure <= 2)).clip(0, .92)
    return pd.DataFrame({
        "employee_number": np.arange(n),
        "Attrition": np.where(r.random(n) < p, "Yes", "No"),
        "OverTime": overtime, "YearsAtCompany": tenure, "JobRole": role,
        "MonthlyIncome": r.uniform(2500, 19000, n).round(2),
        "JobSatisfaction": r.integers(1, 5, n),
    })


@pytest.fixture(scope="module")
def finance_df():
    r = np.random.default_rng(5)
    n = 300
    return pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=n, freq="D"),
        "department": r.choice(["Ops", "Sales", "R&D"], n),
        "revenue": r.uniform(5e4, 4e5, n),
        "cost": r.uniform(2e4, 2e5, n),
        "gross_profit": r.uniform(1e4, 2e5, n),
        "budget": r.uniform(5e4, 4e5, n),
        "operating_expense": r.uniform(1e4, 9e4, n),
        "ebitda": r.uniform(-1e4, 1e5, n),
    })


@pytest.fixture(scope="module")
def drivers(churn_df):
    target = find_binary_target(churn_df)
    assert target == "Attrition"
    return compute_drivers(churn_df, target), find_top_cluster(churn_df, target)


# ── Report at a Glance ────────────────────────────────────

def test_glance_page_is_always_present(finance_df):
    """It is the page an executive reads first; it must not depend on
    optional inputs being supplied."""
    text = _text(build_pdf(df=finance_df, config=CONFIG, domain="finance"))
    assert "Report at a Glance" in text
    assert "RECORDS ANALYSED" in text
    assert "Scope of Analysis" in text


def test_glance_page_is_listed_in_the_contents(finance_df):
    text = _text(build_pdf(df=finance_df, config=CONFIG, domain="finance"))
    assert text.count("Report at a Glance") >= 2, \
        "section is in the report but missing from the contents page"


# ── Predictive Risk ───────────────────────────────────────

def test_predictive_section_reaches_the_report(churn_df, drivers):
    dr, cluster = drivers
    assert dr is not None and dr.top_drivers, "no drivers computed"
    text = _text(build_pdf(df=churn_df, config=CONFIG, domain="hr",
                           predictive=dr, top_cluster=cluster))
    assert "Predictive Risk Analysis" in text
    assert "MODEL AUC" in text
    assert "Top Predictive Drivers" in text


def test_predictive_section_is_absent_when_no_model_was_built(churn_df):
    """Omitted, not rendered empty — a heading over nothing is worse than
    no heading."""
    text = _text(build_pdf(df=churn_df, config=CONFIG, domain="hr"))
    assert "Predictive Risk Analysis" not in text


def test_model_finds_the_planted_drivers(drivers):
    dr, _ = drivers
    names = " ".join(str(d[0]).lower() for d in dr.top_drivers)
    assert "overtime" in names or "yearsatcompany" in names, \
        f"model missed both planted drivers: {dr.top_drivers}"
    assert dr.auc > 0.65, f"model barely separates: AUC {dr.auc}"


def test_predictive_section_never_prints_nan(churn_df, drivers):
    dr, cluster = drivers
    text = _text(build_pdf(df=churn_df, config=CONFIG, domain="hr",
                           predictive=dr, top_cluster=cluster))
    i = text.find("Predictive Risk Analysis")
    assert "nan" not in text[i:].lower().replace("finance", "")


def test_unit_cost_is_labelled_as_an_assumption(churn_df, drivers):
    """Costing avoidable events is only honest if the assumed unit cost is
    presented as an assumption rather than a measurement."""
    dr, cluster = drivers
    text = _text(build_pdf(df=churn_df, config=CONFIG, domain="hr",
                           predictive=dr, top_cluster=cluster,
                           avg_salary_k=60.0))
    assert "assumption" in text.lower()


# ── Domain deep pages ─────────────────────────────────────

def test_finance_deep_page_renders(finance_df):
    text = _text(build_pdf(df=finance_df, config=CONFIG, domain="finance"))
    assert "Finance Analysis" in text
    assert "P&L Summary" in text
    assert "Gross Profit" in text


def test_finance_page_escapes_its_ampersands(finance_df):
    """ReportLab parses Paragraph text as markup, so a bare '&' renders as
    'P&L;' on the page of a finance report."""
    text = _text(build_pdf(df=finance_df, config=CONFIG, domain="finance"))
    assert "P&L;" not in text


def test_deep_page_only_appears_for_domains_that_have_one(finance_df):
    """A domain without a deep page must not get an empty section."""
    text = _text(build_pdf(df=finance_df, config=CONFIG, domain="general"))
    assert "Finance Analysis" not in text


def test_deep_pages_are_registered_not_hardcoded():
    """Wired through DomainSpec.deep_page so adding one for a new domain
    stays a one-file change."""
    from app.engines.domains.registry import spec_for
    import app.engines.pdf.domain_sections  # noqa: F401  (attaches the page)
    assert spec_for("finance").deep_page is not None
    assert spec_for("general").deep_page is None


def test_attaching_a_deep_page_to_an_unknown_domain_fails_loudly():
    from app.engines.domains.registry import attach_deep_page
    with pytest.raises(KeyError):
        attach_deep_page("not_a_domain", lambda *a, **k: None)


# ── The whole report still holds together ─────────────────

def test_report_grows_when_the_new_sections_are_supplied(churn_df, drivers):
    dr, cluster = drivers
    without = build_pdf(df=churn_df, config=CONFIG, domain="hr")
    with_pred = build_pdf(df=churn_df, config=CONFIG, domain="hr",
                          predictive=dr, top_cluster=cluster)
    n_without = len(PdfReader(io.BytesIO(without)).pages)
    n_with = len(PdfReader(io.BytesIO(with_pred)).pages)
    assert n_with > n_without, "predictive section added no pages"


def test_existing_callers_still_work_without_the_new_arguments(finance_df):
    """Every added parameter is optional; a pre-existing call must build."""
    pdf = build_pdf(
        df=finance_df, config=CONFIG, profile=None, cleaning_summary=None,
        stats_report=None, bi_report=None, ml_report=None, chart_data=None,
        executive_summary="", findings=[], risks=[], opportunities=[],
        recommendations=[], top_insights=[], attrition=None,
        domain="finance")
    assert pdf[:5] == b"%PDF-"


@pytest.mark.parametrize("cols", [
    {"revenue": "num"},                                  # revenue only
    {"revenue": "num", "cost": "num"},                   # revenue and cost
    {"revenue": "num", "budget": "num", "cat": "str"},   # no cost column
    {"x": "num"},                                        # nothing financial
])
def test_finance_page_survives_partial_finance_data(cols):
    """The gross-profit highlight was written as
    `(...) if cost_col else ("",)`, and a bare ("",) is not a ReportLab
    style command — every dataset without a cost column lost the entire
    finance page to an unpack error that only showed up in a warning log.
    """
    from app.engines.pdf.domain_sections import _finance_page
    from app.engines.pdf.theme import THEMES, _styles

    r = np.random.default_rng(4)
    n = 120
    df = pd.DataFrame({
        name: (r.uniform(1e3, 9e4, n) if kind == "num"
               else r.choice(["A", "B"], n))
        for name, kind in cols.items()
    })
    T = THEMES["Corporate Light"]
    flowables = []
    _finance_page(flowables, _styles(T), T, df, {}, 500.0, profile=None)
    assert flowables, "finance page produced nothing"


def test_glance_page_applies_the_insight_guard():
    """The guard withholds unsupported claims from the findings section.
    The glance page renders insight titles directly, so it has to run the
    same guard or the claim is withheld in one place and printed in the
    other — on the page most likely to be read on its own."""
    class _Ins:
        def __init__(self, title, problem, evidence):
            self.title, self.problem, self.evidence = title, problem, evidence
            self.cause = self.action = self.impact = ""
            self.severity, self.category = "high", "test"

    df = pd.DataFrame({"revenue": np.random.default_rng(1).normal(1e4, 1e3, 150),
                       "category": ["A", "B"] * 75})
    speculative = _Ins("Revenue will collapse next quarter",
                       "Revenue is 10,000 and will collapse next quarter.",
                       "n=150")
    text = _text(build_pdf(df=df, config=CONFIG, domain="finance",
                           top_insights=[speculative]))
    assert "will collapse next quarter" not in text
