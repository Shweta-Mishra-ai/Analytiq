"""
Every engine against every registered domain, and a full report build for
each.

This is the guardrail for adding domains. The original audit found that a
domain could be detectable without being analysable, that benchmarks could
exist for domains nothing could reach, and that a misrouted dataset would
be analysed by an engine speaking the wrong language — and the unit suite
was green through all of it, because nothing exercised the whole pipeline
per domain.

A new domain added without its engine, blueprint, prompts, theme or
benchmarks fails here rather than in production.
"""
import io

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from app.engines.domains.registry import REGISTRY, detect_domain, run_insights

# Engines every dataset must survive, whatever its domain.
from app.engines.bi_engine import run_bi
from app.engines.data_profiler import profile_dataset
from app.engines.domains.base import col_stats, correlations
from app.engines.eda_engine import run_eda
from app.engines.health_engine import build_report_payload
from app.engines.pdf_builder import build_pdf
from app.engines.stats_engine import analyze as stats_analyze
from app.engines.story_engine import generate_story
from app.engines.chart_exporter import generate_all_charts

N = 300
CONFIG = {"title": "Matrix", "client_name": "Test", "subtitle": "",
          "confidential": True, "theme_name": "", "logo_path": None,
          "prepared_by": "", "source_table": "src"}


def _r():
    return np.random.default_rng(23)


def _hr():
    r = _r()
    return pd.DataFrame({
        "EmployeeNumber": np.arange(N),
        "Attrition": r.choice(["Yes", "No"], N, p=[.16, .84]),
        "JobRole": r.choice(["Sales Rep", "Engineer", "Manager"], N),
        "MonthlyIncome": r.uniform(2500, 19000, N).round(2),
        "JobSatisfaction": r.integers(1, 5, N),
        "OverTime": r.choice(["Yes", "No"], N),
        "YearsAtCompany": r.integers(1, 25, N),
        "EducationField": r.choice(["Life Sciences", "Medical", "Other"], N),
    })


def _ecommerce():
    r = _r()
    return pd.DataFrame({
        "sku": [f"S{i}" for i in range(N)],
        "product_name": r.choice(["Widget", "Gadget"], N),
        "discounted_price": r.uniform(50, 5000, N).round(2),
        "actual_price": r.uniform(60, 6000, N).round(2),
        "discount_percentage": r.uniform(0, .7, N).round(3),
        "rating": r.uniform(1, 5, N).round(1),
        "rating_count": r.integers(0, 5000, N),
        "category": r.choice(["A", "B", "C"], N),
        "fulfilment": r.choice(["FBA", "FBM"], N),
    })


def _sales():
    r = _r()
    return pd.DataFrame({
        "opportunity_id": np.arange(N),
        "sales_rep": r.choice(["A", "B", "C"], N),
        "territory": r.choice(["West", "East", "North"], N),
        "deal_size": r.uniform(1e3, 2e5, N).round(2),
        "stage": r.choice(["Won", "Lost", "Open"], N),
        "quota": r.uniform(5e4, 3e5, N).round(2),
        "revenue": r.uniform(0, 2e5, N).round(2),
        "lead_source": r.choice(["inbound", "outbound"], N),
        "forecast": r.uniform(0, 2e5, N).round(2),
    })


def _finance():
    r = _r()
    return pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=N, freq="D"),
        "department": r.choice(["Ops", "Sales", "R&D"], N),
        "revenue": r.uniform(1e4, 9e5, N).round(2),
        "operating_expense": r.uniform(5e3, 4e5, N).round(2),
        "gross_margin": r.uniform(.05, .6, N).round(3),
        "ebitda": r.uniform(-1e4, 3e5, N).round(2),
        "cash_flow": r.uniform(-5e4, 4e5, N).round(2),
        "receivable": r.uniform(0, 2e5, N).round(2),
        "budget_variance": r.uniform(-.4, .4, N).round(3),
    })


def _marketing():
    r = _r()
    imp = r.integers(5000, 90000, N)
    clicks = np.maximum((imp * r.uniform(.004, .03, N)).astype(int), 1)
    return pd.DataFrame({
        "campaign_id": np.arange(N),
        "channel": r.choice(["Search", "Social", "Email", "Display"], N),
        "impressions": imp, "clicks": clicks,
        "conversions": np.maximum((clicks * r.uniform(.01, .08, N)).astype(int), 0),
        "spend": (clicks * r.uniform(.5, 3.0, N)).round(2),
        "revenue": r.uniform(1e3, 9e4, N).round(2),
        "ctr": (clicks / imp).round(4),
        "roas": r.uniform(.2, 8, N).round(2),
    })


def _saas():
    r = _r()
    return pd.DataFrame({
        "account_id": np.arange(N),
        "plan": r.choice(["free", "pro", "enterprise"], N),
        "churned": r.choice([0, 1], N, p=[.85, .15]),
        "mrr": r.uniform(20, 3000, N).round(2),
        "arr": r.uniform(240, 36000, N).round(2),
        "seats": r.integers(1, 250, N),
        "nps": r.integers(-100, 100, N),
        "tenure_months": r.integers(1, 60, N),
        "expansion_revenue": r.uniform(0, 200, N).round(2),
        "active_users": r.integers(1, 200, N),
    })


def _operations():
    r = _r()
    return pd.DataFrame({
        "work_order_id": np.arange(N),
        "cycle_time_hours": r.lognormal(3, .8, N).round(1),
        "defect_rate": r.uniform(.01, .09, N).round(4),
        "throughput": r.integers(20, 800, N),
        "downtime_minutes": r.integers(0, 500, N),
        "utilization": r.uniform(.60, .99, N).round(3),
        "on_time_delivery": r.choice([0, 1], N, p=[.2, .8]),
        "inventory_turns": r.uniform(2, 14, N).round(2),
        "plant": r.choice(["Pune", "Chennai", "Noida"], N),
    })


def _healthcare():
    r = _r()
    return pd.DataFrame({
        "patient_id": np.arange(N),
        "length_of_stay": np.maximum(r.lognormal(1.4, .9, N).astype(int), 1),
        "readmission_30d": r.choice([0, 1], N, p=[.83, .17]),
        "bed_occupancy": r.uniform(.60, .99, N).round(3),
        "cost_per_case": r.lognormal(9, .7, N).round(2),
        "department": r.choice(["Cardiology", "Oncology", "ICU", "Ortho"], N),
        "admission_type": r.choice(["Emergency", "Elective"], N),
        "diagnosis_code": r.choice(["A01", "B02", "C03"], N),
    })


def _general():
    r = _r()
    return pd.DataFrame({
        "alpha": r.normal(50, 12, N).round(2),
        "beta": r.uniform(0, 500, N).round(2),
        "gamma": r.integers(1, 100, N),
        "grouping": r.choice(["one", "two", "three"], N),
    })


BUILDERS = {
    "hr": _hr, "ecommerce": _ecommerce, "sales": _sales, "finance": _finance,
    "marketing": _marketing, "saas": _saas, "operations": _operations,
    "healthcare": _healthcare, "general": _general,
}

ENGINES = {
    "profile": lambda df: profile_dataset(df),
    "stats":   lambda df: stats_analyze(df),
    "bi":      lambda df: run_bi(df),
    "eda":     lambda df: run_eda(df),
    "story":   lambda df: generate_story(df),
    "charts":  lambda df: generate_all_charts(df, "Corporate Light", 5),
}


def test_every_registered_domain_has_a_matrix_fixture():
    """Adding a domain without adding its fixture here would leave it
    untested end to end — which is how the original defects survived."""
    missing = sorted(set(REGISTRY) - set(BUILDERS))
    assert not missing, (
        f"domains registered but not covered by the matrix: {missing}. "
        f"Add a fixture to BUILDERS in this file.")


@pytest.mark.parametrize("domain", sorted(BUILDERS))
@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_engine_runs_on_every_domain(domain, engine):
    """No engine may crash on any domain's data."""
    ENGINES[engine](BUILDERS[domain]())


@pytest.mark.parametrize("domain", sorted(set(BUILDERS) - {"general"}))
def test_each_fixture_detects_as_its_own_domain(domain):
    detected, conf = detect_domain(BUILDERS[domain]())
    assert detected == domain, f"{domain} fixture detected as {detected}"
    assert conf > 0.4


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_domain_engine_returns_the_agreed_shape(domain):
    df = BUILDERS[domain]()
    stats = {c: col_stats(df[c]) for c in df.select_dtypes("number").columns}
    raw = run_insights(domain, df, {k: v for k, v in stats.items() if v},
                       correlations(df))
    assert set(raw) >= {"findings", "risks", "opportunities", "actions",
                        "insights"}


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_full_report_builds_for_every_domain(domain):
    """The end-to-end path: detect, analyse, chart, render. This is what
    the user actually receives."""
    df = BUILDERS[domain]()
    story = generate_story(df)
    charts = [(t, b, "Chart narrative.")
              for t, b, _spec in generate_all_charts(df, "Corporate Light", 3) if b]
    pdf = build_pdf(
        df=df, config=dict(CONFIG), profile=profile_dataset(df),
        stats_report=stats_analyze(df), bi_report=run_bi(df),
        chart_data=charts, executive_summary=story.executive_summary,
        findings=story.key_findings, risks=story.business_risks,
        opportunities=story.opportunities,
        recommendations=story.recommended_actions,
        top_insights=story.top_insights, attrition=story.attrition,
        domain=domain)
    assert pdf[:5] == b"%PDF-"
    pages = PdfReader(io.BytesIO(pdf)).pages
    assert len(pages) >= 5, f"{domain} report is only {len(pages)} pages"


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_report_prose_never_contains_nan_or_infinity(domain):
    """'nan%' in a client report is the tell that a computation failed and
    nothing checked."""
    import re
    df = BUILDERS[domain]()
    story = generate_story(df)
    pdf = build_pdf(df=df, config=dict(CONFIG), profile=profile_dataset(df),
                    executive_summary=story.executive_summary,
                    findings=story.key_findings, risks=story.business_risks,
                    top_insights=story.top_insights, domain=domain)
    text = "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf)).pages).lower()
    for token in ("nan%", "nan ", "-inf", "infinity%"):
        assert token not in text, f"{domain} report printed {token!r}"
    assert not re.search(r"\b\d{5,}\s*%", text), \
        f"{domain} report printed an absurd percentage"


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_health_report_builds_for_every_domain(domain):
    payload = build_report_payload(BUILDERS[domain](), domain)
    assert isinstance(payload, dict)


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_report_is_labelled_with_the_domain_it_was_analysed_as(domain):
    """The specific marketing failure: a report headed 'marketing' whose
    findings came from the general engine. Label and analysis must agree."""
    from app.engines.report_blueprints import blueprint_for
    df = BUILDERS[domain]()
    story = generate_story(df)
    pdf = build_pdf(df=df, config=dict(CONFIG), profile=profile_dataset(df),
                    top_insights=story.top_insights, domain=domain)
    text = "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf)).pages)
    label = blueprint_for(domain).label
    assert label.split()[0] in text, \
        f"{domain} report does not carry its own blueprint label {label!r}"


@pytest.mark.parametrize("domain", sorted(set(BUILDERS) - {"hr"}))
def test_attrition_analysis_only_runs_for_hr(domain):
    """SaaS and healthcare data both used to land on the HR engine and come
    back with an employee attrition rate."""
    assert generate_story(BUILDERS[domain]()).attrition is None
