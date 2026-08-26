"""
Insight engines for the expansion domains: marketing, SaaS, operations,
healthcare.

Each of these datasets used to be routed to an engine that spoke the wrong
language — SaaS and healthcare to HR, operations to e-commerce, marketing
to the general engine under a "marketing" heading. These tests check the
engines produce domain-appropriate output and, just as importantly, do not
borrow another domain's vocabulary.
"""
import re

import numpy as np
import pandas as pd
import pytest

from app.engines.domains.base import col_stats, correlations
from app.engines.domains.registry import REGISTRY, detect_domain, run_insights
from app.engines.story_engine import generate_story


def _stats_and_corrs(df):
    stats = {c: col_stats(df[c]) for c in df.select_dtypes("number").columns}
    return {k: v for k, v in stats.items() if v}, correlations(df)


def _marketing(n=500):
    r = np.random.default_rng(7)
    imp = r.integers(5000, 90000, n)
    channel = r.choice(["Search", "Social", "Email", "Display"], n)
    # Display is deliberately inefficient so the waste rule has something
    # real to find.
    mult = np.where(channel == "Display", 0.15, 1.0)
    clicks = np.maximum((imp * r.uniform(.004, .03, n) * mult).astype(int), 1)
    conv = np.maximum((clicks * r.uniform(.01, .08, n) * mult).astype(int), 0)
    spend = np.round(clicks * r.uniform(.5, 3.0, n), 2)
    return pd.DataFrame({
        "campaign_id": np.arange(n), "channel": channel, "impressions": imp,
        "clicks": clicks, "conversions": conv, "spend": spend,
        "revenue": np.round(conv * r.uniform(30, 150, n), 2),
        "ctr": np.round(clicks / imp, 4),
    })


def _saas(n=500):
    r = np.random.default_rng(7)
    plan = r.choice(["free", "pro", "enterprise"], n, p=[.5, .35, .15])
    # 'free' churns much harder — the plan-level leak the engine looks for.
    p_churn = np.where(plan == "free", .38, np.where(plan == "pro", .10, .04))
    return pd.DataFrame({
        "account_id": np.arange(n), "plan": plan,
        "churned": (r.random(n) < p_churn).astype(int),
        "mrr": np.round(r.uniform(20, 3000, n), 2),
        "seats": r.integers(1, 250, n), "nps": r.integers(-100, 100, n),
        "tenure_months": r.integers(1, 60, n),
        "expansion_revenue": np.round(r.uniform(0, 200, n), 2),
        "active_users": r.integers(1, 200, n),
    })


def _operations(n=500):
    r = np.random.default_rng(7)
    return pd.DataFrame({
        "work_order_id": np.arange(n),
        "cycle_time_hours": np.round(r.lognormal(3, .8, n), 1),
        "defect_rate": np.round(r.uniform(.01, .09, n), 4),
        "throughput": r.integers(20, 800, n),
        "downtime_minutes": r.integers(0, 500, n),
        "utilization": np.round(r.uniform(.80, .99, n), 3),
        "on_time_delivery": r.choice([0, 1], n, p=[.22, .78]),
        "inventory_turns": np.round(r.uniform(2, 14, n), 2),
        "plant": r.choice(["Pune", "Chennai", "Noida"], n),
    })


def _healthcare(n=500):
    r = np.random.default_rng(7)
    return pd.DataFrame({
        "patient_id": np.arange(n),
        "length_of_stay": np.maximum(r.lognormal(1.4, .9, n).astype(int), 1),
        "readmission_30d": r.choice([0, 1], n, p=[.82, .18]),
        "bed_occupancy": np.round(r.uniform(.80, .99, n), 3),
        "cost_per_case": np.round(r.lognormal(9, .7, n), 2),
        "department": r.choice(["Cardiology", "Oncology", "ICU", "Ortho"], n),
        "admission_type": r.choice(["Emergency", "Elective"], n),
        "diagnosis_code": r.choice(["A01", "B02", "C03"], n),
    })


BUILDERS = {"marketing": _marketing, "saas": _saas,
            "operations": _operations, "healthcare": _healthcare}

# Words that mean this domain got someone else's engine.
FOREIGN_VOCAB = {
    "marketing": ("employee", "attrition", "headcount", "patient", "bed "),
    "saas": ("employee attrition", "headcount", "patient", "defect rate"),
    "operations": ("employee", "attrition", "patient", "churn"),
    "healthcare": ("employee", "attrition", "headcount", "campaign", "roas"),
}


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_domain_is_registered(domain):
    assert domain in REGISTRY


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_engine_produces_insights(domain):
    df = BUILDERS[domain]()
    stats, corrs = _stats_and_corrs(df)
    raw = run_insights(domain, df, stats, corrs)
    assert raw["insights"], f"{domain}: engine produced no insights"
    assert raw["findings"], f"{domain}: engine produced no findings"


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_insights_are_well_formed(domain):
    df = BUILDERS[domain]()
    stats, corrs = _stats_and_corrs(df)
    for ins in run_insights(domain, df, stats, corrs)["insights"]:
        for field in ("title", "problem", "cause", "evidence", "action",
                      "impact"):
            val = getattr(ins, field)
            assert val and val.strip(), f"{domain}: empty {field}"
        assert ins.severity in ("critical", "high", "warning", "medium",
                                "info", "low", "positive")
        assert ins.category, f"{domain}: insight has no category"


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_no_nan_or_inf_leaks_into_prose(domain):
    """'nan%' in a client-facing report is the tell that a computation
    failed and nothing checked."""
    df = BUILDERS[domain]()
    stats, corrs = _stats_and_corrs(df)
    raw = run_insights(domain, df, stats, corrs)
    blob = " ".join(
        raw["findings"] + raw["risks"] + raw["opportunities"] + raw["actions"]
        + [f"{i.title} {i.problem} {i.evidence} {i.impact}"
           for i in raw["insights"]]
    ).lower()
    # Whole words only: "infection" and "nanometre" are not failures, a
    # bare "nan" or "inf" is.
    for bad in ("nan", "inf", "-inf", "none"):
        assert not re.search(rf"(?<![a-z0-9]){re.escape(bad)}(?![a-z0-9])",
                             blob), \
            f"{domain}: bare {bad!r} leaked into report prose"
    assert "%%" not in blob and "nan%" not in blob


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_engine_does_not_borrow_another_domains_vocabulary(domain):
    df = BUILDERS[domain]()
    stats, corrs = _stats_and_corrs(df)
    raw = run_insights(domain, df, stats, corrs)
    blob = " ".join(
        raw["findings"] + raw["risks"]
        + [f"{i.title} {i.problem} {i.cause} {i.impact}"
           for i in raw["insights"]]).lower()
    for word in FOREIGN_VOCAB[domain]:
        assert word not in blob, \
            f"{domain} report used {word!r} — wrong domain's vocabulary"


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_full_story_routes_and_labels_consistently(domain):
    """The report's label and its analysis must come from the same engine.
    Marketing used to be labelled 'marketing' and analysed as general."""
    df = BUILDERS[domain]()
    story = generate_story(df)
    assert story.domain == domain
    assert story.domain_confidence > 0.4
    assert story.top_insights


@pytest.mark.parametrize("domain", sorted(BUILDERS))
def test_engine_survives_a_degenerate_frame(domain):
    """A domain engine must not crash the report on thin or empty input."""
    for df in (pd.DataFrame({"a": [1]}),
               pd.DataFrame({"a": [np.nan, np.nan]}),
               pd.DataFrame({"channel": ["x"], "spend": [0.0]})):
        stats, corrs = _stats_and_corrs(df)
        raw = run_insights(domain, df, stats, corrs)
        assert isinstance(raw, dict)
        assert set(raw) >= {"findings", "risks", "opportunities", "actions",
                            "insights"}


def test_saas_churn_is_customer_churn_not_attrition():
    """The specific regression: SaaS data detected as HR came back with an
    employee attrition rate computed from subscription churn."""
    story = generate_story(_saas())
    assert story.domain == "saas"
    assert story.attrition is None, \
        "attrition pipeline ran on subscription data again"


def test_healthcare_output_stays_administrative():
    """This engine analyses operations, not patients. It must not emit
    anything that reads as clinical guidance."""
    df = _healthcare()
    stats, corrs = _stats_and_corrs(df)
    raw = run_insights("healthcare", df, stats, corrs)
    blob = " ".join(
        raw["findings"] + raw["risks"] + raw["actions"]
        + [f"{i.action} {i.cause} {i.impact}" for i in raw["insights"]]).lower()
    for phrase in ("should be treated", "prescribe", "diagnos" + "e the",
                   "recommend treatment", "the patient should"):
        assert phrase not in blob, \
            f"healthcare engine emitted clinical guidance: {phrase!r}"


def test_marketing_finds_the_inefficient_channel():
    """Display is built to be ~6x less efficient. If the engine cannot see
    a gap that large, the waste rule is not doing anything."""
    df = _marketing()
    stats, corrs = _stats_and_corrs(df)
    raw = run_insights("marketing", df, stats, corrs)
    blob = " ".join(raw["findings"] + raw["risks"]
                    + [i.title for i in raw["insights"]]).lower()
    assert "display" in blob, "engine missed the deliberately inefficient channel"
