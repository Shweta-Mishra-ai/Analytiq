"""
Tests for app/engines/domains/ — the per-domain insight engines ported
from dataforge-ai, plus story_engine's assembly of their output.

Analytiq previously had shallower HR/ecommerce/sales insight logic inline
in story_engine.py and no finance engine at all. The report shape
(headline / top_insights / critical_issues / positive_findings) is
Analytiq's own and is read directly by the frontend, so these tests pin
that contract as well as the new domain depth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.story_engine import detect_domain, generate_story

_REQUIRED_KEYS = {"findings", "risks", "opportunities", "actions", "insights"}


# ══════════════════════════════════════════════════════════
#  Fixtures — one representative frame per domain
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def finance_df():
    rng = np.random.default_rng(3)
    n = 400
    revenue = rng.normal(100_000, 25_000, n).round(2)
    return pd.DataFrame({
        "invoice_id": range(1, n + 1),
        "revenue": revenue,
        "expense": (revenue * rng.uniform(0.5, 0.95, n)).round(2),
        "profit_margin": rng.uniform(0.02, 0.35, n).round(3),
        "budget": rng.normal(90_000, 10_000, n).round(2),
        "cost_center": rng.choice(["Ops", "Sales", "R&D", "Admin"], n),
        "tax": (revenue * 0.18).round(2),
    })


@pytest.fixture()
def ecommerce_df():
    rng = np.random.default_rng(4)
    n = 400
    return pd.DataFrame({
        "order_id": range(1, n + 1),
        "product": rng.choice(["Widget", "Gadget", "Doohickey"], n),
        "category": rng.choice(["Home", "Tech", "Toys"], n),
        "price": rng.uniform(5, 500, n).round(2),
        "discount": rng.uniform(0, 0.5, n).round(2),
        "rating": rng.uniform(1, 5, n).round(1),
        "stock": rng.integers(0, 500, n),
    })


@pytest.fixture()
def sales_df():
    rng = np.random.default_rng(5)
    n = 400
    return pd.DataFrame({
        "deal_id": range(1, n + 1),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "revenue": rng.normal(50_000, 12_000, n).round(2),
        "quota": rng.normal(45_000, 5_000, n).round(2),
        "pipeline": rng.normal(120_000, 30_000, n).round(2),
        "conversion": rng.uniform(0.05, 0.6, n).round(3),
        "closed": rng.choice(["Yes", "No"], n, p=[0.4, 0.6]),
    })


# ══════════════════════════════════════════════════════════
#  Domain engines honour the shared contract
# ══════════════════════════════════════════════════════════

def _stats_and_corrs(df):
    from app.engines.domains.base import col_stats, correlations
    num_cols = df.select_dtypes(include="number").columns.tolist()
    stats = {c: col_stats(df[c]) for c in num_cols}
    return {k: v for k, v in stats.items() if v}, correlations(df)


@pytest.mark.parametrize("engine_path,fn_name", [
    ("app.engines.domains.finance",   "_insights_finance"),
    ("app.engines.domains.ecommerce", "_insights_ecommerce"),
    ("app.engines.domains.sales",     "_insights_sales"),
    ("app.engines.domains.general",   "_insights_general"),
])
def test_domain_engine_returns_required_keys(engine_path, fn_name, hr_df):
    import importlib
    fn = getattr(importlib.import_module(engine_path), fn_name)
    stats, corrs = _stats_and_corrs(hr_df)
    raw = fn(hr_df, stats, corrs)
    assert _REQUIRED_KEYS <= set(raw), f"{fn_name} missing keys: {_REQUIRED_KEYS - set(raw)}"
    for key in ("findings", "risks", "opportunities", "actions", "insights"):
        assert isinstance(raw[key], list)


def test_hr_engine_returns_required_keys(hr_df):
    from app.engines.domains.hr import _insights_hr, _run_attrition
    stats, corrs = _stats_and_corrs(hr_df)
    raw = _insights_hr(hr_df, stats, corrs, _run_attrition(hr_df))
    assert _REQUIRED_KEYS <= set(raw)


# ══════════════════════════════════════════════════════════
#  Finance — a domain Analytiq did not have before
# ══════════════════════════════════════════════════════════

def test_finance_domain_is_detected(finance_df):
    domain, confidence = detect_domain(finance_df)
    assert domain == "finance", f"expected finance, got {domain}"
    assert confidence > 0


def test_finance_story_produces_content(finance_df):
    story = generate_story(finance_df)
    assert story.domain == "finance"
    assert story.key_findings
    assert story.headline
    assert story.executive_summary


# ══════════════════════════════════════════════════════════
#  Report shape the frontend depends on
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("fixture_name", [
    "hr_df", "finance_df", "ecommerce_df", "sales_df"])
def test_story_report_shape_is_stable_across_domains(fixture_name, request):
    """InsightsPage.tsx reads story.headline and maps story.top_insights;
    the PDF builder reads critical_issues/positive_findings. None of these
    may go missing for any domain."""
    df = request.getfixturevalue(fixture_name)
    story = generate_story(df)

    assert isinstance(story.headline, str) and story.headline
    assert isinstance(story.top_insights, list)
    assert isinstance(story.critical_issues, list)
    assert isinstance(story.positive_findings, list)
    assert isinstance(story.key_findings, list) and story.key_findings
    assert isinstance(story.recommended_actions, list)
    assert story.data_quality_verdict and story.analysis_confidence

    for ins in story.top_insights:
        for attr in ("title", "problem", "cause", "evidence",
                     "action", "impact", "severity", "category"):
            assert hasattr(ins, attr), f"insight missing '{attr}'"


def test_insights_carry_confidence_from_ported_engines(hr_df):
    """The ported engines add an evidence-strength rating the old inline
    ones lacked."""
    story = generate_story(hr_df)
    if story.top_insights:
        assert hasattr(story.top_insights[0], "confidence")


# ══════════════════════════════════════════════════════════
#  Severity ladder regression
# ══════════════════════════════════════════════════════════

def test_high_severity_outranks_info_in_top_insights():
    """The ported engines emit "high"/"medium"/"low", which story_engine's
    original severity map didn't know — unmapped severities sorted to 99,
    pushing HIGH findings below INFO ones and off the top-6 list."""
    from app.engines.domains.base import Insight
    import app.engines.story_engine as se

    fake = [
        Insight(title="Info item", problem="p", cause="c", evidence="e",
                action="a", impact="i", severity="info", category="general"),
        Insight(title="High item", problem="p", cause="c", evidence="e",
                action="a", impact="i", severity="high", category="general"),
        Insight(title="Critical item", problem="p", cause="c", evidence="e",
                action="a", impact="i", severity="critical", category="general"),
    ]
    sev_order = {"critical": 0, "high": 1, "warning": 2, "medium": 2,
                 "info": 3, "low": 3, "positive": 4}
    ordered = sorted(fake, key=lambda x: sev_order.get(x.severity, 99))
    assert [i.title for i in ordered] == ["Critical item", "High item", "Info item"]

    # and the module actually uses a map covering those severities
    import inspect
    src = inspect.getsource(se.generate_story)
    for sev in ("high", "medium", "low"):
        assert f'"{sev}"' in src, f"story_engine severity map missing '{sev}'"


def test_high_severity_counts_as_critical_issue():
    """'high' from the domain engines means act-now, same as 'critical' —
    it must not silently drop out of critical_issues."""
    import inspect
    from app.engines import story_engine as se
    src = inspect.getsource(se.generate_story)
    assert '("critical", "high")' in src or "('critical', 'high')" in src


# ══════════════════════════════════════════════════════════
#  General top-up merge
# ══════════════════════════════════════════════════════════

def test_general_domain_does_not_double_merge_itself():
    """When the detected domain IS general, merging the general engine's
    output again duplicated every finding."""
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "alpha": rng.normal(10, 2, 200),
        "beta": rng.normal(50, 9, 200),
        "label": rng.choice(["x", "y", "z"], 200),
    })
    story = generate_story(df)
    assert len(story.key_findings) == len(set(story.key_findings)), \
        "duplicate key findings — general engine merged into itself"
    assert len(story.business_risks) == len(set(story.business_risks))
