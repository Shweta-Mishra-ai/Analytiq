"""
Domain registry completeness and detection.

These tests exist because of a specific bug: `marketing` was a detectable
domain with no insight engine, so a marketing dataset was labelled
"marketing" in the report and then analysed by the general engine. Nothing
failed — the report just quietly described the wrong thing.

Domain facets used to live in seven modules that had to be kept in sync by
hand. `test_every_domain_is_complete` is the guard: register a domain
without a blueprint, prompt, theme or benchmark set and this fails, rather
than production degrading silently.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.domains.registry import (
    REGISTRY, detect_domain, score_domains, spec_for,
)

SAMPLE = "/home/user/dataforge-ai/sample_data"


# ── completeness ──────────────────────────────────────────

@pytest.mark.parametrize("key", sorted(REGISTRY.keys()))
def test_every_domain_is_complete(key):
    """Every registered domain must carry every facet the app will ask for."""
    spec = REGISTRY[key]

    assert spec.key == key
    assert spec.label and spec.label.strip(), f"{key}: empty label"
    assert callable(spec.insight_fn), f"{key}: insight_fn not callable"

    # PDF theme must actually exist, or the report falls back mid-build.
    from app.engines.pdf_builder import THEMES
    assert spec.pdf_theme in THEMES, \
        f"{key}: pdf_theme {spec.pdf_theme!r} is not a defined theme"

    # Report structure.
    from app.engines.report_blueprints import BLUEPRINTS
    assert spec.blueprint is not None or key in BLUEPRINTS, \
        f"{key}: no report blueprint"

    # LLM prompts — a domain with no prompt of its own must not silently
    # borrow another domain's voice.
    from app.ai.prompt_builder import EXECUTIVE_PROMPTS, INSIGHT_PROMPTS
    assert key in EXECUTIVE_PROMPTS, f"{key}: no executive prompt"
    assert key in INSIGHT_PROMPTS, f"{key}: no insight prompt"

    if key != "general":
        assert spec.signature, f"{key}: no signature keywords — undetectable"
        # KPI specs: without them the panel falls back to generic
        # measures, which is how it came to sum employee ID numbers.
        assert spec.kpis, f"{key}: no KPI specs"
        assert spec.chart_metrics, f"{key}: no chart metric priorities"
        from app.engines.industry_benchmarks import DOMAIN_BENCHMARKS
        assert spec.benchmarks is not None or key in DOMAIN_BENCHMARKS, \
            f"{key}: no benchmark set"


def test_general_is_registered_as_the_fallback():
    assert "general" in REGISTRY
    assert REGISTRY["general"].signature == ()
    assert "general" not in score_domains(pd.DataFrame({"a": [1, 2]}))


def test_signature_keywords_are_not_shared_between_domains():
    """A signature word is meant to be decisive. If two domains claim the
    same one, neither can be."""
    seen = {}
    for key, spec in REGISTRY.items():
        assert len(set(spec.signature)) == len(spec.signature), \
            f"{key}: duplicate keyword within its own signature list"
        for kw in spec.signature:
            assert kw not in seen, \
                f"signature {kw!r} claimed by both {seen[kw]!r} and {key!r}"
            seen[kw] = key


def test_spec_for_unknown_domain_falls_back_to_general():
    assert spec_for("does_not_exist").key == "general"
    assert spec_for("").key == "general"
    assert spec_for(None).key == "general"


def test_only_hr_runs_attrition():
    """Attrition ran for anything detected as HR, so a misrouted SaaS
    dataset came back with an employee attrition rate."""
    runners = [k for k, s in REGISTRY.items() if s.runs_attrition]
    assert runners == ["hr"], f"unexpected attrition domains: {runners}"


# ── detection ─────────────────────────────────────────────

def _rng():
    return np.random.default_rng(42)


def _marketing_df(n=400):
    r = _rng()
    imp = r.integers(1000, 90000, n)
    clicks = (imp * r.uniform(0.005, 0.06, n)).astype(int)
    conv = (clicks * r.uniform(0.01, 0.14, n)).astype(int)
    return pd.DataFrame({
        "campaign_id": np.arange(n), "channel": r.choice(["Search", "Social"], n),
        "impressions": imp, "clicks": clicks, "conversions": conv,
        "spend": r.uniform(100, 9000, n), "ctr": clicks / imp,
        "cpa": r.uniform(5, 400, n), "roas": r.uniform(0.2, 8, n),
        "lead_source": r.choice(["organic", "paid"], n),
    })


def _saas_df(n=400):
    r = _rng()
    return pd.DataFrame({
        "account_id": np.arange(n), "mrr": r.uniform(50, 4000, n),
        "arr": r.uniform(600, 48000, n), "seats": r.integers(1, 400, n),
        "churned": r.choice([0, 1], n), "plan": r.choice(["free", "pro"], n),
        "nps": r.integers(-100, 100, n), "tenure_months": r.integers(1, 72, n),
        "expansion_revenue": r.uniform(0, 900, n),
        "active_users": r.integers(1, 350, n),
    })


def _operations_df(n=400):
    r = _rng()
    return pd.DataFrame({
        "work_order_id": np.arange(n), "cycle_time_hours": r.uniform(1, 180, n),
        "defect_rate": r.uniform(0, 0.13, n), "throughput": r.integers(10, 900, n),
        "downtime_minutes": r.integers(0, 700, n), "utilization": r.uniform(.35, .99, n),
        "on_time_delivery": r.choice([0, 1], n), "inventory_turns": r.uniform(1, 20, n),
        "plant": r.choice(["Pune", "Chennai"], n),
    })


def _healthcare_df(n=400):
    r = _rng()
    return pd.DataFrame({
        "patient_id": np.arange(n), "length_of_stay": r.integers(1, 35, n),
        "readmission_30d": r.choice([0, 1], n), "mortality_flag": r.choice([0, 1], n),
        "bed_occupancy": r.uniform(.4, 1, n), "cost_per_case": r.uniform(800, 42000, n),
        "department": r.choice(["Cardiology", "ICU"], n), "age": r.integers(18, 95, n),
        "diagnosis_code": r.choice(["A01", "B02"], n),
    })


def test_hr_dataset_detects_as_hr():
    df = pd.read_csv(f"{SAMPLE}/hr_attrition.csv")
    df.columns = [c.replace("﻿", "") for c in df.columns]
    domain, conf = detect_domain(df)
    assert domain == "hr"
    assert conf > 0.5, f"real HR data should be confident, got {conf}"


def test_telco_churn_does_not_detect_as_hr():
    """Customer churn is not employee attrition. This dataset used to
    detect as 'hr' at confidence 0.07 and get an attrition analysis."""
    df = pd.read_csv(f"{SAMPLE}/telco_churn.csv")
    df.columns = [c.replace("﻿", "") for c in df.columns]
    domain, _ = detect_domain(df)
    assert domain != "hr", "telco churn misrouted to the HR engine again"


def test_weak_evidence_returns_general_not_a_guess():
    """One incidental keyword must not win a domain. The old scoring
    accepted a match at 0.04 confidence."""
    df = pd.DataFrame({"salary": [1, 2, 3], "x": [4, 5, 6], "y": [7, 8, 9]})
    domain, conf = detect_domain(df)
    assert domain == "general"
    assert conf == 0.0


def test_empty_and_degenerate_frames_are_safe():
    assert detect_domain(pd.DataFrame()) == ("general", 0.0)
    assert detect_domain(None) == ("general", 0.0)


def test_confidence_is_bounded():
    df = pd.read_csv(f"{SAMPLE}/hr_attrition.csv")
    df.columns = [c.replace("﻿", "") for c in df.columns]
    _, conf = detect_domain(df)
    assert 0.0 <= conf <= 1.0


@pytest.mark.parametrize("name,builder", [
    ("marketing", _marketing_df),
    ("saas", _saas_df),
    ("operations", _operations_df),
    ("healthcare", _healthcare_df),
])
def test_expansion_domains_route_correctly(name, builder):
    """Each expansion domain must claim its own data. Skipped until the
    domain is registered, so this file lands with Phase 1."""
    if name not in REGISTRY:
        pytest.skip(f"{name} not registered yet")
    domain, conf = detect_domain(builder())
    assert domain == name, f"{name} data detected as {domain}"
    assert conf > 0.4, f"{name} detected at only {conf} confidence"


@pytest.mark.parametrize("name,builder", [
    ("marketing", _marketing_df), ("saas", _saas_df),
    ("operations", _operations_df), ("healthcare", _healthcare_df),
])
def test_expansion_domains_never_route_to_hr(name, builder):
    """Holds whether or not the domain is registered: none of these is HR
    data, and all four used to land on the HR engine."""
    domain, _ = detect_domain(builder())
    assert domain != "hr", f"{name} data routed to the HR engine"
