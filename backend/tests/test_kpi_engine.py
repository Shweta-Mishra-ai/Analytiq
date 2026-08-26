"""
The headline numbers, per domain.

What this replaced: the KPI panel returned four data-quality counts
(rows, columns, missing %, duplicates) and then the *sum of the first
four numeric columns*, whatever they were. On an HR extract that meant
"Σ EmployeeNumber" — a total of employee ID numbers — and "Σ Age".

Two failures in one. The aggregation ignored what the metric was, and
the panel had no idea what business it was looking at.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.domains.registry import REGISTRY
from app.engines.kpi_engine import (
    KpiSpec, compute_kpi, compute_kpis, data_quality_cards,
)

SAMPLE = "/home/user/dataforge-ai/sample_data"


def _load(name):
    df = pd.read_csv(f"{SAMPLE}/{name}")
    df.columns = [c.replace("﻿", "") for c in df.columns]
    return df


@pytest.fixture(scope="module")
def hr():
    return _load("hr_attrition.csv")


@pytest.fixture
def marketing():
    r = np.random.default_rng(7)
    n = 400
    imp = r.integers(5000, 90000, n)
    clicks = np.maximum((imp * r.uniform(.004, .03, n)).astype(int), 1)
    return pd.DataFrame({
        "campaign_id": np.arange(n),
        "channel": r.choice(["Search", "Social"], n),
        "impressions": imp, "clicks": clicks,
        "conversions": np.maximum((clicks * r.uniform(.01, .08, n)).astype(int), 1),
        "spend": (clicks * r.uniform(.5, 3.0, n)).round(2),
        "revenue": r.uniform(1e3, 9e4, n).round(2),
    })


# ── the original defect ───────────────────────────────────

def test_identifiers_never_become_a_kpi(hr):
    for card in compute_kpis(hr, "hr"):
        assert "EmployeeNumber" not in card.source_column
        assert "employee_number" not in card.source_column.lower()


def test_no_kpi_is_a_total_of_a_rating(hr):
    """A total of a 1-5 satisfaction score is not a quantity."""
    for card in compute_kpis(hr, "hr"):
        if "satisfaction" in card.source_column.lower():
            assert card.label.lower().startswith("average")


def test_hr_leads_with_attrition(hr):
    labels = [c.label for c in compute_kpis(hr, "hr")]
    assert "Attrition Rate" in labels
    assert "Headcount" in labels


def test_attrition_is_a_percentage_not_a_sum(hr):
    card = next(c for c in compute_kpis(hr, "hr") if c.label == "Attrition Rate")
    assert card.unit == "%"
    assert 0 < card.value < 100


def test_kpis_carry_benchmark_context_where_one_exists(hr):
    card = next(c for c in compute_kpis(hr, "hr") if c.label == "Attrition Rate")
    assert card.benchmark, "no published range cited for attrition"


def test_kpis_name_the_column_they_came_from(hr):
    for card in compute_kpis(hr, "hr"):
        if card.label not in ("Headcount",):
            assert card.source_column, f"{card.label} cannot be checked"


# ── domain awareness ──────────────────────────────────────

def test_marketing_leads_with_spend_efficiency(marketing):
    labels = [c.label for c in compute_kpis(marketing, "marketing")]
    assert "Return on Ad Spend" in labels
    assert "Cost per Acquisition" in labels


def test_ratio_kpis_divide_the_right_way(marketing):
    """ROAS is revenue over spend. Inverted it looks like a disaster."""
    card = next(c for c in compute_kpis(marketing, "marketing")
                if c.label == "Return on Ad Spend")
    expected = marketing["revenue"].sum() / marketing["spend"].sum()
    assert card.value == pytest.approx(expected, rel=0.01)


def test_ctr_is_expressed_as_a_percentage(marketing):
    card = next(c for c in compute_kpis(marketing, "marketing")
                if c.label == "Click-Through Rate")
    assert card.unit == "%"
    assert 0 < card.value < 100


@pytest.mark.parametrize("domain", sorted(set(REGISTRY) - {"general"}))
def test_every_domain_declares_its_kpis(domain):
    """A domain without KPI specs falls back to generic measures, which is
    how the panel ended up summing identifiers."""
    assert REGISTRY[domain].kpis, f"{domain} declares no KPIs"


@pytest.mark.parametrize("domain", sorted(set(REGISTRY) - {"general"}))
def test_kpi_specs_are_well_formed(domain):
    from app.engines.kpi_engine import KINDS
    for spec in REGISTRY[domain].kpis:
        assert spec.kind in KINDS, f"{domain}/{spec.key}: unknown kind"
        assert spec.label and spec.label.strip()
        if spec.kind != "count":
            assert spec.columns, f"{domain}/{spec.key}: nothing to resolve"
        if spec.kind == "ratio":
            assert spec.denominator, f"{domain}/{spec.key}: ratio needs a denominator"


# ── behaviour when the data does not support a KPI ────────

def test_a_kpi_that_does_not_apply_is_omitted_not_shown_empty():
    """'Gross Margin —' claims the question was asked and unanswered."""
    df = pd.DataFrame({"revenue": [100.0, 200.0], "region": ["N", "S"]})
    spec = KpiSpec("gm", "Gross Margin", "ratio", ("grossprofit",),
                   denominator=("revenue",), unit="%")
    assert compute_kpi(df, spec, "finance") is None


def test_a_ratio_with_a_zero_denominator_is_omitted():
    df = pd.DataFrame({"clicks": [5.0, 5.0], "impressions": [0.0, 0.0]})
    spec = KpiSpec("ctr", "CTR", "ratio", ("clicks",),
                   denominator=("impressions",), unit="%")
    assert compute_kpi(df, spec, "marketing") is None


def test_fallback_uses_the_right_aggregation_per_metric():
    """With no domain specs, measures still get the aggregation they
    deserve — and identifiers are still excluded."""
    df = pd.DataFrame({
        "order_id": range(200),
        "revenue": np.linspace(10, 900, 200),
        "rating": np.random.default_rng(1).integers(1, 6, 200),
    })
    cards = compute_kpis(df, "general")
    labels = " ".join(c.label for c in cards)
    assert "order_id" not in labels.lower() and "order id" not in labels.lower()
    assert "Total Revenue" in labels or "Total revenue" in labels
    rating = [c for c in cards if "rating" in c.label.lower()]
    if rating:
        assert rating[0].label.lower().startswith("average")


def test_empty_frame_produces_no_kpis():
    assert compute_kpis(pd.DataFrame(), "hr") == []


@pytest.mark.parametrize("domain", sorted(REGISTRY))
def test_kpis_never_return_nan_or_infinity(domain):
    r = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame({"a": r.normal(0, 1, n), "b": np.zeros(n),
                       "c": r.choice(["x", "y"], n)})
    for card in compute_kpis(df, domain):
        assert card.value is None or np.isfinite(card.value), \
            f"{domain}/{card.label} returned {card.value}"


# ── the file-shape figures ────────────────────────────────

def test_data_quality_is_returned_separately(hr):
    """'Duplicates: 3' beside 'Revenue: 4.2M' are not the same kind of
    question, and mixing them is what made the old panel unreadable."""
    quality = {c.label for c in data_quality_cards(hr)}
    business = {c.label for c in compute_kpis(hr, "hr")}
    assert "Records" in quality
    assert not (quality & business)


def test_repeated_rows_are_described_as_reported_not_removed(hr):
    card = next(c for c in data_quality_cards(hr) if c.label == "Repeated rows")
    assert "not removed" in card.note.lower()
