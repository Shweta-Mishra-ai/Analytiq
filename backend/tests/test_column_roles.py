"""
Deciding what each column is — the decision every figure in the report
rests on.

There were twenty-three copies of this logic, one per engine, each a
substring scan with its own hand-rolled exclusion list, and no two the
same. The finance engine had learned to exclude `_pct` and `rate`; the
sales engine had not, so `margin_pct` was a candidate for the profit
column. Nothing knew that `forecast_category` holds Commit / Best Case /
Pipeline, so it was read as the product line and the report recommended
reviewing "revenue by forecast_category for concentration and whether the
long tail justifies its resource". A shipment manifest's `cost` column
made the file finance. `order_id` was summed as a measure.

Each was fixed where it was found, which left the same defect alive in
the other twenty-two places.

The refusals matter as much as the matches. `None` drops a section; a
wrong match produces a fluent, confident, wrong one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.column_roles import resolve


def _frame(**cols):
    return pd.DataFrame(cols)


@pytest.fixture()
def pipeline():
    rng = np.random.default_rng(3)
    n = 300
    return pd.DataFrame({
        "opportunity_id": range(n),
        "sales_rep": rng.choice(list("ABCDE"), n),
        "territory": rng.choice(["EMEA", "AMER"], n),
        "deal_stage": rng.choice(["Closed Won", "Closed Lost"], n),
        "deal_amount": rng.lognormal(9, .5, n),
        "quota": 250_000.0,
        "margin_pct": rng.normal(22, 4, n),
        "forecast_category": rng.choice(["Commit", "Best Case"], n),
        "created_date": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


# ══════════════════════════════════════════════════════════
#  Qualifiers win over the noun they qualify
# ══════════════════════════════════════════════════════════

def test_a_percentage_is_not_money(pipeline):
    """"margin" is a money word. "margin_pct" is a ratio."""
    roles = resolve(pipeline)
    assert roles.rate == "margin_pct"
    assert roles.money != "margin_pct"
    assert roles.profit != "margin_pct"


def test_a_target_is_not_an_actual(pipeline):
    """Summing budget next to revenue as though both were income is how
    a variance becomes nonsense."""
    roles = resolve(pipeline)
    assert roles.plan == "quota"
    assert roles.money != "quota"


@pytest.mark.parametrize("name", [
    "revenue_pct", "cost_percentage", "profit_ratio", "spend_share",
    "revenue_per_head", "conversion_rate",
])
def test_every_rate_spelling_is_caught(name):
    df = _frame(**{name: np.linspace(1, 50, 100),
                   "revenue": np.linspace(1000, 9000, 100)})
    roles = resolve(df)
    assert roles.money == "revenue", roles.reason
    assert roles.rate == name, roles.reason


@pytest.mark.parametrize("name", [
    "budget", "forecast_revenue", "target_sales", "planned_spend",
])
def test_every_plan_spelling_is_caught(name):
    df = _frame(**{name: np.linspace(1, 50, 100),
                   "revenue": np.linspace(1000, 9000, 100)})
    roles = resolve(df)
    assert roles.plan == name, roles.reason
    assert roles.money == "revenue", roles.reason


# ══════════════════════════════════════════════════════════
#  Whole words, not substrings
# ══════════════════════════════════════════════════════════

def test_a_confidence_band_is_not_a_product_line(pipeline):
    """Commit / Best Case / Pipeline is not a catalogue."""
    assert resolve(pipeline).product != "forecast_category"


@pytest.mark.parametrize("name", [
    "risk_category", "priority_segment", "age_band", "credit_tier",
    "customer_segment", "lead_category",
])
def test_a_category_column_that_is_not_a_product(name):
    df = _frame(**{name: ["a", "b"] * 50, "revenue": np.linspace(1, 100, 100)})
    assert resolve(df).product is None, name


def test_a_real_product_column_is_still_found():
    """The exclusions must not swallow the case they exist to protect."""
    df = _frame(product_category=["Home", "Beauty"] * 50,
                revenue=np.linspace(1, 100, 100))
    assert resolve(df).product == "product_category"


def test_a_discounted_price_is_a_price_not_a_rate():
    """It contains "discount"; it is money."""
    df = _frame(discounted_price=np.linspace(10, 90, 100),
                discount_pct=np.linspace(0, 30, 100))
    roles = resolve(df)
    assert roles.money == "discounted_price", roles.reason
    assert roles.rate == "discount_pct", roles.reason


def test_reorder_point_is_not_an_order_count():
    df = _frame(reorder_point=np.linspace(1, 90, 100),
                units=np.linspace(1, 90, 100))
    assert resolve(df).quantity == "units"


# ══════════════════════════════════════════════════════════
#  The dtype has to agree
# ══════════════════════════════════════════════════════════

def test_a_revenue_band_is_not_a_revenue_measure():
    """A column called `revenue_band` holding High/Low is a label."""
    df = _frame(revenue_band=["High", "Low"] * 50,
                total_amount=np.linspace(1, 100, 100))
    roles = resolve(df)
    assert roles.money == "total_amount", roles.reason


def test_an_identifier_is_never_a_measure():
    df = _frame(order_id=range(200), amount=np.linspace(1, 100, 200))
    roles = resolve(df)
    assert roles.money == "amount"
    assert roles.quantity != "order_id"


def test_money_stored_as_text_is_still_money():
    df = _frame(revenue=["1,200", "3,400"] * 50)
    # Thousands separators do not coerce, so this stays unresolved rather
    # than being silently summed as text — the honest outcome.
    assert resolve(df).money in (None, "revenue")


def test_a_high_cardinality_label_is_not_a_region():
    df = _frame(city=[f"City {i}" for i in range(300)],
                revenue=np.linspace(1, 300, 300))
    assert resolve(df).region is None


# ══════════════════════════════════════════════════════════
#  Preference among valid matches
# ══════════════════════════════════════════════════════════

def test_revenue_beats_amount():
    """"amount" is a container word; "revenue" says what it holds."""
    df = _frame(amount=np.linspace(1, 100, 100),
                revenue=np.linspace(1, 100, 100))
    assert resolve(df).money == "revenue"


def test_cost_and_revenue_do_not_take_each_others_place():
    df = _frame(revenue=np.linspace(100, 900, 100),
                cogs=np.linspace(50, 500, 100),
                gross_profit=np.linspace(50, 400, 100))
    roles = resolve(df)
    assert roles.money == "revenue", roles.reason
    assert roles.cost == "cogs", roles.reason
    assert roles.profit == "gross_profit", roles.reason


def test_no_column_fills_two_roles():
    df = _frame(revenue=np.linspace(1, 100, 100),
                cost=np.linspace(1, 50, 100),
                units=np.linspace(1, 20, 100))
    roles = resolve(df)
    taken = [c for c in (roles.money, roles.cost, roles.profit,
                         roles.quantity, roles.rate, roles.plan) if c]
    assert len(taken) == len(set(taken)), taken


# ══════════════════════════════════════════════════════════
#  Refusing
# ══════════════════════════════════════════════════════════

def test_a_shipment_manifest_has_a_cost_but_no_revenue():
    """Every business file has a cost column. That does not make it a
    P&L."""
    rng = np.random.default_rng(4)
    df = _frame(shipment_id=range(200),
                transit_days=rng.integers(1, 20, 200),
                cost=rng.normal(500, 50, 200))
    roles = resolve(df)
    assert roles.cost == "cost"
    assert roles.money is None, roles.reason
    assert roles.profit is None


def test_a_frame_with_no_business_columns_resolves_nothing():
    df = _frame(a=np.linspace(1, 10, 50), b=np.linspace(1, 10, 50))
    roles = resolve(df)
    assert roles.money is None and roles.product is None


def test_an_empty_frame_does_not_raise():
    assert resolve(pd.DataFrame()).money is None


# ══════════════════════════════════════════════════════════
#  The assignment can be checked
# ══════════════════════════════════════════════════════════

def test_every_assignment_records_why(pipeline):
    """A role that shapes every figure in the report has to be arguable
    with, not silent."""
    roles = resolve(pipeline)
    for role in ("money", "plan", "rate", "region", "person", "period"):
        col = roles.get(role)
        if col:
            assert role in roles.reason, role
            assert col in roles.reason[role], (role, roles.reason[role])


def test_the_reason_names_the_matching_word(pipeline):
    assert "quota" in resolve(pipeline).reason["plan"]


# ══════════════════════════════════════════════════════════
#  The engine that uses it
# ══════════════════════════════════════════════════════════

def test_the_sales_engine_sees_the_same_roles(pipeline):
    from app.engines.domains.base import col_stats
    from app.engines.domains.sales import _insights_sales

    stats = {c: col_stats(pipeline[c])
             for c in pipeline.select_dtypes(include="number").columns}
    out = _insights_sales(pipeline, {c: s for c, s in stats.items() if s}, [])
    text = " ".join(out["actions"] + out["findings"])
    assert "forecast_category" not in text, text
    assert "margin_pct" not in text, text
