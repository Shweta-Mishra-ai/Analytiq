"""What a projection is allowed to claim.

A fitted line returns a value for any input. That it does so says
nothing about whether the business has ever operated at that level, and
a number produced past the edge of the evidence is the most damaging
kind this app can produce: precise, confident and invented.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.bi_engine import analyze_scenario


@pytest.fixture()
def discounts():
    """Discount observed only between 0% and 20%, with a strong, real,
    negative relationship to revenue."""
    rng = np.random.default_rng(0)
    n = 500
    discount = rng.uniform(0, 20, n)
    return pd.DataFrame({
        "discount_pct": discount,
        "revenue": 1000 - 12 * discount + rng.normal(0, 20, n),
    })


def test_a_projection_inside_the_data_is_reliable(discounts):
    """The relationship is real and the question is answerable."""
    result = analyze_scenario(discounts, "discount_pct", "revenue", 10)
    assert result.within_observed_range is True
    assert result.reliable is True


def test_a_projection_past_the_evidence_is_not_reliable(discounts):
    """Asking for a 500% rise in a discount rate that never exceeded 20%
    projected a 223% discount and revenue of -1,665 — and reported it as
    reliable, because only r² and p were checked."""
    result = analyze_scenario(discounts, "discount_pct", "revenue", 2000)
    assert result.within_observed_range is False
    assert result.reliable is False


def test_the_relationship_is_still_strong_when_the_request_is_not(discounts):
    """The refusal must be about the question, not the data. Reporting
    this as "too weak" would misdiagnose a relationship that explains
    almost everything."""
    result = analyze_scenario(discounts, "discount_pct", "revenue", 2000)
    assert result.r_squared > 0.9
    assert result.p_value < 0.001


def test_the_reason_names_the_range_the_data_covers(discounts):
    result = analyze_scenario(discounts, "discount_pct", "revenue", 2000)
    assert "outside the" in result.interpretation
    assert "range the data actually covers" in result.interpretation


def test_the_result_carries_where_the_projection_lands(discounts):
    """So a caller can show the reader what was asked for, not just that
    it was refused."""
    result = analyze_scenario(discounts, "discount_pct", "revenue", 2000)
    assert result.projected_driver_value > result.driver_observed_max
    assert result.driver_observed_min == pytest.approx(0, abs=0.2)
    assert result.driver_observed_max == pytest.approx(20, abs=0.2)


@pytest.mark.parametrize("change,expected", [
    (5, True), (10, True), (50, True),      # inside
    (200, False), (500, False), (2000, False),   # past the edge
])
def test_the_boundary_holds_across_sizes(discounts, change, expected):
    result = analyze_scenario(discounts, "discount_pct", "revenue", change)
    assert result.within_observed_range is expected


def test_a_small_step_past_the_edge_is_still_a_forecast(discounts):
    """A tenth of the observed span beyond the edge is a step past what
    was seen; refusing that would make the tool useless."""
    result = analyze_scenario(discounts, "discount_pct", "revenue", 95)
    assert result.projected_driver_value > result.driver_observed_max
    assert result.within_observed_range is True


def test_a_reduction_below_the_floor_is_caught_too():
    """The check has to hold in both directions. A driver observed only
    between 50 and 100 cannot be reduced to 5."""
    rng = np.random.default_rng(1)
    n = 400
    price = rng.uniform(50, 100, n)
    df = pd.DataFrame({"price": price,
                       "units": 500 - 3 * price + rng.normal(0, 8, n)})
    result = analyze_scenario(df, "price", "units", -90)
    assert result.projected_driver_value < result.driver_observed_min
    assert result.within_observed_range is False
    assert result.reliable is False


def test_a_weak_relationship_is_still_reported_as_weak(discounts):
    """The range check is an addition, not a replacement."""
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({"noise": rng.normal(10, 2, n),
                       "revenue": rng.normal(500, 100, n)})
    result = analyze_scenario(df, "noise", "revenue", 10)
    assert result.within_observed_range is True
    assert result.reliable is False
    assert "too weak or not statistically significant" in result.interpretation


# ══════════════════════════════════════════════════════════
#  A driver that is the target rewritten
# ══════════════════════════════════════════════════════════
#
# The most convincing-looking output the scenario engine can produce is
# also its emptiest: point it at a column derived from the target and the
# fit is perfect, the p-value vanishes, and the projection is exactly the
# change requested. "If revenue_k rises 10%, revenue rises 10% (R²=1.00,
# p<0.0001)" is arithmetic on one column recorded twice, dressed as a
# finding — and it outranked every real driver.

@pytest.fixture(scope="module")
def restated() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 400
    qty = rng.integers(1, 20, n)
    price = rng.normal(50, 12, n).round(2)
    rev = qty * price
    return pd.DataFrame({
        "revenue":       rev,
        "revenue_k":     rev / 1000.0,        # same money, different unit
        "total_revenue": rev,                 # exact copy, second name
        "log_revenue":   np.log(rev + 1),     # monotone transform
        "quantity":      qty,                 # a real, independent driver
        "unit_price":    price,
    })


@pytest.mark.parametrize("driver", ["revenue_k", "total_revenue", "log_revenue"])
def test_a_restated_driver_is_never_reliable(restated, driver):
    r = analyze_scenario(restated, driver, "revenue", 10.0)
    assert r is not None
    assert r.driver_restates_target is True
    assert r.reliable is False, (
        "a perfect fit against the target rewritten is the emptiest "
        "result the engine can produce, not its most reliable")


def test_the_restatement_is_explained_in_words(restated):
    """A flag the UI could ignore is not enough — say why, in the text."""
    r = analyze_scenario(restated, "revenue_k", "revenue", 10.0)
    text = r.interpretation.lower()
    assert "same measurement" in text
    assert "revenue_k" in r.interpretation and "revenue" in r.interpretation
    # And it must tell the reader what to do instead.
    assert "pick a driver" in text


def test_a_real_driver_is_still_reliable(restated):
    """The guard must not cost the app its genuine findings."""
    r = analyze_scenario(restated, "quantity", "revenue", 10.0)
    assert r.driver_restates_target is False
    assert r.reliable is True
    assert "same measurement" not in r.interpretation


def test_a_merely_strong_relationship_is_not_called_a_restatement():
    """r≈0.95 is a strong driver, not a duplicate column. The threshold is
    extreme on purpose: below a near-perfect fit, the finding stands."""
    rng = np.random.default_rng(11)
    n = 500
    spend = rng.uniform(100, 900, n)
    sales = spend * 2.4 + rng.normal(0, 90, n)   # strong but genuinely noisy
    df = pd.DataFrame({"ad_spend": spend, "sales": sales})

    r = analyze_scenario(df, "ad_spend", "sales", 10.0)
    assert r.r_squared > 0.8
    assert r.driver_restates_target is False
    assert r.reliable is True
