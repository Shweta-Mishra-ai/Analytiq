"""Offering the choice instead of the dead end.

Column detection reads names, so a file whose customer column is called
`rep` or `account` failed it — and the page stopped there: "this dataset
doesn't look like transaction data". The endpoint had accepted explicit
column overrides all along; its own docstring said the suggestions were
"for a frontend column picker" that was never built.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.rfm_engine import detect_rfm_columns, rfm_candidates


@pytest.fixture()
def sales():
    """A real transaction file whose customer column is named `rep` —
    detection cannot find it, and every RFM ingredient is present."""
    rng = np.random.default_rng(0)
    n = 4_000
    return pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="2h"),
        "region": rng.choice(["North", "South", "East"], n),
        "rep": rng.choice([f"rep_{i:03d}" for i in range(40)], n),
        "units": rng.integers(1, 40, n),
        "revenue": rng.gamma(3, 90, n).round(2),
        "profit": (rng.gamma(3, 90, n) - 120).round(2),   # goes negative
    })


def test_detection_really_does_fail_here(sales):
    """The premise of the picker."""
    assert detect_rfm_columns(sales) is None


def test_every_role_gets_candidates(sales):
    c = rfm_candidates(sales)
    assert c["customer"] and c["date"] and c["monetary"]


def test_a_measurement_is_not_offered_as_a_customer(sales):
    """`profit` repeats too, at about 1.7 rows a value. Offering it as
    "who the customer is" would be worse than offering nothing."""
    names = [x["column"] for x in rfm_candidates(sales)["customer"]]
    assert "profit" not in names
    assert "revenue" not in names


def test_the_customer_column_is_offered_first(sales):
    """Labels before numbers, most distinct first: the column that
    identifies people rather than grouping them."""
    assert rfm_candidates(sales)["customer"][0]["column"] == "rep"


def test_money_is_offered_before_counts(sales):
    """Left in column order, `units` was the default answer to "what it
    was worth", and the page then reported a total value that was a
    count of items."""
    assert rfm_candidates(sales)["monetary"][0]["column"] == "revenue"


def test_a_column_with_one_value_per_row_is_not_a_customer(sales):
    """`order_id` identifies the transaction, not the buyer. RFM on it
    gives every customer a frequency of exactly one."""
    names = [x["column"] for x in rfm_candidates(sales)["customer"]]
    assert "order_id" not in names


def test_the_date_column_is_found_even_when_stored_as_text():
    df = pd.DataFrame({
        "who": ["a", "b"] * 60,
        "when": ["2024-03-0{}".format(i % 9 + 1) for i in range(120)],
        "spend": range(120),
    })
    assert "when" in [x["column"] for x in rfm_candidates(df)["date"]]


def test_candidates_carry_what_the_data_says(sales):
    """Each option is checkable rather than a bare column name."""
    top = rfm_candidates(sales)["customer"][0]
    assert "distinct" in top["note"] and "rows each" in top["note"]


def test_a_file_with_nothing_to_score_offers_nothing():
    """One row per value everywhere, and no date: there is no customer
    to score and the page should say so rather than offer a choice that
    cannot work."""
    df = pd.DataFrame({"a": range(50), "b": [f"x{i}" for i in range(50)]})
    c = rfm_candidates(df)
    assert not c["customer"]
    assert not c["date"]


def test_the_picked_columns_actually_change_the_answer(sales):
    """The point of the picker: a different money column gives a
    different total, rather than a cached one."""
    from app.engines.rfm_engine import RFMColumns, run_rfm

    by_revenue = run_rfm(sales, RFMColumns(
        customer_id="rep", date_col="order_date", monetary_col="revenue"))
    by_units = run_rfm(sales, RFMColumns(
        customer_id="rep", date_col="order_date", monetary_col="units"))
    assert by_revenue.total_revenue != by_units.total_revenue
    assert by_revenue.n_customers == by_units.n_customers == 40
