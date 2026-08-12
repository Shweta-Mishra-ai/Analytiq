"""Unit tests for engines/rfm_engine.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.rfm_engine import RFMColumns, detect_rfm_columns, run_rfm


def test_detect_rfm_columns_finds_standard_names(transactions_df):
    cols = detect_rfm_columns(transactions_df)
    assert cols is not None
    assert cols.customer_id == "customer_id"
    assert cols.date_col == "order_date"
    assert cols.monetary_col == "amount"


def test_detect_rfm_columns_returns_none_without_id_or_date():
    df = pd.DataFrame({"amount": [1, 2, 3], "notes": ["a", "b", "c"]})
    assert detect_rfm_columns(df) is None


def test_run_rfm_auto_detects_and_segments(transactions_df):
    report = run_rfm(transactions_df)
    assert report.n_customers > 0
    assert report.n_customers <= transactions_df["customer_id"].nunique()
    assert len(report.segment_summary) > 0
    # every customer must land in exactly one named segment
    all_customers = sum(s.n_customers for s in report.segment_summary)
    assert all_customers == report.n_customers
    # percentages of customers across segments should sum to ~100
    assert abs(sum(s.pct_customers for s in report.segment_summary) - 100.0) < 1.0
    assert set(report.customer_table.columns) == {
        "CustomerID", "Recency", "Frequency", "Monetary",
        "R", "F", "M", "RFM_Score", "Segment",
    }


def test_run_rfm_too_few_customers_raises():
    df = pd.DataFrame({
        "customer_id": [1, 1, 2, 2, 3],
        "order_date": pd.date_range("2025-01-01", periods=5),
        "amount": [10, 20, 30, 40, 50],
    })
    with pytest.raises(ValueError, match="unique customers"):
        run_rfm(df)


def test_run_rfm_no_monetary_column_falls_back_to_frequency(transactions_df):
    df = transactions_df.drop(columns=["amount"])
    report = run_rfm(df)
    assert any("Monetary score approximated" in w for w in report.warnings)


def test_run_rfm_derives_monetary_from_price_times_quantity():
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({
        "customer_id": rng.integers(1, 40, n),
        "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D"),
        "unit_price": rng.uniform(5, 50, n).round(2),
        "quantity": rng.integers(1, 5, n),
    })
    columns = RFMColumns(customer_id="customer_id", date_col="order_date",
                          price_col="unit_price", quantity_col="quantity")
    report = run_rfm(df, columns=columns)
    assert report.total_revenue > 0
    assert any("derived from" in w for w in report.warnings)


def test_run_rfm_rejects_non_dataframe():
    with pytest.raises(TypeError):
        run_rfm([1, 2, 3])  # type: ignore[arg-type]
