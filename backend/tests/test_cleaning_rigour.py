"""
Missing-data and duplicate handling, judged the way a reviewer would.

The cleaner logged every action, but three things it did were not
defensible in a client report:
  - it median-filled columns up to 60% missing, then downstream statistics
    treated the fabricated values as observed;
  - it imputed missingness that carried signal, erasing the pattern;
  - it removed only byte-identical rows, so one entity recorded twice with
    differing values silently double-counted in every sum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.data_cleaner import (
    _describe_key_duplicates,
    _missingness_is_informative,
    auto_clean,
)


@pytest.fixture()
def leavers_df():
    """Satisfaction is absent precisely for the people who left — the
    absence itself is the finding."""
    rng = np.random.default_rng(3)
    n = 400
    left = rng.choice([True, False], n, p=[.3, .7])
    return pd.DataFrame({
        "satisfaction": np.where(left, np.nan, rng.uniform(3, 5, n)),
        "tenure_years": np.where(left, rng.uniform(0, 2, n),
                                 rng.uniform(4, 12, n)).round(1),
        "dept": rng.choice(["A", "B", "C"], n),
    })


# ══════════════════════════════════════════════════════════
#  Missingness that carries signal
# ══════════════════════════════════════════════════════════

def test_informative_missingness_is_detected(leavers_df):
    assert _missingness_is_informative(leavers_df, "satisfaction") is True


def test_random_missingness_is_not_flagged():
    rng = np.random.default_rng(9)
    n = 400
    vals = rng.normal(50, 10, n)
    vals[rng.random(n) < 0.25] = np.nan          # missing completely at random
    df = pd.DataFrame({"metric": vals, "other": rng.normal(0, 1, n)})
    assert _missingness_is_informative(df, "metric") is False


def test_informative_column_is_not_imputed(leavers_df):
    """Median-filling here would erase the pattern and report a mean partly
    computed from invented values."""
    cleaned, report = auto_clean(leavers_df)
    assert "satisfaction" in report.informative_missingness
    assert cleaned["satisfaction"].isna().sum() > 0, \
        "informative missingness was imputed away"
    assert "satisfaction" not in report.imputed_columns


def _wide_frame(rng, n, missing_col, missing_rate):
    """A frame wide enough that rows are not accidentally identical.

    With only two columns, every row sharing a NaN is a genuine exact
    duplicate and is removed before imputation ever runs — which makes a
    narrow fixture measure deduplication rather than imputation.
    """
    vals = rng.normal(60_000, 12_000, n)
    vals[rng.random(n) < missing_rate] = np.nan
    return pd.DataFrame({
        "row_ref": range(n),
        missing_col: vals,
        "score": rng.normal(50, 9, n).round(2),
        "dept": rng.choice(["A", "B", "C"], n),
        "region": rng.choice(["N", "S", "E", "W"], n),
    })


def test_ordinary_missingness_is_still_imputed_and_recorded():
    rng = np.random.default_rng(11)
    df = _wide_frame(rng, 300, "salary", 0.25)
    cleaned, report = auto_clean(df)
    assert cleaned["salary"].isna().sum() == 0
    assert "salary" in report.imputed_columns
    assert report.imputed_columns["salary"] > 15


def test_heavy_imputation_is_surfaced_for_caveating():
    rng = np.random.default_rng(12)
    df = _wide_frame(rng, 300, "metric", 0.45)
    _cleaned, report = auto_clean(df)
    assert "metric" in report.imputed_columns
    assert "metric" in report.heavily_imputed
    action = next(a for a in report.actions
                  if a.column == "metric" and "median" in str(a.action))
    assert "synthetic" in action.action


def test_column_beyond_the_usable_threshold_is_dropped():
    rng = np.random.default_rng(13)
    n = 200
    vals = rng.normal(10, 2, n)
    vals[rng.random(n) < 0.8] = np.nan           # 80% missing
    df = pd.DataFrame({"sparse": vals, "keep": rng.normal(0, 1, n)})
    cleaned, _report = auto_clean(df)
    assert "sparse" not in cleaned.columns


# ══════════════════════════════════════════════════════════
#  Duplicates
# ══════════════════════════════════════════════════════════

def test_exact_duplicate_rows_are_removed():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    cleaned, report = auto_clean(df)
    assert report.duplicates_removed == 1
    assert len(cleaned) == 3


def test_repeated_identity_keys_are_reported_not_silently_kept():
    """One entity recorded twice with differing values double-counts in
    every downstream sum; exact-row dedup cannot catch it."""
    rng = np.random.default_rng(14)
    n = 200
    df = pd.DataFrame({
        "customer_id": list(range(n - 25)) + list(range(25)),
        "amount": rng.normal(100, 20, n).round(2),
        "region": rng.choice(["N", "S"], n),
    })
    note = _describe_key_duplicates(df)
    assert note, "repeated identity keys were not detected"
    assert "customer_id" in note
    _cleaned, report = auto_clean(df)
    assert report.key_duplicate_note


def test_unique_keys_produce_no_false_alarm():
    rng = np.random.default_rng(15)
    n = 150
    df = pd.DataFrame({
        "order_id": range(n),
        "value": rng.normal(50, 5, n),
    })
    assert _describe_key_duplicates(df) == ""


def test_cleaning_never_silently_loses_rows():
    """Row count must be explainable from the report alone."""
    rng = np.random.default_rng(16)
    n = 250
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.choice(["p", "q"], n)})
    df = pd.concat([df, df.head(10)], ignore_index=True)   # 10 exact dupes
    cleaned, report = auto_clean(df)
    assert len(df) - len(cleaned) == report.duplicates_removed
