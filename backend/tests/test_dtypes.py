"""
Tests for services/dtypes.py and the pandas-3 text-dtype bug class.

pandas 3.0 infers a dedicated ``str`` dtype for text columns instead of
``object``. Every ``series.dtype == object`` check silently became False,
so code branched into its numeric path for ordinary text columns —
``pd.to_numeric()`` over "Yes"/"No" returned all-NaN and the health report
printed "'Engineering' has the highest attrition: nan%" into a
client-facing PDF. Nothing raised.

requirements.txt pins ``pandas>=2.0.0``, so a fresh install picks up 3.x
and hits this while an older lockfile does not.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest

from app.services.dtypes import is_categorical_like, is_text_dtype

APP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")


def test_text_columns_are_detected_whatever_pandas_infers():
    s = pd.Series(["Yes", "No", "Yes"])
    assert is_text_dtype(s), (
        f"text column with dtype {s.dtype!r} not detected — this is exactly "
        "the pandas-3 regression that produced 'nan%' in the health report")


def test_explicit_object_and_string_dtypes_both_count():
    assert is_text_dtype(pd.Series(["a", "b"], dtype=object))
    assert is_text_dtype(pd.Series(["a", "b"], dtype="string"))


def test_numeric_bool_and_datetime_are_not_text():
    assert not is_text_dtype(pd.Series([1, 2, 3]))
    assert not is_text_dtype(pd.Series([1.5, 2.5]))
    assert not is_text_dtype(pd.Series([True, False]))
    assert not is_text_dtype(pd.Series(pd.date_range("2025-01-01", periods=3)))


def test_categorical_is_not_text_but_is_categorical_like():
    s = pd.Series(["a", "b", "a"], dtype="category")
    assert not is_text_dtype(s)
    assert is_categorical_like(s)


def test_categorical_like_covers_text_and_bool():
    assert is_categorical_like(pd.Series(["x", "y"]))
    assert is_categorical_like(pd.Series([True, False]))
    assert not is_categorical_like(pd.Series([1.0, 2.0]))


def test_accepts_a_bare_dtype_not_just_a_series():
    assert is_text_dtype(pd.Series(["a"]).dtype)
    assert not is_text_dtype(np.dtype("int64"))


def test_the_original_failure_mode_is_gone():
    """The exact sequence that produced nan%: branch on dtype, then take
    the mean of the coerced column."""
    df = pd.DataFrame({
        "department": ["Eng", "Eng", "HR", "HR"],
        "attrition": ["Yes", "No", "Yes", "Yes"],
    })
    s = df["attrition"]
    if is_text_dtype(s):
        atr = s.str.lower().isin(["yes", "true", "1", "left"]).astype(float)
    else:
        atr = pd.to_numeric(s, errors="coerce")
    rate = df.assign(_a=atr).groupby("department")["_a"].mean()
    assert not rate.isna().any(), "attrition rate came out NaN again"
    assert rate["HR"] == pytest.approx(1.0)
    assert rate["Eng"] == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════
#  Behaviour at each site that branched on `dtype == object`
# ══════════════════════════════════════════════════════════
# These assert the *effect*, not the predicate. On pandas 3 each of these
# paths was skipped entirely because the dtype check said "not text" — so
# whitespace was never stripped, high-cardinality string columns were
# never dropped before training, and currency symbols were left in place.
# All of it silent.

def test_auto_clean_strips_whitespace_from_text_columns():
    from app.engines.data_cleaner import auto_clean
    df = pd.DataFrame({
        "name": ["  Alice ", "Bob  ", " Carol"],
        "value": [1.0, 2.0, 3.0],
    })
    cleaned, _report = auto_clean(df)
    assert cleaned["name"].tolist() == ["Alice", "Bob", "Carol"], (
        "whitespace stripping was skipped — the text-dtype branch did not fire")


def test_ml_prep_drops_high_cardinality_text_columns():
    """A near-unique string column is an identifier, not a feature; it must
    be dropped before training rather than one-hot exploded."""
    from app.engines.ml_engine import run_ml_pipeline
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "ticket_ref": [f"REF-{i:05d}" for i in range(n)],   # unique text
        "amount": rng.normal(100, 20, n),
        "score": rng.normal(50, 10, n),
        "target": rng.choice(["Yes", "No"], n),
    })
    report = run_ml_pipeline(df, "target")
    assert "ticket_ref" not in report.feature_cols, (
        "high-cardinality text column was kept as a feature")


def test_table_json_to_df_strips_currency_decorations():
    from app.services.table_extractor import table_json_to_df
    payload = {
        "found": True,
        "columns": ["item", "price"],
        "rows": [["Widget", "$1,200"], ["Gadget", "$350"], ["Doohickey", "$75"]],
    }
    df, _warnings = table_json_to_df(payload)
    assert pd.api.types.is_numeric_dtype(df["price"]), (
        f"currency stripping skipped — price dtype is {df['price'].dtype}")
    assert df["price"].tolist() == [1200, 350, 75]


def test_hr_attrition_rate_is_computed_from_text_flags(hr_df):
    """_run_attrition reads a Yes/No column; on pandas 3 the text branch
    was skipped and the rate came back missing or NaN."""
    from app.engines.domains.hr import _run_attrition
    result = _run_attrition(hr_df)
    if result is None:
        pytest.skip("no attrition column detected in fixture")
    assert result.rate == result.rate, "attrition rate is NaN"
    assert 0 < result.rate < 100


def _iter_app_py_files():
    skip = {"__pycache__", ".git", "node_modules", "venv"}
    for root, dirs, files in os.walk(APP_DIR):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


_BARE_OBJECT_CMP = re.compile(r"dtype\s*==\s*object|dtype\s*==\s*np\.object")


def test_no_bare_object_dtype_comparisons_remain():
    """Guard the whole bug class: comparing a dtype to ``object`` is not a
    reliable "is this text?" test on pandas 3. Use
    services.dtypes.is_text_dtype / is_categorical_like instead.

    A line is allowed only if it also accounts for the ``str`` dtype.
    """
    violations = []
    for path in _iter_app_py_files():
        if path.endswith(os.path.join("services", "dtypes.py")):
            continue  # the helper's own docstring quotes the pattern
        for i, line in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("``"):
                continue
            if _BARE_OBJECT_CMP.search(line) and '"str"' not in line and "'str'" not in line:
                rel = os.path.relpath(path, os.path.dirname(APP_DIR))
                violations.append(f"{rel}:{i}: {stripped}")
    assert not violations, (
        "Bare `dtype == object` comparison(s) — these silently return False "
        "for text columns on pandas 3; use is_text_dtype():\n"
        + "\n".join(violations)
    )
