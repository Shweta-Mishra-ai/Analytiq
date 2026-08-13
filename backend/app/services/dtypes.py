"""
services/dtypes.py — dtype predicates that behave the same across pandas
versions.

Why this exists: pandas 3.0 infers a dedicated ``str`` dtype for text
columns instead of the historical ``object`` dtype. Every
``series.dtype == object`` check therefore silently became False for
ordinary text columns, and code that branches on it took the numeric
path instead — e.g. the health report ran ``pd.to_numeric()`` over
"Yes"/"No" values, got all-NaN, and printed
"'Engineering' has the highest attrition: nan%" into a client-facing PDF.

Nothing raised. The requirement pin is ``pandas>=2.0.0``, so a fresh
install picks up 3.x and hits this; an older lockfile does not, which is
what makes it easy to miss.

Use ``is_text_dtype()`` for "is this a text/categorical-ish column?"
rather than comparing dtypes directly.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def is_text_dtype(obj) -> bool:
    """True for text-like columns under any supported pandas version.

    Accepts a Series or a dtype. Covers the legacy ``object`` dtype and
    pandas 3's ``str`` / ``StringDtype``. Numeric, boolean, datetime and
    categorical dtypes return False (use ``is_categorical_like`` when a
    category column should also count).
    """
    dtype = getattr(obj, "dtype", obj)
    try:
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            return False
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return False
        if isinstance(dtype, pd.CategoricalDtype):
            return False
        return (pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype))
    except Exception:
        logger.debug("is_text_dtype: unrecognised dtype %r", dtype, exc_info=True)
        return False


def text_columns(df) -> list:
    """The text-like column names of a frame, in order.

    Replaces ``df.select_dtypes(include="object")``. That call still finds
    pandas 3's ``str`` columns, but only through a deprecation shim that
    warns and is scheduled for removal — at which point every categorical
    breakdown in the app would quietly return an empty list and the
    reports would lose their segment analysis without erroring.
    """
    return [c for c in df.columns if is_text_dtype(df[c])]


def month_end_rule() -> str:
    """The resample alias for month-end, valid on this pandas version.

    pandas 3 removed the ``"M"`` alias in favour of ``"ME"``; calling
    ``resample("M")`` now raises. Every call site here sat inside a
    try/except that logged at debug level, so the failures were invisible:
    the time-series trend chart vanished from every report and Deep EDA's
    stationarity/trend section silently returned nothing.
    """
    test = pd.Series(
        [0.0, 1.0],
        index=pd.to_datetime(["2025-01-01", "2025-02-01"]),
    )
    for alias in ("ME", "M"):
        try:
            test.resample(alias).mean()
            return alias
        except (ValueError, KeyError):
            continue
    logger.warning("no usable month-end resample alias found; defaulting to 'ME'")
    return "ME"


MONTH_END = month_end_rule()


def is_categorical_like(obj) -> bool:
    """True for text columns *and* pandas categorical/boolean columns —
    i.e. anything that groups into discrete buckets rather than measuring
    a quantity."""
    dtype = getattr(obj, "dtype", obj)
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    try:
        if pd.api.types.is_bool_dtype(dtype):
            return True
    except Exception:
        logger.debug("is_categorical_like: unrecognised dtype %r", dtype, exc_info=True)
    return is_text_dtype(dtype)
