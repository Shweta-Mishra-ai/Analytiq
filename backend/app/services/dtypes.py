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
