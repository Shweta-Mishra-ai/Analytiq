"""
services/filters.py — Power BI-style cross-filtering.
A filter is {"column": str, "op": str, "value": any}.
Applied server-side before any chart/KPI computation, so every
dashboard tile reflects the same slicer state.
"""
from __future__ import annotations
import logging

from typing import Any, List

import pandas as pd

logger = logging.getLogger(__name__)


class FilterError(ValueError):
    """A filter that cannot be applied as written."""


def apply_filters(df: pd.DataFrame, filters: List[dict] | None) -> pd.DataFrame:
    """Apply every filter, or raise FilterError naming the one that failed.

    A filter that could not be applied used to be skipped silently, so the
    dashboard showed unfiltered totals under a filter bar saying otherwise
    — a stale filter on a column cleaning had dropped, a non-numeric bound,
    or a "contains" value with a regex metacharacter like "(" all did it.
    """
    if not filters:
        return df
    out = df
    for f in filters:
        col = f.get("column")
        op = (f.get("op") or "eq").lower()
        val = f.get("value")
        if col not in out.columns:
            raise FilterError(
                f"The filter on '{col}' no longer applies — that column is not "
                f"in the dataset (cleaning may have removed it). Remove the filter.")
        s = out[col]
        try:
            if op == "eq":
                mask = s.astype(str) == str(val)
            elif op == "ne":
                mask = s.astype(str) != str(val)
            elif op == "in":
                vals = [str(v) for v in (val if isinstance(val, list) else [val])]
                mask = s.astype(str).isin(vals)
            elif op == "contains":
                # Literal match: the value is what the user typed, not a regex.
                mask = s.astype(str).str.contains(str(val), case=False, na=False,
                                                  regex=False)
            elif op in ("gt", "lt", "gte", "lte"):
                num = pd.to_numeric(s, errors="coerce")
                v = float(val)
                mask = {"gt": num > v, "lt": num < v,
                        "gte": num >= v, "lte": num <= v}[op]
            elif op == "between":
                lo, hi = val
                if pd.api.types.is_datetime64_any_dtype(s):
                    lo_t, hi_t = pd.to_datetime(lo), pd.to_datetime(hi)
                    if getattr(s.dt, "tz", None) is not None:
                        lo_t = lo_t.tz_localize(s.dt.tz) if lo_t.tzinfo is None else lo_t
                        hi_t = hi_t.tz_localize(s.dt.tz) if hi_t.tzinfo is None else hi_t
                    mask = (s >= lo_t) & (s <= hi_t)
                else:
                    num = pd.to_numeric(s, errors="coerce")
                    mask = (num >= float(lo)) & (num <= float(hi))
            else:
                raise FilterError(f"Unknown filter operator '{op}'")
        except FilterError:
            raise
        except Exception as e:
            raise FilterError(
                f"The filter {op} {val!r} cannot be applied to '{col}': {e}") from e
        out = out[mask]
    return out


def field_catalog(df: pd.DataFrame, max_unique: int = 50) -> list[dict]:
    """Column metadata for the dashboard builder / slicer dropdowns."""
    fields = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        else:
            kind = "categorical"
        entry: dict[str, Any] = {
            "name": str(col),
            "kind": kind,
            "missing_pct": round(float(s.isna().mean()) * 100, 1),
            "unique": int(s.nunique()),
        }
        if kind == "categorical" and s.nunique() <= max_unique:
            entry["values"] = [str(v) for v in s.dropna().unique().tolist()[:max_unique]]
        if kind == "numeric":
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if len(clean):
                entry["min"] = float(clean.min())
                entry["max"] = float(clean.max())
        if kind == "datetime":
            clean = s.dropna()
            if len(clean):
                entry["min"] = str(clean.min())
                entry["max"] = str(clean.max())
        fields.append(entry)
    return fields
