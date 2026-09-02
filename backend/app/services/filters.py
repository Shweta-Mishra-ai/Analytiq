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


def apply_filters(df: pd.DataFrame, filters: List[dict] | None) -> pd.DataFrame:
    if not filters:
        return df
    out = df
    for f in filters:
        col = f.get("column")
        op = (f.get("op") or "eq").lower()
        val = f.get("value")
        if col not in out.columns:
            continue
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
                mask = s.astype(str).str.contains(str(val), case=False, na=False)
            elif op in ("gt", "lt", "gte", "lte"):
                num = pd.to_numeric(s, errors="coerce")
                v = float(val)
                mask = {"gt": num > v, "lt": num < v,
                        "gte": num >= v, "lte": num <= v}[op]
            elif op == "between":
                lo, hi = val
                if pd.api.types.is_datetime64_any_dtype(s):
                    mask = (s >= pd.to_datetime(lo)) & (s <= pd.to_datetime(hi))
                else:
                    num = pd.to_numeric(s, errors="coerce")
                    mask = (num >= float(lo)) & (num <= float(hi))
            else:
                continue
            out = out[mask]
        except Exception:
            logger.debug("apply_filters: suppressed exception", exc_info=True)
            continue
    return out


def field_catalog(df: pd.DataFrame, max_unique: int = 50) -> list[dict]:
    """Column metadata for the dashboard builder and slicer dropdowns.

    Identifiers are marked rather than presented as measures. The
    dashboard picks its default tiles from the first numeric field, and
    on an HR extract that was EmployeeNumber — so a new user's first
    screen was a bar chart of summed employee ID numbers by department,
    a pie of the same, and a correlation matrix with a row number in it.
    """
    from app.engines.domains.base import is_id_column
    fields = []
    for col in df.columns:
        s = df[col]
        identifier = False
        if pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            identifier = is_id_column(col, s)
            kind = "identifier" if identifier else "numeric"
        else:
            kind = "categorical"
        entry: dict[str, Any] = {
            "name": str(col),
            "kind": kind,
            "is_identifier": identifier,
            "missing_pct": round(float(s.isna().mean()) * 100, 1),
            "unique": int(s.nunique()),
        }
        if kind == "categorical" and s.nunique() <= max_unique:
            entry["values"] = [str(v) for v in s.dropna().unique().tolist()[:max_unique]]
        if kind in ("numeric", "identifier"):
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
