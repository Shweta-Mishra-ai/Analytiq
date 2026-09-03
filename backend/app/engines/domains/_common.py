"""
engines/domains/_common.py — column-finding and comparison helpers shared
by the domain insight engines.

Each domain engine needs the same three things before it can say anything:
locate the column that holds a concept ("which column is the spend?"),
turn a binary column into a rate, and compare groups on a measure. Written
once here so four engines cannot disagree about, say, what counts as a
usable grouping column.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app.engines.domains.base import is_id_column

logger = logging.getLogger(__name__)

# A categorical column is only useful for grouping when it has more than
# one group and few enough to read in a table.
MIN_GROUPS = 2
MAX_GROUPS = 25


def _norm(col: str) -> str:
    """'MonthlyCharges' / 'monthly_charges' -> 'monthlycharges'."""
    text = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ',
                  re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', str(col)))
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def find_col(df: pd.DataFrame, keywords: Sequence[str],
             exclude: Sequence[str] = (), numeric_only: bool = False
             ) -> Optional[str]:
    """First column whose normalised name contains any keyword.

    Keywords are tried in order, so callers list the most specific first
    ("costpercase" before "cost") — otherwise a loose keyword claims the
    column a precise one was meant to find.
    """
    if df is None or df.empty:
        return None
    excl = tuple(_norm(e) for e in exclude)
    cols = [c for c in df.columns
            if not numeric_only or pd.api.types.is_numeric_dtype(df[c])]
    for kw in keywords:
        k = _norm(kw)
        for c in cols:
            n = _norm(c)
            if k in n and not any(e in n for e in excl):
                return c
    return None


def find_measure(df: pd.DataFrame, keywords: Sequence[str],
                 exclude: Sequence[str] = ()) -> Optional[str]:
    """Like find_col but numeric and never an identifier — a 'total spend'
    that is really an account number produces confident nonsense."""
    col = find_col(df, keywords, exclude=exclude, numeric_only=True)
    if col is not None and is_id_column(col, df[col]):
        return None
    return col


def grouping_columns(df: pd.DataFrame) -> List[str]:
    """Categorical columns with a usable number of groups."""
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or is_id_column(c, df[c]):
            continue
        n = df[c].nunique(dropna=True)
        if MIN_GROUPS <= n <= MAX_GROUPS:
            out.append(c)
    return out


def binary_rate(series: pd.Series) -> Optional[float]:
    """Percentage of 'yes' in a binary column, or None if it isn't binary.

    Accepts 0/1, True/False and the usual word pairs, because a churn flag
    arrives as any of them.
    """
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    vals = set(str(v).strip().lower() for v in s.unique())
    if len(vals) != 2:
        return None
    positives = {"1", "yes", "y", "true", "t", "churned", "left", "exited",
                 "attrited", "cancelled", "canceled", "lost", "failed",
                 "readmitted", "positive"}
    hit = vals & positives
    if not hit:
        # Numeric 0/1 stored as floats still counts.
        if vals <= {"0.0", "1.0", "0", "1"}:
            hit = {"1.0", "1"} & vals
        if not hit:
            return None
    marker = next(iter(hit))
    match = s.astype(str).str.strip().str.lower() == marker
    return round(float(match.mean() * 100), 2)


def segment_gap(df: pd.DataFrame, group_col: str, value_col: str,
                agg: str = "mean", min_n: int = 5
                ) -> Optional[Dict]:
    """Best and worst group on a measure, with the size of the gap.

    Groups smaller than `min_n` are dropped: a "worst region" that is one
    record is noise, and printing it as a finding is how a report loses
    a reader's trust.
    """
    if group_col not in df.columns or value_col not in df.columns:
        return None
    try:
        work = df[[group_col, value_col]].dropna()
        if work.empty:
            return None
        grouped = work.groupby(group_col)[value_col]
        sizes = grouped.size()
        keep = sizes[sizes >= min_n].index
        if len(keep) < 2:
            return None
        agg_vals = getattr(grouped, agg)().loc[keep].sort_values()
        worst_name, worst_val = agg_vals.index[0], float(agg_vals.iloc[0])
        best_name, best_val = agg_vals.index[-1], float(agg_vals.iloc[-1])
        if not np.isfinite(best_val) or not np.isfinite(worst_val):
            return None
        # Ratio is undefined against a zero baseline — report the absolute
        # gap instead of dividing and emitting inf.
        ratio = (best_val / worst_val) if worst_val not in (0, 0.0) else None
        return {
            "group_col": group_col, "value_col": value_col,
            "best": str(best_name), "best_val": best_val,
            "worst": str(worst_name), "worst_val": worst_val,
            "gap": best_val - worst_val,
            "ratio": ratio,
            "n_groups": int(len(keep)),
            "n_best": int(sizes.loc[best_name]),
            "n_worst": int(sizes.loc[worst_name]),
        }
    except Exception:
        logger.debug("segment_gap failed for %s by %s", value_col, group_col,
                     exc_info=True)
        return None


def concentration(df: pd.DataFrame, group_col: str, value_col: str,
                  top_n: int = 1) -> Optional[Dict]:
    """Share of a total held by the largest group(s) — the Pareto check
    behind 'one channel carries 68% of spend'."""
    if group_col not in df.columns or value_col not in df.columns:
        return None
    try:
        totals = df.groupby(group_col)[value_col].sum().sort_values(
            ascending=False)
        grand = float(totals.sum())
        if grand <= 0 or totals.empty:
            return None
        top = totals.head(top_n)
        return {
            "top_names": [str(i) for i in top.index],
            "top_share": round(float(top.sum()) / grand * 100, 1),
            "n_groups": int(len(totals)),
            "total": grand,
        }
    except Exception:
        logger.debug("concentration failed", exc_info=True)
        return None


def variability(series: pd.Series) -> Optional[float]:
    """Coefficient of variation as a percentage — how unstable a process
    measure is, independent of its units."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return None
    mean = float(s.mean())
    if mean == 0:
        return None
    return round(float(s.std() / abs(mean)) * 100, 1)


def fmt(value: float, unit: str = "") -> str:
    """Numbers a reader can scan: thousands separated, sensible precision."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if abs(value) >= 1_000_000:
        body = f"{value/1_000_000:,.1f}M"
    elif abs(value) >= 1000:
        body = f"{value:,.0f}"
    elif abs(value) >= 10:
        body = f"{value:,.1f}"
    else:
        body = f"{value:,.2f}"
    return f"{body}{unit}"


def benchmark_note(domain: str, column: str, value: float) -> str:
    """One sentence placing a value against its published range, or ''.

    Returns empty rather than inventing a comparison when no range applies
    — an unsupported benchmark is worse than none.
    """
    try:
        from app.engines.industry_benchmarks import (
            lookup_benchmark, format_benchmark_context)
        bm = lookup_benchmark(domain, column)
        if bm is None:
            return ""
        return format_benchmark_context(bm)
    except Exception:
        logger.debug("benchmark lookup failed for %s/%s", domain, column,
                     exc_info=True)
        return ""
