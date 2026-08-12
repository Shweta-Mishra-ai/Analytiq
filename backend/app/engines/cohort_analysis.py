"""
core/cohort_analysis.py — shared quantile-cohort and Pareto/concentration
helpers.

Finance (cost/revenue concentration), Ecommerce (Price Band Analysis,
Category Concentration), and Sales (Deal Size Cohort Analysis) in
pages/4_Business_Insights.py each re-implemented the same two patterns
independently:

1. Quantile cohorts — bin a numeric column into ~5 quantile bands and
   aggregate other columns per band (Ecommerce's Price Band Analysis and
   Sales' Deal Size Cohort Analysis were byte-for-byte the same shape of
   logic with different column names).
2. Pareto / concentration — group by a category, sum a value column,
   and report % of total / cumulative % (Finance's 80/20 concentration
   chart and Ecommerce's Category Concentration table were the same
   computation, one with a chart, one without).

Both are extracted here so all three domains share one tested
implementation instead of three that could silently drift apart.
"""
from __future__ import annotations
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def build_quantile_cohorts(
    df: pd.DataFrame,
    value_col: str,
    q: int = 5,
    agg_cols: Optional[dict] = None,
    band_label: str = "Band",
) -> pd.DataFrame:
    """Bin df[value_col] into up to `q` quantile bands and aggregate.

    `agg_cols` is an optional dict of {col: agg_func_or_list} for
    additional columns to aggregate per band (value_col itself always
    gets count + sum + mean). Returns a DataFrame indexed by band, with
    flattened "col_agg" column names, sorted by band order.

    Raises ValueError if value_col has no usable (non-null) data — callers
    should catch and show a friendly message, same as the call sites did
    before extraction.
    """
    series = df[value_col].dropna()
    if series.empty:
        raise ValueError(f"'{value_col}' has no non-null values to bin")

    bands = pd.qcut(series, q=min(q, series.nunique()), duplicates="drop")
    work = df.loc[series.index].copy()
    work["_band"] = bands

    agg_dict = {value_col: ["count", "sum", "mean"]}
    if agg_cols:
        agg_dict.update(agg_cols)

    tbl = work.groupby("_band", observed=True).agg(agg_dict).round(2)
    tbl.columns = ["_".join(c).strip() for c in tbl.columns]
    tbl.index.name = band_label
    return tbl


def concentration_analysis(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    pareto_threshold: float = 80.0,
) -> dict:
    """Group by `group_col`, sum `value_col`, and report concentration.

    Returns a dict with:
      - table: DataFrame indexed by group, sorted descending by total,
        with 'total', 'pct_of_total', 'cum_pct' columns
      - top_n_for_threshold: how many top groups account for
        `pareto_threshold`% of the total (the Pareto point)
      - total_groups: total number of groups
      - top3_pct: % of total held by the top 3 groups (a quick
        concentration-risk signal independent of the 80/20 threshold)
    """
    totals = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    grand_total = totals.sum()
    pct = (totals / grand_total * 100).round(1) if grand_total else totals * 0
    cum_pct = pct.cumsum()

    table = pd.DataFrame({
        "total": totals,
        "pct_of_total": pct,
        "cum_pct": cum_pct,
    })
    top_n = int((cum_pct <= pareto_threshold).sum()) + 1
    top_n = min(top_n, len(totals))

    return {
        "table": table,
        "top_n_for_threshold": top_n,
        "total_groups": len(totals),
        "top3_pct": float(pct.head(3).sum()) if len(pct) else 0.0,
    }
