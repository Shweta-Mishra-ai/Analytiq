"""
engines/bi/pareto.py — how concentrated a measure is across a dimension.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)



from app.engines.bi.results import ParetoResult


#  PARETO ANALYSIS
# ══════════════════════════════════════════════════════════

def analyze_pareto(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    agg_fn: str = "sum",
) -> ParetoResult:
    """80/20 Pareto analysis."""
    fn  = np.sum if agg_fn == "sum" else np.mean
    agg = (df.groupby(group_col)[value_col]
             .agg(fn)
             .sort_values(ascending=False))

    total     = agg.sum()
    cum_pct   = (agg.cumsum() / total * 100).round(2)
    n         = len(agg)
    top_20_n  = max(1, int(np.ceil(n * 0.20)))

    groups_list = []
    for i, (name, val) in enumerate(agg.items()):
        groups_list.append({
            "rank":           i + 1,
            "name":           str(name)[:30],
            "value":          round(float(val), 4),
            "pct_of_total":   round(float(val / total * 100), 2),
            "cumulative_pct": round(float(cum_pct[name]), 2),
            "in_top_20":      i < top_20_n,
        })

    top_share = round(float(agg.iloc[:top_20_n].sum() / total * 100), 2)
    pareto_holds = top_share >= 60

    interp = (
        "Top {:.0f}% of '{}' groups ({} out of {}) account for {:.1f}% "
        "of total '{}'. {}".format(
            20, group_col, top_20_n, n, top_share, value_col,
            "Pareto principle HOLDS — concentrate on top performers." if pareto_holds
            else "Pareto principle does NOT hold — value is distributed evenly.")
    )

    return ParetoResult(
        group_col=group_col, value_col=value_col, agg_fn=agg_fn,
        groups=groups_list, top_20_pct_groups=top_20_n,
        top_groups_share=top_share, pareto_holds=pareto_holds,
        interpretation=interp,
    )


# ══════════════════════════════════════════════════════════