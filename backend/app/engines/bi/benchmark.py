"""
engines/bi/benchmark.py — how a measure compares to its own history and
to a published range.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)



from app.engines.bi.results import BenchmarkResult


#  BENCHMARKING
# ══════════════════════════════════════════════════════════

def analyze_benchmark(df: pd.DataFrame, col: str) -> BenchmarkResult:
    """Full benchmark analysis for one numeric column."""
    s = df[col].dropna()

    mean   = float(s.mean())
    median = float(s.median())
    std    = float(s.std())
    cv     = std / abs(mean) if mean != 0 else 0

    p25  = float(s.quantile(0.25))
    p75  = float(s.quantile(0.75))
    p90  = float(s.quantile(0.90))
    p10  = float(s.quantile(0.10))

    above_avg_pct = float((s > mean).mean() * 100)

    if cv < 0.1:
        label = "Very Consistent — low variation across records"
    elif cv < 0.3:
        label = "Consistent — moderate variation"
    elif cv < 0.6:
        label = "Variable — significant spread"
    else:
        label = "Highly Variable — large spread, investigate outliers"

    # Mean vs median interpretation
    diff_pct = abs(mean - median) / abs(median) * 100 if median != 0 else 0
    if diff_pct > 20:
        central = ("Mean ({:.2f}) is {:.0f}% away from median ({:.2f}) — "
                   "skewed distribution, use median for central tendency.").format(
                       mean, diff_pct, median)
    else:
        central = ("Mean ({:.2f}) and median ({:.2f}) are close — "
                   "symmetric distribution.").format(mean, median)

    interp = (
        "{} | {:.1f}% of records are above average. "
        "Top 10% threshold: {:.2f}. Bottom 10%: {:.2f}. {}".format(
            label, above_avg_pct, p90, p10, central)
    )

    return BenchmarkResult(
        column=col, mean=round(mean, 4), median=round(median, 4),
        p25=round(p25, 4), p75=round(p75, 4), p90=round(p90, 4),
        top_10_pct=round(p90, 4), bottom_10_pct=round(p10, 4),
        above_avg_pct=round(above_avg_pct, 2),
        cv=round(cv, 4), benchmark_label=label,
        interpretation=interp,
    )


# ══════════════════════════════════════════════════════════