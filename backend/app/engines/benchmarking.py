"""
core/benchmarking.py
Honest benchmarking / contextualisation for the report.

A consulting report never leaves a number naked — it contextualises it against
a reference. We have no licence to invent industry figures (that was the old
fabricated-citation bug), so we benchmark two ways, both fully data-derived:

  1. Against an explicit TARGET the analyst supplies per metric (white-label /
     client-agreed goals). Only used when given.
  2. Against the dataset's OWN top-quartile ("internal best") — the standard
     move when no external benchmark exists: the average is N, the best quartile
     already achieves M, so M−N is the realistic headroom. This is exactly how
     a McKinsey/Deloitte "close the gap to your top performers" analysis reads.

A benchmark only makes sense when we know which DIRECTION is good. We assert a
direction ONLY for well-known metric families; anything ambiguous (salary/price —
good or bad depends on whose side you're on) is skipped rather than guessed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Metric families where a higher value is unambiguously better.
_HIGHER_BETTER = (
    "satisfaction", "engagement", "retention", "nps", "csat", "rating",
    "score", "revenue", "sales", "profit", "margin", "conversion",
    "productivity", "quality", "performance", "attendance", "uptime",
    "fulfilment", "fulfillment", "throughput", "win", "renewal",
)
# Metric families where a lower value is unambiguously better.
_LOWER_BETTER = (
    "attrition", "churn", "turnover", "cost", "expense", "defect",
    "error", "complaint", "absence", "absenteeism", "wait", "delay",
    "downtime", "risk", "late", "backlog", "rework", "cancellation",
    "refund", "return", "escalation", "overtime",
)
# Ambiguous / identifier-ish — never benchmark these (direction unknowable).
_SKIP = (
    "id", "index", "number", "code", "zip", "phone", "year", "age",
    "salary", "income", "price", "wage", "pay", "count", "qty", "quantity",
    "amount", "balance", "tenure", "hour", "day", "month", "date",
)


def metric_direction(col_name: str) -> int:
    """+1 higher-is-better, -1 lower-is-better, 0 unknown (skip)."""
    cl = col_name.lower()
    if any(k in cl for k in _SKIP):
        # An explicit good/bad word still wins over a generic skip word
        # (e.g. 'cost_score' -> lower better; 'overtime' -> lower better).
        if any(k in cl for k in _LOWER_BETTER):
            return -1
        if any(k in cl for k in _HIGHER_BETTER):
            return 1
        return 0
    if any(k in cl for k in _LOWER_BETTER):
        return -1
    if any(k in cl for k in _HIGHER_BETTER):
        return 1
    return 0


@dataclass
class BenchmarkContext:
    metric: str
    value: float               # dataset average
    reference: float           # target if given, else internal top-quartile
    reference_kind: str        # "target" | "internal top-quartile"
    direction: int             # +1 / -1
    gap: float                 # signed: reference - value (in metric units)
    headroom_pct: float        # % improvement available closing to reference
    meets: bool                # already at/above (good side of) the reference
    interpretation: str = ""


def _quartile_best(s: pd.Series, direction: int) -> float:
    """Mean of the 'good' quartile — top 25% if higher-better, else bottom 25%."""
    if direction > 0:
        cut = s.quantile(0.75)
        good = s[s >= cut]
    else:
        cut = s.quantile(0.25)
        good = s[s <= cut]
    return float(good.mean()) if len(good) else float(s.mean())


def benchmark_metric(
    series: pd.Series, name: str, target: Optional[float] = None
) -> Optional[BenchmarkContext]:
    """Benchmark one metric vs target (if given) or its own top quartile."""
    direction = metric_direction(name)
    if direction == 0:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 8 or s.nunique() < 4:
        return None

    value = float(s.mean())

    if target is not None:
        reference, kind = float(target), "target"
    else:
        reference, kind = _quartile_best(s, direction), "internal top-quartile"

    # "meets" = value already on the good side of the reference.
    meets = (value >= reference) if direction > 0 else (value <= reference)

    gap = reference - value
    denom = abs(value) if value != 0 else (abs(reference) or 1.0)
    headroom_pct = abs(gap) / denom * 100

    dir_word = "higher" if direction > 0 else "lower"
    ref_label = "target" if kind == "target" else "top-quartile benchmark"
    if meets:
        interp = (
            "Average {:.2f} already meets the {} ({:.2f}); "
            "{} is better here — hold and protect this level.".format(
                value, ref_label, reference, dir_word)
        )
    else:
        interp = (
            "Average {:.2f} vs {} {:.2f} — closing the gap is a "
            "{:.0f}% improvement ({} is better).".format(
                value, ref_label, reference, headroom_pct, dir_word)
        )

    return BenchmarkContext(
        metric=name, value=round(value, 4), reference=round(reference, 4),
        reference_kind=kind, direction=direction, gap=round(gap, 4),
        headroom_pct=round(headroom_pct, 1), meets=meets, interpretation=interp,
    )


def compute_benchmarks(
    df: pd.DataFrame,
    num_cols: Optional[List[str]] = None,
    targets: Optional[Dict[str, float]] = None,
    max_metrics: int = 5,
) -> List[BenchmarkContext]:
    """Benchmark the directional numeric metrics in the frame.

    Metrics with an explicit target are always included; the rest fill up to
    ``max_metrics``, most-headroom first (biggest opportunities lead).
    """
    targets = {k.lower(): v for k, v in (targets or {}).items()}
    if num_cols is None:
        num_cols = df.select_dtypes(include="number").columns.tolist()

    out: List[BenchmarkContext] = []
    for col in num_cols:
        try:
            tgt = targets.get(col.lower())
            ctx = benchmark_metric(df[col], col, tgt)
            if ctx is not None:
                out.append(ctx)
        except Exception:
            logger.warning("benchmark_metric failed for column '%s'", col, exc_info=True)
            continue

    # Targets first, then largest headroom (the biggest opportunities).
    out.sort(key=lambda c: (c.reference_kind != "target", -c.headroom_pct))
    return out[:max_metrics]
