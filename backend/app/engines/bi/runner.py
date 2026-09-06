"""
engines/bi/runner.py — sequencing the analyses into one report.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from app.engines.domains.base import is_id_column


from app.services.dtypes import text_columns
from app.services.stat_guards import is_restatement

from app.engines.bi.results import BIReport
from app.engines.bi.benchmark import analyze_benchmark
from app.engines.bi.root_cause import analyze_root_cause
from app.engines.bi.cohort import analyze_cohort
from app.engines.bi.pareto import analyze_pareto
from app.engines.bi.segments import analyze_segment_health
from app.engines.bi.insights import _generate_key_insights


# ══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

# Words that name something a business tries to move. Root-cause
# analysis asks "why is this low", which is only a question for a metric
# where low is worse — not for an age, a headcount or a postcode.
_PERFORMANCE_TOKENS = {
    "revenue", "sales", "profit", "margin", "income", "salary", "pay",
    "value", "spend", "cost", "price", "amount", "units", "volume",
    "quantity", "orders", "conversion", "rate", "score", "rating",
    "satisfaction", "engagement", "performance", "productivity",
    "efficiency", "utilisation", "utilization", "throughput", "output",
    "retention", "growth", "mrr", "arr", "ltv", "aov", "nps", "csat",
    "quality", "yield", "uptime", "accuracy",
}

# Facts about a row rather than a result it produced.
_DEMOGRAPHIC_TOKENS = {
    "age", "gender", "sex", "birth", "dob", "tenure", "years", "year",
    "month", "day", "date", "distance", "count", "number", "id", "code",
    "zip", "postcode", "level", "band", "grade",
}


def _is_performance_metric(col) -> bool:
    """True when "low {col}" describes underperformance."""
    import re as _re
    spaced = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(col)).lower()
    tokens = {t for t in _re.split(r"[^a-z0-9]+", spaced) if t}
    if tokens & _PERFORMANCE_TOKENS:
        return True
    return not (tokens & _DEMOGRAPHIC_TOKENS)


# How many distinct numeric columns the analyses below can consume.
# Benchmarks take four; everything else takes two.
_DEDUP_ENOUGH = 4


def run_bi(df: pd.DataFrame, max_rows: int = 50_000) -> BIReport:
    """
    Full BI pipeline.
    Auto-selects best columns for each analysis.
    """
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    # An identifier is not a metric. The report opened with "Root cause —
    # low EmployeeNumber: 368 low performers (25.0%)", which analyses a
    # row number as underperformance, and listed it as an insight.
    all_numeric = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in all_numeric if not is_id_column(c, df[c])]
    if len(num_cols) < len(all_numeric):
        logger.info("BI: excluded identifier column(s) %s",
                    ", ".join(c for c in all_numeric if c not in num_cols))

    # Every analysis below takes the first two or four numeric columns.
    # A file carrying both `revenue` and `revenue_k` spends two of those
    # slots on one measurement and prints the finding twice — "Region
    # splits Revenue: A averages 62.1% more than B" followed by "Region
    # splits Revenue K: A averages 76.6% more than D" — while a column
    # with something else to say never gets looked at. The first spelling
    # of a measurement is kept; later ones are dropped.
    #
    # Nothing below looks past the fourth distinct column, so once that
    # many are in hand the rest go through unchecked. That bound matters:
    # checking every pair at full length took a 50,000-row, 40-column run
    # from 0.5s to 16s.
    deduped, dropped = [], []
    for c in num_cols:
        if len(deduped) >= _DEDUP_ENOUGH:
            deduped.append(c)
            continue
        if any(is_restatement(df[c], df[kept]) for kept in deduped):
            dropped.append(c)
            continue
        deduped.append(c)
    if dropped:
        logger.info("BI: %s restate a column already being analysed — "
                    "dropped so the slots go to columns with something "
                    "else to say", ", ".join(dropped))
    num_cols = deduped
    cat_cols = [c for c in text_columns(df)
                if 2 <= df[c].nunique() <= 25]

    report = BIReport()

    # 1. Benchmarks — top 4 numeric cols
    for col in num_cols[:4]:
        try:
            report.benchmarks.append(analyze_benchmark(df, col))
        except Exception:
            logger.debug("run_bi: suppressed exception", exc_info=True)
            continue

    # 2. Root cause — on metrics that can meaningfully be "low"
    #
    # "Root cause of low Age: 368 low performers" treats a demographic
    # fact as underperformance. Root-cause analysis only means something
    # against a metric where more is better (or worse) by definition.
    performance_cols = [c for c in num_cols if _is_performance_metric(c)]
    if not performance_cols:
        logger.info("BI: no performance metric to run root cause against "
                    "among %s", ", ".join(num_cols))
    for col in performance_cols[:2]:
        try:
            report.root_causes.append(
                analyze_root_cause(df, col, threshold_pct=25))
        except Exception:
            logger.debug("run_bi: suppressed exception", exc_info=True)
            continue

    # 3. Cohort analysis — top cat × top numeric
    for cat in cat_cols[:2]:
        for num in num_cols[:2]:
            try:
                report.cohorts.append(analyze_cohort(df, cat, num))
            except Exception:
                logger.debug("run_bi: suppressed exception", exc_info=True)
                continue

    # 4. Pareto — top cat × top numeric
    for cat in cat_cols[:1]:
        for num in num_cols[:2]:
            try:
                agg = "mean" if "rate" in num.lower() or "score" in num.lower() \
                      or "rating" in num.lower() else "sum"
                report.pareto.append(analyze_pareto(df, cat, num, agg))
            except Exception:
                logger.debug("run_bi: suppressed exception", exc_info=True)
                continue

    # 5. Segment health
    if cat_cols and num_cols:
        try:
            report.segments = analyze_segment_health(
                df, cat_cols[0], num_cols[:4])
        except Exception:
            logger.debug("run_bi: suppressed exception", exc_info=True)

    # 6. Key insights + brief
    report.key_insights, report.executive_brief = _generate_key_insights(report, df)

    return report

