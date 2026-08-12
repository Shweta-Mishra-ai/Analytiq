"""
core/industry_benchmarks.py — publicly-known general industry ranges,
by domain, for the Performance vs Benchmark section of the report.

IMPORTANT — what this is and isn't:
  These are widely-cited, PUBLICLY KNOWN general ranges (the kind found in
  industry commentary, analyst notes, and common business literature) —
  NOT a verified, licensed benchmark database. They vary significantly by
  company size, sector, region, and business model, and every report that
  surfaces one carries an explicit caveat to that effect. Never present
  these as precise external data; they exist to give a reader a rough
  sense of "is this number in a plausible range" — nothing more.

Each entry: (low, high, unit, note). `note` is shown alongside the range
for context (e.g. what "healthy" looks like, or a caveat specific to that
metric).
"""
from __future__ import annotations

import logging
from typing import Optional, NamedTuple

logger = logging.getLogger(__name__)


class BenchmarkRange(NamedTuple):
    low: float
    high: float
    unit: str
    note: str


# ══════════════════════════════════════════════════════════
#  HR & PEOPLE ANALYTICS
# ══════════════════════════════════════════════════════════
HR_BENCHMARKS = {
    "attrition_rate": BenchmarkRange(10, 15, "%",
        "Annual voluntary attrition — varies widely by sector (retail/hospitality "
        "often 20-30%+, tech/professional services often single digits to mid-teens)"),
    "satisfaction_score": BenchmarkRange(3.5, 4.2, "/5",
        "Typical employee-satisfaction survey range on a 5-point scale"),
    "enps": BenchmarkRange(10, 30, "",
        "Employee Net Promoter Score — positive is generally considered healthy, 50+ is excellent"),
    "time_to_fill_days": BenchmarkRange(30, 45, "days",
        "Average time to fill an open role — varies heavily by role seniority/specialization"),
    "absenteeism_rate": BenchmarkRange(1.5, 3.0, "%",
        "Unplanned absence as a share of scheduled work days"),
    "training_hours_per_employee": BenchmarkRange(20, 40, "hrs/year",
        "Annual training investment per employee"),
}

# ══════════════════════════════════════════════════════════
#  SALES
# ══════════════════════════════════════════════════════════
SALES_BENCHMARKS = {
    "win_rate": BenchmarkRange(20, 30, "%",
        "Opportunities won as a share of opportunities closed (won+lost) — B2B SaaS "
        "medians often cited in the 20-30% range, varies a lot by deal complexity"),
    "sales_cycle_days": BenchmarkRange(30, 90, "days",
        "Time from opportunity creation to close-won — shorter for transactional, "
        "much longer for enterprise/complex sales"),
    "quota_attainment_pct_of_reps": BenchmarkRange(60, 75, "%",
        "Share of reps hitting 100% of quota — widely-cited commentary suggests "
        "many organizations see well under half"),
    "avg_deal_size_growth": BenchmarkRange(5, 15, "% YoY",
        "Healthy year-over-year growth in average deal size"),
    "sales_rep_turnover": BenchmarkRange(15, 25, "%",
        "Annual sales rep attrition — sales roles typically run higher than company-wide average"),
    "pipeline_coverage_ratio": BenchmarkRange(3, 4, "x",
        "Open pipeline value vs remaining quota — common rule-of-thumb minimum coverage"),
}

# ══════════════════════════════════════════════════════════
#  FINANCE
# ══════════════════════════════════════════════════════════
FINANCE_BENCHMARKS = {
    "gross_margin": BenchmarkRange(30, 50, "%",
        "Varies enormously by industry — software often 70-90%, retail often 20-40%, "
        "this range is a rough cross-industry midpoint, not sector-specific"),
    "net_margin": BenchmarkRange(5, 15, "%",
        "Net profit margin — highly sector-dependent"),
    "current_ratio": BenchmarkRange(1.5, 3.0, "x",
        "Current assets / current liabilities — below 1.0 is often a liquidity flag, "
        "above ~3 may indicate underused assets"),
    "quick_ratio": BenchmarkRange(1.0, 1.5, "x",
        "(Current assets - inventory) / current liabilities"),
    "opex_ratio": BenchmarkRange(60, 80, "%",
        "Operating expenses as a share of revenue — lower generally indicates better efficiency"),
    "budget_variance": BenchmarkRange(-5, 5, "%",
        "Actual vs budget — within +/-5% is commonly treated as 'on plan'"),
    "days_sales_outstanding": BenchmarkRange(30, 45, "days",
        "Average collection period — lower is generally better for cash flow"),
}

# ══════════════════════════════════════════════════════════
#  ECOMMERCE
# ══════════════════════════════════════════════════════════
ECOMMERCE_BENCHMARKS = {
    "conversion_rate": BenchmarkRange(1, 4, "%",
        "Visitors-to-purchase conversion — widely-cited overall web averages, "
        "varies a lot by category and traffic source"),
    "cart_abandonment": BenchmarkRange(60, 80, "%",
        "Share of carts started but not completed — commonly-cited range across studies"),
    "avg_order_value_growth": BenchmarkRange(3, 10, "% YoY",
        "Healthy year-over-year AOV growth"),
    "return_rate": BenchmarkRange(15, 30, "%",
        "Online return rate — apparel/fashion tends toward the higher end, "
        "other categories often lower"),
    "customer_rating": BenchmarkRange(4.0, 4.5, "/5",
        "Typical range for a healthy product/seller rating"),
    "repeat_purchase_rate": BenchmarkRange(20, 30, "%",
        "Share of customers who purchase again within a year"),
    "customer_acquisition_cost_payback": BenchmarkRange(3, 12, "months",
        "Time to recover CAC from a customer's gross margin — shorter is healthier"),
}

DOMAIN_BENCHMARKS = {
    "hr": HR_BENCHMARKS,
    "sales": SALES_BENCHMARKS,
    "finance": FINANCE_BENCHMARKS,
    "ecommerce": ECOMMERCE_BENCHMARKS,
}

# Column-name keywords -> benchmark key, per domain. First match wins.
_COLUMN_KEYWORD_MAP = {
    "hr": [
        (("attrition", "churn", "left", "exited"), "attrition_rate"),
        (("satisfaction",), "satisfaction_score"),
        (("enps", "net promoter"), "enps"),
        (("time_to_fill", "time to fill", "days_to_fill"), "time_to_fill_days"),
        (("absentee",), "absenteeism_rate"),
        (("training_hours", "training hours"), "training_hours_per_employee"),
    ],
    "sales": [
        (("win_rate", "win rate"), "win_rate"),
        (("sales_cycle", "cycle_days", "cycle length"), "sales_cycle_days"),
        (("quota_attainment", "quota attainment"), "quota_attainment_pct_of_reps"),
        (("deal_size_growth",), "avg_deal_size_growth"),
        (("rep_turnover", "sales turnover"), "sales_rep_turnover"),
        (("pipeline_coverage",), "pipeline_coverage_ratio"),
    ],
    "finance": [
        (("gross_margin", "gross margin"), "gross_margin"),
        (("net_margin", "net margin", "net_profit_margin"), "net_margin"),
        (("current_ratio",), "current_ratio"),
        (("quick_ratio",), "quick_ratio"),
        (("opex_ratio", "operating_expense_ratio"), "opex_ratio"),
        (("budget_variance",), "budget_variance"),
        (("dso", "days_sales_outstanding"), "days_sales_outstanding"),
    ],
    "ecommerce": [
        (("conversion_rate", "conversion rate"), "conversion_rate"),
        (("cart_abandon",), "cart_abandonment"),
        (("aov_growth", "order_value_growth"), "avg_order_value_growth"),
        (("return_rate", "return_pct"), "return_rate"),
        (("rating",), "customer_rating"),
        (("repeat_purchase", "repeat_rate"), "repeat_purchase_rate"),
        (("cac_payback", "payback_period"), "customer_acquisition_cost_payback"),
    ],
}


def lookup_benchmark(domain: str, column_name: str) -> Optional[BenchmarkRange]:
    """Find a general industry range for a column, if this domain has one
    that plausibly matches the column's name. Returns None if there's no
    confident match — callers should fall back to internal (top-quartile)
    benchmarking rather than guessing.
    """
    keyword_map = _COLUMN_KEYWORD_MAP.get(domain.lower())
    benchmarks = DOMAIN_BENCHMARKS.get(domain.lower())
    if not keyword_map or not benchmarks:
        return None
    col_norm = column_name.lower().replace("_", "").replace(" ", "")
    for keywords, key in keyword_map:
        for kw in keywords:
            kw_norm = kw.lower().replace("_", "").replace(" ", "")
            if kw_norm in col_norm:
                return benchmarks.get(key)
    return None


def format_benchmark_context(bm: BenchmarkRange) -> str:
    """Render a BenchmarkRange as the 'Context / Reference' table cell text
    used throughout the Performance vs Benchmark PDF section."""
    unit = bm.unit
    # Symbol-style units (%, x, /5) attach directly; word-style units
    # (days, months, hrs/year) get a space.
    if unit in ("%", "x") or unit.startswith("/"):
        range_str = f"{bm.low:g}-{bm.high:g}{unit}"
    elif unit:
        range_str = f"{bm.low:g}-{bm.high:g} {unit}"
    else:
        range_str = f"{bm.low:g}-{bm.high:g}"
    return f"General guidance: {range_str} — {bm.note}"
