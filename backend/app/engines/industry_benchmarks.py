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
    # Where the range comes from, named so a reader can go and check it.
    # An unattributed number in a client report is worth nothing — the
    # first question anyone senior asks is "says who?", and "industry
    # standard" is not an answer.
    source: str = "General industry commentary"


# ══════════════════════════════════════════════════════════
#  HR & PEOPLE ANALYTICS
# ══════════════════════════════════════════════════════════
HR_BENCHMARKS = {
    "attrition_rate": BenchmarkRange(10, 15, "%",
        "Annual voluntary attrition — varies widely by sector (retail/hospitality "
        "often 20-30%+, tech/professional services often single digits to mid-teens)",
        "SHRM, State of the Workplace / annual turnover reporting"),
    "satisfaction_score": BenchmarkRange(3.5, 4.2, "/5",
        "Typical employee-satisfaction survey range on a 5-point scale",
        "Common survey-instrument convention (5-point Likert)"),
    "enps": BenchmarkRange(10, 30, "",
        "Employee Net Promoter Score — positive is generally considered healthy, 50+ is excellent",
        "Bain & Company NPS methodology, applied to employees"),
    "time_to_fill_days": BenchmarkRange(30, 45, "days",
        "Average time to fill an open role — varies heavily by role seniority/specialization",
        "SHRM talent-acquisition benchmarking"),
    "absenteeism_rate": BenchmarkRange(1.5, 3.0, "%",
        "Unplanned absence as a share of scheduled work days",
        "US Bureau of Labor Statistics absence-rate series"),
    "training_hours_per_employee": BenchmarkRange(20, 40, "hrs/year",
        "Annual training investment per employee",
        "ATD State of the Industry"),
}

# ══════════════════════════════════════════════════════════
#  SALES
# ══════════════════════════════════════════════════════════
SALES_BENCHMARKS = {
    "win_rate": BenchmarkRange(20, 30, "%",
        "Opportunities won as a share of opportunities closed (won+lost) — B2B SaaS "
        "medians often cited in the 20-30% range, varies a lot by deal complexity",
        "Widely-reported B2B sales-operations medians"),
    "sales_cycle_days": BenchmarkRange(30, 90, "days",
        "Time from opportunity creation to close-won — shorter for transactional, "
        "much longer for enterprise/complex sales",
        "Widely-reported B2B sales-cycle medians"),
    "quota_attainment_pct_of_reps": BenchmarkRange(60, 75, "%",
        "Share of reps hitting 100% of quota — widely-cited commentary suggests "
        "many organizations see well under half",
        "Sales-compensation survey commentary"),
    "avg_deal_size_growth": BenchmarkRange(5, 15, "% YoY",
        "Healthy year-over-year growth in average deal size",
        "General sales-operations commentary"),
    "sales_rep_turnover": BenchmarkRange(15, 25, "%",
        "Annual sales rep attrition — sales roles typically run higher than company-wide average",
        "SHRM turnover reporting, sales-function commentary"),
    "pipeline_coverage_ratio": BenchmarkRange(3, 4, "x",
        "Open pipeline value vs remaining quota — common rule-of-thumb minimum coverage",
        "Standard sales-operations rule of thumb"),
}

# ══════════════════════════════════════════════════════════
#  FINANCE
# ══════════════════════════════════════════════════════════
FINANCE_BENCHMARKS = {
    "gross_margin": BenchmarkRange(30, 50, "%",
        "Varies enormously by industry — software often 70-90%, retail often 20-40%, "
        "this range is a rough cross-industry midpoint, not sector-specific",
        "Damodaran (NYU Stern) published sector margin data"),
    "net_margin": BenchmarkRange(5, 15, "%",
        "Net profit margin — highly sector-dependent",
        "Damodaran (NYU Stern) published sector margin data"),
    "current_ratio": BenchmarkRange(1.5, 3.0, "x",
        "Current assets / current liabilities — below 1.0 is often a liquidity flag, "
        "above ~3 may indicate underused assets",
        "CFA Institute liquidity-ratio conventions"),
    "quick_ratio": BenchmarkRange(1.0, 1.5, "x",
        "(Current assets - inventory) / current liabilities",
        "CFA Institute liquidity-ratio conventions"),
    "opex_ratio": BenchmarkRange(60, 80, "%",
        "Operating expenses as a share of revenue — lower generally indicates better efficiency",
        "CFA Institute ratio conventions"),
    "budget_variance": BenchmarkRange(-5, 5, "%",
        "Actual vs budget — within +/-5% is commonly treated as 'on plan'",
        "Standard FP&A variance-review convention"),
    "days_sales_outstanding": BenchmarkRange(30, 45, "days",
        "Average collection period — lower is generally better for cash flow",
        "Working-capital management convention (CFA / CIMA)"),
}

# ══════════════════════════════════════════════════════════
#  ECOMMERCE
# ══════════════════════════════════════════════════════════
ECOMMERCE_BENCHMARKS = {
    "conversion_rate": BenchmarkRange(1, 4, "%",
        "Visitors-to-purchase conversion — widely-cited overall web averages, "
        "varies a lot by category and traffic source",
        "Published e-commerce conversion studies"),
    "cart_abandonment": BenchmarkRange(60, 80, "%",
        "Share of carts started but not completed — commonly-cited range across studies",
        "Baymard Institute cart-abandonment meta-study"),
    "avg_order_value_growth": BenchmarkRange(3, 10, "% YoY",
        "Healthy year-over-year AOV growth",
        "General retail commentary"),
    "return_rate": BenchmarkRange(15, 30, "%",
        "Online return rate — apparel/fashion tends toward the higher end, "
        "other categories often lower",
        "National Retail Federation returns reporting"),
    "customer_rating": BenchmarkRange(4.0, 4.5, "/5",
        "Typical range for a healthy product/seller rating",
        "Marketplace rating-distribution studies"),
    "repeat_purchase_rate": BenchmarkRange(20, 30, "%",
        "Share of customers who purchase again within a year",
        "Published e-commerce cohort studies"),
    "customer_acquisition_cost_payback": BenchmarkRange(3, 12, "months",
        "Time to recover CAC from a customer's gross margin — shorter is healthier",
        "SaaS/DTC unit-economics convention"),
}

HR_BENCHMARKS.update({
    "cost_per_hire": BenchmarkRange(3000, 5000, "USD",
        "Total recruiting cost per hire — scales sharply with seniority",
        "SHRM talent-acquisition benchmarking"),
    "offer_acceptance_rate": BenchmarkRange(85, 95, "%",
        "Offers accepted as a share of offers extended — a falling rate usually "
        "points at compensation or candidate experience",
        "Talent-acquisition benchmarking commentary"),
    "internal_mobility_rate": BenchmarkRange(6, 12, "%",
        "Share of roles filled by internal moves in a year",
        "LinkedIn Workforce / talent-mobility reporting"),
    "first_year_attrition": BenchmarkRange(15, 25, "%",
        "Leavers within twelve months of joining — a hiring and onboarding "
        "measure rather than a retention one",
        "SHRM onboarding research"),
    "span_of_control": BenchmarkRange(6, 10, "reports",
        "Direct reports per manager — below ~4 usually signals an over-layered "
        "structure, above ~12 a stretched one",
        "Organisational-design convention"),
    "gender_pay_gap": BenchmarkRange(0, 5, "%",
        "Unadjusted median pay difference; anything above a few per cent "
        "warrants a like-for-like analysis before any conclusion",
        "ILO / national pay-gap reporting conventions"),
})

FINANCE_BENCHMARKS.update({
    "ebitda_margin": BenchmarkRange(10, 20, "%",
        "Earnings before interest, tax, depreciation and amortisation as a "
        "share of revenue — strongly sector-dependent",
        "Damodaran (NYU Stern) published sector data"),
    "days_payable_outstanding": BenchmarkRange(30, 60, "days",
        "Average time taken to pay suppliers",
        "Working-capital management convention"),
    "days_inventory_outstanding": BenchmarkRange(30, 90, "days",
        "Average time inventory is held before sale",
        "Working-capital management convention"),
    "cash_conversion_cycle": BenchmarkRange(30, 60, "days",
        "DSO + DIO − DPO. Negative is possible and healthy for businesses "
        "paid before they pay",
        "Working-capital management convention"),
    "revenue_per_employee": BenchmarkRange(150_000, 300_000, "USD",
        "A productivity measure only comparable within an industry",
        "Cross-industry published company filings"),
    "bad_debt_ratio": BenchmarkRange(0.5, 2.0, "%",
        "Receivables written off as a share of revenue",
        "Credit-management convention"),
})

SALES_BENCHMARKS.update({
    "lead_to_opportunity_rate": BenchmarkRange(10, 20, "%",
        "Leads that become qualified opportunities",
        "B2B demand-generation commentary"),
    "quota_attainment_avg": BenchmarkRange(80, 100, "%",
        "Average attainment across the team, distinct from the share of reps "
        "who hit quota",
        "Sales-compensation survey commentary"),
    "churn_rate_annual": BenchmarkRange(5, 15, "%",
        "Annual customer churn for a subscription business",
        "SaaS benchmarking commentary"),
    "discount_rate_avg": BenchmarkRange(5, 15, "%",
        "Average discount off list — persistent higher discounting usually "
        "indicates a pricing or qualification problem rather than a sales one",
        "Pricing-practice commentary"),
})

ECOMMERCE_BENCHMARKS.update({
    "email_capture_rate": BenchmarkRange(2, 5, "%",
        "Visitors who provide an email address",
        "Published e-commerce marketing studies"),
    "mobile_share_of_traffic": BenchmarkRange(60, 75, "%",
        "Share of sessions from mobile devices",
        "Published web-traffic reporting"),
    "customer_lifetime_to_cac": BenchmarkRange(3, 5, "x",
        "Lifetime value to acquisition cost — below 3x usually means growth "
        "is buying revenue rather than building it",
        "SaaS/DTC unit-economics convention"),
    "gross_margin_retail": BenchmarkRange(30, 50, "%",
        "Retail gross margin before fulfilment and marketing",
        "Retail sector reporting"),
})

# ══════════════════════════════════════════════════════════
#  MARKETING
# ══════════════════════════════════════════════════════════
MARKETING_BENCHMARKS = {
    "email_open_rate": BenchmarkRange(20, 30, "%",
        "Opens as a share of delivered email — inflated by privacy features "
        "that pre-fetch images, so treat as directional only",
        "Published email-marketing benchmark reports"),
    "email_click_rate": BenchmarkRange(2, 5, "%",
        "Clicks as a share of delivered email — a more reliable measure than "
        "opens for the same reason",
        "Published email-marketing benchmark reports"),
    "paid_search_ctr": BenchmarkRange(2, 5, "%",
        "Click-through rate on paid search",
        "Published search-advertising benchmark reports"),
    "landing_page_conversion": BenchmarkRange(2, 6, "%",
        "Visitors completing the page's goal",
        "Published conversion-optimisation studies"),
    "marketing_cost_of_revenue": BenchmarkRange(5, 15, "%",
        "Marketing spend as a share of revenue — higher for growth-stage "
        "businesses, lower for established ones",
        "Published CMO spend surveys"),
    "customer_acquisition_cost_ratio": BenchmarkRange(3, 5, "x",
        "Lifetime value to acquisition cost",
        "SaaS/DTC unit-economics convention"),
}

# ══════════════════════════════════════════════════════════
#  OPERATIONS & MANUFACTURING
# ══════════════════════════════════════════════════════════
OPERATIONS_BENCHMARKS = {
    "on_time_delivery": BenchmarkRange(95, 98, "%",
        "Orders delivered by the promised date",
        "Supply-chain performance convention (SCOR model)"),
    "order_accuracy": BenchmarkRange(97, 99.5, "%",
        "Orders shipped without error",
        "Supply-chain performance convention (SCOR model)"),
    "oee": BenchmarkRange(60, 85, "%",
        "Overall equipment effectiveness — availability x performance x "
        "quality. 85% is the widely-cited world-class threshold",
        "TPM / OEE standard definition"),
    "defect_rate": BenchmarkRange(0.5, 3.0, "%",
        "Units failing quality inspection",
        "Quality-management convention"),
    "inventory_turnover": BenchmarkRange(4, 12, "x/year",
        "Cost of goods sold divided by average inventory",
        "Operations-management convention"),
    "capacity_utilisation": BenchmarkRange(70, 85, "%",
        "Sustained utilisation above ~85% typically removes the slack needed "
        "to absorb demand variation",
        "Operations-management convention"),
}

# ══════════════════════════════════════════════════════════
#  SUBSCRIPTION / SAAS
# ══════════════════════════════════════════════════════════
SAAS_BENCHMARKS = {
    "gross_revenue_retention": BenchmarkRange(85, 95, "%",
        "Revenue retained from existing customers before expansion",
        "SaaS benchmarking commentary"),
    "net_revenue_retention": BenchmarkRange(100, 120, "%",
        "Revenue retained including expansion — above 100% means the existing "
        "base grows without new customers",
        "SaaS benchmarking commentary"),
    "logo_churn_monthly": BenchmarkRange(0.5, 2.0, "%",
        "Customers lost per month",
        "SaaS benchmarking commentary"),
    "cac_payback_months": BenchmarkRange(12, 18, "months",
        "Months of gross profit to recover acquisition cost",
        "SaaS unit-economics convention"),
    "rule_of_40": BenchmarkRange(40, 100, "",
        "Growth rate plus profit margin; 40 is the widely-used threshold for "
        "a healthy balance between the two",
        "Widely-used SaaS investor heuristic"),
    "trial_to_paid": BenchmarkRange(15, 25, "%",
        "Free trials converting to paid",
        "SaaS conversion benchmarking commentary"),
}

# ══════════════════════════════════════════════════════════
#  HEALTHCARE
# ══════════════════════════════════════════════════════════
HEALTHCARE_BENCHMARKS = {
    "bed_occupancy": BenchmarkRange(80, 85, "%",
        "Sustained occupancy above ~85% is associated with rising delays and "
        "infection risk, so higher is not better",
        "Health-service capacity planning literature"),
    "readmission_rate_30d": BenchmarkRange(8, 15, "%",
        "Unplanned readmission within 30 days",
        "Published health-system quality reporting"),
    "average_length_of_stay": BenchmarkRange(4, 7, "days",
        "Highly dependent on case mix; only comparable within a specialty",
        "Published health-system reporting"),
    "did_not_attend_rate": BenchmarkRange(5, 10, "%",
        "Booked appointments not attended",
        "Outpatient-services reporting"),
    "patient_satisfaction": BenchmarkRange(4.0, 4.5, "/5",
        "Typical range on a 5-point patient experience survey",
        "Patient-experience survey convention"),
}


DOMAIN_BENCHMARKS = {
    "hr": HR_BENCHMARKS,
    "sales": SALES_BENCHMARKS,
    "finance": FINANCE_BENCHMARKS,
    "ecommerce": ECOMMERCE_BENCHMARKS,
    "marketing": MARKETING_BENCHMARKS,
    "operations": OPERATIONS_BENCHMARKS,
    "saas": SAAS_BENCHMARKS,
    "healthcare": HEALTHCARE_BENCHMARKS,
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
        (("cost_per_hire",), "cost_per_hire"),
        (("offer_acceptance",), "offer_acceptance_rate"),
        (("internal_mobility",), "internal_mobility_rate"),
        (("first_year_attrition", "early_attrition"), "first_year_attrition"),
        (("span_of_control", "direct_reports"), "span_of_control"),
        (("pay_gap", "gender_pay"), "gender_pay_gap"),
    ],
    "sales": [
        (("win_rate", "win rate"), "win_rate"),
        (("sales_cycle", "cycle_days", "cycle length"), "sales_cycle_days"),
        (("quota_attainment", "quota attainment"), "quota_attainment_pct_of_reps"),
        (("deal_size_growth",), "avg_deal_size_growth"),
        (("rep_turnover", "sales turnover"), "sales_rep_turnover"),
        (("pipeline_coverage",), "pipeline_coverage_ratio"),
        (("lead_to_opportunity", "lead_conversion"), "lead_to_opportunity_rate"),
        (("avg_quota", "quota_avg"), "quota_attainment_avg"),
        (("churn",), "churn_rate_annual"),
        (("discount",), "discount_rate_avg"),
    ],
    "finance": [
        (("gross_margin", "gross margin"), "gross_margin"),
        (("net_margin", "net margin", "net_profit_margin"), "net_margin"),
        (("current_ratio",), "current_ratio"),
        (("quick_ratio",), "quick_ratio"),
        (("opex_ratio", "operating_expense_ratio"), "opex_ratio"),
        (("budget_variance",), "budget_variance"),
        (("dso", "days_sales_outstanding"), "days_sales_outstanding"),
        (("ebitda",), "ebitda_margin"),
        (("dpo", "days_payable"), "days_payable_outstanding"),
        (("dio", "days_inventory"), "days_inventory_outstanding"),
        (("cash_conversion",), "cash_conversion_cycle"),
        (("revenue_per_employee",), "revenue_per_employee"),
        (("bad_debt", "write_off"), "bad_debt_ratio"),
    ],
    "ecommerce": [
        (("conversion_rate", "conversion rate"), "conversion_rate"),
        (("cart_abandon",), "cart_abandonment"),
        (("aov_growth", "order_value_growth"), "avg_order_value_growth"),
        (("return_rate", "return_pct"), "return_rate"),
        (("rating",), "customer_rating"),
        (("repeat_purchase", "repeat_rate"), "repeat_purchase_rate"),
        (("cac_payback", "payback_period"), "customer_acquisition_cost_payback"),
        (("email_capture", "signup_rate"), "email_capture_rate"),
        (("mobile_share", "mobile_traffic"), "mobile_share_of_traffic"),
        (("ltv_cac", "ltv_to_cac"), "customer_lifetime_to_cac"),
    ],
    "marketing": [
        (("open_rate",), "email_open_rate"),
        (("click_rate", "ctr_email"), "email_click_rate"),
        (("ctr", "click_through"), "paid_search_ctr"),
        (("landing_conversion", "page_conversion"), "landing_page_conversion"),
        (("marketing_spend_ratio", "marketing_cost"), "marketing_cost_of_revenue"),
        (("ltv_cac", "cac_ratio"), "customer_acquisition_cost_ratio"),
    ],
    "operations": [
        (("on_time", "otd", "delivery_rate"), "on_time_delivery"),
        (("order_accuracy", "fulfilment_accuracy"), "order_accuracy"),
        (("oee",), "oee"),
        (("defect", "reject_rate", "scrap"), "defect_rate"),
        (("inventory_turn", "stock_turn"), "inventory_turnover"),
        (("utilisation", "utilization", "capacity"), "capacity_utilisation"),
    ],
    "saas": [
        (("gross_retention", "grr"), "gross_revenue_retention"),
        (("net_retention", "nrr"), "net_revenue_retention"),
        (("logo_churn", "monthly_churn"), "logo_churn_monthly"),
        (("cac_payback", "payback"), "cac_payback_months"),
        (("rule_of_40", "rule40"), "rule_of_40"),
        (("trial_conversion", "trial_to_paid"), "trial_to_paid"),
    ],
    "healthcare": [
        (("occupancy", "bed_occupancy"), "bed_occupancy"),
        (("readmission",), "readmission_rate_30d"),
        (("length_of_stay", "los"), "average_length_of_stay"),
        (("dna_rate", "no_show", "did_not_attend"), "did_not_attend_rate"),
        (("patient_satisfaction",), "patient_satisfaction"),
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
