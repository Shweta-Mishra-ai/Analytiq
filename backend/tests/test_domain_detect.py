"""
Deciding what a dataset is about.

This one decision selects the KPIs, the benchmarks, the report structure
and the findings the engines look for. Getting it wrong does not degrade
the report — it invalidates it, while leaving every figure arithmetically
correct.

The previous detector matched bare substrings and accepted any domain
scoring above one keyword in sixteen. Measured against realistic files it
read `reorder_point` and `stockout_flag` as e-commerce, a shipment
manifest's `cost` column as finance, and a support queue's `closed`
column as sales. Each of those produced a full, confident, wrong report.

The refusals matter as much as the detections. "general" produces an
honest domain-agnostic report; a wrong domain produces a fluent one.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.engines.domain_detect import DOMAIN_TERMS, detect, detect_domain, tokenise


def _cols(*names):
    return pd.DataFrame(columns=list(names))


# ══════════════════════════════════════════════════════════
#  Tokenising
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,expected", [
    ("reorder_point", ["reorder", "point"]),
    ("stockOutFlag", ["stock", "out", "flag"]),
    ("Total Revenue (USD)", ["total", "revenue", "usd"]),
    ("customer-id", ["customer", "id"]),
])
def test_column_names_split_into_whole_words(name, expected):
    assert tokenise(name) == expected


def test_a_substring_is_not_a_match():
    """The whole class of error: "reorder" is not an order, "stockout" is
    not stock, and "closed" in a ticket queue is not a closed deal."""
    v = detect(_cols("reorder_point", "stockout_flag", "warehouse_bay"))
    assert "order" not in v.matched.get("ecommerce", [])
    assert "stock" not in v.matched.get("ecommerce", [])


# ══════════════════════════════════════════════════════════
#  Correct identification
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain,columns", [
    ("hr", ("employee_id", "department", "job_title", "salary",
            "tenure_years", "attrition", "manager_id")),
    ("sales", ("opportunity_id", "deal_stage", "deal_amount", "sales_rep",
               "territory", "quota", "forecast_category")),
    ("finance", ("period", "cost_centre", "revenue", "cogs", "gross_profit",
                 "opex", "ebitda", "budget", "variance")),
    ("ecommerce", ("order_id", "customer_id", "product_sku", "category",
                   "unit_price", "discount_pct", "rating", "cart_id")),
    ("marketing", ("campaign_id", "channel", "impressions", "clicks", "ctr",
                   "spend", "cpa")),
    ("operations", ("work_order", "machine", "defect_rate", "oee",
                    "downtime_minutes", "batch_id", "shift")),
    ("saas", ("account_id", "mrr", "plan", "trial_start", "renewal_date",
              "seats")),
    ("healthcare", ("patient_id", "admission_date", "discharge_date",
                    "diagnosis_code", "ward", "readmission_flag")),
])
def test_a_clear_dataset_is_identified(domain, columns):
    v = detect(_cols(*columns))
    assert v.domain == domain, (v.domain, v.matched.get(v.domain))
    assert v.confidence > 0.5


def test_a_factory_log_is_operations_not_ecommerce():
    """`reorder_point` and `stockout_flag` made this an e-commerce
    catalogue, so a production line was analysed for basket value."""
    v = detect(_cols("work_order", "machine", "lead_time_days",
                     "reorder_point", "stockout_flag", "output_units",
                     "defect_rate", "oee"))
    assert v.domain == "operations"


def test_the_verdict_names_the_evidence():
    """A domain claim with nothing behind it cannot be checked or
    overridden by the person delivering the report."""
    v = detect(_cols("employee_id", "attrition", "salary", "department"))
    assert "attrition" in v.reason
    assert v.matched["hr"]


# ══════════════════════════════════════════════════════════
#  Honest refusals
# ══════════════════════════════════════════════════════════

def test_a_single_shared_column_decides_nothing():
    """One "customer" column used to select the sales report."""
    v = detect(_cols("record_id", "value", "customer"))
    assert v.domain == "general"


def test_a_shipment_manifest_is_not_a_finance_report():
    """It has a `cost` column, which every business file has."""
    v = detect(_cols("shipment_id", "origin", "destination", "transit_days",
                     "cost", "on_time_flag"))
    assert v.domain != "finance"


def test_a_support_queue_is_not_a_sales_pipeline():
    v = detect(_cols("ticket_id", "priority", "opened_at", "closed",
                     "resolution_hours", "agent"))
    assert v.domain != "sales"


def test_an_ambiguous_file_says_why_it_refused():
    """Where two domains are close, applying either produces figures that
    look right and mean nothing — the report has to know that."""
    v = detect(_cols("respondent_id", "nps_score", "segment", "comment"))
    assert v.domain == "general"
    assert v.reason


def test_a_dataset_with_no_business_columns_is_general():
    v = detect(_cols("a", "b", "c", "value1", "value2"))
    assert v.domain == "general"
    assert v.confidence == 0.0


def test_an_empty_frame_does_not_raise():
    assert detect(pd.DataFrame()).domain == "general"


# ══════════════════════════════════════════════════════════
#  Weighting
# ══════════════════════════════════════════════════════════

def test_shared_words_are_worth_less_than_identifying_ones():
    """"revenue" and "cost" appear in every export ever made; "ebitda"
    and "quota" appear in one kind of file each."""
    for weak in ("revenue", "cost", "customer"):
        for domain, terms in DOMAIN_TERMS.items():
            if weak in terms:
                assert terms[weak] <= 1.0, f"{domain}/{weak}"
    assert DOMAIN_TERMS["finance"]["ebitda"] >= 3.0
    assert DOMAIN_TERMS["sales"]["quota"] >= 3.0


def test_weak_terms_alone_cannot_reach_a_verdict():
    v = detect(_cols("revenue", "cost", "customer", "region"))
    assert v.domain == "general"


def test_plurals_match_their_singular():
    v = detect(_cols("campaigns", "impressions", "clicks", "conversions",
                     "spend"))
    assert v.domain == "marketing"


# ══════════════════════════════════════════════════════════
#  The compatible entry point
# ══════════════════════════════════════════════════════════

def test_detect_domain_returns_the_old_shape():
    domain, confidence = detect_domain(_cols("employee_id", "attrition",
                                             "salary", "department"))
    assert domain == "hr"
    assert 0.0 < confidence <= 1.0


def test_story_engine_uses_the_new_detector():
    from app.engines.story_engine import detect_domain as story_detect
    domain, _c = story_detect(_cols("work_order", "machine", "oee",
                                    "defect_rate", "batch_id", "shift"))
    assert domain == "operations"
