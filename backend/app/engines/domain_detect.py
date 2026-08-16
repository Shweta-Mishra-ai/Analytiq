"""
engines/domain_detect.py — deciding what a dataset is about.

This decision drives everything downstream: which KPIs are computed,
which benchmarks are cited, which report structure is used and which
findings are looked for. Getting it wrong does not degrade the report, it
invalidates it — a factory's production log analysed as an e-commerce
catalogue produces figures that are arithmetically correct and describe
nothing.

The previous detector matched keywords as bare substrings against the
joined column names and accepted any domain scoring above 0.04, which is
a single keyword out of sixteen. Measured against realistic files:

    reorder_point, stockout_flag   -> "ecommerce"  ("order", "stock")
    shipment cost, transit_days    -> "finance"    ("cost")
    ticket closed_at, agent        -> "sales"      ("closed")
    a lone "customer" column       -> "sales"

Three changes fix that class of error:

1. **Whole-word matching.** Column names are split on underscores, camel
   case and punctuation, so "reorder" is not "order" and "stockout" is
   not "stock".
2. **Weighted evidence.** "quota", "attrition", "ebitda" and "sku"
   identify a domain almost on their own. "revenue", "cost" and
   "customer" appear in every business dataset ever exported and are
   worth very little. Scoring them equally is what let one shared word
   decide a report.
3. **A margin requirement.** A domain has to beat the runner-up
   convincingly. Where two are close the answer is "general", which
   produces an honest domain-agnostic report rather than a confident
   wrong one.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Weights. A strong term is one that essentially only appears in its own
# domain; a weak one is shared across most business data.
STRONG, MEDIUM, WEAK = 3.0, 1.5, 0.5

DOMAIN_TERMS: Dict[str, Dict[str, float]] = {
    "hr": {
        "attrition": STRONG, "headcount": STRONG, "onboarding": STRONG,
        "employee": STRONG, "workforce": STRONG, "recruiter": STRONG,
        "tenure": MEDIUM, "salary": MEDIUM, "promotion": MEDIUM,
        "manager": MEDIUM, "department": MEDIUM, "hire": MEDIUM,
        "termination": MEDIUM, "absence": MEDIUM, "engagement": MEDIUM,
        "satisfaction": WEAK, "performance": WEAK, "bonus": WEAK,
        "job": WEAK, "grade": WEAK, "leaver": MEDIUM,
    },
    "sales": {
        "quota": STRONG, "pipeline": STRONG, "opportunity": STRONG,
        "deal": STRONG, "territory": STRONG, "prospect": STRONG,
        "forecast": MEDIUM, "stage": MEDIUM, "lead": MEDIUM,
        "rep": MEDIUM, "salesperson": MEDIUM, "win": MEDIUM,
        "loss": WEAK, "target": MEDIUM, "closed": WEAK,
        "customer": WEAK, "region": WEAK, "revenue": WEAK,
        "conversion": WEAK, "account": WEAK,
    },
    "finance": {
        "ebitda": STRONG, "cogs": STRONG, "ledger": STRONG,
        "payable": STRONG, "receivable": STRONG, "accrual": STRONG,
        "depreciation": STRONG, "liability": STRONG, "creditor": STRONG,
        "invoice": MEDIUM, "budget": MEDIUM, "variance": MEDIUM,
        "opex": MEDIUM, "capex": MEDIUM, "tax": MEDIUM,
        "asset": MEDIUM, "cashflow": MEDIUM, "gross": MEDIUM,
        "profit": WEAK, "margin": WEAK, "cost": WEAK, "expense": WEAK,
        "income": WEAK, "revenue": WEAK,
    },
    "ecommerce": {
        "sku": STRONG, "basket": STRONG, "cart": STRONG,
        "asin": STRONG, "marketplace": STRONG, "checkout": STRONG,
        "listing": STRONG, "fulfilment": STRONG, "fulfillment": STRONG,
        "product": MEDIUM, "catalogue": MEDIUM, "catalog": MEDIUM,
        "rating": MEDIUM, "review": MEDIUM, "discount": MEDIUM,
        "shipping": MEDIUM, "refund": MEDIUM, "seller": MEDIUM,
        "order": WEAK, "category": WEAK, "price": WEAK, "customer": WEAK,
        "quantity": WEAK, "stock": WEAK,
    },
    "marketing": {
        "impression": STRONG, "ctr": STRONG, "cpa": STRONG,
        "cpc": STRONG, "campaign": STRONG, "adset": STRONG,
        "creative": MEDIUM, "audience": MEDIUM, "keyword": MEDIUM,
        "roas": STRONG, "utm": STRONG, "bounce": MEDIUM,
        "click": MEDIUM, "spend": WEAK, "channel": WEAK,
        "conversion": WEAK, "traffic": WEAK, "lead": WEAK,
    },
    "operations": {
        "oee": STRONG, "downtime": STRONG, "throughput": STRONG,
        "defect": STRONG, "scrap": STRONG, "batch": STRONG,
        "shift": MEDIUM, "machine": STRONG, "line": WEAK,
        "yield": MEDIUM, "backlog": MEDIUM, "wip": STRONG,
        "leadtime": MEDIUM, "reorder": MEDIUM, "stockout": MEDIUM,
        "warehouse": MEDIUM, "shipment": MEDIUM, "transit": MEDIUM,
        "utilisation": MEDIUM, "utilization": MEDIUM, "capacity": WEAK,
        "inventory": MEDIUM, "supplier": MEDIUM,
    },
    "saas": {
        "mrr": STRONG, "arr": STRONG, "churn": MEDIUM,
        "subscription": STRONG, "trial": MEDIUM, "seat": MEDIUM,
        "plan": WEAK, "renewal": STRONG, "upgrade": MEDIUM,
        "downgrade": MEDIUM, "activation": MEDIUM, "retention": WEAK,
    },
    "healthcare": {
        "patient": STRONG, "diagnosis": STRONG, "admission": STRONG,
        "discharge": STRONG, "readmission": STRONG, "clinician": STRONG,
        "ward": MEDIUM, "referral": MEDIUM, "triage": STRONG,
        "appointment": MEDIUM, "specialty": MEDIUM, "bed": MEDIUM,
    },
}

# A domain must reach this before it is claimed at all, and beat the
# runner-up by this ratio. Both were absent before, which is why one
# shared word could decide a report.
MIN_SCORE = 4.0
MIN_MARGIN = 1.3
# One strong term alone is not a domain — "customer" plus nothing else
# was enough previously.
MIN_DISTINCT_TERMS = 2


@dataclass
class DomainVerdict:
    domain: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    matched: Dict[str, List[str]] = field(default_factory=dict)
    runner_up: str = ""
    reason: str = ""

    @property
    def is_confident(self) -> bool:
        return self.domain != "general"


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenise(name: str) -> List[str]:
    """Column name to whole words.

    "reorder_point", "stockOutFlag" and "Total Revenue (USD)" become
    ["reorder","point"], ["stock","out","flag"], ["total","revenue","usd"].
    Substring matching on the raw string is what made "reorder" an order
    and "stockout" a stock.
    """
    spaced = _CAMEL.sub(" ", str(name))
    return [t for t in re.split(r"[^a-zA-Z0-9]+", spaced.lower()) if t]


def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def detect(df: pd.DataFrame) -> DomainVerdict:
    """What this dataset is about, or "general" when it is not clear."""
    tokens: List[str] = []
    for col in df.columns:
        parts = tokenise(col)
        tokens.extend(parts)
        tokens.extend(_singular(p) for p in parts)
        # Multi-word terms that lose their meaning when split.
        if len(parts) > 1:
            tokens.append("".join(parts))
    token_set = set(tokens)

    scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {}
    for domain, terms in DOMAIN_TERMS.items():
        hits = [t for t in terms if t in token_set]
        scores[domain] = sum(terms[t] for t in hits)
        matched[domain] = hits

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_score = ranked[0]
    second, second_score = ranked[1] if len(ranked) > 1 else ("", 0.0)

    verdict = DomainVerdict(domain="general", confidence=0.0, scores=scores,
                            matched=matched, runner_up=second)

    if best_score < MIN_SCORE or len(matched[best]) < MIN_DISTINCT_TERMS:
        verdict.reason = (
            "No domain is clearly indicated by the column names, so the "
            "analysis is domain-agnostic: every comparison is against this "
            "dataset's own distribution.")
        return verdict

    if second_score > 0 and best_score < second_score * MIN_MARGIN:
        verdict.reason = (
            "The columns point about equally at {} and {}, so neither was "
            "assumed. Applying one domain's measures to the other's data "
            "produces figures that look right and mean nothing.".format(
                best, second))
        return verdict

    # Confidence scaled so a comfortable win reads near 1 without ever
    # claiming certainty from keyword matching alone.
    verdict.domain = best
    verdict.confidence = round(min(best_score / (MIN_SCORE * 3), 1.0), 2)
    verdict.reason = "Identified as {} from {}.".format(
        best, ", ".join(sorted(matched[best])[:6]))
    return verdict


def detect_domain(df: pd.DataFrame) -> Tuple[str, float]:
    """Backwards-compatible entry point: (domain, confidence)."""
    v = detect(df)
    return v.domain, v.confidence
