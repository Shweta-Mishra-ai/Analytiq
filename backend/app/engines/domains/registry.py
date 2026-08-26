"""
engines/domains/registry.py — the single source of truth for domains.

WHY THIS EXISTS
---------------
Domain knowledge used to live in seven independent places that had to be
edited in lockstep: detection keywords and an if/elif dispatch in
story_engine, report_blueprints, industry_benchmarks, the PDF theme map,
the narrator's prompt and label maps, and insights_builder's fallbacks.
They drifted apart, and the drift was invisible:

  * `marketing` was detectable (and scored 0.91 on marketing data) but had
    no insight engine, so a marketing dataset was labelled "marketing" in
    the report and then analysed by the general engine.
  * benchmarks existed for saas, operations and healthcare that nothing
    could ever reach, because detection had never heard of them.

Registering a domain here, with a spec that must be complete, makes that
class of bug structurally impossible — `tests/test_domain_registry.py`
fails if any registered domain is missing any facet.

ADDING A DOMAIN
---------------
See docs/ADDING_A_DOMAIN.md. Short version: write
`app/engines/domains/<name>.py` exporting an insight function (plus its
own BLUEPRINT and BENCHMARKS if it needs them), then add one DomainSpec
below. No frontend change is needed — the UI renders whatever domain
string the API returns.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  SPEC
# ══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DomainSpec:
    """Everything the app needs to know to treat a dataset as this domain."""

    key: str
    # Prose label, used in report sentences. Acronym domains keep their
    # capitalisation ("HR", "SaaS") — "Published hr ranges" reads as
    # machine output.
    label: str
    # Distinctive column vocabulary. A hit here is strong evidence: these
    # words should not appear in another domain's data by coincidence.
    signature: Tuple[str, ...]
    # Supporting vocabulary. Common across domains, so worth little alone.
    keywords: Tuple[str, ...]
    # (df, stats, corrs) -> raw insight dict. HR additionally takes the
    # attrition result, so the caller passes it positionally when
    # runs_attrition is set.
    insight_fn: Callable
    # Key into pdf_builder.THEMES.
    pdf_theme: str = "Corporate Light"
    # Optional dedicated exit/churn pipeline, run before insights and
    # passed into insight_fn as a 4th argument. Only HR has one today; it
    # used to run for anything detected as HR, which is how SaaS
    # subscription data ended up with an "employee attrition rate".
    attrition_fn: Optional[Callable] = None
    # Optional per-domain overrides. When None the shared lookups in
    # report_blueprints / industry_benchmarks are used.
    blueprint: Optional[object] = None
    benchmarks: Optional[dict] = None

    @property
    def runs_attrition(self) -> bool:
        return self.attrition_fn is not None

    def run_attrition(self, df):
        """The domain's exit/churn analysis, or None if it has none."""
        return self.attrition_fn(df) if self.attrition_fn is not None else None

    def __post_init__(self):
        if not self.key or not self.key.strip():
            raise ValueError("DomainSpec.key must be a non-empty string")
        if not callable(self.insight_fn):
            raise TypeError(f"{self.key}: insight_fn must be callable")


# ══════════════════════════════════════════════════════════
#  COLUMN VOCABULARY
# ══════════════════════════════════════════════════════════

_CAMEL_1 = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_CAMEL_2 = re.compile(r'(?<=[A-Z])(?=[A-Z][a-z])')


def _normalise(col: str) -> str:
    """'MonthlyCharges' -> 'monthly charges'; 'ship-service-level' -> the
    same words separated. Detection reads column *words*, so camelCase
    headers have to be split or 'MonthlyIncome' matches nothing."""
    text = _CAMEL_2.sub(' ', _CAMEL_1.sub(' ', str(col)))
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def column_vocabulary(df: pd.DataFrame) -> Tuple[frozenset, frozenset]:
    """(tokens, squashed column names) for keyword matching."""
    tokens, squashed = set(), set()
    for col in df.columns:
        norm = _normalise(col)
        if not norm:
            continue
        squashed.add(norm.replace(' ', ''))
        tokens.update(t for t in norm.split() if t)
    return frozenset(tokens), frozenset(squashed)


def _matches(keyword: str, tokens: frozenset, squashed: frozenset) -> bool:
    """A keyword matches on a whole word, or as a substring of a column
    name when it is long enough to be unambiguous.

    The length floor is what stops the old behaviour where short keywords
    matched inside unrelated words — the previous detector joined every
    column into one string and used `kw in text`, so 'id' matched
    'ProvIDer' and 'age' matched 'average'.
    """
    kw = keyword.lower().strip()
    if not kw:
        return False
    if kw in tokens:
        return True
    return len(kw) >= 5 and any(kw in s for s in squashed)


# ══════════════════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════════════════

_SIGNATURE_WEIGHT = 3.0
_KEYWORD_WEIGHT = 1.0

# A domain must clear this much weighted evidence to be claimed at all.
# Roughly two signature hits, or one signature plus two supporting hits.
MIN_SIGNAL = 6.0
# ...and beat the runner-up by this factor. Two domains that score alike
# means the vocabulary is genuinely ambiguous, and a coin-flip between
# them is worse than saying "general".
MIN_MARGIN = 1.35
# Confidence saturates rather than dividing by keyword-list length. The
# old score was hits/len(keywords), which rewarded whichever domain had
# the *fewest* keywords and let 0.07 count as a match.
_CONFIDENCE_SCALE = 12.0


def score_domains(df: pd.DataFrame) -> Dict[str, float]:
    """Weighted evidence for every registered domain. Exposed so the
    detection tests can show *why* a dataset routed where it did."""
    tokens, squashed = column_vocabulary(df)
    scores: Dict[str, float] = {}
    for key, spec in REGISTRY.items():
        if key == "general":          # the fallback never competes
            continue
        score = 0.0
        for kw in spec.signature:
            if _matches(kw, tokens, squashed):
                score += _SIGNATURE_WEIGHT
        for kw in spec.keywords:
            if _matches(kw, tokens, squashed):
                score += _KEYWORD_WEIGHT
        scores[key] = round(score, 2)
    return scores


def detect_domain(df: pd.DataFrame) -> Tuple[str, float]:
    """(domain_key, confidence 0..1).

    Returns ("general", 0.0) when the evidence is weak or ambiguous.
    That matters: a wrong domain is worse than no domain, because it
    routes the dataset to an engine that speaks the wrong language and
    prints confident numbers about the wrong thing.
    """
    if df is None or df.empty or not len(df.columns):
        return ("general", 0.0)

    scores = score_domains(df)
    if not scores:
        return ("general", 0.0)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score < MIN_SIGNAL:
        return ("general", 0.0)
    if runner_up > 0 and best_score < runner_up * MIN_MARGIN:
        logger.info(
            "domain detection ambiguous: %s=%.1f vs %s=%.1f — using general",
            best, best_score, ranked[1][0], runner_up)
        return ("general", 0.0)

    confidence = best_score / (best_score + _CONFIDENCE_SCALE)
    return (best, round(confidence, 2))


# ══════════════════════════════════════════════════════════
#  ACCESSORS
# ══════════════════════════════════════════════════════════

def spec_for(domain: str) -> DomainSpec:
    """The spec for a domain, falling back to general."""
    return REGISTRY.get(str(domain or "").strip().lower(), REGISTRY["general"])


def label_for(domain: str) -> str:
    return spec_for(domain).label


def theme_for(domain: str) -> str:
    return spec_for(domain).pdf_theme


def registered_domains() -> Tuple[str, ...]:
    return tuple(REGISTRY.keys())


def run_insights(domain: str, df, stats, corrs, attrition=None) -> dict:
    """Dispatch to the domain's insight engine.

    Replaces the if/elif chain in story_engine, which was the reason a
    domain could be detectable without being analysable: the chain's
    `else` swallowed any unlisted domain into the general engine.
    """
    spec = spec_for(domain)
    if spec.runs_attrition:
        return spec.insight_fn(df, stats, corrs, attrition)
    return spec.insight_fn(df, stats, corrs)


# ══════════════════════════════════════════════════════════
#  REGISTRY
# ══════════════════════════════════════════════════════════
# Imported here, at the bottom, so the domain modules can import helpers
# from this module without a cycle.

from app.engines.domains.hr        import _insights_hr, _run_attrition   # noqa: E402
from app.engines.domains.ecommerce import _insights_ecommerce            # noqa: E402
from app.engines.domains.sales     import _insights_sales                # noqa: E402
from app.engines.domains.finance   import _insights_finance              # noqa: E402
from app.engines.domains.general   import _insights_general              # noqa: E402
from app.engines.domains.marketing import _insights_marketing            # noqa: E402
from app.engines.domains.saas      import _insights_saas                 # noqa: E402
from app.engines.domains.operations import _insights_operations          # noqa: E402
from app.engines.domains.healthcare import _insights_healthcare          # noqa: E402


REGISTRY: Dict[str, DomainSpec] = {}


def register(spec: DomainSpec) -> DomainSpec:
    if spec.key in REGISTRY:
        raise ValueError(f"domain '{spec.key}' is already registered")
    REGISTRY[spec.key] = spec
    return spec


register(DomainSpec(
    key="hr",
    label="HR",
    signature=("attrition", "employee", "headcount", "jobrole", "joblevel",
               "jobsatisfaction", "overtime", "worklifebalance", "promotion",
               "workforce", "businesstravel", "educationfield"),
    keywords=("salary", "department", "satisfaction", "tenure", "performance",
              "hire", "job", "manager", "bonus", "income", "gender", "age",
              "training", "years"),
    insight_fn=_insights_hr,
    pdf_theme="HR Blue",
    attrition_fn=_run_attrition,
))

register(DomainSpec(
    key="ecommerce",
    label="e-commerce",
    signature=("sku", "asin", "marketplace", "fulfilment", "fulfillment",
               "cart", "productname", "discountpercentage", "ratingcount",
               "shipservicelevel", "courier", "listingprice"),
    keywords=("price", "discount", "rating", "product", "category", "order",
              "review", "seller", "inventory", "stock", "shipping", "qty",
              "quantity", "amount"),
    insight_fn=_insights_ecommerce,
    pdf_theme="Ecommerce Orange",
))

register(DomainSpec(
    key="sales",
    label="sales",
    signature=("quota", "pipeline", "opportunity", "territory", "salesrep",
               "dealsize", "winrate", "closedate", "leadsource", "forecast",
               "stage", "accountexecutive"),
    keywords=("revenue", "sales", "target", "deal", "customer", "region",
              "conversion", "lead", "closed", "margin", "profit", "rep"),
    insight_fn=_insights_sales,
    pdf_theme="Sales Green",
))

register(DomainSpec(
    key="finance",
    label="finance",
    signature=("ebitda", "grossmargin", "netincome", "cashflow", "liability",
               "receivable", "payable", "ledger", "invoice", "depreciation",
               "budgetvariance", "operatingexpense", "capex", "opex"),
    keywords=("profit", "loss", "expense", "income", "budget", "cost",
              "margin", "asset", "tax", "revenue", "balance", "account"),
    insight_fn=_insights_finance,
    pdf_theme="Corporate Light",
))


register(DomainSpec(
    key="marketing",
    label="marketing",
    signature=("impressions", "clicks", "ctr", "cpa", "cpc", "cpm", "roas",
               "campaign", "adspend", "clickthrough", "adgroup", "creative",
               "placement", "adset"),
    keywords=("spend", "channel", "conversions", "reach", "budget", "medium",
              "source", "audience", "engagement", "traffic", "roi"),
    insight_fn=_insights_marketing,
    pdf_theme="Ecommerce Orange",
))

register(DomainSpec(
    key="saas",
    label="SaaS",
    signature=("mrr", "arr", "churn", "seats", "nps", "expansionrevenue",
               "netrevenueretention", "trialtopaid", "activeusers",
               "subscription", "renewal", "licenses"),
    keywords=("plan", "tier", "account", "revenue", "tenure", "upgrade",
              "downgrade", "contract", "trial", "usage", "customer"),
    insight_fn=_insights_saas,
    pdf_theme="Dark Tech",
))

register(DomainSpec(
    key="operations",
    label="operations",
    signature=("cycletime", "defectrate", "throughput", "downtime", "oee",
               "scraprate", "inventoryturns", "leadtime", "firstpassyield",
               "ontimedelivery", "utilisation", "utilization", "stoppage",
               "workorder"),
    keywords=("plant", "shift", "line", "machine", "capacity", "output",
              "quality", "delivery", "inventory", "yield", "process",
              "warehouse"),
    insight_fn=_insights_operations,
    pdf_theme="Corporate Light",
))

register(DomainSpec(
    key="healthcare",
    label="healthcare",
    signature=("patient", "readmission", "lengthofstay", "bedoccupancy",
               "diagnosis", "mortality", "admission", "discharge", "ward",
               "triage", "icd", "clinical", "inpatient", "outpatient"),
    keywords=("department", "cost", "age", "satisfaction", "specialty",
              "treatment", "care", "hospital", "bed", "case", "clinic"),
    insight_fn=_insights_healthcare,
    pdf_theme="HR Blue",
))

# The fallback. Never competes in scoring; used whenever evidence is weak
# or two domains tie.
register(DomainSpec(
    key="general",
    label="Business Analytics",
    signature=(),
    keywords=(),
    insight_fn=_insights_general,
    pdf_theme="Corporate Light",
))
