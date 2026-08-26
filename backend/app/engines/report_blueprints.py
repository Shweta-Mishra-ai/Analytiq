"""
engines/report_blueprints.py — what a report should contain, and in what
order, for the domain it is about.

Every report the app produced had the same skeleton: executive summary,
data quality, a flat list of insights, statistics, charts,
recommendations. That is a generic analytics output, and a reader in a
specific function notices immediately — an HR director opens a workforce
report expecting the workforce profile first and attrition second, and a
finance director expects the P&L position before anything else. A single
ordering cannot serve both, and the one that served neither read as a
tool's default output rather than as a piece of work.

A blueprint states, per domain:

  - the section order and the names that function actually uses
    ("Workforce Profile", not "Dataset Overview");
  - which findings belong in which section, by the category the domain
    engines already tag them with;
  - the headline metrics a reader of that domain looks for first;
  - the reference frameworks the figures should be read against.

Structure only. Nothing here computes or invents a number — the domain
engines do that, and a section with nothing to say is dropped rather than
padded.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

# This module is pure data and has no failure path of its own, but every
# module in app/ carries a logger — a blanket rule is easier to enforce,
# and to reason about, than a list of exemptions.
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Section:
    """One section of a report."""
    key: str
    title: str
    # Why this section exists, in the language of the domain. Printed
    # under the heading so a reader knows what they are looking at.
    purpose: str
    # Insight categories (as tagged by the domain engines) that belong
    # here. Empty means the section is built from something other than
    # insights (charts, tables, methodology).
    categories: tuple = ()


@dataclass(frozen=True)
class Blueprint:
    domain: str
    label: str                      # what this kind of report is called
    headline_metrics: tuple         # what the reader looks for first
    sections: tuple
    references: tuple = ()
    # Named because a reader asks "against what?" — and the honest answer
    # for most of these is a convention, not a licensed dataset.
    reference_note: str = ""

    def section(self, key: str):
        for s in self.sections:
            if s.key == key:
                return s
        return None

    def categories_for(self, key: str) -> tuple:
        s = self.section(key)
        return s.categories if s else ()


_COMMON_OPENING = (
    Section("basis", "Basis of Preparation",
            "What this report covers, what it was computed from, and the "
            "limits of what it can support."),
    Section("summary", "Executive Summary",
            "The position, the material findings, and what they call for."),
)

_COMMON_CLOSING = (
    Section("recommendations", "Recommendations",
            "What to do, in the order the evidence supports."),
    Section("method", "Basis of Preparation — Method",
            "The tests applied, why each was chosen, and what they do and "
            "do not establish."),
)


FINANCE = Blueprint(
    domain="finance",
    label="Financial Performance Review",
    headline_metrics=("gross margin", "cost structure", "break-even",
                      "budget variance"),
    sections=_COMMON_OPENING + (
        Section("position", "Financial Position",
                "Revenue, cost and margin for the period covered.",
                ("finance_margin", "finance_trend")),
        Section("structure", "Cost Structure & Operating Leverage",
                "How much of the cost base moves with trading, and what "
                "that implies for break-even.",
                ("finance_structure", "finance_margin_stability")),
        Section("variance", "Budget Variance",
                "Actual against plan, and where the difference sits.",
                ("finance_budget",)),
        Section("segments", "Segment Profitability",
                "Which parts of the business earn and which consume.",
                ("finance_segment_loss", "finance_concentration")),
        Section("controls", "Data Integrity Observations",
                "Patterns in the ledger worth a second look before the "
                "figures are relied on.",
                ("finance_integrity",)),
        Section("charts", "Supporting Analysis",
                "The figures above, plotted."),
    ) + _COMMON_CLOSING,
    references=(
        "IFRS / GAAP presentation conventions for revenue, cost of sales "
        "and operating expense classification.",
        "CFA Institute financial-ratio conventions — gross margin, "
        "operating expense ratio and their standard interpretation bands.",
        "Nigrini's mean absolute deviation thresholds for first-digit "
        "conformity, as used in forensic and internal-audit practice.",
    ),
    reference_note=(
        "Sector margin ranges vary too widely for an external figure to be "
        "meaningful on its own. The organisation's own prior periods are "
        "the more reliable comparison and are used wherever the data "
        "supports it."),
)


HR = Blueprint(
    domain="hr",
    label="Workforce Analytics Review",
    headline_metrics=("headcount", "attrition rate", "tenure",
                      "compensation spread"),
    sections=_COMMON_OPENING + (
        Section("profile", "Workforce Profile",
                "Headcount, structure and tenure across the population.",
                ("workforce", "demographics")),
        Section("attrition", "Attrition & Retention",
                "Who is leaving, from where, and what the data can and "
                "cannot say about why.",
                ("attrition", "retention", "hr_attrition")),
        Section("engagement", "Engagement & Satisfaction",
                "What the survey and behavioural measures show.",
                ("satisfaction", "engagement", "hr_engagement")),
        Section("reward", "Compensation & Equity",
                "Pay distribution and differences between groups.",
                ("compensation", "pay", "hr_pay")),
        Section("risk", "Talent Risk",
                "Where the workforce data points to exposure.",
                ("hr_risk", "risk")),
        Section("charts", "Supporting Analysis",
                "The figures above, plotted."),
    ) + _COMMON_CLOSING,
    references=(
        "SHRM, State of the Workplace — voluntary attrition benchmarks and "
        "direct replacement cost.",
        "Gallup, State of the Global Workplace — engagement measures and "
        "preventable-exit rates.",
        "Mercer, Global Talent Trends — career growth as a leading "
        "voluntary exit driver.",
    ),
    reference_note=(
        "Attrition norms differ sharply by sector — retail and hospitality "
        "commonly run several times the rate of professional services — so "
        "an external range is context, never a target."),
)


SALES = Blueprint(
    domain="sales",
    label="Sales Performance Review",
    headline_metrics=("bookings", "win rate", "cycle length",
                      "quota attainment"),
    sections=_COMMON_OPENING + (
        Section("bookings", "Bookings & Revenue",
                "What was sold, and how it is distributed.",
                ("revenue", "sales_revenue")),
        Section("conversion", "Win Rates & Conversion",
                "What converts, and whether differences between people "
                "and segments are larger than deal volume explains.",
                ("sales_rep_performance", "conversion", "target")),
        Section("velocity", "Pipeline Velocity",
                "How long deals take, and what separates the ones that "
                "close from the ones that do not.",
                ("sales_cycle", "sales_cycle_outcome")),
        Section("coverage", "Territory & Segment Performance",
                "Where performance concentrates.",
                ("regional", "segment", "product")),
        Section("charts", "Supporting Analysis",
                "The figures above, plotted."),
    ) + _COMMON_CLOSING,
    references=(
        "Quota-attainment and pipeline-coverage conventions as used in "
        "standard sales-operations practice.",
        "Win-rate and cycle-length ranges are highly sector-specific.",
    ),
    reference_note=(
        "The internal comparison — a rep or segment against the rest of "
        "this data — is more reliable than any published win-rate range, "
        "and is what the findings here rest on."),
)


ECOMMERCE = Blueprint(
    domain="ecommerce",
    label="Commercial & Customer Review",
    headline_metrics=("revenue", "average order value", "repeat rate",
                      "retention"),
    sections=_COMMON_OPENING + (
        Section("demand", "Demand & Revenue",
                "What sold, at what value, and how that moved.",
                ("revenue", "pricing", "aov")),
        Section("customers", "Customer Base & Retention",
                "Who buys, who comes back, and where the revenue "
                "concentrates.",
                ("ecommerce_retention", "ecommerce_rfm",
                 "ecommerce_concentration", "repeat_purchase")),
        Section("catalogue", "Product & Category Performance",
                "Which categories carry the business and which drag.",
                ("category", "product", "rating")),
        Section("experience", "Rating & Experience Signals",
                "What the review and rating data indicates.",
                ("rating", "review", "satisfaction")),
        Section("charts", "Supporting Analysis",
                "The figures above, plotted."),
    ) + _COMMON_CLOSING,
    references=(
        "Baymard Institute, cart-abandonment and checkout usability "
        "research.",
        "RFM segmentation and cohort retention as standard commercial "
        "analytics practice; the quintile cut-offs used here are computed "
        "within this dataset, not taken from an external norm.",
    ),
    reference_note=(
        "Conversion, return and rating norms differ sharply by vertical "
        "and price point, so external figures are indicative only."),
)


GENERAL = Blueprint(
    domain="general",
    label="Data Analysis Report",
    headline_metrics=(),
    sections=_COMMON_OPENING + (
        Section("overview", "Dataset Overview",
                "What the data describes and how it is distributed."),
        Section("findings", "Findings",
                "What the analysis established, strongest first."),
        Section("relationships", "Relationships & Drivers",
                "What moves with what, and how strongly.",
                ("correlation", "driver")),
        Section("charts", "Supporting Analysis",
                "The figures above, plotted."),
    ) + _COMMON_CLOSING,
    references=(
        "All comparisons in this report are internal: each metric is "
        "measured against its own distribution within the supplied data.",
    ),
    reference_note=(
        "No external benchmark set applies to this dataset, so none is "
        "cited. An internal comparison is the stronger evidence in any "
        "case, because it holds the business constant."),
)



# ══════════════════════════════════════════════════════════
#  EXPANSION DOMAINS
#  Categories below match those emitted by the matching engine in
#  app/engines/domains/<domain>.py — group_insights routes on them, so
#  the two have to agree.
# ══════════════════════════════════════════════════════════

MARKETING = Blueprint(
    domain="marketing",
    label="Marketing Performance Review",
    headline_metrics=("return on spend", "cost per acquisition",
                      "channel mix", "funnel conversion"),
    sections=_COMMON_OPENING + (
        Section("efficiency", "Spend & Return",
                "What was spent, what it returned, and at what efficiency.",
                ("marketing_efficiency",)),
        Section("channels", "Channel Performance",
                "Which channels carry the result and which consume budget "
                "without returning it.",
                ("marketing_channel", "marketing_waste")),
        Section("funnel", "Funnel Conversion",
                "Where prospects are lost between impression, click and "
                "conversion.",
                ("marketing_funnel",)),
        Section("risk", "Concentration & Risk",
                "Dependencies in the current channel mix.",
                ("marketing_risk", "risk")),
        Section("charts", "Supporting Analysis",
                "The figures above, plotted."),
    ) + _COMMON_CLOSING,
    reference_note=(
        "Channel benchmarks vary by sector, auction and season, so a "
        "published range is context rather than a target. The internal "
        "comparison between this account's own channels is the stronger "
        "evidence, because it holds the business constant."),
)

SAAS = Blueprint(
    domain="saas",
    label="Subscription Performance Review",
    headline_metrics=("recurring revenue", "customer churn",
                      "net revenue retention", "expansion"),
    sections=_COMMON_OPENING + (
        Section("position", "Recurring Revenue Position",
                "Scale and shape of the recurring revenue base.",
                ("saas_revenue",)),
        Section("retention", "Retention & Churn",
                "Where the base is leaking, and which segment carries it. "
                "Churn here is customer churn, not staff attrition.",
                ("saas_retention", "retention")),
        Section("growth", "Expansion & Growth",
                "Revenue available from the existing base.",
                ("saas_growth", "opportunity")),
        Section("risk", "Concentration & Risk",
                "Dependence on a single segment, tier or cohort.",
                ("saas_risk", "risk")),
        Section("charts", "Supporting Analysis",
                "The figures above, plotted."),
    ) + _COMMON_CLOSING,
    reference_note=(
        "Subscription benchmarks are heavily stage- and segment-dependent: "
        "an early-stage self-serve product and a mature enterprise one are "
        "not comparable on the same churn range. Treat published figures as "
        "orientation only."),
)

OPERATIONS = Blueprint(
    domain="operations",
    label="Operational Performance Review",
    headline_metrics=("cycle time", "defect rate", "on-time delivery",
                      "capacity utilisation"),
    sections=_COMMON_OPENING + (
        Section("stability", "Process Stability",
                "How predictable the process is, not only how fast.",
                ("ops_stability",)),
        Section("quality", "Quality & Yield",
                "Defects, rework and first-pass yield.",
                ("ops_quality", "quality")),
        Section("delivery", "Delivery Reliability",
                "Commitments met, and what drives the misses.",
                ("ops_delivery",)),
        Section("capacity", "Capacity & Utilisation",
                "Whether the operation has the slack to absorb variation.",
                ("ops_capacity",)),
        Section("variance", "Site & Line Variance",
                "Differences between sites running the same process.",
                ("ops_variance",)),
        Section("charts", "Supporting Analysis",
                "The figures above, plotted."),
    ) + _COMMON_CLOSING,
    reference_note=(
        "Operational ranges such as OEE and on-time delivery come from "
        "widely-used definitions (SCOR, TPM) rather than a licensed "
        "dataset, and differ by process type. Internal site-to-site "
        "comparison is the more actionable evidence."),
)

HEALTHCARE = Blueprint(
    domain="healthcare",
    label="Healthcare Operations Review",
    headline_metrics=("bed occupancy", "length of stay",
                      "readmission rate", "cost per case"),
    sections=_COMMON_OPENING + (
        Section("capacity", "Capacity & Occupancy",
                "Bed availability and the buffer left to absorb demand.",
                ("healthcare_capacity",)),
        Section("flow", "Patient Flow",
                "Length of stay and the non-clinical delays within it.",
                ("healthcare_flow",)),
        Section("quality", "Quality Indicators",
                "Readmission and related operational quality measures.",
                ("healthcare_quality", "quality")),
        Section("cost", "Cost Variation",
                "Cost per case across departments, and what case mix "
                "explains.",
                ("healthcare_cost",)),
        Section("charts", "Supporting Analysis",
                "The figures above, plotted."),
    ) + _COMMON_CLOSING,
    reference_note=(
        "This is an operational and administrative analysis of aggregate "
        "records. It is not clinical guidance, contains no diagnosis or "
        "treatment recommendation, and draws no conclusion about any "
        "individual patient. Case mix drives most variation between "
        "departments, so comparisons hold only within a specialty and any "
        "clinical interpretation requires the responsible clinical team."),
)


BLUEPRINTS: Dict[str, Blueprint] = {
    "finance": FINANCE,
    "hr": HR,
    "sales": SALES,
    "ecommerce": ECOMMERCE,
    "marketing": MARKETING,
    "saas": SAAS,
    "operations": OPERATIONS,
    "healthcare": HEALTHCARE,
    "general": GENERAL,
}


def blueprint_for(domain: str) -> Blueprint:
    """The blueprint for a domain, falling back to the general one."""
    return BLUEPRINTS.get(str(domain or "").strip().lower(), GENERAL)


def group_insights(blueprint: Blueprint, insights: List) -> List:
    """Assign each insight to its section, keeping the blueprint's order.

    Returns [(Section, [insight, ...])] for sections that have something
    to show. Anything whose category matches no section is appended to
    the first section that carries findings, rather than dropped — a
    finding the report does not print is worse than one under a slightly
    wrong heading.
    """
    by_category: Dict[str, List] = {}
    for ins in insights:
        by_category.setdefault(str(getattr(ins, "category", "")).lower(),
                               []).append(ins)

    used: set = set()
    grouped: List = []
    for section in blueprint.sections:
        if not section.categories:
            continue
        items: List = []
        for cat in section.categories:
            for ins in by_category.get(cat, []):
                if id(ins) not in used:
                    used.add(id(ins))
                    items.append(ins)
        if items:
            grouped.append((section, items))

    leftovers = [i for i in insights if id(i) not in used]
    if leftovers:
        if grouped:
            grouped[0][1].extend(leftovers)
        else:
            fallback = next((s for s in blueprint.sections
                             if s.categories), None)
            if fallback is None:
                fallback = Section("findings", "Findings",
                                   "What the analysis established.")
            grouped.append((fallback, leftovers))
    return grouped
