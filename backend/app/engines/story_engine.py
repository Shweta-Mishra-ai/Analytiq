"""
story_engine.py — Senior analyst insights, domain-aware.
Format: Problem → Cause → Evidence → Action → Impact

Domain detection and report assembly live here; the heavy per-domain
insight logic lives in app/engines/domains/{hr,ecommerce,sales,finance,
general}.py (ported from dataforge-ai, which carries materially deeper
rules per domain plus a finance engine this app previously lacked).

The StoryReport shape is deliberately Analytiq's own — `headline`,
`top_insights`, `critical_issues`, `positive_findings` are read
directly by the frontend (InsightsPage.tsx) and the PDF builder, so
the richer domain engines are adapted into that shape rather than
replacing it.
"""
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy import stats as scipy_stats

from app.engines.domains.base import (
    Insight, AttritionAnalysis, build_insight, col_stats, correlations,
)
from app.engines.domains.hr        import _insights_hr, _run_attrition
from app.engines.domains.ecommerce import _insights_ecommerce
from app.engines.domains.sales     import _insights_sales
from app.engines.domains.finance   import _insights_finance
from app.engines.domains.general   import _insights_general

logger = logging.getLogger(__name__)

# Re-exported so existing importers of these names keep working.
__all__ = ["detect_domain", "generate_story", "StoryReport",
           "Insight", "AttritionAnalysis"]

# ══════════════════════════════════════════════════════════
#  DOMAIN DETECTION
# ══════════════════════════════════════════════════════════

DOMAIN_KEYWORDS = {
    "ecommerce": ["price", "discount", "rating", "product", "category",
                  "order", "revenue", "sales", "sku", "review", "seller",
                  "cart", "inventory", "stock", "asin", "marketplace"],
    "hr":        ["employee", "salary", "department", "attrition", "satisfaction",
                  "tenure", "performance", "hire", "job", "left", "manager",
                  "bonus", "promotion", "headcount", "workforce"],
    "sales":     ["revenue", "sales", "profit", "margin", "target", "quota",
                  "pipeline", "deal", "customer", "region", "territory",
                  "forecast", "conversion", "lead", "opportunity", "closed"],
    "finance":   ["profit", "loss", "expense", "income", "budget", "cost",
                  "margin", "cashflow", "asset", "liability", "tax", "invoice"],
    "marketing": ["campaign", "click", "impression", "conversion", "lead",
                  "channel", "spend", "roi", "ctr", "cpa", "traffic"],
}


def detect_domain(df: pd.DataFrame) -> Tuple[str, float]:
    """What this dataset is about, or ("general", 0.0) when unclear.

    Delegates to engines/domain_detect, which matches whole words rather
    than substrings and requires a margin over the runner-up. The old
    inline version read "reorder_point" as an order and "stockout_flag"
    as stock, so a factory's production log was analysed as an
    e-commerce catalogue.
    """
    from app.engines.domain_detect import detect_domain as _detect
    return _detect(df)


# ══════════════════════════════════════════════════════════
#  REPORT SHAPE
# ══════════════════════════════════════════════════════════

@dataclass
class StoryReport:
    domain:              str
    domain_confidence:   float
    headline:            str
    executive_summary:   str
    top_insights:        List[Insight]
    critical_issues:     List[Insight]
    positive_findings:   List[Insight]
    attrition:           Optional[AttritionAnalysis]
    key_findings:        List[str]
    business_risks:      List[str]
    opportunities:       List[str]
    recommended_actions: List[str]
    data_quality_verdict: str
    analysis_confidence:  str
    anomalies:           List[str] = field(default_factory=list)
    column_insights:     List = field(default_factory=list)


# ══════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════

def _detect_anomalies(df: pd.DataFrame, stats: Dict) -> List[str]:
    anomalies = []
    for col, st in stats.items():
        if not st: continue
        if st.get("outlier_pct",0) > 10:
            anomalies.append(
                "'{}' has {:.1f}% outliers — normal range {:.2f} to {:.2f}. Validate.".format(
                    col, st["outlier_pct"],
                    st["q1"]-1.5*st["iqr"], st["q3"]+1.5*st["iqr"]))
        if abs(st.get("skew",0)) > 2:
            anomalies.append(
                "'{}' heavily skewed ({:.2f}). Median {:.2f} more reliable than mean {:.2f}.".format(
                    col, st["skew"], st["median"], st["mean"]))
        if st.get("missing_pct",0) > 20:
            anomalies.append(
                "'{}' is {:.1f}% missing — imputed values may affect results.".format(
                    col, st["missing_pct"]))
    return anomalies


# ══════════════════════════════════════════════════════════
#  EXECUTIVE NARRATIVE
# ══════════════════════════════════════════════════════════
# Ported from dataforge-ai. Synthesises one headline claim and
# supporting sentences that connect back to it, instead of the
# template sentence this module used to emit
# ("This N-row X dataset analysis identified N critical issue(s)…"),
# which read as filler in a client-facing report.

def _is_tautological_pair(a: str, b: str) -> bool:
    """
    True when a correlation is mechanically obvious and therefore worthless as
    a headline: senior people earn more (level~pay), tenure metrics track each
    other (years~years), prices move together (price~mrp). Leading a report
    with 'JobLevel and MonthlyIncome correlate' reads as AI padding.
    """
    a, b = a.lower(), b.lower()
    tenure = ("year", "tenure", "month", "duration", "since", "age", "experience")
    level  = ("level", "grade", "band", "seniority", "rank")
    pay    = ("income", "salary", "pay", "wage", "compensation", "rate", "ctc")
    price  = ("price", "mrp", "cost", "amount", "value", "revenue")

    def has(s, kws):
        return any(k in s for k in kws)

    # Both tenure/duration-like
    if has(a, tenure) and has(b, tenure):
        return True
    # One is a level, the other is pay (or vice versa)
    if (has(a, level) and has(b, pay)) or (has(a, pay) and has(b, level)):
        return True
    # Both money/price-like
    if has(a, price) and has(b, price):
        return True
    # Both are counts of the same thing
    if ("count" in a and "count" in b) or ("total" in a and "total" in b):
        return True
    return False


def _first_meaningful_corr(corrs: List[Dict]) -> Optional[Dict]:
    """Strongest correlation that is NOT a tautology.

    A near-perfect |r| (>= 0.99) is excluded regardless of column names: it
    means the two columns are the same measurement stored twice (a
    duplicated export column, an "actual" copied from "revenue", a unit
    conversion). Reporting that as a discovered relationship is worse than
    saying nothing — it tells a paying client something they already know
    and signals the analysis is mechanical.
    """
    for c in corrs:
        r = c.get("r")
        if r is not None and abs(float(r)) >= 0.99:
            logger.debug("skipping near-perfect correlation %s~%s (r=%.4f) — "
                         "almost certainly a duplicated column",
                         c.get("col_a"), c.get("col_b"), float(r))
            continue
        if c.get("strength") == "strong" and not _is_tautological_pair(c["col_a"], c["col_b"]):
            return c
    return None


def _build_narrative_summary(
    df: pd.DataFrame, domain: str, confidence: float,
    deduped: List[Insight], corrs: List[Dict],
    attrition: Optional[AttritionAnalysis], raw: Dict,
) -> str:
    """
    Synthesises a single headline narrative claim instead of listing facts.
    Priority order for the headline: attrition signal > critical insight >
    strongest correlation > data quality. Supporting sentences follow,
    each connecting back to the headline rather than standing alone.
    """
    n_crit = sum(1 for i in deduped if i.severity == "critical")
    n_warn = sum(1 for i in deduped if i.severity == "warning")
    miss   = round(df.isna().mean().mean() * 100, 1)
    n_rows = len(df)

    # ── HEADLINE: pick the single most important claim ─────────────────────
    headline = None

    if attrition is not None and attrition.severity in ("critical", "high"):
        cohort_note = ""
        if attrition.top_drivers:
            d = attrition.top_drivers[0]
            cohort_note = f", concentrated among {d.get('label', 'a specific cohort')}"
        headline = (
            f"The {attrition.rate:.1f}% attrition rate ({attrition.n_left:,} of "
            f"{attrition.n_total:,} employees){cohort_note} is the dominant signal "
            f"in this dataset and warrants immediate retention review."
        )
    elif n_crit > 0:
        top_critical = next((i for i in deduped if i.severity == "critical"), None)
        if top_critical:
            headline = (
                f"{top_critical.problem} This is the most urgent finding in the "
                f"dataset — {n_crit} critical issue{'s' if n_crit > 1 else ''} total."
            )

    # A real business finding (even 'warning' severity) leads over a correlation.
    if headline is None:
        top_business = next(
            (i for i in deduped
             if i.severity in ("high", "warning")
             and i.category not in ("correlation", "data_quality")),
            None,
        )
        if attrition is not None and attrition.rate > 0:
            headline = (
                f"Attrition stands at {attrition.rate:.1f}% ({attrition.n_left:,} of "
                f"{attrition.n_total:,}), the headline signal in this dataset; "
                f"the sections below break it down by segment and driver."
            )
        elif top_business is not None:
            headline = (
                f"{top_business.problem} This is the most material finding in the "
                f"dataset and is detailed, with evidence, in the sections below."
            )

    # Only fall back to a correlation if it is NOT a mechanical tautology.
    if headline is None:
        mc = _first_meaningful_corr(corrs)
        if mc is not None:
            headline = (
                f"'{mc['col_a']}' and '{mc['col_b']}' show a strong "
                f"{mc['direction']} relationship (Spearman r={mc['r']:+.2f}), "
                f"explaining {mc['r']**2*100:.0f}% of shared variance — the clearest "
                f"non-trivial structural pattern in this dataset."
            )

    if headline is None and miss > 15:
        headline = (
            f"Data completeness is the primary concern: {miss:.1f}% of values "
            f"are missing across {n_rows:,} records, which will materially affect "
            f"any downstream analysis or modelling."
        )
    if headline is None:
        # Only claim a domain when detection is confident. Printing
        # "HR domain (detection confidence: 7%)" in a client PDF both
        # mislabels the data and advertises the uncertainty.
        domain_phrase = (
            f"in the {domain.upper()} domain " if confidence >= 0.4 else ""
        )
        headline = (
            f"Analysis of {n_rows:,} records across {len(df.columns)} variables "
            f"{domain_phrase}did not surface a single dominant risk — findings "
            f"below are of comparable priority."
        )

    # ── SUPPORTING sentences — connect to headline, don't repeat it ────────
    support = []

    # Don't repeat attrition if the headline already covers it.
    if attrition is not None and headline and "attrition" not in headline.lower()[:80]:
        support.append(
            f"Separately, attrition stands at {attrition.rate:.1f}% "
            f"({attrition.severity} severity)."
        )

    if n_warn > 0:
        support.append(
            f"{n_warn} additional warning{'s' if n_warn > 1 else ''} "
            f"{'requires' if n_warn == 1 else 'require'} review but "
            f"{'is' if n_warn == 1 else 'are'} not urgent."
        )

    if miss > 0 and miss <= 15:
        support.append(f"Data completeness is acceptable ({100-miss:.1f}% complete).")
    elif miss == 0:
        support.append("Data is fully complete with no missing values.")

    if "Spearman r=" not in (headline or ""):
        # First correlation that is real (not NaN from a constant column) and
        # not a mechanical tautology — otherwise the summary prints
        # "'Age' vs 'EmployeeCount' (r=+nan)" which looks broken.
        import math
        # Same exclusions as _first_meaningful_corr: skip NaN, tautological
        # pairs, and near-perfect |r| (a duplicated column stored twice).
        # This branch previously applied only the tautology check, so an
        # exact-duplicate column still reached the summary as
        # "Notable relationship: 'revenue' vs 'actual' (r=+1.00)".
        top = next(
            (c for c in corrs
             if c.get("r") is not None and not math.isnan(c["r"])
             and abs(float(c["r"])) < 0.99
             and not _is_tautological_pair(c["col_a"], c["col_b"])),
            None,
        )
        if top is not None and abs(top["r"]) >= 0.25:
            support.append(
                f"Notable relationship: '{top['col_a']}' vs '{top['col_b']}' "
                f"(r={top['r']:+.2f}, {top['strength']})."
            )

    n_actions = len(raw.get("actions", []))
    if n_actions:
        support.append(
            f"{n_actions} recommendation{'s' if n_actions > 1 else ''} "
            f"{'follows' if n_actions == 1 else 'follow'} below."
        )

    return headline + (" " + " ".join(support) if support else "")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

def generate_story(df: pd.DataFrame) -> StoryReport:
    domain, confidence = detect_domain(df)

    num_cols  = df.select_dtypes(include="number").columns.tolist()
    all_stats = {col: col_stats(df[col]) for col in num_cols}
    all_stats = {k:v for k,v in all_stats.items() if v}

    corrs     = correlations(df)
    attrition = _run_attrition(df) if domain == "hr" else None

    if domain == "hr":
        raw = _insights_hr(df, all_stats, corrs, attrition)
    elif domain == "ecommerce":
        raw = _insights_ecommerce(df, all_stats, corrs)
    elif domain == "sales":
        raw = _insights_sales(df, all_stats, corrs)
    elif domain == "finance":
        raw = _insights_finance(df, all_stats, corrs)
    else:
        raw = _insights_general(df, all_stats, corrs)

    # Top up from the general engine only where the domain engine came back
    # thin. Previously this merged unconditionally, which meant the general
    # findings were appended even to a rich HR/sales report (and duplicated
    # outright when the domain WAS general).
    if domain != "general":
        try:
            gen = _insights_general(df, all_stats, corrs)
            if len(raw.get("findings", [])) < 2:
                raw.setdefault("findings", []).extend(gen["findings"])
            if len(raw.get("risks", [])) < 2:
                raw.setdefault("risks", []).extend(gen["risks"])
            if len(raw.get("opportunities", [])) < 2:
                raw.setdefault("opportunities", []).extend(gen["opportunities"])
            if len(raw.get("insights", [])) < 4:
                raw.setdefault("insights", []).extend(gen.get("insights", []))
        except Exception:
            logger.warning("general insight merge failed", exc_info=True)

    # De-duplicate the flat string lists (identical findings never repeat)
    for key in ("findings", "risks", "opportunities"):
        seen_k, uniq = set(), []
        for item in raw.get(key, []):
            if item not in seen_k:
                seen_k.add(item)
                uniq.append(item)
        raw[key] = uniq

    insights = raw.get("insights",[])
    # The domain engines emit a richer severity ladder than this module
    # originally knew about ("high"/"medium"/"low"). Any severity missing
    # from this map sorts to 99 — i.e. HIGH-severity insights would have
    # been pushed below "info" ones and dropped off the top-6 list.
    sev_order = {"critical":0,"high":1,"warning":2,"medium":2,
                 "info":3,"low":3,"positive":4}
    insights  = sorted(insights, key=lambda x: sev_order.get(x.severity,99))

    # Deduplicate
    seen, deduped = set(), []
    for ins in insights:
        if ins.title not in seen:
            seen.add(ins.title)
            deduped.append(ins)

    # "critical" for headline/critical_issues purposes includes the
    # domain engines' "high" band — both mean "act on this now".
    critical  = [i for i in deduped if i.severity in ("critical", "high")]
    positive  = [i for i in deduped if i.severity=="positive"]

    # Flat lists for PDF.
    # key_findings previously came straight off raw["findings"], which the
    # domain-specific insight builders often leave empty even when they
    # produced rich `insights` — the report then rendered a blank "Key
    # Findings" section while the rest of the page was full, with no error
    # anywhere. Fall back to the insight titles, then to a dataset-shape
    # summary, so this section is never silently empty.
    findings_flat = raw["findings"][:6]
    if not findings_flat and deduped:
        findings_flat = ["{}: {}".format(ins.severity.upper(), ins.title)
                          for ins in deduped[:6]]
    if not findings_flat:
        n_cat = len(df.select_dtypes(include=["object", "string"]).columns)
        miss_pct = round(df.isna().mean().mean() * 100, 1)
        findings_flat = [
            "{:,} records × {} columns ({} numeric, {} categorical)".format(
                len(df), len(df.columns), len(num_cols), n_cat),
            "Missing data: {:.1f}%{}".format(
                miss_pct, " — fully complete" if miss_pct == 0
                else " — imputation applied"),
        ]

    risks_flat    = raw["risks"][:6]
    opps_flat     = raw["opportunities"][:4]
    actions_flat  = ["[{}] {}".format(
        "CRITICAL" if i<2 else "SHORT TERM" if i<4 else "LONG TERM", a)
        for i, a in enumerate(raw["actions"][:8])]

    # Executive summary — a synthesised narrative that leads with the single
    # most important claim, not a count of how many issues were found. The
    # old template ("This N-row X dataset analysis identified N critical
    # issue(s) and N risk(s)…") said nothing a reader couldn't see from the
    # section headings, and reads as filler at the top of a paid report.
    n_crit = len(critical)
    try:
        exec_s = _build_narrative_summary(
            df=df, domain=domain, confidence=confidence,
            deduped=deduped, corrs=corrs, attrition=attrition, raw=raw)
    except Exception:
        logger.warning("narrative summary failed — falling back to a factual "
                       "summary line", exc_info=True)
        exec_s = ""
    if not exec_s:
        parts = ["{:,} rows × {} columns analysed.".format(len(df), len(df.columns))]
        if deduped:
            parts.append("Most material finding: {}.".format(deduped[0].title))
        if attrition:
            parts.append("Attrition: {:.1f}%.".format(attrition.rate))
        exec_s = " ".join(parts)

    headline = ("CRITICAL: " + critical[0].title) if critical else (
        deduped[0].title if deduped else "Analysis complete")

    # Quality
    avg_miss = sum(st.get("missing_pct",0) for st in all_stats.values()) / max(len(all_stats),1)
    quality  = ("GOOD — data suitable for reliable analysis." if avg_miss<5
                else "FAIR — {:.1f}% avg missing. Imputation applied.".format(avg_miss)
                if avg_miss<20
                else "NEEDS ATTENTION — {:.1f}% missing. Treat findings with caution.".format(avg_miss))

    conf = ("HIGH — sufficient data for reliable conclusions." if len(df)>=1000 and len(num_cols)>=3
            else "MEDIUM — directional, more data improves reliability." if len(df)>=100
            else "LOW — small dataset, treat as directional only.")

    return StoryReport(
        domain=domain, domain_confidence=confidence,
        headline=headline, executive_summary=exec_s,
        top_insights=deduped[:6],
        critical_issues=critical,
        positive_findings=positive,
        attrition=attrition,
        key_findings=findings_flat,
        business_risks=risks_flat,
        opportunities=opps_flat,
        recommended_actions=actions_flat,
        data_quality_verdict=quality,
        analysis_confidence=conf,
        anomalies=_detect_anomalies(df, all_stats),
    )
