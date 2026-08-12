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
    col_text = " ".join(df.columns.str.lower().tolist())
    scores   = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in col_text)
        scores[domain] = hits / len(keywords)
    best  = max(scores, key=scores.get)
    score = scores[best]
    return (best, round(score, 2)) if score > 0.04 else ("general", 0.0)


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

    # Executive summary
    n_crit = len(critical)
    exec_s = "This {:,}-row {} dataset analysis identified {} critical issue(s) and {} risk(s). ".format(
        len(df), domain, n_crit, len(risks_flat))
    if attrition:
        exec_s += "Attrition: {:.1f}% ({} severity). ".format(
            attrition.rate, attrition.severity.upper())
    if deduped:
        exec_s += "Priority: {}. ".format(deduped[0].title)
    exec_s += "{} actionable recommendations provided.".format(len(actions_flat))

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
