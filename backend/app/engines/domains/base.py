"""
core/engines/base.py
Shared dataclasses, helpers, and stat primitives used by all domain engines.
Single source of truth — import from here, never from story_engine directly.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class Insight:
    title:    str
    problem:  str
    cause:    str
    evidence: str
    action:   str
    impact:   str
    severity: str        # critical | high | warning | positive | info
    category: str = "general"
    confidence: str = ""  # High | Medium | Low — how strong the evidence is


# Display ladder: clients read Critical/High/Medium/Low, not the internal
# 'warning'/'info' mix (the reviewer's "everything reads HIGH" complaint).
_SEVERITY_DISPLAY = {
    "critical": "CRITICAL", "high": "HIGH", "warning": "MEDIUM",
    "medium": "MEDIUM", "info": "LOW", "low": "LOW",
    "positive": "STRENGTH",
}
_SEVERITY_RANK = {"critical": 0, "high": 1, "warning": 2, "medium": 2,
                  "info": 3, "low": 3, "positive": 4}


def severity_display(sev: str) -> str:
    return _SEVERITY_DISPLAY.get(str(sev).lower(), "INFO")


_ID_NAME_TOKENS = ("id", "ids", "uuid", "guid", "index", "idx", "code",
                   "ref", "reference", "key", "number", "no", "sku", "rowid")


def is_id_column(col: str, series=None) -> bool:
    """
    True when a column is an identifier, not a measure. Identifiers must be
    excluded from correlation, benchmarking, Pareto, root-cause and predictive
    features — a 'Pareto of Response ID' is meaningless. Combines a name check
    (whole-word 'id'/'code'/'ref'…) with a value check (all-unique or a
    monotonic 1-step sequence)."""
    import re as _re
    name = _re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', str(col)).lower()
    tokens = set(_re.split(r'[^a-z0-9]+', name))
    if tokens & set(_ID_NAME_TOKENS):
        return True
    # Value-based: only a near-perfect consecutive INTEGER sequence (1,2,3,…)
    # is treated as an identifier. A continuous measure can be all-unique too,
    # so 'all distinct' alone must NOT flag it — that mislabels real metrics.
    if series is not None:
        try:
            import pandas as _pd
            s = series.dropna()
            if len(s) > 20 and _pd.api.types.is_integer_dtype(s) \
                    and s.nunique() >= 0.98 * len(s):
                d = s.sort_values().diff().dropna()
                if len(d) and (d == 1).mean() > 0.95:
                    return True
        except Exception:
            logger.debug("is_id_column value check failed for %r", col)
    return False


def infer_confidence(evidence: str, severity: str = "") -> str:
    """
    Grade how strongly the DATA supports an insight, from its evidence text.
    Turns the significance work already computed (p-values, sample sizes,
    'hypothesis', 'not significant') into a client-readable High/Medium/Low.
    """
    t = (evidence or "").lower()
    # Explicitly weak evidence
    if any(k in t for k in ("not statistically significant", "sampling noise",
                            "sample too small", "not testable", "could not",
                            "significance not")):
        return "Low"
    # Strong significance
    import re
    if ("p<0.001" in t.replace(" ", "")) or ("p < 0.001" in t):
        return "High"
    m = re.search(r'p\s*[=<]\s*([0-9.]+)', t)
    if m:
        try:
            pval = float(m.group(1))
            return "High" if pval < 0.01 else "Medium" if pval < 0.05 else "Low"
        except (ValueError, TypeError):
            logger.debug("confidence: unparseable p-value %r", m.group(1))
    if "statistically significant" in t:
        return "High"
    # Hypotheses / association-only framing = medium at best
    if any(k in t for k in ("hypothesis", "association only", "not causation",
                            "consistent with", "may ")):
        return "Medium"
    return "Medium"


@dataclass
class AttritionAnalysis:
    rate:             float
    n_left:           int
    n_total:          int
    severity:         str
    top_drivers:      List[Dict]
    dept_attrition:   Dict
    salary_attrition: Dict
    n_flight_risk:    int
    flight_risk_pct:  float
    cost_estimate:    str
    interpretation:   str


@dataclass
class StoryReport:
    domain:              str = "general"
    domain_confidence:   float = 0.0
    executive_summary:   str = ""
    key_findings:        List[str] = field(default_factory=list)
    business_risks:      List[str] = field(default_factory=list)
    opportunities:       List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    insights:            List[Insight] = field(default_factory=list)
    anomalies:           List[str] = field(default_factory=list)
    attrition:           Optional[AttritionAnalysis] = None
    data_quality_verdict: str = ""
    analysis_confidence:  str = ""


# ══════════════════════════════════════════════════════════
#  STAT HELPERS  (Spearman throughout — non-parametric)
# ══════════════════════════════════════════════════════════

def infer_scale_bounds(obs_min: float, obs_max: float):
    """Infer a rating/score scale's THEORETICAL bounds from its observed max.

    Satisfaction/rating/engagement scores come on a handful of standard scales
    (0-1, 1-5, 1-7, 1-10, 0-100). Thresholding on observed min/max mis-reads
    data that doesn't span the whole scale (all 4-5 on a 1-5 Likert reads as
    "mediocre"); hard-coding a single scale (e.g. treating every rating as /5)
    mis-reads 1-10 or 0-100 data. Map to the nearest standard scale and only
    widen to observed values if the data spills outside it. Returns (lo, hi).
    """
    if obs_max <= 1.0:
        lo, hi = 0.0, 1.0
    elif obs_max <= 5.0:
        lo, hi = 1.0, 5.0
    elif obs_max <= 7.0:
        lo, hi = 1.0, 7.0
    elif obs_max <= 10.0:
        lo, hi = 1.0, 10.0
    elif obs_max <= 100.0:
        lo, hi = 0.0, 100.0
    else:
        lo, hi = 0.0, obs_max
    return min(lo, obs_min), max(hi, obs_max)


def col_stats(s: pd.Series) -> Dict:
    """
    Per-column statistics. Uses robust (non-parametric) measures throughout.
    Raises on empty series — callers must guard.
    """
    s = s.dropna()
    if len(s) < 3:
        return {}
    # Boolean/object-backed numeric columns fail numpy percentile with
    # "boolean subtract not supported" — same failure family as the
    # data_cleaner.py production crash. Force a clean float64 array first.
    # NOTE: pd.to_numeric does NOT convert bool dtype — must cast explicitly.
    if s.dtype == bool:
        s = s.astype(float)
    elif not pd.api.types.is_numeric_dtype(s):
        try:
            s = pd.to_numeric(s, errors="coerce").dropna()
        except Exception:
            logger.warning("col_stats: numeric coercion failed for non-numeric series", exc_info=True)
            return {}
        if len(s) < 3:
            return {}
    n = len(s)
    q1, med, q3 = float(np.percentile(s, 25)), float(np.percentile(s, 50)), float(np.percentile(s, 75))
    iqr = q3 - q1
    mean_v = float(s.mean())
    std_v  = float(s.std())
    try:
        skew_v = float(scipy_stats.skew(s))
    except Exception:
        skew_v = 0.0
        logger.debug("skew computation failed for column — data nearly constant")
    try:
        kurt_v = float(scipy_stats.kurtosis(s))
    except Exception:
        kurt_v = 0.0
        logger.debug("kurtosis computation failed for column — data nearly constant")
    out_mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
    out_pct  = float(out_mask.mean() * 100)
    cv       = std_v / abs(mean_v) if mean_v != 0 else 0.0
    return {
        "n": n, "mean": mean_v, "median": med,
        "std": std_v, "cv": cv,
        "q1": q1, "q3": q3, "iqr": iqr,
        "min": float(s.min()), "max": float(s.max()),
        "skew": skew_v, "kurtosis": kurt_v,
        "outlier_pct": out_pct,
        "outliers": int(out_mask.sum()),
        "p10": float(np.percentile(s, 10)),
        "p90": float(np.percentile(s, 90)),
    }


def correlations(df: pd.DataFrame, min_r: float = 0.25) -> List[Dict]:
    """
    Spearman correlations — robust to non-normal distributions.
    Returns pairs sorted by |r| descending, filtered to |r| >= min_r.
    Raises ValueError if df has fewer than 2 numeric columns.
    """
    # Identifiers must not be correlated (a 'Response ID vs X' correlation is
    # spurious). Exclude them up front.
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if not is_id_column(c, df[c])]
    if len(num_cols) < 2:
        return []
    results = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            a, b = num_cols[i], num_cols[j]
            common = df[[a, b]].dropna()
            if len(common) < 10:
                continue
            # A constant column has no variance — Spearman is undefined (NaN)
            # and SciPy emits ConstantInputWarning. Skip it at the source.
            if common[a].nunique() < 2 or common[b].nunique() < 2:
                continue
            try:
                r, p = scipy_stats.spearmanr(common[a], common[b])
                r = float(r)
                if np.isnan(r) or abs(r) < min_r:
                    continue
                strength = (
                    "strong"   if abs(r) >= 0.7 else
                    "moderate" if abs(r) >= 0.4 else
                    "weak"
                )
                results.append({
                    "col_a": a, "col_b": b,
                    "r": round(r, 4), "p": round(float(p), 6),
                    "r2": round(r ** 2, 4),
                    "strength": strength,
                    "direction": "positive" if r > 0 else "negative",
                    "significant": p < 0.05,
                })
            except Exception:
                logger.warning("Spearman failed for %s/%s", a, b, exc_info=True)
    results.sort(key=lambda x: abs(x["r"]), reverse=True)
    return results


def build_insight(
    title: str, problem: str, cause: str,
    evidence: str, action: str, impact: str,
    severity: str = "info", category: str = "general",
) -> Insight:
    return Insight(
        title=title, problem=problem, cause=cause,
        evidence=evidence, action=action, impact=impact,
        severity=severity, category=category,
    )
