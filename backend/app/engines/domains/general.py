"""
core/engines/general.py — General / unknown domain engine.
Handles any dataset without a detectable domain.
Provides outlier flagging, correlation surfacing, and generic opportunity detection.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from app.engines.domains.base import Insight, build_insight, col_stats, correlations
from app.engines.domains.general_depth import run_general_depth

logger = logging.getLogger(__name__)

# A segment gap has to clear one of these to be worth a client's attention:
# either the groups differ by a quarter, or by a quarter of the column's own
# spread. Below both, the difference is real and irrelevant.
MIN_SEGMENT_RATIO = 1.25


def _num(x: float) -> str:
    """A number as a reader writes it: 320, 1,250, 0.042 — never 3.2e+02."""
    x = float(x)
    if abs(x) >= 100:
        return "{:,.0f}".format(x)
    if abs(x) >= 1:
        return "{:,.2f}".format(x).rstrip("0").rstrip(".")
    return "{:.3g}".format(x)
MIN_SEGMENT_EFFECT = 0.25


def _insights_general(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    from app.engines.domains.base import is_id_column
    findings, risks, opps, actions = [], [], [], []
    insights = []

    # Identifiers are not measures — exclude from every analysis below.
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if not is_id_column(c, df[c])]
    analysable = [c for c in stats.keys() if not is_id_column(c, df[c] if c in df else None)]

    for col in analysable[:8]:
        st = stats.get(col, {})
        if not st:
            continue
        skew    = st.get("skew", 0)
        out_pct = st.get("outlier_pct", 0)
        mean    = st.get("mean", 0)
        median  = st.get("median", 0)

        if out_pct > 10:
            # A high-outlier column is a genuine data-quality RISK — count it.
            risks.append(
                "'{}' has {:.0f}% outliers — verify before using it in any "
                "decision; extremes distort averages and models.".format(col, out_pct))
            insights.append(build_insight(
                title="'{}': {:.0f}% Outliers — Data Quality Issue".format(col, out_pct),
                problem="{:.0f}% of '{}' values are statistical outliers".format(out_pct, col),
                cause="Data entry errors, measurement anomalies, or genuine extreme values",
                evidence="IQR method: {:.0f}% outliers. Range: {:.2f} to {:.2f}".format(
                    out_pct, st.get("min", 0), st.get("max", 0)),
                action="1. Inspect outlier records  2. Determine error or genuine  "
                       "3. Cap or remove confirmed errors  4. Document decisions",
                impact="Outliers distort all statistical analyses and reduce ML accuracy",
                severity="warning", category="data_quality"
            ))
        if abs(skew) > 1.5:
            findings.append(
                "'{}' is {}-skewed (mean {:.2f} vs median {:.2f}). "
                "Report median for this column.".format(
                    col, "right" if skew > 0 else "left", mean, median))

    for corr in corrs[:3]:
        if corr.get("strength") in ("strong", "moderate"):
            # correlations() computes the real p-value — report it instead of
            # asserting "statistically significant" unconditionally.
            p = corr.get("p")
            if corr.get("significant"):
                sig_txt = "p<0.001" if (p is not None and p < 0.001) else \
                          ("p={:.3f}".format(p) if p is not None else "significant")
            else:
                sig_txt = "not significant (p={:.2f})".format(p) if p is not None \
                          else "significance unknown"
            findings.append(
                "{} {} relationship: '{}' and '{}' (r={:.2f}, {})".format(
                    corr["strength"].title(), corr["direction"],
                    corr["col_a"], corr["col_b"], corr["r"], sig_txt))

    # ── Generic opportunity detection — works on ANY column names ────────────
    # Looks for P90/median uplift potential, high-variability improvement room
    for col in num_cols[:8]:
        try:
            s = df[col].dropna()
            if len(s) < 20:
                continue
            p10  = float(s.quantile(0.10))
            p50  = float(s.quantile(0.50))
            p90  = float(s.quantile(0.90))
            mean_v = float(s.mean())
            cv   = s.std() / abs(mean_v) * 100 if mean_v != 0 else 0

            # Uplift opportunity: large spread between P10 and P90
            if p50 > 0 and p90 / p50 >= 2.0 and cv > 40:
                # This called the 10th percentile "the bottom quartile" and
                # quoted the top-decile gap as the gain from lifting it.
                opps.append(
                    f"'{col}': the top 10% of rows are above {_num(p90)}, "
                    f"{p90/p50:.1f}× the median ({_num(p50)}); the bottom 10% are "
                    f"below {_num(p10)}. Worth checking what the highest rows have "
                    f"in common."
                )
            # High concentration risk: >50% in one value for numeric col
            top_val_pct = float(s.value_counts(normalize=True).iloc[0]) * 100
            if top_val_pct > 60 and s.nunique() > 3:
                risks.append(
                    f"'{col}': {top_val_pct:.0f}% of records share the same value "
                    f"({_num(s.value_counts().index[0])}). "
                    "Potential data collection bias or limited diversity."
                )
        except Exception:
            logger.warning("Opportunity check failed for %s", col, exc_info=True)

    # ── Segment-difference insight — works on ANY dataset ────────────────
    # Best categorical × numeric pair, tested with Kruskal-Wallis so the
    # report can say "segment X outperforms Y" with a defensible p-value.
    seg = _best_segment_difference(df, num_cols)
    if seg:
        seg = {**seg, "lo_txt": _num(seg["lo"]), "hi_txt": _num(seg["hi"])}
        # NOTE: neutral highest/lowest labels — whether high is good or bad
        # depends on the metric (revenue vs delay), which we cannot know here.
        insights.append(build_insight(
            title="'{cat}' Segments Differ Significantly on '{num}'".format(**seg),
            problem=("Median '{num}' ranges from {lo_txt} in '{lo_seg}' to {hi_txt} "
                     "in '{hi_seg}' — a {ratio:.1f}× spread across '{cat}' segments."
                     .format(**seg)),
            cause="A gap this size is statistically unlikely to be chance "
                  "(Kruskal-Wallis p<0.01, see evidence). That points to a "
                  "structural or operational difference between the segments — "
                  "a hypothesis to confirm, not yet a proven cause.",
            evidence=("Kruskal-Wallis H={h:.1f}, {ptxt}, {k} groups, n={n:,}. "
                      "Highest: '{hi_seg}' (median {hi_txt}) | Lowest: '{lo_seg}' "
                      "(median {lo_txt}).".format(**seg)),
            action=("1. Determine which end of the '{num}' range is desirable, then "
                    "profile what separates '{hi_seg}' from '{lo_seg}'  "
                    "2. Check the gap persists after controlling for volume/size  "
                    "3. Pilot the stronger segment's practices in the weaker one"
                    .format(**seg)),
            impact=("Moving the weaker segment to the overall median is the measured "
                    "upper bound of the '{num}' opportunity in this dataset."
                    .format(**seg)),
            severity="high", category="segmentation",
        ))
        # A significant segment gap is both a risk (the weak segment) and an
        # opportunity (the gap to close) — count both so the exec summary
        # doesn't read '0 risks, 0 opportunities' next to a real finding.
        # Which end is weak depends on the metric: the lowest-delay segment is
        # the strong one. Unknown direction gets neutral wording.
        from app.engines.benchmarking import metric_direction
        direction = metric_direction(seg["num"])
        if direction:
            weak, strong = ((seg["lo_seg"], seg["hi_seg"]) if direction > 0
                            else (seg["hi_seg"], seg["lo_seg"]))
            weak_v, strong_v = ((seg["lo_txt"], seg["hi_txt"]) if direction > 0
                                else (seg["hi_txt"], seg["lo_txt"]))
            risks.append(
                "'{}' trails on '{}' ({} vs {} in '{}'), a statistically significant "
                "{:.1f}× gap.".format(weak, seg["num"], weak_v, strong_v, strong, seg["ratio"]))
            opps.append(
                "Closing the '{}' gap on '{}' — moving '{}' toward the '{}' level — is "
                "a quantified, testable improvement.".format(
                    seg["cat"], seg["num"], weak, strong))
        else:
            risks.append(
                "'{cat}' segments differ on '{num}' ({lo_txt} in '{lo_seg}' vs {hi_txt} "
                "in '{hi_seg}'), a statistically significant {ratio:.1f}× gap — decide "
                "which end is desirable before acting.".format(**seg))
            opps.append(
                "Find out what separates '{hi_seg}' from '{lo_seg}' on '{num}' — the "
                "{ratio:.1f}× gap is real, and testable.".format(**seg))

    # ── Concentration (Pareto) risk on categorical columns ───────────────
    for col in df.select_dtypes(include=["object", "string"]).columns[:6]:
        try:
            vc = df[col].dropna().value_counts(normalize=True)
            if 3 <= len(vc) <= 500 and float(vc.iloc[0]) > 0.5:
                findings.append(
                    "Concentration: {:.0f}% of records fall in a single '{}' value "
                    "('{}'). Aggregate metrics are dominated by this segment — "
                    "report it separately.".format(
                        float(vc.iloc[0]) * 100, col, str(vc.index[0])[:40]))
        except Exception:
            logger.warning("concentration check failed for %s", col, exc_info=True)

    # ── Trend, outlier concentration, segment sufficiency ────
    run_general_depth(df, insights, findings, risks, opps)

    # Each only where the data shows the reason for it.
    if any(stats.get(c, {}).get("outlier_pct", 0) > 1 for c in analysable):
        actions.append("Validate the outliers flagged above before analysis or modelling")
    if any(abs(stats.get(c, {}).get("skew", 0)) > 1 for c in analysable):
        actions.append("Use the median, not the mean, for the skewed columns in executive reports")
    if any(2 <= df[c].nunique() <= 30 for c in df.select_dtypes(include=["object", "string"]).columns):
        actions.append("Segment analysis — subgroups may tell different stories")
    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}


def _is_obvious_segment_pair(cat: str, num: str) -> bool:
    """True for mechanically-obvious pairings a senior analyst would never
    headline: age/tenure by seniority-level/role (older people hold senior
    roles), or a metric grouped by a near-synonym of itself. Statistically
    significant but worthless as a finding — 'managers are older than interns'.
    """
    c, n = cat.lower(), num.lower()
    demographic = ("age", "tenure", "years", "experience", "yearsat",
                   "yearsin", "yearssince")
    seniority   = ("level", "grade", "band", "seniority", "rank", "role",
                   "title", "position", "designation", "job")
    if any(d in n for d in demographic) and any(s in c for s in seniority):
        return True
    # Metric grouped by a column whose name contains it (near-tautology).
    stem = n.replace("_", "").replace(" ", "")[:6]
    if len(stem) >= 4 and stem in c.replace("_", "").replace(" ", ""):
        return True
    return False


def _best_segment_difference(df: pd.DataFrame, num_cols) -> dict | None:
    """
    Find the categorical × numeric pair with the strongest, statistically
    significant group difference (Kruskal-Wallis, p<0.01). Returns a dict of
    display fields, or None when nothing defensible exists. Skips mechanically
    obvious pairs (age-by-seniority) so the report never headlines a truism.
    """
    from scipy import stats as scipy_stats
    cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns
                if 2 <= df[c].nunique(dropna=True) <= 12]
    best = None
    for cat in cat_cols[:5]:
        for num in num_cols[:6]:
            if _is_obvious_segment_pair(cat, num):
                continue
            try:
                sub = df[[cat, num]].dropna()
                if len(sub) < 60:
                    continue
                groups = [g[num].values for _, g in sub.groupby(cat) if len(g) >= 15]
                if len(groups) < 2:
                    continue
                h, p = scipy_stats.kruskal(*groups)
                if np.isnan(h) or p >= 0.01:
                    continue
                med = sub.groupby(cat)[num].median().sort_values()
                lo, hi = float(med.iloc[0]), float(med.iloc[-1])
                if hi == lo:
                    continue
                ratio = abs(hi / lo) if lo != 0 else float("inf")

                # Significance alone is not a finding. On a few hundred
                # rows Kruskal-Wallis returns p<0.001 for a 6% median
                # difference, and the report then says one segment
                # "underperforms" over a gap nobody would act on — often
                # produced by a handful of outliers in the other group.
                # The effect has to be large enough to matter as well as
                # unlikely enough to be real.
                spread = float(sub[num].std())
                if spread <= 0:
                    continue
                standardised = abs(hi - lo) / spread
                if ratio < MIN_SEGMENT_RATIO and standardised < MIN_SEGMENT_EFFECT:
                    continue
                score = float(h)
                if best is None or score > best["h"]:
                    best = {
                        "cat": cat, "num": num, "h": float(h),
                        "ptxt": "p<0.001" if p < 0.001 else "p={:.3f}".format(p),
                        "k": len(groups), "n": len(sub),
                        "lo_seg": str(med.index[0])[:30], "hi_seg": str(med.index[-1])[:30],
                        "lo": lo, "hi": hi,
                        "ratio": min(ratio, 999.0),
                    }
            except Exception:
                logger.warning("segment difference test failed (%s × %s)",
                               cat, num, exc_info=True)
    return best


# ══════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════

