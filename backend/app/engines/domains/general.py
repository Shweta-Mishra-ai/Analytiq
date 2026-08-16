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

from app.engines.domains.base import (
    build_insight, outliers_explained_by_group,
)
from app.engines.domains.general_depth import run_general_depth
from app.services.numfmt import human_number

logger = logging.getLogger(__name__)

# A segment gap has to clear one of these to be worth a client's attention:
# either the groups differ by a quarter, or by a quarter of the column's own
# spread. Below both, the difference is real and irrelevant.
MIN_SEGMENT_RATIO = 1.25
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
            # One IQR fence across a mixed population is not an outlier
            # test. See `outliers_explained_by_group`: a retail file with
            # Electronics at 258 and Grocery at 12 produced three
            # data-quality warnings on columns that were entirely correct.
            grouping = outliers_explained_by_group(df, col)
            if grouping:
                findings.append(
                    "'{}' is not one distribution but several: the {:.0f}% of "
                    "values that sit outside the overall range are ordinary "
                    "within their own '{}'. Read it per group, not "
                    "pooled.".format(col, out_pct, grouping))
                insights.append(build_insight(
                    title="'{}' Splits by '{}' Rather Than Being Dirty".format(
                        col, grouping),
                    problem="{:.0f}% of '{}' values fall outside the fence "
                            "computed across the whole file, but not outside "
                            "the fence computed within their own "
                            "'{}'.".format(out_pct, col, grouping),
                    cause="The column mixes groups with genuinely different "
                          "levels. A pooled quartile range describes none of "
                          "them.",
                    evidence="IQR across all rows flags {:.0f}%. Computing the "
                             "same fence within each '{}' removes most of "
                             "them. Overall range: {:,.2f} to {:,.2f}.".format(
                                 out_pct, grouping, st.get("min", 0),
                                 st.get("max", 0)),
                    action="1. Report and model '{}' per '{}' rather than "
                           "pooled  2. Treat any pooled average of this "
                           "column as a mix, not a typical value".format(
                               col, grouping),
                    impact="Nothing needs cleaning here. Pooling the groups "
                           "is what produces the wrong number.",
                    severity="info", category="data_quality"
                ))
            else:
                # A genuine data-quality risk — count it.
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
                uplift_pct = (p90 - p50) / p50 * 100
                opps.append(
                    f"'{col}': Top decile ({human_number(p90)}) is {p90/p50:.1f}× the median "
                    f"({human_number(p50)}). Bringing the bottom quartile (currently {human_number(p10)}) "
                    f"to median would represent a {uplift_pct:.0f}% improvement. "
                    f"Identify what high performers have in common."
                )
            # High concentration risk: >50% in one value for numeric col
            top_val_pct = float(s.value_counts(normalize=True).iloc[0]) * 100
            if top_val_pct > 60 and s.nunique() > 3:
                risks.append(
                    f"'{col}': {top_val_pct:.0f}% of records share the same value "
                    f"({human_number(s.value_counts().index[0])}). "
                    "Potential data collection bias or limited diversity."
                )
        except Exception:
            logger.warning("Opportunity check failed for %s", col, exc_info=True)

    # ── Segment-difference insight — works on ANY dataset ────────────────
    # Best categorical × numeric pair, tested with Kruskal-Wallis so the
    # report can say "segment X outperforms Y" with a defensible p-value.
    seg = _best_segment_difference(df, num_cols)
    if seg:
        # `%g` switched to scientific notation on anything above five
        # figures, so the lead sentence of a finance report read "ranges
        # from 7.72e+04 in 'Support' to 8.89e+05 in 'Retail'".
        seg = dict(seg)
        seg["lo_txt"] = human_number(seg["lo"])
        seg["hi_txt"] = human_number(seg["hi"])
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
            # This is a cross-tab that runs on any dataset. Marked "high"
            # it outranked the domain engines' own findings: a finance
            # report led with the revenue spread between a sales cost
            # centre and a support one, ahead of margin and cost
            # structure. It is a real result and a supporting one.
            severity="warning", category="segmentation",
        ))
        # The insight above is careful to stay neutral about which end of
        # the range is desirable — and this line then called the low end
        # underperformance. On the finance file that read as "'Support'
        # underperforms on 'revenue'", which is what a support cost centre
        # is meant to do. Stated as a spread, it is true either way.
        risks.append(
            "'{num}' differs {ratio:.1f}× across '{cat}' ({lo_txt} in "
            "'{lo_seg}' against {hi_txt} in '{hi_seg}'); if the two are "
            "expected to be comparable, the gap is larger than sampling "
            "explains.".format(**seg))
        opps.append(
            "Closing the '{cat}' gap on '{num}' — lifting '{lo_seg}' toward the "
            "'{hi_seg}' level — is a quantified, testable improvement.".format(**seg))

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

    actions.extend([
        "Validate all outliers before analysis or modeling",
        "Use median for skewed distributions in executive reports",
        "Segment analysis — subgroups may tell different stories",
    ])
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

