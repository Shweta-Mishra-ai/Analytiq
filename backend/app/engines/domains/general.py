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

# Everything a reader sees goes through present: column names as
# words rather than identifiers, numbers as figures rather than
# scientific notation.
from app.engines.present import (label as _L, num as _N,
                                 truncate as _T)

# A segment gap has to clear one of these to be worth a client's attention:
# either the groups differ by a quarter, or by a quarter of the column's own
# spread. Below both, the difference is real and irrelevant.
MIN_SEGMENT_RATIO = 1.25
MIN_SEGMENT_EFFECT = 0.25


def _is_rating_scale(series) -> bool:
    """True for an ordinal rating rather than a measurement.

    IQR flags anything past 1.5 IQR from the quartiles. On a 1-4
    satisfaction score or a 3/4 performance rating that is not an
    anomaly, it is the top of the scale — and the report was telling
    clients that 16% of their performance ratings were a data quality
    issue to "cap or remove". A rating has no outliers; it has levels.
    """
    if series is None:
        return False
    try:
        values = series.dropna()
        if values.empty:
            return False
        if values.nunique() > 11:
            return False
        as_float = values.astype(float)
        if not (as_float % 1 == 0).all():
            return False
        # A small contiguous range near zero: the shape of a Likert or
        # star rating, not of an age, a count or an amount.
        return bool(as_float.min() >= 0 and as_float.max() <= 10)
    except Exception:
        logger.debug("rating-scale check failed", exc_info=True)
        return False


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

        if out_pct > 10 and not _is_rating_scale(df[col] if col in df else None):
            # A high-outlier column is a genuine data-quality RISK — count it.
            risks.append(
                "{} has {:.0f}% outliers — verify these before using the field "
                "in any decision, because extremes distort both averages and "
                "models.".format(_L(col), out_pct))
            insights.append(build_insight(
                title="{} carries {:.0f}% outliers".format(_L(col), out_pct),
                problem="{:.0f}% of {} values sit outside the expected range"
                        .format(out_pct, _L(col)),
                cause="Data entry errors, measurement anomalies, or genuine "
                      "extreme values — which of the three it is cannot be "
                      "determined from the data alone",
                evidence="IQR method: {:.0f}% outliers. Range {} to {}.".format(
                    out_pct, _N(st.get("min", 0)), _N(st.get("max", 0))),
                action="1. Inspect the outlying records  2. Determine whether "
                       "each is an error or genuine  3. Correct confirmed "
                       "errors at source  4. Record the decision so the next "
                       "run is comparable",
                impact="Until they are resolved, every average and model built "
                       "on {} carries them.".format(_L(col)),
                severity="warning", category="data_quality"
            ))
        if abs(skew) > 1.5:
            findings.append(
                "{} is {}-skewed (mean {} against median {}), so the median is "
                "the fair summary for this column.".format(
                    _L(col), "right" if skew > 0 else "left",
                    _N(mean), _N(median)))

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
                    f"{_L(col)}: the top decile ({_N(p90)}) is "
                    f"{p90/p50:.1f}x the median ({_N(p50)}). Bringing the "
                    f"bottom quartile (currently {_N(p10)}) to the median "
                    f"would be a {uplift_pct:.0f}% improvement. Identify what "
                    f"the high performers have in common."
                )
            # High concentration risk: >50% in one value for numeric col
            top_val_pct = float(s.value_counts(normalize=True).iloc[0]) * 100
            if top_val_pct > 60 and s.nunique() > 3:
                risks.append(
                    f"{_L(col)}: {top_val_pct:.0f}% of records share one value "
                    f"({_N(s.value_counts().index[0])}). That points to a data "
                    "collection default or limited genuine variation, and it "
                    "flattens any average built on this field."
                )
        except Exception:
            logger.warning("Opportunity check failed for %s", col, exc_info=True)

    # ── Segment-difference insight — works on ANY dataset ────────────────
    # Best categorical × numeric pair, tested with Kruskal-Wallis so the
    # report can say "segment X outperforms Y" with a defensible p-value.
    seg = _best_segment_difference(df, num_cols)
    if seg:
        # NOTE: neutral highest/lowest labels — whether high is good or bad
        # depends on the metric (revenue vs delay), which we cannot know here.
        view = dict(seg, cat_l=_L(seg["cat"]), num_l=_L(seg["num"]),
                    lo_f=_N(seg["lo"]), hi_f=_N(seg["hi"]))
        insights.append(build_insight(
            title="{hi_seg} and {lo_seg} differ on {num_l}".format(**view),
            problem=("Median {num_l} runs from {lo_f} in {lo_seg} to {hi_f} in "
                     "{hi_seg} — a {ratio:.1f}x spread across {cat_l}."
                     .format(**view)),
            cause="A gap this size is statistically unlikely to be chance "
                  "(Kruskal-Wallis p<0.01, see evidence). That points to a "
                  "structural or operational difference between the segments — "
                  "a hypothesis to confirm, not yet a proven cause.",
            evidence=("Kruskal-Wallis H={h:.1f}, {ptxt}, {k} groups, n={n:,}. "
                      "Highest: {hi_seg} (median {hi_f}). Lowest: {lo_seg} "
                      "(median {lo_f}).".format(**view)),
            action=("1. Decide which end of the {num_l} range is the desirable "
                    "one, then profile what separates {hi_seg} from {lo_seg}  "
                    "2. Check the gap persists after controlling for volume "
                    "and size  "
                    "3. Pilot the stronger segment's practices in the weaker one"
                    .format(**view)),
            impact=("Moving the weaker segment to the overall median is the "
                    "measured upper bound of the {num_l} opportunity in this "
                    "dataset.".format(**view)),
            severity="high", category="segmentation",
        ))
        # A significant segment gap is both a risk (the weak segment) and an
        # opportunity (the gap to close) — count both so the exec summary
        # doesn't read '0 risks, 0 opportunities' next to a real finding.
        risks.append(
            "{lo_seg} sits below {hi_seg} on {num_l} ({lo_f} against {hi_f}), "
            "a statistically significant {ratio:.1f}x gap.".format(**view))
        opps.append(
            "Closing the {cat_l} gap on {num_l} — lifting {lo_seg} toward the "
            "{hi_seg} level — is a quantified, testable improvement."
            .format(**view))

    # ── Concentration (Pareto) risk on categorical columns ───────────────
    for col in df.select_dtypes(include=["object", "string"]).columns[:6]:
        try:
            vc = df[col].dropna().value_counts(normalize=True)
            if 3 <= len(vc) <= 500 and float(vc.iloc[0]) > 0.5:
                findings.append(
                    "Concentration: {:.0f}% of records fall in a single {} "
                    "value ({}). Every aggregate is dominated by this segment, "
                    "so report it separately.".format(
                        float(vc.iloc[0]) * 100, _L(col),
                        _T(str(vc.index[0]), 40)))
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
    # Pay by job role is definitional, not a finding: the role IS the pay
    # band. The report was headlining "Research Scientist and HR Specialist
    # differ on Monthly Income" as a HIGH severity result. Pay by
    # department or location stays in — that one is a real question.
    compensation = ("salary", "income", "pay", "wage", "compensation",
                    "remuneration", "ctc", "stipend", "bonus", "rate")
    role_like = ("role", "title", "position", "designation", "level",
                 "grade", "band", "seniority", "rank", "job")
    if any(m in n for m in compensation) and any(r in c for r in role_like):
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

