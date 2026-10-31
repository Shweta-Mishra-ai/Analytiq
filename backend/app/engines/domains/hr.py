"""
core/engines/hr.py — HR & People Analytics domain engine.
Single responsibility: given a DataFrame, produce structured HR insights.
All metrics computed from the submitted dataset — no external benchmarks.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.engines.domains.base import (Insight, AttritionAnalysis, build_insight,
                               col_stats, correlations, infer_scale_bounds)
from app.services.dtypes import is_text_dtype
from app.services.stat_guards import apply_fdr, chi2_association

logger = logging.getLogger(__name__)

# Back-compat alias — scale inference now lives in base.py (shared with the
# ecommerce rating engine so both handle 1-5 / 1-10 / 0-100 scales identically).
_infer_scale_bounds = infer_scale_bounds


def _find_left_mask(df: pd.DataFrame):
    """Boolean 'employee left' mask, or None if no attrition column exists."""
    attr_col = next((c for c in df.columns
                     if "attrition" in c.lower()
                     or c.lower() in ["left", "churned", "resigned"]), None)
    if attr_col is None:
        return None
    return df[attr_col].astype(str).str.lower().str.strip().isin(
        ["yes", "1", "1.0", "true", "left"])


def _segment_attrition_evidence(df: pd.DataFrame, segment_mask, segment_label: str) -> str:
    """
    Compute THIS dataset's attrition rate inside a risk segment vs everyone
    else. Replaces invented multipliers ('2-3x more likely to leave') with a
    measured, defensible number — or honest silence when it can't be measured.
    """
    try:
        left = _find_left_mask(df)
        if left is None:
            return ""
        seg, rest = left[segment_mask], left[~segment_mask]
        if len(seg) < 10 or len(rest) < 10:
            return ""
        r_seg, r_rest = seg.mean() * 100, rest.mean() * 100
        if r_rest > 0 and r_seg > r_rest:
            return ("In this dataset, {} leave at {:.1f}% vs {:.1f}% for everyone else "
                    "({:.1f}× the rate).".format(segment_label, r_seg, r_rest, r_seg / r_rest))
        return ("In this dataset, {} leave at {:.1f}% vs {:.1f}% for everyone else."
                .format(segment_label, r_seg, r_rest))
    except Exception:
        logger.warning("segment attrition evidence failed", exc_info=True)
        return ""


def _segment_exit_rates(df: pd.DataFrame, segment_mask):
    """(segment rate %, everyone-else rate %, chi-square p) or None."""
    try:
        from app.services.stat_guards import chi2_association
        left = _find_left_mask(df)
        if left is None:
            return None
        mask = segment_mask.reindex(left.index, fill_value=False)
        seg, rest = left[mask], left[~mask]
        if len(seg) < 10 or len(rest) < 10:
            return None
        test = chi2_association(pd.crosstab(mask, left))
        return (float(seg.mean() * 100), float(rest.mean() * 100),
                float(test["p"]) if test else float("nan"))
    except Exception:
        logger.warning("segment exit rates failed", exc_info=True)
        return None


def _exit_sentence(label: str, exits) -> str:
    """'Those X leave at a% against b% for everyone else (p=…).'"""
    if not exits:
        return "This data has no record of who left, or too few people in the group."
    a, b, p = exits
    test = "" if p != p else " (p={:.3f})".format(p)      # NaN: test not valid
    return "{} leave at {:.1f}% against {:.1f}% for everyone else{}.".format(label, a, b, test)


def group_exit_gap(groups: pd.Series, left_mask: pd.Series, min_n: int = 20):
    """Exit rate per group and whether the groups really differ.

    Returns None unless at least two groups have ``min_n`` people and the
    chi-square test can be run on them. Otherwise a dict with ``rates``
    (group -> exit %, highest first), ``sizes``, ``worst``, ``best`` and
    ``p``. A gap between the highest and lowest group is a finding only when
    ``p`` is small: with four groups of ~120, ranges of 5–10 points turn up
    by chance alone, and reporting one as "the department with the highest
    attrition" was reporting noise.
    """
    try:
        g = groups.reindex(left_mask.index)
        keep = g.notna()
        g, left = g[keep].astype(str), left_mask[keep].astype(bool)
        sizes = g.value_counts()
        big = sizes[sizes >= min_n].index
        if len(big) < 2:
            return None
        g, left = g[g.isin(big)], left[g.isin(big)]
        test = chi2_association(pd.crosstab(g, left))
        if test is None:
            return None
        rates = (left.groupby(g).mean() * 100).sort_values(ascending=False)
        return {"rates": rates, "sizes": sizes[big], "p": float(test["p"]),
                "worst": str(rates.index[0]), "best": str(rates.index[-1])}
    except Exception:
        logger.warning("group exit gap failed", exc_info=True)
        return None


def describe_rates(gap: dict) -> str:
    """'HR 23% (n=120) · Sales 19% (n=131)' — the figures behind a gap."""
    return " · ".join("{} {:.0f}% (n={:,})".format(k, v, int(gap["sizes"][k]))
                      for k, v in gap["rates"].items())


def compute_flight_risk(df: pd.DataFrame, left_mask=None):
    """Single source of truth for 'flight risk' headcount.

    Definition: among CURRENT (not-left) employees, those whose satisfaction
    score falls below the bottom-25th-percentile of current employees'
    satisfaction. Single-factor (satisfaction only) — this used to be
    computed a second, different way (satisfaction below p25 AND tenure
    above median) in the Health Report, which produced a different headcount
    for the same dataset. Both reports now call this one function.

    Returns (n_flight, flight_pct, sat_col). n_flight/flight_pct are 0 if no
    satisfaction column exists. flight_pct is a percentage of current
    (not-left) employees.
    """
    if left_mask is None:
        left_mask = _find_left_mask(df)
        if left_mask is None:
            left_mask = pd.Series(False, index=df.index)

    n_total = len(df)
    n_left = int(left_mask.sum())
    sat_col = next((c for c in df.columns if "satisfaction" in c.lower()), None)
    n_flight = 0
    if sat_col and sat_col in df.columns:
        sv = df.loc[~left_mask, sat_col].dropna()
        thresh = float(sv.quantile(0.25)) if len(sv) > 0 else 0.4
        n_flight = int((sv < thresh).sum())

    flight_pct = round(n_flight / max(n_total - n_left, 1) * 100, 1)
    return n_flight, flight_pct, sat_col


def _run_attrition(df: pd.DataFrame) -> Optional[AttritionAnalysis]:
    attr_col = next((c for c in df.columns
                     if "attrition" in c.lower()
                     or c.lower() in ["left","churned","resigned"]), None)
    if attr_col is None:
        return None

    left_mask = df[attr_col].astype(str).str.lower().str.strip().isin(
        ["yes","1","1.0","true","left"])
    n_left  = int(left_mask.sum())
    n_total = len(df)
    if n_left == 0:
        return None

    rate = round(n_left / max(n_total,1) * 100, 1)
    severity = ("critical" if rate > 25 else "high" if rate > 18
                else "warning" if rate > 12 else "normal")

    # Numeric drivers
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != attr_col]
    top_drivers = []
    for col in num_cols[:12]:
        try:
            lv = df.loc[left_mask, col].dropna()
            sv = df.loc[~left_mask, col].dropna()
            if len(lv) < 5 or len(sv) < 5: continue
            _, p = scipy_stats.mannwhitneyu(lv, sv, alternative="two-sided")
            if p < 0.05:
                diff_pct = abs(lv.mean()-sv.mean())/abs(sv.mean())*100 if sv.mean()!=0 else 0
                top_drivers.append({
                    "factor":    col,
                    "type":      "numeric",
                    "impact":    round(diff_pct,1),
                    "direction": "lower" if lv.mean()<sv.mean() else "higher",
                    "left_mean": round(float(lv.mean()),3),
                    "stay_mean": round(float(sv.mean()),3),
                    "p_value":   round(float(p),4),
                    "detail":    "Leavers avg {:.2f} vs stayers {:.2f} ({:.0f}% diff)".format(
                        lv.mean(), sv.mean(), diff_pct),
                })
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)
            continue

    # Categorical drivers
    cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns
                if c != attr_col and df[c].nunique() <= 20]
    for col in cat_cols[:6]:
        try:
            ct = pd.crosstab(df[col], left_mask)
            if ct.shape[1] < 2: continue
            # Verified chi-square: skips tables whose expected cell counts
            # are too small for the p-value to mean anything, and gives an
            # effect size so a large-n table can't turn a trivial difference
            # into a headline "attrition driver".
            assoc = chi2_association(ct)
            if assoc is None:
                continue
            p = assoc["p"]
            if p < 0.05 and assoc["cramers_v"] >= 0.1:
                rates = {str(k): round(left_mask[df[col]==k].mean()*100,1)
                         for k in df[col].dropna().unique()}
                worst = max(rates, key=rates.get)
                best  = min(rates, key=rates.get)
                top_drivers.append({
                    "factor":     col, "type": "categorical",
                    "impact":     round(rates[worst]-rates[best],1),
                    "worst_cat":  worst, "worst_rate": rates[worst],
                    "best_cat":   best,  "best_rate":  rates[best],
                    "p_value":    round(float(p),4),
                    "detail":     "'{}' = {:.0f}% vs '{}' = {:.0f}%".format(
                        worst, rates[worst], best, rates[best]),
                })
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)
            continue

    # Up to 12 numeric plus 6 categorical columns are tested here, each at
    # α=0.05 — across ~18 tests about one spurious "attrition driver" is
    # expected from chance alone, and it would be presented as a cause.
    # Benjamini-Hochberg controls the false-discovery rate over the family.
    if top_drivers:
        top_drivers = apply_fdr(top_drivers, p_key="p_value", q_key="q_value")
        survivors = [d for d in top_drivers if d["q_value"] < 0.05]
        if not survivors:
            logger.info("attrition: %d candidate driver(s) failed FDR "
                        "correction — reporting none", len(top_drivers))
        top_drivers = survivors

    top_drivers.sort(key=lambda x: x["impact"], reverse=True)

    # Segment breakdown
    dept_col = next((c for c in df.columns
                     if "department" in c.lower() or "dept" in c.lower()), None)
    sal_col  = next((c for c in df.columns
                     if "salary" in c.lower() and is_text_dtype(df[c])), None)

    dept_attrition = {}
    if dept_col:
        for d in df[dept_col].dropna().unique():
            m = df[dept_col]==d
            dept_attrition[str(d)] = round(left_mask[m].mean()*100,1)

    salary_attrition = {}
    if sal_col:
        for s in df[sal_col].dropna().unique():
            m = df[sal_col]==s
            salary_attrition[str(s)] = round(left_mask[m].mean()*100,1)

    # Flight risk — single shared definition, see compute_flight_risk().
    n_flight, flight_pct, _sat_col = compute_flight_risk(df, left_mask=left_mask)
    # Gallup puts the cost of replacing an employee at one-half to two times
    # their annual salary. The figure is general, so the text says so.
    cost_str   = ("Gallup estimates replacing an employee at one-half to two times "
                  "their annual salary; applied to {:,} leavers with your own salary "
                  "figures, that gives the cost for this organisation".format(n_left))

    dept_gap = group_exit_gap(df[dept_col], left_mask) if dept_col else None

    return AttritionAnalysis(
        rate=rate, n_left=n_left, n_total=n_total,
        severity=severity, top_drivers=top_drivers[:8],
        dept_attrition=dept_attrition,
        salary_attrition=salary_attrition,
        n_flight_risk=n_flight, flight_risk_pct=flight_pct,
        cost_estimate=cost_str,
        dept_p=dept_gap["p"] if dept_gap else None,
        interpretation="{:.1f}% attrition ({:,} employees).{}".format(
            rate, n_left,
            " The factor most associated with leaving is '{}'.".format(
                top_drivers[0]["factor"]) if top_drivers else
            " No factor in this data separates leavers from stayers beyond chance."),
    )


# ══════════════════════════════════════════════════════════
#  HR INSIGHTS
# ══════════════════════════════════════════════════════════

def _insights_hr(df: pd.DataFrame, stats: Dict,
                 corrs: List, attrition: Optional[AttritionAnalysis]) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights = []

    # FIX-050: 50+ HR column synonym patterns — handles any HR dataset structure
    def _hr_col(*keywords, cat_ok=False, max_unique=None):
        """Find first column matching any keyword. Respects type and cardinality."""
        for col in df.columns:
            col_l = col.lower().strip()
            if any(kw in col_l for kw in keywords):
                if max_unique and df[col].nunique() > max_unique:
                    continue
                if not cat_ok and is_text_dtype(df[col]):
                    continue
                return col
        return None

    # Satisfaction — 0-1 scale OR 1-5 scale OR 1-10 scale
    sat_col = _hr_col(
        "satisfaction", "engagement", "survey", "happiness",
        "morale", "sentiment", "wellbeing", "nps", "esat"
    )

    # Performance evaluation
    eval_col = _hr_col(
        "evaluat", "performance", "appraisal", "rating", "score",
        "review", "perfscore", "perfrating", "performancerating"
    )

    # Working hours
    hrs_col = _hr_col(
        "hour", "monthly_hours", "avg_hour", "workhour",
        "overtime", "workedhour", "timeworked"
    )

    # Department
    dept_col = _hr_col(
        "department", "dept", "division", "team", "group",
        "business_unit", "bu", "function", "unit",
        cat_ok=True, max_unique=30
    )

    # Projects
    proj_col = _hr_col(
        "project", "numberproject", "num_project", "activeproject"
    )

    # FIX-051: Satisfaction scale normalizer
    def _normalize_sat(col):
        """Normalize satisfaction to 0-1 scale for consistent benchmarking."""
        if col is None:
            return None, None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            return col, 1.0
        max_val = s.max()
        if max_val <= 1.0:
            return col, 1.0      # Already 0-1
        elif max_val <= 5.0:
            return col, 5.0      # 1-5 scale
        elif max_val <= 10.0:
            return col, 10.0     # 1-10 scale
        else:
            return col, 100.0    # Percentage scale

    sat_col, sat_scale = _normalize_sat(sat_col)

    # ── Attrition ──────────────────────────────────────────
    if attrition:
        rate = attrition.rate
        if attrition.severity in ("critical","high"):
            insights.append(build_insight(
                title="Attrition at {:.1f}% — above the planning band".format(rate),
                problem="{:,} out of {:,} employees left — {:.1f}% rate".format(
                    attrition.n_left, attrition.n_total, rate),
                cause="That is {:.1f} points above the top of the 10–15% planning band "
                      "used here; healthy rates vary by industry.".format(rate - 15),
                evidence=("Leavers differ from stayers most on '{}' ({}).".format(
                              attrition.top_drivers[0]["factor"],
                              attrition.top_drivers[0]["detail"])
                          if attrition.top_drivers else
                          "No measured factor separates leavers from stayers beyond chance."),
                action="Interview leavers where possible to learn why they left, and "
                       "compare pay with the market for the roles with the most exits.",
                impact=attrition.cost_estimate + ".",
                severity="critical", category="attrition"
            ))
            risks.append("{:.1f}% attrition — {:,} employees left, above the 10–15% "
                         "planning band.".format(rate, attrition.n_left))
        elif 10 <= rate <= 15:
            findings.append("Attrition rate {:.1f}% is within the 10–15% planning band.".format(rate))
        elif rate < 10:
            findings.append("Attrition rate {:.1f}% is below the 10–15% planning band.".format(rate))
        else:
            findings.append("Attrition rate {:.1f}% is {:.1f} points above the 10–15% "
                            "planning band.".format(rate, rate - 15))

        # Department and pay-band gaps. Each is reported only when the
        # chi-square test says the groups really differ: with a few groups
        # of ~100 people, a 5–10 point spread between the highest and lowest
        # turns up by chance, and was printed as "the department with the
        # highest attrition".
        left_mask = _find_left_mask(df)
        dept_c = next((c for c in df.columns
                       if "department" in c.lower() or "dept" in c.lower()), None)
        dept_gap = (group_exit_gap(df[dept_c], left_mask)
                    if dept_c is not None and left_mask is not None else None)
        if dept_gap and dept_gap["p"] < 0.05:
            worst_d, best_d = dept_gap["worst"], dept_gap["best"]
            worst_r = float(dept_gap["rates"].iloc[0])
            best_r = float(dept_gap["rates"].iloc[-1])
            excess = round((worst_r - rate) / 100 * int(dept_gap["sizes"][worst_d]))
            insights.append(build_insight(
                title="'{}' loses {:.0f}% of its staff, against {:.0f}% in '{}'".format(
                    worst_d, worst_r, best_r, best_d),
                problem="Attrition in '{}' is {:.1f}%, against {:.1f}% company-wide.".format(
                    worst_d, worst_r, rate),
                # Naming management quality or workload as the cause
                # asserts something no column in this dataset measures.
                cause="The gap is established; its driver is not, because this "
                      "dataset holds no manager, workload or progression fields "
                      "to test it against.",
                evidence="Exit rate by department: {}. Chi-square p={:.3f}.".format(
                    describe_rates(dept_gap), dept_gap["p"]),
                action="Compare '{}' with '{}' on tenure mix, pay and role to see what "
                       "differs; where the data cannot explain the gap, exit interviews "
                       "in '{}' are the next source.".format(worst_d, best_d, worst_d),
                impact="At the company rate, '{}' would have lost about {:,} fewer "
                       "people.".format(worst_d, excess) if excess > 0 else "",
                severity="critical" if worst_r > 25 else "warning",
                category="department"
            ))
            risks.append("'{}' attrition {:.0f}%, against {:.0f}% in '{}' (p={:.3f}).".format(
                worst_d, worst_r, best_r, best_d, dept_gap["p"]))

        pay_c = next((c for c in df.columns
                      if any(k in c.lower() for k in ("salary", "pay", "compensation", "wage"))
                      and not any(k in c.lower() for k in ("percent", "pct", "hike",
                                                           "rate", "ratio", "date"))), None)
        bands = None
        if pay_c is not None and left_mask is not None:
            if pd.api.types.is_numeric_dtype(df[pay_c]) and df[pay_c].nunique() > 8:
                try:
                    bands = pd.qcut(df[pay_c], 4, labels=["Lowest 25%", "Lower-middle",
                                                          "Upper-middle", "Highest 25%"])
                except ValueError:
                    bands = None
            elif df[pay_c].nunique() <= 8:
                bands = df[pay_c]
        pay_gap = group_exit_gap(bands, left_mask) if bands is not None else None
        if pay_gap and pay_gap["p"] < 0.05:
            worst_b, best_b = pay_gap["worst"], pay_gap["best"]
            worst_r = float(pay_gap["rates"].iloc[0])
            best_r = float(pay_gap["rates"].iloc[-1])
            kept = round((worst_r - best_r) / 100 * int(pay_gap["sizes"][worst_b]))
            insights.append(build_insight(
                title="'{}' pay band: {:.0f}% attrition, against {:.0f}% in '{}'".format(
                    worst_b, worst_r, best_r, best_b),
                problem="Exit rates differ by pay band by more than chance explains.",
                cause="Pay band also tracks role and seniority, which this data may "
                      "not separate, so the link does not show that pay causes the exits.",
                evidence="Exit rate by pay band: {}. Chi-square p={:.3f}.".format(
                    describe_rates(pay_gap), pay_gap["p"]),
                action="Compare pay in the '{}' band with the market rate for the same "
                       "roles, and check whether leavers from it cite pay.".format(worst_b),
                impact="At the '{}' rate, about {:,} fewer people would have left the "
                       "'{}' band.".format(best_b, kept, worst_b) if kept > 0 else "",
                severity="critical" if worst_r > 25 else "warning",
                category="pay"
            ))

    # ── Satisfaction ───────────────────────────────────────
    if sat_col and sat_col in stats:
        st     = stats[sat_col]
        mean_s = st.get("mean", 0)
        obs_min = st.get("min", 0)
        obs_max = st.get("max", 1)
        # Infer the THEORETICAL scale bounds, not the observed ones. If everyone
        # scores 4-5 on a 1-5 Likert, the floor is still 1 — using observed min
        # (4) would wrongly report genuine high satisfaction as ~57%.
        min_s, max_s = _infer_scale_bounds(obs_min, obs_max)
        rng    = max_s - min_s
        # Scale-aware normalisation: express the mean as % of the scale RANGE,
        # so it reads correctly whether the data is a 0-1, 1-5, 1-10 or 0-100
        # scale. (A raw mean/max is wrong for scales that don't start at 0 —
        # e.g. a 1-5 Likert midpoint of 3.0 is 50% of range, not 60%.)
        pct = ((mean_s - min_s) / rng * 100) if rng > 0 else 0
        # "Critically dissatisfied" = bottom 40% of the scale range (bottom-box
        # responses). Threshold and count use the SAME scale-relative cutoff,
        # so the number is never a spurious 0 on non-0-1 scales.
        low_thresh = min_s + 0.4 * rng
        low_mask   = df[sat_col].dropna() < low_thresh
        low_n      = int(low_mask.sum()) if sat_col in df.columns else 0
        low_pct    = round(low_n / len(df) * 100, 1) if len(df) else 0.0

        exits = _segment_exit_rates(df, df[sat_col] < low_thresh)
        drives_exits = bool(exits and exits[0] > exits[1] and exits[2] < 0.05)
        if exits is None:
            exit_impact = ("This data has no record of who left, so the effect of low "
                           "satisfaction on exits cannot be measured here.")
        elif drives_exits:
            exit_impact = ("Employees in the bottom band leave at {:.1f}% against {:.1f}% "
                           "for everyone else (p={:.3f}) — the measured cost of low "
                           "satisfaction in this data.".format(*exits))
        else:
            exit_impact = ("Employees in the bottom band leave at about the same rate as "
                           "everyone else ({:.1f}% vs {:.1f}%), so low satisfaction is not "
                           "what separates leavers here.".format(exits[0], exits[1]))
        if pct < 55:
            insights.append(build_insight(
                title="Satisfaction averages {:.0f}% of scale — {:,} employees in the bottom band".format(pct, low_n),
                problem="The average score sits at {:.0f}% of the scale, and {:,} employees "
                        "({:.0f}%) are in its bottom band.".format(pct, low_n, low_pct),
                # The dataset records the score, not why it is low. Naming
                # culture, workload, recognition or pay as the cause would be
                # asserting something this data cannot show — the exact kind
                # of unfounded claim that gets a report dismissed.
                cause="This dataset records the score, not its drivers.",
                evidence="Mean {:.2f} on a {:.0f}–{:.0f} scale; the bottom band is "
                         "below {:.2f}.".format(mean_s, min_s, max_s, low_thresh),
                action="Find out what lies behind the low scores from survey comments "
                       "and exit interviews, and break the scores down by department "
                       "to see where they are lowest.",
                impact=exit_impact,
                severity="critical" if (pct < 40 or drives_exits) else "warning",
                category="satisfaction"
            ))
            risks.append("{:,} employees ({:.0f}%) score in the bottom band of the "
                         "satisfaction scale.".format(low_n, low_pct))
        elif pct < 70:
            insights.append(build_insight(
                title="Satisfaction averages {:.0f}% of the scale".format(pct),
                problem="The average score sits at {:.0f}% of the scale; {:,} employees "
                        "({:.0f}%) are in its bottom band.".format(pct, low_n, low_pct),
                cause="This dataset records the score, not its drivers.",
                evidence="Mean {:.2f} on a {:.0f}–{:.0f} scale; the bottom band is "
                         "below {:.2f}.".format(mean_s, min_s, max_s, low_thresh),
                action="Break the scores down by department to see where they are lowest.",
                impact=exit_impact,
                severity="warning" if drives_exits else "info", category="satisfaction"
            ))
        else:
            insights.append(build_insight(
                title="Satisfaction averages {:.0f}% of the scale".format(pct),
                problem="The average score sits at {:.0f}% of the scale.".format(pct),
                cause="",
                evidence="Mean {:.2f} on a {:.0f}–{:.0f} scale; {:,} employees ({:.0f}%) "
                         "are in the bottom band.".format(mean_s, min_s, max_s, low_n, low_pct),
                action="Keep measuring it on the same scale so a fall shows early.",
                impact=exit_impact,
                severity="positive", category="satisfaction"
            ))

    # ── Overwork ───────────────────────────────────────────
    # Long hours are a fact; burnout and its causes are not in the data. The
    # card states the hours and, where the data records exits, whether the
    # long-hours group actually leaves more.
    if hrs_col and hrs_col in stats:
        st       = stats[hrs_col]
        mean_hrs = st.get("mean",0)
        long_mask = df[hrs_col] > 260
        high_n   = int(long_mask.sum())
        if mean_hrs > 220:
            exits = _segment_exit_rates(df, long_mask)
            leaves_more = bool(exits and exits[0] > exits[1] and exits[2] < 0.05)
            insights.append(build_insight(
                title="Employees average {:.0f} hours a month; {:,} work more than 260".format(
                    mean_hrs, high_n),
                problem="Average monthly hours are {:.0f}, against a typical full-time "
                        "160–200.".format(mean_hrs),
                cause="This dataset records the hours, not why they are worked.",
                evidence=_exit_sentence("Those working more than 260 hours", exits),
                action="Look at where the long hours are concentrated, by department "
                       "or role, before deciding whether to rebalance work or hire.",
                impact=("The long-hours group's higher exit rate is the measured cost here."
                        if leaves_more else ""),
                severity="critical" if (mean_hrs >= 240 and leaves_more) else "warning",
                category="workload"
            ))
            risks.append("Average {:.0f} hours a month; {:,} employees work more than "
                         "260.".format(mean_hrs, high_n))

    # ── Projects ───────────────────────────────────────────
    if proj_col and proj_col in stats:
        st = stats[proj_col]
        if st.get("mean",0) > 5:
            exits = _segment_exit_rates(df, df[proj_col] >= 6)
            insights.append(build_insight(
                title="Employees carry {:.1f} projects on average".format(st["mean"]),
                problem="The average load is {:.1f} projects, ranging from {:.0f} to "
                        "{:.0f}.".format(st["mean"], st["min"], st["max"]),
                cause="This dataset records the count, not how demanding each project is.",
                evidence=_exit_sentence("Those with 6 or more projects", exits),
                action="Check how projects are assigned in the teams with the highest "
                       "loads.",
                impact="",
                severity="warning", category="workload"
            ))

    # ── Performance-Satisfaction link ──────────────────────
    if eval_col and sat_col:
        for corr in corrs:
            if (eval_col in [corr["col_a"],corr["col_b"]] and
                sat_col in [corr["col_a"],corr["col_b"]] and
                abs(corr["r"]) >= 0.3):
                _dir = "positively" if corr["r"] > 0 else "negatively"
                findings.append(
                    "Performance and satisfaction are {} associated (r={:.2f}) in "
                    "this dataset. Association is not causation — the direction of "
                    "effect can't be established from these fields alone; treat as a "
                    "relationship to investigate, not a lever proven to work.".format(
                        _dir, corr["r"]))

    # Department performance gap — only when the departments really differ.
    if dept_col and eval_col and eval_col in df.columns:
        groups = [g[eval_col].dropna() for _, g in df.groupby(dept_col)]
        groups = [g for g in groups if len(g) >= 20]
        dept_eval = df.groupby(dept_col)[eval_col].mean().sort_values()
        if len(groups) >= 2 and len(dept_eval) >= 2:
            try:
                p_kw = float(scipy_stats.kruskal(*groups).pvalue)
            except ValueError:
                p_kw = 1.0
            if p_kw < 0.05:
                findings.append(
                    "'{}' scores {:.2f} on '{}' against {:.2f} in '{}' (Kruskal-Wallis "
                    "p={:.3f}).".format(dept_eval.index[0], dept_eval.iloc[0], eval_col,
                                        dept_eval.iloc[-1], dept_eval.index[-1], p_kw))

    # Actions follow from what this dataset holds and what was found in it.
    # The fixed list recommended a promotion profile and a manager breakdown
    # for files with no promotion or manager column, and "quarterly surveys,
    # tracked monthly" to everyone.
    _cols = " ".join(c.lower() for c in df.columns)
    if attrition is not None and attrition.rate > 15:
        actions.append("Interview recent leavers to learn why they left; this data "
                       "records that they left, not why.")
    if any(i.category == "department" for i in insights):
        actions.append("Compare the department with the highest attrition against "
                       "the lowest on tenure mix, pay and role.")
    if any(i.category == "pay" for i in insights):
        actions.append("Check pay in the band with the most exits against the market "
                       "rate for the same roles.")
    if sat_col is not None and attrition is not None:
        actions.append("Track exit rates by satisfaction band at each survey, so a "
                       "change in who leaves shows early.")
    if "promot" in _cols and attrition is not None:
        actions.append("Test whether time since last promotion separates leavers from "
                       "stayers before treating it as a retention lever.")
    if any(k in _cols for k in ("manager", "supervisor")) and sat_col is not None:
        actions.append("Break satisfaction down by manager to find where scores are "
                       "lowest.")

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  ECOMMERCE INSIGHTS
# ══════════════════════════════════════════════════════════

