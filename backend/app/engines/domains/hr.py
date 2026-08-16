"""
core/engines/hr.py — HR & People Analytics domain engine.
Single responsibility: given a DataFrame, produce structured HR insights.
All metrics computed from the submitted dataset — no external benchmarks.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional

import pandas as pd
from scipy import stats as scipy_stats

from app.engines.domains.base import (AttritionAnalysis, build_insight,
                               infer_scale_bounds)
from app.services.dtypes import is_text_dtype
from app.services.stat_guards import apply_fdr, chi2_association

from app.engines.column_roles import left_mask as role_left_mask
from app.engines.domain_detect import tokenise

logger = logging.getLogger(__name__)

# Back-compat alias — scale inference now lives in base.py (shared with the
# ecommerce rating engine so both handle 1-5 / 1-10 / 0-100 scales identically).
_infer_scale_bounds = infer_scale_bounds


def _find_left_mask(df: pd.DataFrame):
    """Boolean 'employee left' mask, or None if no attrition column exists.

    One definition, in engines/column_roles. There were four, with four
    different keyword lists and two different ideas about how to read the
    column.
    """
    found = role_left_mask(df)
    return found[1] if found else None


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
    sat_col = next((c for c in df.columns
                    if "satisfaction" in tokenise(c)), None)
    n_flight = 0
    if sat_col and sat_col in df.columns:
        sv = df.loc[~left_mask, sat_col].dropna()
        thresh = float(sv.quantile(0.25)) if len(sv) > 0 else 0.4
        n_flight = int((sv < thresh).sum())

    flight_pct = round(n_flight / max(n_total - n_left, 1) * 100, 1)
    return n_flight, flight_pct, sat_col


def _run_attrition(df: pd.DataFrame) -> Optional[AttritionAnalysis]:
    found = role_left_mask(df)
    if not found:
        return None
    attr_col, left_mask = found
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
                     if {"department", "dept"} & set(tokenise(c))), None)
    sal_col  = next((c for c in df.columns
                     if "salary" in tokenise(c) and is_text_dtype(df[c])), None)

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
    cost_str   = "Replacing {:,} employees estimated at 50-150% of annual salary each".format(n_left)

    return AttritionAnalysis(
        rate=rate, n_left=n_left, n_total=n_total,
        severity=severity, top_drivers=top_drivers[:8],
        dept_attrition=dept_attrition,
        salary_attrition=salary_attrition,
        n_flight_risk=n_flight, flight_risk_pct=flight_pct,
        cost_estimate=cost_str,
        interpretation="{:.1f}% attrition ({:,} employees). {} severity. Top driver: {}.".format(
            rate, n_left, severity.upper(),
            top_drivers[0]["factor"] if top_drivers else "unknown"),
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
                cause="Rate is {:.1f}pp above the 10–15% planning band used in this "
                      "analysis (healthy rates vary by industry — verify against your "
                      "sector data). ".format(rate - 12.5) +
                      ("Primary driver: '{}'".format(attrition.top_drivers[0]["factor"])
                       if attrition.top_drivers else "Multiple factors"),
                evidence="Planning band: 10–15% (indicative). Current: {:.1f}%. "
                         "{:,} remaining employees at flight risk ({:.0f}%).".format(
                    rate, attrition.n_flight_risk, attrition.flight_risk_pct),
                action="1. Exit interviews with all {:,} leavers this week  "
                       "2. Salary benchmarking vs market  "
                       "3. Engagement survey for remaining staff  "
                       "4. Identify and retain top performers".format(attrition.n_left),
                impact=attrition.cost_estimate + ". Knowledge loss accelerates with each exit.",
                severity="critical", category="attrition"
            ))
            risks.append("CRITICAL: {:.1f}% attrition — {:,} employees left. {}".format(
                rate, attrition.n_left, attrition.cost_estimate))
        else:
            findings.append("Attrition rate {:.1f}% within the 10–15% planning band.".format(rate))

        # Dept breakdown
        if attrition.dept_attrition:
            sorted_d = sorted(attrition.dept_attrition.items(), key=lambda x:x[1], reverse=True)
            if len(sorted_d)>=2 and sorted_d[0][1] > sorted_d[-1][1]+10:
                worst_d, worst_r = sorted_d[0]
                best_d,  best_r  = sorted_d[-1]
                insights.append(build_insight(
                    title="'{}' Department: {:.0f}% Attrition vs {:.0f}% Best".format(
                        worst_d, worst_r, best_r),
                    problem="'{}' losing {:.0f}% of staff vs company average {:.1f}%".format(
                        worst_d, worst_r, rate),
                    # Naming management quality or workload as the cause
                    # asserts something no column in this dataset measures.
                    # State what is established — the gap — and what would
                    # establish the rest.
                    cause="The gap is established; its driver is not, because this "
                          "dataset holds no manager, workload or progression fields "
                          "to test against. Comparing '{}' with '{}' on tenure, pay "
                          "band and role would narrow it.".format(worst_d, best_d),
                    evidence="{:.0f}pp gap between highest ({}) and lowest ({}) attrition dept".format(
                        worst_r-best_r, worst_d, best_d),
                    action="Review '{}' against '{}' on tenure mix, pay band and role "
                           "profile to establish what differs. Where the data cannot "
                           "explain the gap, exit-interview commentary from '{}' is "
                           "the next source.".format(worst_d, best_d, worst_d),
                    impact="{:.0f}pp of the company-wide attrition rate is concentrated "
                           "in this one department.".format(worst_r - rate),
                    severity="critical" if worst_r>25 else "warning",
                    category="attrition"
                ))
                risks.append("'{}' department attrition {:.0f}% — {:.0f}pp above best performer '{}'".format(
                    worst_d, worst_r, worst_r-best_r, best_d))

        # Salary band attrition
        if attrition.salary_attrition:
            sorted_s = sorted(attrition.salary_attrition.items(), key=lambda x:x[1], reverse=True)
            if sorted_s[0][1] > 20:
                insights.append(build_insight(
                    title="'{}' Salary Band: {:.0f}% Attrition — Pay Issue".format(
                        sorted_s[0][0], sorted_s[0][1]),
                    problem="Lowest salary band losing {:.0f}% of employees".format(sorted_s[0][1]),
                    cause="Below-market compensation driving employees to better-paying companies",
                    evidence="Attrition by pay: " + " | ".join(
                        ["{}: {:.0f}%".format(k,v) for k,v in sorted_s]),
                    action="1. Market salary benchmarking immediately  "
                           "2. Targeted retention bonuses for low-pay high-performers  "
                           "3. Review compensation bands",
                    impact="Pay-driven attrition is fastest to fix but most expensive if ignored",
                    severity="critical", category="attrition"
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

        if pct < 55:
            insights.append(build_insight(
                title="Satisfaction averages {:.0f}% of scale — {:,} employees in the bottom band".format(pct, low_n),
                problem="{:.0f}% satisfaction (scale-normalised). {:,} employees ({:.0f}%) sit in the bottom band of the scale".format(
                    pct, low_n, low_pct),
                # The dataset records the score, not why it is low. Naming
                # culture, workload, recognition or pay as the cause would be
                # asserting something this data cannot show — the exact kind
                # of unfounded claim that gets a report dismissed.
                cause="Not identifiable from this dataset, which records the score "
                      "but not its drivers. Isolating it requires either free-text "
                      "exit/survey commentary or a satisfaction breakdown by "
                      "sub-factor (manager, workload, pay, progression).",
                evidence=("Mean={:.2f} on a {:.0f}-{:.0f} scale ({:.0f}% of range). "
                          "{:,} employees ({:.0f}%) score below {:.2f} (bottom 40% of the scale). "
                          "Internal planning target: 70% of range. ".format(
                              mean_s, min_s, max_s, pct, low_n, low_pct, low_thresh)
                          + _segment_attrition_evidence(
                              df, df[sat_col] < low_thresh,
                              "critically dissatisfied employees")),
                action="1. Anonymous pulse survey — identify top 3 pain points (48 hrs)  "
                       "2. Quick wins: flexible hours, recognition program, manager training  "
                       "3. Publish action plan within 30 days",
                impact="The dissatisfied segment is the highest-priority retention risk — "
                       "its measured exit rate above quantifies the exposure for this dataset.",
                severity="critical", category="satisfaction"
            ))
            risks.append("Satisfaction {:.0f}% — {:,} employees critically disengaged".format(pct, low_n))
        elif pct < 70:
            insights.append(build_insight(
                title="Satisfaction Below Target: {:.0f}% (Target: 70%+)".format(pct),
                problem="{:.0f}% satisfaction with {:,} employees ({:.0f}%) in the bottom-box range".format(pct, low_n, low_pct),
                cause="Specific fixable issues rather than systemic breakdown",
                evidence=("Mean={:.2f} on a {:.0f}-{:.0f} scale ({:.0f}% of range). "
                          "{:,} employees ({:.0f}%) score below {:.2f} (bottom 40% of the scale). "
                          "Internal planning target: 70% of range. ".format(
                              mean_s, min_s, max_s, pct, low_n, low_pct, low_thresh)
                          + _segment_attrition_evidence(
                              df, df[sat_col] < low_thresh, "low-satisfaction employees")),
                action="1. Focus groups to identify top 3 issues  "
                       "2. Manager communication training  "
                       "3. Career development conversations",
                impact="Closing the gap to the 70% target directly shrinks the "
                       "low-satisfaction segment measured above — the group with the "
                       "highest exit rate in this dataset.",
                severity="warning", category="satisfaction"
            ))
        else:
            insights.append(build_insight(
                title="Strong Satisfaction: {:.0f}% — Above 70% Planning Target".format(pct),
                problem="N/A — satisfaction is healthy",
                cause="Effective HR practices and management culture",
                evidence="Mean={:.2f} on a {:.0f}-{:.0f} scale ({:.0f}% of range). Above the 70%-of-range internal planning target.".format(mean_s, min_s, max_s, pct),
                action="Maintain programs. Create career paths for high-performers to prevent exits.",
                impact="Strong satisfaction = lower attrition + higher productivity",
                severity="positive", category="satisfaction"
            ))
            opps.append("Satisfaction {:.0f}% is a competitive advantage — use in recruitment marketing".format(pct))

    # ── Overwork ───────────────────────────────────────────
    if hrs_col and hrs_col in stats:
        st       = stats[hrs_col]
        mean_hrs = st.get("mean",0)
        high_n   = int((df[hrs_col].dropna()>260).sum()) if hrs_col in df.columns else 0
        if mean_hrs > 220:
            insights.append(build_insight(
                title="Overwork Alert: Avg {:.0f} hrs/Month — {:,} Employees at Burnout Risk".format(
                    mean_hrs, high_n),
                problem="Average {:.0f} monthly hours. {:,} employees working 260+ hours (critical zone)".format(
                    mean_hrs, high_n),
                cause="Understaffing, poor task distribution, or culture of overwork",
                evidence=("Mean={:.0f} hrs. Reference range: 160–200. Overwork zone: 240+. "
                          "{:,} employees in critical zone. ".format(mean_hrs, high_n)
                          + _segment_attrition_evidence(
                              df, df[hrs_col] > 260, "employees working 260+ hrs/month")),
                action="1. Workload audit by team/department  "
                       "2. Hiring plan for overloaded teams  "
                       "3. No-overtime policy for 260+ hour employees",
                impact="The measured exit rate of the 260+ hour segment (above) is the "
                       "direct, quantified burnout cost visible in this dataset.",
                severity="warning" if mean_hrs<240 else "critical",
                category="workload"
            ))
            risks.append("Avg {:.0f} hrs/month — overwork driving burnout and attrition".format(mean_hrs))

    # ── Projects ───────────────────────────────────────────
    if proj_col and proj_col in stats:
        st = stats[proj_col]
        if st.get("mean",0) > 5:
            insights.append(build_insight(
                title="High Project Load: Avg {:.1f} Projects Per Employee".format(st["mean"]),
                problem="Employees handling average {:.1f} projects simultaneously".format(st["mean"]),
                cause="Resource allocation issues or insufficient headcount for demand",
                evidence=("Mean={:.1f} projects. Range: {:.0f}-{:.0f}. ".format(
                              st["mean"], st["min"], st["max"])
                          + _segment_attrition_evidence(
                              df, df[proj_col] >= 6, "employees with 6+ projects")),
                action="1. Review project assignment process  "
                       "2. Cap individual project loads at 4-5  "
                       "3. Prioritize projects by strategic value",
                impact="Excessive project loads reduce output quality and increase error rates",
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

    # Department performance gap
    if dept_col and eval_col and eval_col in df.columns:
        dept_eval = df.groupby(dept_col)[eval_col].mean().sort_values()
        if len(dept_eval)>=2:
            gap = dept_eval.iloc[-1]-dept_eval.iloc[0]
            if gap > 0.1:
                findings.append(
                    "Performance gap: '{}' scores {:.2f} vs '{}' at {:.2f} — "
                    "{:.0f}% difference. Share best practices across departments.".format(
                        dept_eval.index[0], dept_eval.iloc[0],
                        dept_eval.index[-1], dept_eval.iloc[-1], gap/dept_eval.iloc[0]*100))

    # FIX-011b: Column-gated HR actions
    # Only recommend actions for columns that exist in this dataset
    _hr_cols = [c.lower() for c in df.columns]

    # Always valid — applies to any HR dataset
    actions.append(
        "Quarterly satisfaction pulse surveys — track trend monthly not annually"
    )

    # Only if attrition column exists
    _atr_present = any(k in _hr_cols for k in ["left", "attrition", "churned", "resigned"])
    if _atr_present:
        actions.append(
            "Identify flight-risk profile: low satisfaction + high tenure + "
            "no recent promotion — target this segment for retention conversations first"
        )

    # Only if salary column exists
    _sal_present = any(k in _hr_cols for k in ["salary", "salary_band", "compensation"])
    if _sal_present:
        actions.append(
            "Benchmark salaries against market within 30 days, then test whether "
            "below-market pay tracks with exits in your own data before acting"
        )

    # Only if promotion column exists
    _promo_present = any(k in _hr_cols for k in ["promotion", "promoted", "promotion_last_5years"])
    if _promo_present:
        actions.append(
            "Review career-development paths — check whether time-since-promotion "
            "correlates with attrition in this dataset before prioritising it"
        )

    # Only if satisfaction or evaluation column exists
    _sat_present = any(k in _hr_cols for k in ["satisfaction", "satisfaction_level", "engagement"])
    if _sat_present:
        actions.append(
            "Break satisfaction down by manager/team to locate the weakest areas, "
            "then target manager coaching where the data shows the largest gaps"
        )

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  ECOMMERCE INSIGHTS
# ══════════════════════════════════════════════════════════

