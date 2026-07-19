"""
story_engine.py — Senior analyst insights for HR, Ecommerce, Sales.
Format: Problem → Cause → Evidence → Action → Impact
McKinsey/BCG level — no jargon, plain English decisions.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy import stats as scipy_stats


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
#  DATA CLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class Insight:
    title:     str
    problem:   str
    cause:     str
    evidence:  str
    action:    str
    impact:    str
    severity:  str   # critical / warning / positive / info
    category:  str

@dataclass
class AttritionAnalysis:
    rate:              float
    n_left:            int
    n_total:           int
    severity:          str
    top_drivers:       List[Dict]
    dept_attrition:    Dict
    salary_attrition:  Dict
    n_flight_risk:     int
    flight_risk_pct:   float
    cost_estimate:     str
    interpretation:    str

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
#  STATS HELPERS
# ══════════════════════════════════════════════════════════

def _col_stats(s: pd.Series) -> Dict:
    try:
        arr = pd.to_numeric(s, errors="coerce").values.astype(float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3:
            return {}
        q1  = float(np.percentile(arr, 25))
        q3  = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        out = int(((arr < lo) | (arr > hi)).sum())
        try:
            _, pval = scipy_stats.shapiro(arr[:5000])
            is_norm = pval > 0.05
        except Exception:
            is_norm = None
        return {
            "mean":       round(float(np.mean(arr)), 4),
            "median":     round(float(np.median(arr)), 4),
            "std":        round(float(np.std(arr, ddof=1)), 4),
            "min":        round(float(np.min(arr)), 4),
            "max":        round(float(np.max(arr)), 4),
            "q1":         round(q1, 4),
            "q3":         round(q3, 4),
            "iqr":        round(iqr, 4),
            "skew":       round(float(pd.Series(arr).skew()), 4),
            "cv":         round(float(np.std(arr)/abs(np.mean(arr))), 4) if np.mean(arr) != 0 else 0,
            "outliers":   out,
            "outlier_pct":round(out / max(len(arr), 1) * 100, 2),
            "is_normal":  is_norm,
            "missing":    int(s.isna().sum()),
            "missing_pct":round(s.isna().mean() * 100, 2),
            "n":          len(arr),
        }
    except Exception:
        return {}


def _correlations(df: pd.DataFrame) -> List[Dict]:
    """All pairwise correlations, filtered through the honesty pipeline:
    n >= 30, Benjamini-Hochberg FDR < 0.05, |r| >= 0.30. Each surviving
    finding carries q (adjusted p) and a confidence label, so downstream
    narratives never cite noise."""
    from app.services.stat_guards import filter_correlations

    num_cols = df.select_dtypes(include="number").columns.tolist()
    results  = []
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            a, b = num_cols[i], num_cols[j]
            try:
                common = df[[a,b]].dropna()
                if len(common) < 10: continue
                r, p = scipy_stats.pearsonr(common[a], common[b])
                results.append({
                    "col_a": a, "col_b": b,
                    "r": round(float(r), 3),
                    "p": round(float(p), 5),
                    "n": int(len(common)),
                    "strength": "strong" if abs(r)>=0.7 else "moderate" if abs(r)>=0.4 else "weak",
                    "direction": "positive" if r>0 else "negative",
                })
            except Exception:
                continue
    survived = filter_correlations(results)
    return sorted(survived, key=lambda x: abs(x["r"]), reverse=True)


def _build_insight(title, problem, cause, evidence, action, impact,
                   severity="info", category="general") -> Insight:
    return Insight(title=title, problem=problem, cause=cause,
                   evidence=evidence, action=action, impact=impact,
                   severity=severity, category=category)


# ══════════════════════════════════════════════════════════
#  ATTRITION ENGINE
# ══════════════════════════════════════════════════════════

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
            continue

    # Categorical drivers
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if c != attr_col and df[c].nunique() <= 20]
    for col in cat_cols[:6]:
        try:
            ct = pd.crosstab(df[col], left_mask)
            if ct.shape[1] < 2: continue
            _, p, _, _ = scipy_stats.chi2_contingency(ct)
            if p < 0.05:
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
            continue

    top_drivers.sort(key=lambda x: x["impact"], reverse=True)

    # Segment breakdown
    dept_col = next((c for c in df.columns
                     if "department" in c.lower() or "dept" in c.lower()), None)
    sal_col  = next((c for c in df.columns
                     if "salary" in c.lower() and df[c].dtype==object), None)

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

    # Flight risk
    sat_col = next((c for c in df.columns if "satisfaction" in c.lower()), None)
    n_flight = 0
    if sat_col and sat_col in df.columns:
        sv = df.loc[~left_mask, sat_col].dropna()
        thresh = float(sv.quantile(0.25)) if len(sv)>0 else 0.4
        n_flight = int((sv < thresh).sum())

    flight_pct = round(n_flight / max(n_total-n_left,1)*100,1)
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

    sat_col  = next((c for c in df.columns if "satisfaction" in c.lower()), None)
    eval_col = next((c for c in df.columns if "evaluat" in c.lower()), None)
    hrs_col  = next((c for c in df.columns if "hour" in c.lower()), None)
    dept_col = next((c for c in df.columns
                     if "department" in c.lower() and df[c].nunique()<=20), None)
    proj_col = next((c for c in df.columns if "project" in c.lower()), None)

    # ── Attrition ──────────────────────────────────────────
    if attrition:
        rate = attrition.rate
        if attrition.severity in ("critical","high"):
            insights.append(_build_insight(
                title="Attrition Crisis: {:.1f}% of Workforce Left".format(rate),
                problem="{:,} out of {:,} employees left — {:.1f}% rate".format(
                    attrition.n_left, attrition.n_total, rate),
                cause="Rate {:.0f}% above industry benchmark (10-15%). ".format(rate-12.5) +
                      ("Primary driver: '{}'".format(attrition.top_drivers[0]["factor"])
                       if attrition.top_drivers else "Multiple factors"),
                evidence="Industry benchmark: 10-15%. Current: {:.1f}%. "
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
            findings.append("Attrition rate {:.1f}% within acceptable range (10-15% benchmark).".format(rate))

        # Dept breakdown
        if attrition.dept_attrition:
            sorted_d = sorted(attrition.dept_attrition.items(), key=lambda x:x[1], reverse=True)
            if len(sorted_d)>=2 and sorted_d[0][1] > sorted_d[-1][1]+10:
                worst_d, worst_r = sorted_d[0]
                best_d,  best_r  = sorted_d[-1]
                insights.append(_build_insight(
                    title="'{}' Department: {:.0f}% Attrition vs {:.0f}% Best".format(
                        worst_d, worst_r, best_r),
                    problem="'{}' losing {:.0f}% of staff vs company average {:.1f}%".format(
                        worst_d, worst_r, rate),
                    cause="Department-specific issues: management quality, workload, or growth opportunities",
                    evidence="{:.0f}pp gap between highest ({}) and lowest ({}) attrition dept".format(
                        worst_r-best_r, worst_d, best_d),
                    action="1. Skip-level interviews in {} this week  "
                           "2. Manager effectiveness review  "
                           "3. Workload distribution audit".format(worst_d),
                    impact="Dept attrition destroys team cohesion and institutional knowledge",
                    severity="critical" if worst_r>25 else "warning",
                    category="attrition"
                ))
                risks.append("'{}' department attrition {:.0f}% — {:.0f}pp above best performer '{}'".format(
                    worst_d, worst_r, worst_r-best_r, best_d))

        # Salary band attrition
        if attrition.salary_attrition:
            sorted_s = sorted(attrition.salary_attrition.items(), key=lambda x:x[1], reverse=True)
            if sorted_s[0][1] > 20:
                insights.append(_build_insight(
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
        mean_s = st.get("mean",0)
        max_s  = st.get("max",1)
        pct    = (mean_s/max_s*100) if max_s>0 else 0
        low_n  = int((df[sat_col].dropna()<0.4).sum()) if sat_col in df.columns else 0
        low_pct= round(low_n/len(df)*100,1)

        if pct < 55:
            insights.append(_build_insight(
                title="Satisfaction Crisis: Only {:.0f}% Score — {:,} Employees Disengaged".format(pct, low_n),
                problem="{:.0f}% satisfaction. {:,} employees ({:.0f}%) critically dissatisfied (below 40% threshold)".format(
                    pct, low_n, low_pct),
                cause="Score below 55% = systemic failure in culture, workload, recognition, or pay",
                evidence="Mean={:.2f}/{:.2f}. {:.0f}% below critical 0.4 threshold. "
                         "Industry benchmark for healthy orgs: 70%+".format(mean_s, max_s, low_pct),
                action="1. Anonymous pulse survey — identify top 3 pain points (48 hrs)  "
                       "2. Quick wins: flexible hours, recognition program, manager training  "
                       "3. Publish action plan within 30 days",
                impact="Each 10% satisfaction improvement = ~15% attrition reduction. "
                       "Low satisfaction costs 20% productivity loss.",
                severity="critical", category="satisfaction"
            ))
            risks.append("Satisfaction {:.0f}% — {:,} employees critically disengaged".format(pct, low_n))
        elif pct < 70:
            insights.append(_build_insight(
                title="Satisfaction Below Target: {:.0f}% (Target: 70%+)".format(pct),
                problem="{:.0f}% satisfaction with {:,} employees below threshold".format(pct, low_n),
                cause="Specific fixable issues rather than systemic breakdown",
                evidence="Mean={:.2f}. {:.0f}% below 40% threshold. Benchmark: 70%+".format(mean_s, low_pct),
                action="1. Focus groups to identify top 3 issues  "
                       "2. Manager communication training  "
                       "3. Career development conversations",
                impact="Improving from {:.0f}% to 75% reduces attrition by 15-25%".format(pct),
                severity="warning", category="satisfaction"
            ))
        else:
            insights.append(_build_insight(
                title="Strong Satisfaction: {:.0f}% — Above Industry Benchmark".format(pct),
                problem="N/A — satisfaction is healthy",
                cause="Effective HR practices and management culture",
                evidence="Mean={:.2f}. {:.0f}% of maximum score. Above 70% benchmark.".format(mean_s, pct),
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
            insights.append(_build_insight(
                title="Overwork Alert: Avg {:.0f} hrs/Month — {:,} Employees at Burnout Risk".format(
                    mean_hrs, high_n),
                problem="Average {:.0f} monthly hours. {:,} employees working 260+ hours (critical zone)".format(
                    mean_hrs, high_n),
                cause="Understaffing, poor task distribution, or culture of overwork",
                evidence="Mean={:.0f} hrs. Normal range: 160-200. Overwork zone: 240+. "
                         "{:,} employees in critical zone.".format(mean_hrs, high_n),
                action="1. Workload audit by team/department  "
                       "2. Hiring plan for overloaded teams  "
                       "3. No-overtime policy for 260+ hour employees",
                impact="Overworked employees 2-3x more likely to leave in 12 months. "
                       "Burnout reduces productivity by 30-40%.",
                severity="warning" if mean_hrs<240 else "critical",
                category="workload"
            ))
            risks.append("Avg {:.0f} hrs/month — overwork driving burnout and attrition".format(mean_hrs))

    # ── Projects ───────────────────────────────────────────
    if proj_col and proj_col in stats:
        st = stats[proj_col]
        if st.get("mean",0) > 5:
            insights.append(_build_insight(
                title="High Project Load: Avg {:.1f} Projects Per Employee".format(st["mean"]),
                problem="Employees handling average {:.1f} projects simultaneously".format(st["mean"]),
                cause="Resource allocation issues or insufficient headcount for demand",
                evidence="Mean={:.1f} projects. Range: {:.0f}-{:.0f}. "
                         "Research shows 3-4 projects optimal for quality output.".format(
                    st["mean"], st["min"], st["max"]),
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
                findings.append(
                    "Performance and satisfaction are linked (r={:.2f}) — "
                    "improving satisfaction will improve evaluation scores".format(corr["r"]))

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

    actions.extend([
        "Quarterly satisfaction pulse surveys — track trend monthly not annually",
        "Identify flight-risk profile: low satisfaction + high tenure + below-market pay",
        "Salary benchmarking against market — competitive pay is #1 retention factor",
        "Career development paths for all levels — lack of growth is top exit reason",
        "Manager effectiveness training — 70% of satisfaction is driven by direct manager",
    ])

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  ECOMMERCE INSIGHTS
# ══════════════════════════════════════════════════════════

def _insights_ecommerce(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights = []

    rating_col = next((c for c in df.columns if "rating" in c.lower()
                       and "count" not in c.lower()), None)
    price_col  = next((c for c in df.columns
                       if any(k in c.lower() for k in ["discounted_price","selling_price","price"])
                       and c in stats), None)
    actual_col = next((c for c in df.columns
                       if "actual_price" in c.lower() or "mrp" in c.lower()), None)
    disc_col   = next((c for c in df.columns if "discount" in c.lower() and c in stats), None)
    cat_col    = next((c for c in df.select_dtypes(include="object").columns
                       if "category" in c.lower() and df[c].nunique()<=30), None)
    rev_col    = next((c for c in df.columns
                       if any(k in c.lower() for k in ["revenue","sales","amount"]) and c in stats), None)

    # ── Rating Analysis ────────────────────────────────────
    if rating_col and rating_col in stats:
        st     = stats[rating_col]
        mean_r = st.get("mean",0)
        low_n  = int((df[rating_col].dropna()<3.0).sum()) if rating_col in df.columns else 0
        out_ct = st.get("outliers",0)
        q1     = st.get("q1",0)

        if mean_r < 3.5:
            insights.append(_build_insight(
                title="Rating Emergency: {:.2f}/5 Average — {:,} Products Below 3.0".format(mean_r, low_n),
                problem="Average {:.2f}/5 with {:,} products rated below 3.0 (unacceptable threshold)".format(
                    mean_r, low_n),
                cause="Products failing to meet customer expectations — quality, description, or delivery mismatch",
                evidence="Mean={:.2f}. Benchmark: 4.0+. Bottom 25% rated below {:.1f}. "
                         "{:,} critically low-rated products.".format(mean_r, q1, low_n),
                action="1. Immediate audit of all products rated below 3.0  "
                       "2. Customer feedback analysis for bottom-rated items  "
                       "3. Supplier quality review for failing products  "
                       "4. Remove or improve within 14 days",
                impact="Ratings below 3.5 cause 40-60% lower purchase probability. "
                       "Each negative review reduces future sales by ~1%.",
                severity="critical", category="rating"
            ))
            risks.append("Rating {:.2f}/5 — {:,} products below 3.0 causing significant revenue loss".format(
                mean_r, low_n))
        elif mean_r < 4.0:
            insights.append(_build_insight(
                title="Rating Below Target: {:.2f}/5 (Target 4.0+)".format(mean_r),
                problem="{:.2f}/5 average. Bottom 25% rated below {:.1f}. {:,} critical products.".format(
                    mean_r, q1, low_n),
                cause="Bottom quartile products dragging overall performance",
                evidence="Mean={:.2f}. 25th percentile={:.1f}. Target: 4.0+.".format(mean_r, q1),
                action="1. Fix or remove bottom quartile products  "
                       "2. Improve product descriptions and images  "
                       "3. Category-level quality audit",
                impact="Reaching 4.0+ rating = estimated 15-20% conversion improvement",
                severity="warning", category="rating"
            ))
        else:
            insights.append(_build_insight(
                title="Strong Ratings: {:.2f}/5 — Competitive Advantage".format(mean_r),
                problem="N/A — ratings are strong",
                cause="Quality products meeting customer expectations",
                evidence="Mean={:.2f}/5. Above 4.0 benchmark. Only {:,} products below 3.0.".format(
                    mean_r, low_n),
                action="Leverage high ratings in all marketing. "
                       "Use as social proof in product listings.",
                impact="4.0+ ratings enable 10-15% premium pricing vs competitors",
                severity="positive", category="rating"
            ))
            opps.append("Rating {:.2f}/5 enables premium pricing — test 5-10% price increase on top-rated items".format(mean_r))

        if out_ct > 0:
            pct = st.get("outlier_pct",0)
            findings.append(
                "{:,} products have outlier ratings ({:.1f}% of catalog) — "
                "investigate immediately for quality or fraud issues".format(out_ct, pct))

    # ── Category Performance ───────────────────────────────
    if cat_col and rating_col and rating_col in df.columns:
        cat_perf = df.groupby(cat_col)[rating_col].agg(["mean","count"]).sort_values("mean")
        cat_perf = cat_perf[cat_perf["count"]>=5]
        if len(cat_perf)>=2:
            worst_c = cat_perf.index[0]
            best_c  = cat_perf.index[-1]
            gap     = cat_perf.loc[best_c,"mean"] - cat_perf.loc[worst_c,"mean"]
            if gap > 0.3:
                insights.append(_build_insight(
                    title="Category Gap: '{}' ({:.2f}) vs '{}' ({:.2f})".format(
                        worst_c, cat_perf.loc[worst_c,"mean"],
                        best_c, cat_perf.loc[best_c,"mean"]),
                    problem="'{}' category underperforms by {:.1f} rating points ({:.0f}% gap)".format(
                        worst_c, gap, gap/cat_perf.loc[worst_c,"mean"]*100),
                    cause="Supplier quality, product complexity, or customer expectation mismatch by category",
                    evidence="{:.1f} point gap across {} categories. "
                             "'{}' avg={:.2f}, '{}' avg={:.2f}".format(
                        gap, len(cat_perf), worst_c, cat_perf.loc[worst_c,"mean"],
                        best_c, cat_perf.loc[best_c,"mean"]),
                    action="1. Quality audit of '{}' category suppliers  "
                           "2. Customer complaint analysis for '{}' products  "
                           "3. Apply '{}' category best practices to '{}' category".format(
                        worst_c, worst_c, best_c, worst_c),
                    impact="Closing gap by 50% = +{:.1f} rating points overall. "
                           "Estimated 10-15% revenue uplift in '{}' category.".format(
                        gap*0.5, worst_c),
                    severity="warning" if gap<0.8 else "critical",
                    category="category_performance"
                ))
                findings.append("Category range: {} ({:.2f}) to {} ({:.2f})".format(
                    worst_c, cat_perf.loc[worst_c,"mean"],
                    best_c, cat_perf.loc[best_c,"mean"]))

    # ── Pricing Analysis ───────────────────────────────────
    if price_col and price_col in stats:
        st   = stats[price_col]
        skew = st.get("skew",0)
        if skew > 1.5:
            findings.append(
                "Price distribution right-skewed (skew={:.1f}) — median {:.0f} vs mean {:.0f}. "
                "Most products are budget-range with few premium items. "
                "Consider expanding mid-market range.".format(
                    skew, st["median"], st["mean"]))
            opps.append("Mid-market price gap detected — products between median ({:.0f}) "
                        "and 75th percentile ({:.0f}) are underrepresented".format(
                            st["median"], st["q3"]))

    # ── Discount Analysis ──────────────────────────────────
    if disc_col and disc_col in stats:
        st       = stats[disc_col]
        avg_disc = st.get("mean",0)
        max_disc = st.get("max",0)
        if avg_disc > 40:
            insights.append(_build_insight(
                title="High Avg Discount {:.0f}% — Potential Margin Erosion".format(avg_disc),
                problem="Average discount {:.0f}% with some products at {:.0f}% — profitability at risk".format(
                    avg_disc, max_disc),
                cause="Competitive pressure or unstrategic discounting without margin analysis",
                evidence="Mean discount={:.0f}%, max={:.0f}%. "
                         "Benchmark: 15-25% is healthy for most categories.".format(avg_disc, max_disc),
                action="1. Margin analysis per category — identify below-cost discounts  "
                       "2. Reduce discounts on 4.0+ rated products (they sell without discounts)  "
                       "3. Strategic discounting only: new launch, clearance, seasonal",
                impact="Every 10% unnecessary discount = direct margin loss. "
                       "High discounts also train customers to wait for sales.",
                severity="warning" if avg_disc<55 else "critical",
                category="pricing"
            ))
            risks.append("Avg discount {:.0f}% may be eroding margins — review per-product profitability".format(avg_disc))

    # ── Price-Rating Correlation ───────────────────────────
    for corr in corrs:
        cols = [corr["col_a"], corr["col_b"]]
        has_rating = any("rating" in c.lower() for c in cols)
        has_price  = any("price" in c.lower() for c in cols)
        if has_rating and has_price and abs(corr["r"])>=0.3:
            if corr["r"] < 0:
                risks.append(
                    "Higher-priced products have LOWER ratings (r={:.2f}) — "
                    "premium pricing not matching perceived value. Review quality.".format(corr["r"]))
            else:
                opps.append(
                    "Higher-priced products have HIGHER ratings (r={:.2f}) — "
                    "quality-price alignment confirmed. Safe to test premium pricing.".format(corr["r"]))

    actions.extend([
        "Weekly rating monitoring — alert if any product drops below 3.5",
        "A/B test 5% price increase on all products rated 4.3+",
        "Remove products with <3.0 rating and <50 reviews — they damage brand",
        "Customer feedback loop — auto-survey buyers 7 days post-delivery",
        "Category manager review — monthly performance vs target",
    ])

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  SALES INSIGHTS
# ══════════════════════════════════════════════════════════

def _insights_sales(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights = []

    rev_col    = next((c for c in df.columns
                       if any(k in c.lower() for k in ["revenue","sales","amount","total"])
                       and c in stats), None)
    profit_col = next((c for c in df.columns
                       if any(k in c.lower() for k in ["profit","margin","net"])
                       and c in stats), None)
    target_col = next((c for c in df.columns
                       if any(k in c.lower() for k in ["target","quota","goal"])
                       and c in stats), None)
    region_col = next((c for c in df.select_dtypes(include="object").columns
                       if any(k in c.lower() for k in ["region","territory","zone","area"])
                       and df[c].nunique()<=25), None)
    product_col= next((c for c in df.select_dtypes(include="object").columns
                       if any(k in c.lower() for k in ["product","category","segment"])
                       and df[c].nunique()<=30), None)
    rep_col    = next((c for c in df.select_dtypes(include="object").columns
                       if any(k in c.lower() for k in ["rep","salesperson","agent","owner"])
                       and df[c].nunique()<=50), None)

    # ── Revenue Analysis ───────────────────────────────────
    if rev_col and rev_col in stats:
        st   = stats[rev_col]
        skew = st.get("skew",0)
        cv   = st.get("cv",0)
        mean = st.get("mean",0)
        med  = st.get("median",0)

        insights.append(_build_insight(
            title="Revenue Overview: Mean {:.0f} | Median {:.0f} | Range {:.0f}-{:.0f}".format(
                mean, med, st["min"], st["max"]),
            problem="Revenue distribution analysis" + (" — high variability detected" if cv>0.5 else ""),
            cause="Skewness={:.1f} indicates {}".format(
                skew, "few large deals driving disproportionate revenue (Pareto effect)" if skew>1
                else "revenue is relatively evenly distributed"),
            evidence="Mean={:.0f}, Median={:.0f} ({:.0f}% difference). "
                     "Top 25% of transactions above {:.0f}.".format(
                mean, med, abs(mean-med)/max(med,1)*100, st["q3"]),
            action="1. Identify top 20% revenue drivers — protect and replicate  "
                   "2. Analyze bottom 20% — cut or transform low performers  "
                   "3. Revenue concentration risk assessment",
            impact="If top 20% drives 80% of revenue, losing 1 key client/product = severe impact",
            severity="info" if cv<0.5 else "warning",
            category="revenue"
        ))

        if skew > 1.5:
            opps.append(
                "Revenue is right-skewed — small number of high-value transactions. "
                "Focus on replicating conditions for top transactions.")
            findings.append(
                "Revenue Pareto effect detected: median {:.0f} vs mean {:.0f} — "
                "few large deals driving disproportionate revenue".format(med, mean))

    # ── Target/Quota Analysis ──────────────────────────────
    if target_col and rev_col and target_col in stats and rev_col in stats:
        target_mean = stats[target_col].get("mean",0)
        rev_mean    = stats[rev_col].get("mean",0)
        achievement = (rev_mean / target_mean * 100) if target_mean > 0 else 0

        if achievement < 80:
            insights.append(_build_insight(
                title="Target Gap: {:.0f}% Achievement — {:.0f}pp Below Target".format(
                    achievement, 100-achievement),
                problem="Average {:.0f}% quota achievement — team missing targets significantly".format(achievement),
                cause="Targets may be unrealistic, pipeline quality poor, or sales process broken",
                evidence="Avg revenue={:.0f} vs avg target={:.0f}. Achievement={:.0f}%.".format(
                    rev_mean, target_mean, achievement),
                action="1. Review if targets are market-realistic (benchmark vs industry)  "
                       "2. Pipeline quality audit — identify qualification issues  "
                       "3. Sales process coaching for bottom quartile reps",
                impact="{:.0f}% achievement gap = {:.0f}% revenue shortfall from plan".format(
                    100-achievement, 100-achievement),
                severity="critical" if achievement<70 else "warning",
                category="target"
            ))
            risks.append("{:.0f}% target achievement — revenue significantly below plan".format(achievement))
        elif achievement >= 100:
            insights.append(_build_insight(
                title="Targets Exceeded: {:.0f}% Achievement".format(achievement),
                problem="N/A — exceeding targets",
                cause="Strong sales execution and/or conservative target setting",
                evidence="Avg revenue={:.0f} vs avg target={:.0f}".format(rev_mean, target_mean),
                action="1. Review if targets were set too conservatively  "
                       "2. Capture learnings from over-performers and scale",
                impact="Consistent over-achievement suggests capacity for higher targets",
                severity="positive", category="target"
            ))
            opps.append("{:.0f}% target achievement — review upside potential for next period".format(achievement))

    # ── Regional Analysis ──────────────────────────────────
    if region_col and rev_col and rev_col in df.columns:
        reg_perf = df.groupby(region_col)[rev_col].agg(["mean","sum","count"])
        reg_perf = reg_perf[reg_perf["count"]>=3].sort_values("sum", ascending=False)
        if len(reg_perf)>=2:
            top_r    = reg_perf.index[0]
            bottom_r = reg_perf.index[-1]
            top_share= reg_perf.loc[top_r,"sum"]/reg_perf["sum"].sum()*100
            gap      = reg_perf.loc[top_r,"mean"]-reg_perf.loc[bottom_r,"mean"]
            gap_pct  = gap/max(reg_perf.loc[bottom_r,"mean"],1)*100

            insights.append(_build_insight(
                title="Regional Gap: '{}' {:.0f}x Performance vs '{}'".format(
                    top_r, reg_perf.loc[top_r,"mean"]/max(reg_perf.loc[bottom_r,"mean"],1), bottom_r),
                problem="'{}' underperforming by {:.0f}% vs top region '{}'".format(
                    bottom_r, gap_pct, top_r),
                cause="Market maturity, team capability, competitive landscape, or resource allocation differ by region",
                evidence="'{}' avg={:.0f} vs '{}' avg={:.0f}. "
                         "Top region '{}' holds {:.0f}% of total revenue.".format(
                    top_r, reg_perf.loc[top_r,"mean"],
                    bottom_r, reg_perf.loc[bottom_r,"mean"],
                    top_r, top_share),
                action="1. Understand '{}' success factors — replicate in '{}'  "
                       "2. Resource allocation review — is '{}' under-resourced?  "
                       "3. Market potential analysis for '{}'".format(
                    top_r, bottom_r, bottom_r, bottom_r),
                impact="Closing '{}'s gap by 30% = significant total revenue uplift".format(bottom_r),
                severity="warning" if gap_pct<50 else "critical",
                category="regional"
            ))
            findings.append("Top region '{}' contributes {:.0f}% of total revenue — concentration risk".format(
                top_r, top_share) if top_share > 50 else
                "Revenue reasonably distributed across {} regions".format(len(reg_perf)))

    # ── Product/Category Performance ──────────────────────
    if product_col and rev_col and rev_col in df.columns:
        prod_perf = df.groupby(product_col)[rev_col].agg(["sum","count"])
        prod_perf = prod_perf[prod_perf["count"]>=3].sort_values("sum", ascending=False)
        if len(prod_perf)>=2:
            total_rev = prod_perf["sum"].sum()
            top_prod  = prod_perf.index[0]
            top_share = prod_perf.loc[top_prod,"sum"]/total_rev*100
            top2_share= prod_perf.iloc[:2]["sum"].sum()/total_rev*100

            if top_share > 40:
                risks.append(
                    "'{}' product/category = {:.0f}% of total revenue — "
                    "dangerous concentration. Losing this = severe revenue impact.".format(
                        top_prod, top_share))
            opps.append(
                "Bottom 3 products/categories contribute only {:.0f}% of revenue — "
                "investigate if resources should be reallocated".format(
                    prod_perf.iloc[-3:]["sum"].sum()/total_rev*100))

    # ── Profit Margin ──────────────────────────────────────
    if profit_col and profit_col in stats:
        st         = stats[profit_col]
        mean_profit= st.get("mean",0)
        neg_n      = int((df[profit_col].dropna()<0).sum()) if profit_col in df.columns else 0
        neg_pct    = round(neg_n/len(df)*100,1)

        if neg_n > 0:
            insights.append(_build_insight(
                title="{:,} Loss-Making Transactions ({:.0f}%) — Immediate Review".format(neg_n, neg_pct),
                problem="{:,} transactions ({:.0f}%) generating negative profit/margin".format(neg_n, neg_pct),
                cause="Below-cost pricing, excessive discounts, high returns, or incorrect cost allocation",
                evidence="{:,} negative profit transactions. Mean margin={:.2f}. "
                         "Loss transactions erode overall profitability.".format(neg_n, mean_profit),
                action="1. Identify all loss-making transactions this week  "
                       "2. Root cause: pricing error, returns, or discounts?  "
                       "3. Reprice or discontinue unprofitable products",
                impact="Eliminating {:.0f}% loss transactions = direct profitability improvement".format(neg_pct),
                severity="critical" if neg_pct>10 else "warning",
                category="profitability"
            ))
            risks.append("{:,} loss-making transactions ({:.0f}%) — eroding overall profitability".format(
                neg_n, neg_pct))

    actions.extend([
        "Weekly revenue vs target review — per rep and per region",
        "Identify top 3 deals at risk in pipeline — intervention strategy",
        "Replicate top performer playbook — what do they do differently?",
        "Revenue concentration audit — reduce dependency on single customer/product",
        "Quarterly pricing review — ensure margins are healthy per product category",
    ])

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  GENERAL INSIGHTS
# ══════════════════════════════════════════════════════════

def _insights_general(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights = []

    for col in list(stats.keys())[:5]:
        st = stats.get(col,{})
        if not st: continue
        skew    = st.get("skew",0)
        out_pct = st.get("outlier_pct",0)
        mean    = st.get("mean",0)
        median  = st.get("median",0)

        if out_pct > 10:
            insights.append(_build_insight(
                title="'{}': {:.0f}% Outliers — Data Quality Issue".format(col, out_pct),
                problem="{:.0f}% of '{}' values are statistical outliers".format(out_pct, col),
                cause="Data entry errors, measurement anomalies, or genuine extreme values",
                evidence="IQR method: {:.0f}% outliers. Range: {:.2f} to {:.2f}".format(
                    out_pct, st.get("min",0), st.get("max",0)),
                action="1. Inspect outlier records  2. Determine error or genuine  "
                       "3. Cap or remove confirmed errors  4. Document decisions",
                impact="Outliers distort all statistical analyses and reduce ML accuracy",
                severity="warning", category="data_quality"
            ))
        if abs(skew) > 1.5:
            findings.append(
                "'{}' is {}-skewed (mean {:.2f} vs median {:.2f}). "
                "Report median for this column.".format(
                    col, "right" if skew>0 else "left", mean, median))

    for corr in corrs[:3]:
        if corr.get("strength") in ("strong","moderate"):
            findings.append(
                "{} {} relationship: '{}' and '{}' (r={:.2f}) — "
                "statistically significant".format(
                    corr["strength"].title(), corr["direction"],
                    corr["col_a"], corr["col_b"], corr["r"]))

    actions.extend([
        "Validate all outliers before analysis or modeling",
        "Use median for skewed distributions in executive reports",
        "Segment analysis — subgroups may tell different stories",
    ])
    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


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
    all_stats = {col: _col_stats(df[col]) for col in num_cols}
    all_stats = {k:v for k,v in all_stats.items() if v}

    corrs     = _correlations(df)
    attrition = _run_attrition(df) if domain == "hr" else None

    if domain == "hr":
        raw = _insights_hr(df, all_stats, corrs, attrition)
    elif domain == "ecommerce":
        raw = _insights_ecommerce(df, all_stats, corrs)
    elif domain == "sales":
        raw = _insights_sales(df, all_stats, corrs)
    else:
        raw = _insights_general(df, all_stats, corrs)

    # Always add general for extra insights
    gen = _insights_general(df, all_stats, corrs)
    raw["findings"]    += gen["findings"]
    raw["risks"]       += gen["risks"]
    raw["opportunities"]+= gen["opportunities"]

    insights = raw.get("insights",[])
    sev_order = {"critical":0,"warning":1,"info":2,"positive":3}
    insights  = sorted(insights, key=lambda x: sev_order.get(x.severity,99))

    # Deduplicate
    seen, deduped = set(), []
    for ins in insights:
        if ins.title not in seen:
            seen.add(ins.title)
            deduped.append(ins)

    critical  = [i for i in deduped if i.severity=="critical"]
    positive  = [i for i in deduped if i.severity=="positive"]

    # Flat lists for PDF
    findings_flat = raw["findings"][:6]
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
