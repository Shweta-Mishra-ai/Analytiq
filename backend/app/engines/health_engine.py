"""
engines/health_engine.py — data-health scoring and niche insight cards
behind the client-facing Health Report.

Ported from dataforge-ai, where this logic lived inside the Streamlit
page (pages/11_Health_Report.py) rather than in a reusable module. It is
pure pandas/numpy — no UI framework — so it moves across unchanged apart
from the data_profiler import path.

compute_health() deliberately takes its headline score from
data_profiler.profile_dataset() rather than computing a second, independent
formula. Two different quality scores for the same file is a directly
client-visible contradiction when both PDFs are delivered together.

build_insights() returns plain dicts shaped
{tag, title, body, action, severity, border, bg, tag_color} — the contract
health_pdf_builder renders.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
from app.services.dtypes import is_text_dtype

logger = logging.getLogger(__name__)


def compute_health(df: pd.DataFrame) -> dict:
    rows, cols   = len(df), len(df.columns)
    missing_pct  = (df.isna().sum().sum() / max(df.size, 1)) * 100
    dup_pct      = df.duplicated().sum() / max(rows, 1) * 100
    num_cols     = df.select_dtypes(include="number").columns.tolist()
    # Use 1.5x IQR — the SAME threshold the analysis report and data_profiler
    # use. The old 3x IQR found almost nothing, so every dataset scored a
    # non-credible 100/100 while the same report elsewhere listed outlier
    # warnings. A skewed column also dents column health.
    outlier_cols = 0
    skewed_cols  = 0
    for c in num_cols:
        s = df[c].dropna()
        if len(s) > 10:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0 and ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).mean() > 0.05:
                outlier_cols += 1
            try:
                if abs(float(s.skew())) > 1.5:
                    skewed_cols += 1
            except Exception:
                logger.warning("skew calc failed for column '%s'", c, exc_info=True)
    outlier_pct = outlier_cols / max(len(num_cols), 1) * 100

    # Headline score: SAME computation the Main Report uses
    # (core.data_profiler.profile_dataset), not a second independent
    # formula. Two different scores for the identical dataset (this used
    # to compute its own subtractive score here — 53/100 on one real
    # dataset — while the Main Report's completeness/dedup/column-health
    # formula gave 86/100 for the SAME file) is a direct, client-visible
    # contradiction between the two PDFs a client receives together.
    from app.engines.data_profiler import profile_dataset
    try:
        score = float(profile_dataset(df).overall_quality_score)
    except Exception:
        logger.warning("profile_dataset() failed in compute_health — falling back "
                       "to a completeness-only estimate", exc_info=True)
        score = max(0.0, 100.0 - missing_pct)
    score = max(int(round(score)), 0)

    # The score above measures how *clean* the data is — completeness,
    # duplication, per-column health. It says nothing about whether the
    # data can be *analysed at all*: a file with a repeating identifier
    # has zero missing values and zero duplicate rows by the row-level
    # definition, so it scored a clean 100 here while the Main Report's
    # separate readiness check — which looks at whether the same entity
    # appears twice under a supposedly unique key — printed "this
    # dataset is not ready to analyse" on the very next page a reader
    # turned to. Two documents from the same engagement is one place a
    # client is certain to notice a contradiction like that. Readiness
    # is folded in here as a ceiling: a blocking issue caps the grade at
    # C regardless of how clean the rest of the file is, because a
    # reader has no way to tell "clean but analysable" from "clean but
    # not" from the number alone.
    not_ready_reason = ""
    try:
        from app.engines.readiness import assess_readiness

        readiness = assess_readiness(df)
        if not readiness.ready:
            blockers = readiness.blockers
            score = min(score, 69)
            not_ready_reason = (
                "{} — {}".format(blockers[0].column, blockers[0].issue)
                if blockers else "the dataset failed a readiness check")
    except Exception:
        logger.warning("readiness check failed in compute_health — the score "
                       "will not reflect it", exc_info=True)

    grade_map = [(90,"A+","Excellent","#22d3a5"),
                 (80,"A", "Very Good","#42b983"),
                 (70,"B+","Good",     "#60a5fa"),
                 (60,"B", "Fair",     "#fbbf24"),
                 (50,"C", "Needs Work","#f97316"),
                 (0, "D", "Poor",    "#ef4444")]
    grade, label, color = next(
        (g, ln, c) for thresh, g, ln, c in grade_map if score >= thresh)
    if not_ready_reason:
        label = "Not ready to analyse"
        color = "#f97316"

    return {
        "score":       score,
        "grade":       grade,
        "label":       label,
        "color":       color,
        "missing_pct": round(missing_pct, 1),
        "dup_pct":     round(dup_pct, 1),
        "outlier_pct": round(outlier_pct, 1),
        "rows":        rows,
        "cols":        cols,
        "num_cols":    len(num_cols),
        "not_ready_reason": not_ready_reason,
    }


def build_insights(df: pd.DataFrame, niche: str) -> list:
    insights = []
    cols_lower = {c.lower(): c for c in df.columns}

    def _find(keywords, numeric_only=False):
        for kw in keywords:
            for cl, c in cols_lower.items():
                if numeric_only and not pd.api.types.is_numeric_dtype(df[c]):
                    continue
                if kw in cl:
                    return c
        return None

    def _ins(tag, title, body, action, severity):
        # FIX: use rgba with low alpha so cards are visible on both light and dark Streamlit themes.
        # Hardcoded light hex (#fef2f2, #eff6ff etc.) vanish on dark backgrounds.
        COLOR = {
            "critical": ("#ef4444", "rgba(239,68,68,0.08)"),
            "warning":  ("#f97316", "rgba(249,115,22,0.08)"),
            "positive": ("#22d3a5", "rgba(34,211,165,0.08)"),
            "info":     ("#3b82f6", "rgba(59,130,246,0.08)"),
        }
        border, bg = COLOR.get(severity, ("#3b82f6", "rgba(59,130,246,0.08)"))
        return {"tag": tag, "title": title, "body": body, "action": action,
                "severity": severity, "border": border, "bg": bg, "tag_color": border}

    # ── Universal: skew check ──────────────────────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for c in num_cols[:3]:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        skew = float(s.skew()) if len(s) > 3 else 0
        if abs(skew) > 2:
            direction = "right" if skew > 0 else "left"
            diff_pct = abs(s.mean() - s.median()) / max(s.std(), 0.001) * 100
            insights.append(_ins(
                "DATA SKEW",
                "'{}' is heavily skewed (skew={:.2f})".format(c, skew),
                "The distribution of **{}** is {}-skewed. Mean ({:.2f}) differs from Median ({:.2f}) by {:.0f}%. Averages in reports based on this column may be misleading.".format(c, direction, s.mean(), s.median(), diff_pct),
                "Use median for '{}' in client reports, not mean.".format(c),
                "warning"
            ))

    # ── HR-specific ────────────────────────────────────────
    if niche == "hr":
        attrition_col = _find(["attrition","left","churned","resigned","exited","turnover"])
        sat_col       = _find(["satisfaction","engagement","score","rating"])
        dept_col      = _find(["department","dept","team","division"])

        if attrition_col:
            s = df[attrition_col]
            atr_rate = None
            if str(s.dtype) in ["bool","int64","float64"] and float(s.max()) <= 1:
                atr_rate = float(s.mean()) * 100
            elif is_text_dtype(s):
                atr_rate = float(s.str.lower().isin(["yes","true","1","left"]).mean()) * 100
            if atr_rate is not None:
                sev = "critical" if atr_rate > 20 else "warning" if atr_rate > 10 else "positive"
                note = "CRITICAL — more than double the 10% planning threshold." if sev == "critical" else "Above the 10% planning threshold." if sev == "warning" else "Below 10% — excellent retention."
                action = "IMMEDIATE: Conduct stay-interviews in high-risk departments." if sev == "critical" else "Build a quarterly pulse survey. Flag departments above 15%." if sev == "warning" else "Document what drives retention — use as a competitive advantage."
                insights.append(_ins(
                    "ATTRITION RISK" if sev == "critical" else "ATTRITION" if sev == "warning" else "HEALTHY RETENTION",
                    "Attrition rate is {:.1f}% (planning threshold: <10%)".format(atr_rate),
                    "Your dataset shows **{:.1f}% attrition**. {} Replacement cost is commonly modelled at **50–200% of annual salary** — substitute your actual figures to quantify the exposure.".format(atr_rate, note),
                    action, sev
                ))

        if sat_col:
            mean_sat = float(df[sat_col].dropna().mean())
            max_sat  = float(df[sat_col].dropna().max())
            sat_norm = mean_sat / max_sat if max_sat > 1 else mean_sat
            sev = "critical" if sat_norm < 0.5 else "warning" if sat_norm < 0.7 else "positive"
            note = "low satisfaction is a leading indicator of attrition — compare exit rates of low scorers vs the rest to quantify it for this organisation." if sev in ("critical","warning") else "scores above 70% indicate an engaged workforce."
            insights.append(_ins(
                "EMPLOYEE SATISFACTION",
                "Avg satisfaction: {:.2f} / {:.0f} ({:.0f}%)".format(mean_sat, max_sat, sat_norm*100),
                "Mean satisfaction score is **{:.2f}** (normalized: {:.0f}%) — {}".format(mean_sat, sat_norm*100, note),
                "Implement manager 1:1s and recognition programs, and track exits by satisfaction band to measure impact." if sev != "positive" else "Publish satisfaction data in employer branding.",
                sev
            ))

        if dept_col and attrition_col:
            try:
                s = df[attrition_col]
                tmp = df.copy()
                if is_text_dtype(s):
                    tmp["_atr"] = s.str.lower().isin(["yes","true","1","left"]).astype(float)
                else:
                    tmp["_atr"] = pd.to_numeric(s, errors="coerce")
                grp = tmp.groupby(dept_col)["_atr"]
                sizes = grp.size()
                dept_risk = grp.mean()[sizes >= 20].sort_values(ascending=False)
                if len(dept_risk) == 0:
                    raise ValueError("no department with n>=20 — skip insight")
                top_dept  = str(dept_risk.index[0])
                top_rate  = float(dept_risk.iloc[0]) * 100
                sev = "critical" if top_rate > 20 else "warning"
                insights.append(_ins(
                    "DEPARTMENT RISK",
                    "'{}' has the highest attrition: {:.1f}%".format(top_dept, top_rate),
                    "The **{}** department shows {:.1f}% attrition — {} the 10% planning threshold. Department-level attrition often signals a specific manager, workload, or pay equity issue.".format(top_dept, top_rate, "critically above" if top_rate > 20 else "above"),
                    "Prioritize '{}' for skip-level interviews and exit interview analysis.".format(top_dept),
                    sev
                ))
            except Exception:
                logger.warning("Health Report section failure", exc_info=True)

        # ── HR: Tenure cohort analysis ─────────────────
        tenure_col = _find(["tenure","years","seniority","experience","time_spend"])
        if tenure_col and attrition_col:
            try:
                tmp2 = df.copy()
                if is_text_dtype(df[attrition_col]):
                    tmp2["_atr2"] = df[attrition_col].str.lower().isin(["yes","true","1","left"]).astype(float)
                else:
                    tmp2["_atr2"] = pd.to_numeric(df[attrition_col], errors="coerce")
                # Bin tenure into 3 cohorts
                tmp2["_tenure_bin"] = pd.cut(tmp2[tenure_col],
                    bins=[0, 2, 5, float("inf")],
                    labels=["0–2 yrs", "3–5 yrs", "6+ yrs"])
                cohort_atr = tmp2.groupby("_tenure_bin")["_atr2"].mean() * 100
                new_hire_atr = float(cohort_atr.get("0–2 yrs", 0))
                vet_atr      = float(cohort_atr.get("6+ yrs", 0))
                if new_hire_atr > 25:
                    insights.append(_ins(
                        "ONBOARDING RISK",
                        "New hires (0–2 yrs) attrition: {:.1f}%".format(new_hire_atr),
                        "Employees in their first 2 years show {:.1f}% attrition — a signal of poor onboarding, misaligned expectations, or poor manager support. "
                        "Veteran employees (6+ yrs) show {:.1f}% attrition by comparison. "
                        "Replacing a new hire is commonly modelled at 50–150% of salary before full productivity.".format(new_hire_atr, vet_atr),
                        "Implement a structured 90-day onboarding program. Assign mentors to all new hires. "
                        "Run a 30/60/90 check-in survey to catch at-risk employees early.",
                        "critical" if new_hire_atr > 35 else "warning"
                    ))
                elif vet_atr > 15:
                    insights.append(_ins(
                        "VETERAN FLIGHT RISK",
                        "Senior employees (6+ yrs) attrition: {:.1f}%".format(vet_atr),
                        "Experienced employees (6+ years tenure) are leaving at {:.1f}%. "
                        "This is a knowledge drain — these employees carry institutional memory, client relationships, and domain expertise. "
                        "Senior replacements are commonly modelled at up to 200% of annual salary because of lost institutional knowledge.".format(vet_atr),
                        "Run skip-level interviews with the 6+ year cohort this quarter. "
                        "Review compensation and career growth opportunities for this group specifically.",
                        "warning"
                    ))
            except Exception:
                logger.warning("Health Report section failure", exc_info=True)

        # ── HR: Overwork risk ─────────────────────────────
        hours_col = _find(["hours","montly_hours","monthly_hours","avg_hours","work_hour"])
        if hours_col:
            try:
                mean_h = float(df[hours_col].dropna().mean())
                pct_overwork = float((df[hours_col].dropna() > 210).mean() * 100)
                if pct_overwork > 20 or mean_h > 195:
                    sev = "critical" if pct_overwork > 40 or mean_h > 220 else "warning"
                    insights.append(_ins(
                        "OVERWORK RISK",
                        "Avg {:.0f} hrs/month — {:.0f}% working >210 hrs".format(mean_h, pct_overwork),
                        "Average monthly hours are {:.0f} (standard: 160–180 hrs). "
                        "{:.0f}% of employees work more than 210 hours per month — a burnout risk zone. "
                        "Compare the exit rate of the >210-hour group against the rest of the workforce "
                        "to quantify the burnout cost visible in this dataset.".format(mean_h, pct_overwork),
                        "Audit workload distribution immediately — identify if overwork is concentrated in specific teams. "
                        "Hire contractors or redistribute tasks. Target: bring >90% of workforce under 200 hrs/month.",
                        sev
                    ))
            except Exception:
                logger.warning("Health Report section failure", exc_info=True)

        # ── HR: Salary band vs attrition ─────────────────
        # A 'salary band' must be a small set of bands (low/medium/high) — NOT
        # a rate/percent column like 'PercentSalaryHike' whose values (11, 22,
        # 24…) would be printed as fake bands ("'24' band"). Exclude percent/
        # rate columns; bin a continuous salary into quartiles.
        _sal_candidates = [c for c in df.columns
                           if any(k in c.lower() for k in ("salary", "pay", "compensation", "wage", "band"))
                           and not any(k in c.lower() for k in ("percent", "pct", "hike", "rate", "ratio", "%"))]
        salary_col = _sal_candidates[0] if _sal_candidates else None
        if salary_col and attrition_col:
            try:
                tmp3 = df.copy()
                if is_text_dtype(df[attrition_col]):
                    tmp3["_atr3"] = df[attrition_col].str.lower().isin(["yes","true","1","left"]).astype(float)
                else:
                    tmp3["_atr3"] = pd.to_numeric(df[attrition_col], errors="coerce")
                # Continuous salary with many values → quartile bands.
                band_col = salary_col
                if pd.api.types.is_numeric_dtype(df[salary_col]) and df[salary_col].nunique() > 8:
                    tmp3["_band"] = pd.qcut(df[salary_col], 4,
                                            labels=["Lowest 25%", "Lower-mid", "Upper-mid", "Highest 25%"],
                                            duplicates="drop")
                    band_col = "_band"
                grp3 = tmp3.groupby(band_col, observed=True)["_atr3"]
                sal_atr = (grp3.mean()[grp3.size() >= 20]
                           .sort_values(ascending=False) * 100)
                if len(sal_atr) >= 2:
                    worst_band = str(sal_atr.index[0])
                    worst_rate = float(sal_atr.iloc[0])
                    best_band  = str(sal_atr.index[-1])
                    best_rate  = float(sal_atr.iloc[-1])
                    gap        = worst_rate - best_rate
                    if gap > 8:
                        insights.append(_ins(
                            "PAY-DRIVEN ATTRITION",
                            "'{}' band: {:.1f}% attrition vs '{}': {:.1f}%".format(worst_band, worst_rate, best_band, best_rate),
                            "The '{}' salary band has {:.1f}% attrition vs {:.1f}% for the '{}' band — a {:.0f} percentage point gap. "
                                                        "Pay-driven attrition is the fastest to fix but most expensive if ignored — each exit in a low band still costs 50–100% of annual salary.".format(
                                worst_band, worst_rate, best_rate, best_band, gap),
                            "Run market salary benchmarking for the '{}' band within 30 days. "
                            "Model the ROI of a 10–15% pay increase vs replacement cost for the highest-risk employees.".format(worst_band),
                            "critical" if worst_rate > 25 else "warning"
                        ))
            except Exception:
                logger.warning("Health Report section failure", exc_info=True)

        # ── HR: Promotion gap → flight risk ───────────────
        promo_col = _find(["promotion","promoted","promotion_last"])
        if promo_col and sat_col and attrition_col:
            try:
                promo_s = df[promo_col]
                if not pd.api.types.is_numeric_dtype(promo_s):
                    promo_s = promo_s.str.lower().isin(["yes","true","1"]).astype(float)
                promo_rate = float(promo_s.mean()) * 100
                # Satisfaction for un-promoted employees
                not_promoted_sat = float(df.loc[promo_s == 0, sat_col].dropna().mean()) if (promo_s == 0).any() else None
                promoted_sat     = float(df.loc[promo_s == 1, sat_col].dropna().mean()) if (promo_s == 1).any() else None
                if promo_rate < 5 and not_promoted_sat is not None:
                    insights.append(_ins(
                        "PROMOTION GAP",
                        "Only {:.1f}% promoted — unpromoted satisfaction: {:.2f}".format(promo_rate, not_promoted_sat),
                        "Only {:.1f}% of employees received a promotion in the last 5 years. "
                        "Employees without promotion show {:.2f} satisfaction vs {:.2f} for promoted staff — a {:.0f}% gap. "
                        "Mercer 2024: lack of career growth is the #1 voluntary exit driver. "
                        "Employees without a promotion path are 3× more likely to leave within 12 months.".format(
                            promo_rate,
                            not_promoted_sat,
                            promoted_sat if promoted_sat else not_promoted_sat,
                            abs((promoted_sat or not_promoted_sat) - not_promoted_sat) / max(not_promoted_sat, 0.01) * 100
                        ),
                        "Create transparent promotion criteria for all levels. "
                        "Implement individual development plans (IDPs) for the bottom 30% satisfaction + no-promotion segment. "
                        "Target: increase promotion rate to at least 10% per year.",
                        "critical" if promo_rate < 3 else "warning"
                    ))
            except Exception:
                logger.warning("Health Report section failure", exc_info=True)

        # ── HR: Flight risk segment ───────────────────────
        # Headline count/% MUST match the Main Report's flight-risk figure —
        # both now call the one shared core.engines.hr.compute_flight_risk()
        # definition (satisfaction only). Tenure is reported as additional
        # context on top of that shared count, not folded into the count
        # itself — previously this section used its own two-factor formula
        # (satisfaction AND tenure), which produced a different headcount
        # than the Main Report for the same dataset.
        if attrition_col and sat_col:
            try:
                from app.engines.domains.hr import compute_flight_risk, _find_left_mask
                left_mask = _find_left_mask(df)
                n_flight, risk_pct, _sat_col = compute_flight_risk(df, left_mask=left_mask)
                if n_flight > 0 and risk_pct > 10:
                    tenure_note = ""
                    if tenure_col and left_mask is not None:
                        try:
                            current = df.loc[~left_mask]
                            ten_median = float(current[tenure_col].median())
                            sat_q25 = float(current[sat_col].quantile(0.25))
                            at_risk_mask = (~left_mask) & (df[sat_col] <= sat_q25)
                            long_tenured = int((df.loc[at_risk_mask, tenure_col] >= ten_median).sum())
                            tenure_note = (
                                " Of these, {:,} are also long-tenured (≥{:.0f} years) — "
                                "the highest-value flight risks, since they carry the most "
                                "institutional knowledge to lose.".format(long_tenured, ten_median)
                            )
                        except Exception:
                            logger.warning("Health Report tenure sub-segment failed", exc_info=True)
                    insights.append(_ins(
                        "FLIGHT RISK SEGMENT",
                        "{:.0f}% of current workforce is at high flight risk".format(risk_pct),
                        "**{:,} current employees** ({:.0f}% of workforce) fall in the bottom "
                        "quartile of satisfaction — the flight risk profile.{}".format(
                            n_flight, risk_pct, tenure_note),
                        "Pull this segment's names from your HRIS immediately. "
                        "Schedule 1:1 career conversations within 2 weeks. "
                        "This is your highest-priority retention action — act before they decide.",
                        "critical" if risk_pct > 20 else "warning"
                    ))
            except Exception:
                logger.warning("Health Report section failure", exc_info=True)

    # ── SALES-specific ────────────────────────────────────
    elif niche == "sales":
        rev_col    = _find(["revenue","amount","deal_value","deal_size","value","gmv","arr"], numeric_only=True)
        status_col = _find(["status","stage","outcome","result","win","lost"])

        if rev_col:
            s     = df[rev_col].dropna()
            total = float(s.sum())
            top10 = float(s.nlargest(max(1, int(len(s)*0.1))).sum())
            conc  = top10 / total * 100 if total > 0 else 0
            sev   = "critical" if conc > 80 else "warning" if conc > 60 else "positive"
            note  = "CRITICAL concentration risk — losing 1-2 key accounts could collapse revenue." if sev == "critical" else "High concentration — moderate dependency on key accounts." if sev == "warning" else "Healthy revenue distribution."
            insights.append(_ins(
                "REVENUE CONCENTRATION",
                "Top 10% of deals = {:.0f}% of total revenue".format(conc),
                "Your top 10% deals account for **{:.0f}% of {}** (${:,.0f} total). {}".format(conc, rev_col, total, note),
                "Immediately build 3–5 additional pipeline accounts at the same deal size." if sev == "critical" else "Implement account health scoring. Trigger executive relationships for top 10%." if sev == "warning" else "Replicate the profile of top-performing deals in prospecting strategy.",
                sev
            ))

        if status_col:
            s    = df[status_col].dropna().astype(str).str.lower()
            won  = int(s.isin(["won","win","closed won","success","yes"]).sum())
            lost = int(s.isin(["lost","lose","closed lost","loss","no","failed"]).sum())
            total = won + lost
            win_rate = won / total * 100 if total > 0 else 0
            sev  = "critical" if win_rate < 20 else "warning" if win_rate < 35 else "positive"
            note = "Well below benchmark — significant pipeline efficiency problem." if sev == "critical" else "Below benchmark — improvement needed." if sev == "warning" else "At or above benchmark — strong sales execution."
            insights.append(_ins(
                "WIN RATE",
                "Win rate: {:.1f}% (Benchmark: 25-40%)".format(win_rate),
                "Out of **{} qualified deals**, {} were won ({:.1f}%). A 25–40% win rate is a common planning band — validate against your market. {}".format(total, won, win_rate, note),
                "Implement MEDDIC or BANT qualification framework. Review lost deal reasons." if sev != "positive" else "Document the winning sales playbook. Scale to underperformers.",
                sev
            ))

    # ── E-COMMERCE-specific ──────────────────────────────
    elif niche == "ecommerce":
        order_val_col = _find(["order_value","gmv","amount","total","price","revenue","aov"], numeric_only=True)
        customer_col  = _find(["customer","user","buyer","client_id","customer_id"])

        if order_val_col:
            s   = df[order_val_col].dropna()
            aov = float(s.mean())
            top20_avg = float(s.nlargest(max(1, int(len(s)*0.2))).mean())
            insights.append(_ins(
                "AVERAGE ORDER VALUE",
                "AOV = ${:,.2f} per transaction".format(aov),
                "Average Order Value is **${:,.2f}**. Top 20% of orders average ${:,.2f}. Increasing AOV by 10% through upselling is more profitable than acquiring new customers.".format(aov, top20_avg),
                "Deploy bundled products for orders below ${:.0f}. Target: lift AOV by 15% in 90 days.".format(aov * 0.7),
                "info"
            ))

        if customer_col and order_val_col:
            try:
                cust_spend = df.groupby(customer_col)[order_val_col].sum()
                top20_pct  = float(cust_spend.nlargest(max(1, int(len(cust_spend)*0.2))).sum() / cust_spend.sum() * 100)
                sev = "critical" if top20_pct > 80 else "warning" if top20_pct > 65 else "positive"
                insights.append(_ins(
                    "PARETO PRINCIPLE",
                    "Top 20% customers = {:.0f}% of revenue".format(top20_pct),
                    "**{:.0f}% of revenue** comes from your top 20% customers. {} ".format(top20_pct, "Classic Pareto — but high concentration means churn of top customers = major revenue loss." if sev != "positive" else "Healthy spread — revenue reasonably distributed."),
                    "Build a VIP tier for top 20% customers. Assign dedicated account managers.",
                    sev
                ))
            except Exception:
                logger.warning("Health Report section failure", exc_info=True)

    # ── FINANCE-specific ─────────────────────────────────
    elif niche == "finance":
        rev_col  = _find(["revenue","income","sales","turnover","gross"], numeric_only=True)
        cost_col = _find(["cost","expense","cogs","expenditure","opex"], numeric_only=True)

        if rev_col and cost_col:
            rev    = float(df[rev_col].dropna().sum())
            cost   = float(df[cost_col].dropna().sum())
            margin = (rev - cost) / rev * 100 if rev > 0 else 0
            sev    = "critical" if margin < 5 else "warning" if margin < 15 else "positive"
            note   = "CRITICAL: At risk of operating loss." if sev == "critical" else "Below benchmark — cost control or pricing review needed." if sev == "warning" else "Healthy margin — above the planning band."
            insights.append(_ins(
                "GROSS MARGIN",
                "Gross Margin: {:.1f}% (Benchmark: >15%)".format(margin),
                "Total Revenue: **${:,.0f}** | Total Cost: **${:,.0f}** | Gross Margin: **{:.1f}%**. McKinsey 2024: healthy businesses target >15% gross margin. {}".format(rev, cost, margin, note),
                "IMMEDIATE: Conduct cost structure analysis. Identify top 3 cost drivers." if sev == "critical" else "Review pricing strategy and renegotiate top 3 supplier contracts." if sev == "warning" else "Model the impact of a 5% price increase to protect margin.",
                sev
            ))

    # ── Universal: Missing data risk ──────────────────────
    miss = df.isna().sum()
    bad_cols = miss[miss / len(df) > 0.3].sort_values(ascending=False)
    if len(bad_cols) > 0:
        col_list = ", ".join(
            ["'{}' ({:.0f}%)".format(c, miss[c]/len(df)*100) for c in bad_cols.index[:3]])
        worst_pct = float(bad_cols.iloc[0] / len(df))
        sev = "critical" if worst_pct > 0.5 else "warning"
        insights.append(_ins(
            "DATA QUALITY RISK",
            "{} column(s) have >30% missing data".format(len(bad_cols)),
            "Columns with high missing rates: **{}**. Any analysis or model trained on these columns will be statistically unreliable.".format(col_list),
            "Either impute with domain-appropriate values, or exclude from client-facing insights.",
            sev
        ))

    if not insights:
        insights.append(_ins(
            "DATA HEALTH",
            "Dataset is clean and analysis-ready",
            "No critical issues detected. Data completeness, distribution, and structure are within acceptable thresholds for business analysis.",
            "Proceed directly to Dashboard and ML Predictions pages.",
            "positive"
        ))

    return insights


# ══════════════════════════════════════════════════════════
#  FULL INSIGHT SET  (domain engines + data-quality cards)
# ══════════════════════════════════════════════════════════

# The domain engines emit a richer severity ladder than the report's card
# renderer understands. Anything unmapped would fall back to "info" and be
# rendered as a neutral blue card, quietly demoting a critical finding.
_SEVERITY_TO_CARD = {
    "critical": "critical",
    "high":     "critical",
    "warning":  "warning",
    "medium":   "warning",
    "info":     "info",
    "low":      "info",
    "positive": "positive",
}

_CARD_COLORS = {
    "critical": ("#ef4444", "rgba(239,68,68,0.08)"),
    "warning":  ("#f97316", "rgba(249,115,22,0.08)"),
    "positive": ("#22d3a5", "rgba(34,211,165,0.08)"),
    "info":     ("#3b82f6", "rgba(59,130,246,0.08)"),
}


def _insight_to_card(ins) -> Dict:
    """Adapt a domains/*.Insight dataclass into the flat card dict the
    Health Report renders.

    The engines already write in Problem → Cause → Evidence → Action →
    Impact form, which maps cleanly onto the report's
    What → Why it matters → What to do layout.
    """
    severity = _SEVERITY_TO_CARD.get(str(ins.severity).lower(), "info")
    border, bg = _CARD_COLORS[severity]

    body_parts = [p for p in (ins.problem, ins.cause, ins.evidence) if p]
    body = " ".join(str(p).strip() for p in body_parts)
    if ins.impact:
        body = f"{body} Impact: {str(ins.impact).strip()}"

    tag = str(getattr(ins, "category", "") or severity).upper().replace("_", " ")
    return {"tag": tag, "title": str(ins.title), "body": body,
            "action": str(ins.action), "severity": severity,
            "border": border, "bg": bg, "tag_color": border,
            "confidence": str(getattr(ins, "confidence", "") or "")}


def build_report_payload(df: pd.DataFrame, niche: str,
                          max_cards: int = 12) -> Dict:
    """Everything the Health Report renders, in one call.

    The domain engines produce four distinct kinds of output — insight
    cards, key findings, risks and opportunities — and several analyses
    contribute only to the latter three. Finance, for example, emits 6
    findings, 2 risks and 2 opportunities alongside just 3 cards; the
    cost-concentration and budget-variance analyses produce no card at
    all. Rendering cards alone therefore threw most of the analysis away.

    Returning all four (plus the executive summary and recommended
    actions) is also what makes the output read like a consulting
    deliverable rather than a list of alerts.
    """
    payload: Dict = {
        "executive_summary": "",
        "insights": [],
        "key_findings": [],
        "risks": [],
        "opportunities": [],
        "actions": [],
    }

    try:
        from app.engines.story_engine import generate_story
        story = generate_story(df)
        payload["executive_summary"] = story.executive_summary or ""
        payload["key_findings"] = list(story.key_findings or [])
        payload["risks"] = list(story.business_risks or [])
        payload["opportunities"] = list(story.opportunities or [])
        payload["actions"] = list(story.recommended_actions or [])
    except Exception:
        logger.warning("story engine failed while building the health report "
                       "payload — narrative sections will be empty",
                       exc_info=True)

    payload["insights"] = build_full_insights(df, niche, max_cards=max_cards)
    return payload


def build_full_insights(df: pd.DataFrame, niche: str, max_cards: int = 12) -> List[Dict]:
    """Every insight the Health Report should carry, best-first.

    Sources, in priority order:
      1. the per-domain engines in app/engines/domains/ — the deep business
         analysis (regional gaps, loss-making segments, quota attainment,
         rep spread, cohort/retention effects, …)
      2. this module's own niche cards — data-quality angles the domain
         engines don't cover (skew warnings, "use median not mean", …)

    Previously the report used source 2 alone, which produced a single
    insight card for a sales/ecommerce/finance dataset even when the domain
    engine found several critical findings in the same file. A one-card
    report is not a deliverable a client will pay for.
    """
    cards: List[Dict] = []

    try:
        from app.engines.story_engine import generate_story
        story = generate_story(df)
        for ins in story.top_insights:
            cards.append(_insight_to_card(ins))
    except Exception:
        logger.warning("domain insight engines failed for the health report — "
                       "falling back to data-quality cards only", exc_info=True)

    try:
        cards.extend(build_insights(df, niche))
    except Exception:
        logger.warning("niche data-quality cards failed", exc_info=True)

    # De-duplicate on title; keep the first (domain engines rank higher).
    seen, unique = set(), []
    for c in cards:
        key = c["title"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
    unique.sort(key=lambda c: order.get(c["severity"], 9))
    return unique[:max_cards]
