"""
core/engines/sales.py — Sales Performance domain engine.
Single responsibility: revenue, quota, margin, and rep performance insights.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import pandas as pd

from app.engines.domains.base import build_insight
from app.engines.domains._common import fmt
from app.engines.domains.sales_performance import run_sales_performance
from app.engines.industry_benchmarks import lookup_benchmark
from app.engines.present import label as _L, num as _N

logger = logging.getLogger(__name__)


def _sales_cycle_insights(df: pd.DataFrame, insights: List, findings: List,
                          risks: List, opps: List) -> None:
    """Sales Cycle Length — days from opportunity open to close. Not a
    column that exists in most datasets; derived from two date columns.
    This is the single most commonly-requested sales KPI that wasn't
    computed anywhere in the app before.
    """
    start_kw = ("created", "opened", "open_date", "start_date", "opportunity_date")
    end_kw   = ("closed", "close_date", "won_date", "end_date", "close date")

    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    if len(date_cols) < 2:
        return

    start_col = next((c for c in date_cols if any(k in c.lower() for k in start_kw)), None)
    end_col   = next((c for c in date_cols
                      if any(k in c.lower() for k in end_kw) and c != start_col), None)
    if not start_col or not end_col:
        return

    try:
        cycle_days = (df[end_col] - df[start_col]).dt.days
        # Only positive, plausible cycles — negative means data-entry error
        # (closed before opened), and >2 years is almost always a bad join
        # or a placeholder date rather than a real sales cycle.
        cycle_days = cycle_days[(cycle_days > 0) & (cycle_days <= 730)]
        if len(cycle_days) < 10:
            return

        mean_cycle = float(cycle_days.mean())
        med_cycle  = float(cycle_days.median())
        p90_cycle  = float(cycle_days.quantile(0.9))

        findings.append(
            "Sales cycle: median {:.0f} days ({} to {}), 90th percentile {:.0f} days.".format(
                med_cycle, start_col, end_col, p90_cycle))

        bm = lookup_benchmark("sales", "sales_cycle_days")
        bm_text = ""
        if bm:
            if med_cycle < bm.low:
                bm_text = (" This is faster than the {:.0f}-{:.0f} day general guidance range "
                          "— worth confirming deals aren't being rushed or "
                          "under-qualified.").format(bm.low, bm.high)
            elif med_cycle > bm.high:
                bm_text = (" This is longer than the {:.0f}-{:.0f} day general guidance range "
                          "— worth investigating where deals are stalling.").format(
                              bm.low, bm.high)
            else:
                bm_text = " This falls within the {:.0f}-{:.0f} day general guidance range.".format(
                    bm.low, bm.high)

        insights.append(build_insight(
            title="Sales Cycle: {:.0f} Days Median ({:.0f} at 90th Percentile)".format(
                med_cycle, p90_cycle),
            problem="Median time from {} to {} is {:.0f} days.".format(
                start_col, end_col, med_cycle),
            cause="Cycle length reflects deal complexity, qualification quality, and "
                  "how many approval steps a deal goes through — not measured directly here.",
            evidence="n={:,} deals with a valid cycle. Mean={:.1f}, median={:.0f}, "
                     "90th percentile={:.0f} days.{}".format(
                         len(cycle_days), mean_cycle, med_cycle, p90_cycle, bm_text),
            action="1. Segment cycle length by deal size and rep — find where it's "
                   "concentrated  2. Map the stages where deals sit longest  "
                   "3. Compare cycle length for won vs lost deals if outcome data exists",
            impact="Shortening the median cycle by even a few days compounds across every "
                   "deal in the pipeline — worth tracking as a trend, not just a snapshot.",
            severity="info", category="sales_cycle"
        ))

        if bm and med_cycle > bm.high * 1.5:
            risks.append(
                "Sales cycle (median {:.0f} days) is well above the {:.0f}-{:.0f} day general "
                "guidance range — deals may be stalling somewhere in the process.".format(
                    med_cycle, bm.low, bm.high))
    except Exception:
        logger.warning("sales cycle length analysis failed", exc_info=True)


# A ratio outside this band says the two columns are not measuring the
# same thing. A per-deal revenue column against an annual team quota
# produced "6% achievement, 94pp below target" and marked it CRITICAL --
# on a file whose own data-quality page said that quota column was
# constant and should be dropped.
ATTAINMENT_FLOOR = 1.0
ATTAINMENT_CEILING = 1000.0


def _quota_attainment(df, rev_col: str, target_col: str):
    """(attainment %, how it was computed) or (None, why it cannot be).

    Mean revenue over mean target is only meaningful when both are
    recorded at the same grain -- one row per owner per period, each
    carrying that owner's number and that owner's target. A quota
    repeated unchanged on every row is not that: it is one figure for a
    team and a period the file never names, and no arithmetic recovers
    which.
    """
    try:
        pair = df[[rev_col, target_col]].dropna()
        if len(pair) < 20:
            return None, "Too few complete rows to compare revenue against target."

        targets = pair[target_col]
        if targets.nunique() <= 1:
            return None, ("A single quota figure is repeated on every row, so "
                          "the period and the owner it applies to are not "
                          "recorded.")
        if (targets <= 0).all():
            return None, "Every target is zero or negative."

        usable = pair[targets > 0]
        if len(usable) < 20:
            return None, "Too few rows carry a positive target."

        # Row-wise, then the median: one row with a tiny target must not
        # carry the whole figure, which is exactly what a ratio of means
        # allows.
        ratios = (usable[rev_col] / usable[target_col]) * 100.0
        attainment = float(ratios.median())
        if not (ATTAINMENT_FLOOR <= attainment <= ATTAINMENT_CEILING):
            return None, ("Revenue and target differ by a factor of {:,.0f}, "
                          "which means they are recorded at different grains "
                          "-- per deal against per period, most likely."
                          .format(max(attainment, 100.0 / max(attainment, 1e-9))
                                  / 100.0))
        return attainment, ("Median of revenue divided by target, row by row, "
                            "across {:,} records".format(len(usable)))
    except Exception:
        logger.debug("quota attainment failed", exc_info=True)
        return None, "Target attainment could not be computed from these columns."


def _insights_sales(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights = []

    # "deal_value", "opportunity_value", "acv" and "contract_value" are
    # what a CRM export actually calls this column, and none of them
    # contain revenue/sales/amount/total. Without them the whole revenue
    # branch below was skipped on ordinary pipeline data and the engine
    # fell back to generic outlier notes.
    rev_col    = next((c for c in df.columns
                       if any(k in c.lower().replace("_", "")
                              for k in ["revenue", "sales", "amount", "total",
                                        "dealvalue", "dealsize", "opportunityvalue",
                                        "contractvalue", "acv", "tcv", "bookings"])
                       and c in stats), None)
    profit_col = next((c for c in df.columns
                       if any(k in c.lower() for k in ["profit","margin","net"])
                       and c in stats), None)
    target_col = next((c for c in df.columns
                       if any(k in c.lower() for k in ["target","quota","goal"])
                       and c in stats), None)
    region_col = next((c for c in df.select_dtypes(include=["object", "string"]).columns
                       if any(k in c.lower() for k in ["region","territory","zone","area"])
                       and df[c].nunique()<=25), None)
    product_col= next((c for c in df.select_dtypes(include=["object", "string"]).columns
                       if any(k in c.lower() for k in ["product","category","segment"])
                       and df[c].nunique()<=30), None)
    rep_col    = next((c for c in df.select_dtypes(include=["object", "string"]).columns
                       if any(k in c.lower() for k in ["rep","salesperson","agent","owner",
                                                       "employee","seller","account manager"])
                       and 2 <= df[c].nunique() <= 200), None)

    # ── Revenue Analysis ───────────────────────────────────
    if rev_col and rev_col in stats:
        st   = stats[rev_col]
        skew = st.get("skew",0)
        cv   = st.get("cv",0)
        mean = st.get("mean",0)
        med  = st.get("median",0)

        insights.append(build_insight(
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
            impact=("High positive skew means a few large deals dominate revenue in "
                    "this dataset — concentration risk worth quantifying on the Pareto "
                    "page before relying on any single account." if skew > 1 else
                    "Revenue is relatively evenly spread here — low single-account "
                    "concentration risk."),
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
        achievement, basis = _quota_attainment(df, rev_col, target_col)

        if achievement is None:
            # The target column is there and its grain cannot be
            # established, so no attainment figure is honest. Say what
            # is actually known instead of printing a ratio.
            insights.append(build_insight(
                title="Quota is recorded but attainment cannot be computed",
                problem=basis,
                cause="A quota is set for a person and a period. This file "
                      "carries one figure on every row without saying which "
                      "person or which period it covers, so dividing one "
                      "column by the other compares quantities of different "
                      "grain.",
                evidence="{} holds a single value of {} across all {:,} rows, "
                         "while {} is recorded per record (mean {:,.0f}).".format(
                             _L(target_col),
                             _N(float(stats[target_col].get("mean", 0))),
                             len(df), _L(rev_col),
                             float(stats[rev_col].get("mean", 0))),
                action="1. Add the period and the owner the quota belongs to  "
                       "2. Or supply attainment as its own column  "
                       "3. Re-run — rep-level attainment is then a direct "
                       "comparison",
                impact="Until then this report ranks reps on win rate and "
                       "revenue, which the data does support, and makes no "
                       "claim about target attainment.",
                severity="info", category="target"))
        elif achievement < 80:
            insights.append(build_insight(
                title="Target Gap: {:.0f}% Achievement — {:.0f}pp Below Target".format(
                    achievement, 100-achievement),
                problem="Average {:.0f}% quota achievement — team missing targets significantly".format(achievement),
                cause="Possible drivers to investigate (not yet confirmed): targets set "
                      "too high, pipeline quality, or sales-process gaps.",
                evidence="{}. Achievement={:.0f}%.".format(basis, achievement),
                action="1. Check whether targets are realistic against your own historical "
                       "attainment  2. Pipeline quality audit — qualification issues  "
                       "3. Sales-process coaching for bottom-quartile reps",
                impact="{:.0f}% achievement gap = {:.0f}% revenue shortfall from plan".format(
                    100-achievement, 100-achievement),
                severity="critical" if achievement<70 else "warning",
                category="target"
            ))
            risks.append("{:.0f}% target achievement — revenue significantly below plan".format(achievement))
        elif achievement >= 100:
            insights.append(build_insight(
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

    # ── Regional Analysis (significance-tested, quantified) ──────────
    if region_col and rev_col and rev_col in df.columns:
        reg_perf = df.groupby(region_col)[rev_col].agg(["mean", "median", "sum", "count"])
        reg_perf = reg_perf[reg_perf["count"] >= 3].sort_values("median", ascending=False)
        if len(reg_perf) >= 2:
            top_r    = reg_perf.index[0]
            bottom_r = reg_perf.index[-1]
            top_share = reg_perf.loc[top_r, "sum"] / reg_perf["sum"].sum() * 100
            top_med   = reg_perf.loc[top_r, "median"]
            bot_med   = reg_perf.loc[bottom_r, "median"]
            gap_pct   = (top_med - bot_med) / max(abs(bot_med), 1) * 100
            overall_med = float(df[rev_col].median())

            # Is the regional difference real, or sampling noise? Kruskal-Wallis
            # across regions (non-parametric — revenue is skewed).
            sig_txt, significant = "difference not statistically tested", True
            try:
                from scipy import stats as _sc
                groups = [g[rev_col].dropna().values
                          for _, g in df.groupby(region_col) if len(g) >= 5]
                if len(groups) >= 2:
                    h, p = _sc.kruskal(*groups)
                    significant = bool(p < 0.05)
                    sig_txt = ("Kruskal-Wallis H={:.1f}, p<0.001".format(h) if p < 0.001
                               else "Kruskal-Wallis H={:.1f}, p={:.3f}".format(h, p))
            except Exception:
                logger.warning("regional significance test failed", exc_info=True)

            # Quantified opportunity: lift the weakest region's records to the
            # overall median — a measured upper bound, not a vague "significant".
            bot_n = int(reg_perf.loc[bottom_r, "count"])
            opp_value = max(0.0, (overall_med - bot_med)) * bot_n

            if significant and gap_pct >= 10:
                insights.append(build_insight(
                    title="Regional Gap: '{}' median {:.0f} vs '{}' {:.0f} ({:.0f}% spread)".format(
                        top_r, top_med, bottom_r, bot_med, gap_pct),
                    problem="'{}' trails the top region '{}' by {:.0f}% on median revenue "
                            "per record.".format(bottom_r, top_r, gap_pct),
                    cause="A gap this size is statistically unlikely to be chance ({}). "
                          "Likely drivers to confirm: market maturity, team capability, "
                          "competition, or resource allocation — not yet proven.".format(sig_txt),
                    evidence="'{}' median={:.0f} vs '{}' median={:.0f}. {}. Top region holds "
                             "{:.0f}% of total revenue.".format(
                                 top_r, top_med, bottom_r, bot_med, sig_txt, top_share),
                    action="1. Profile what separates '{}' from '{}' (mix, deal size, cycle)  "
                           "2. Check the gap persists after controlling for account size  "
                           "3. Pilot the stronger region's playbook in '{}'".format(
                               top_r, bottom_r, bottom_r),
                    impact="Lifting '{}' ({:,} records) to the overall median is worth about "
                           "{:,.0f} in revenue — the measured upper bound of this gap.".format(
                               bottom_r, bot_n, opp_value),
                    severity="warning" if gap_pct < 50 else "critical",
                    category="regional"
                ))
                opps.append("Lift '{}' to the overall median revenue: ~{:,.0f} upside "
                            "({:,} records).".format(bottom_r, opp_value, bot_n))
            findings.append(
                "Top region '{}' contributes {:.0f}% of total revenue — concentration "
                "risk worth monitoring.".format(top_r, top_share) if top_share > 50 else
                "Revenue reasonably distributed across {} regions.".format(len(reg_perf)))

    # ── Product/Category Performance ──────────────────────
    if product_col and rev_col and rev_col in df.columns:
        prod_perf = df.groupby(product_col)[rev_col].agg(["sum","count"])
        prod_perf = prod_perf[prod_perf["count"]>=3].sort_values("sum", ascending=False)
        if len(prod_perf)>=2:
            total_rev = prod_perf["sum"].sum()
            top_prod  = prod_perf.index[0]
            top_share = prod_perf.loc[top_prod,"sum"]/total_rev*100
            _top2_share= prod_perf.iloc[:2]["sum"].sum()/total_rev*100

            if top_share > 40:
                risks.append(
                    "'{}' product/category accounts for {:.0f}% of total revenue — "
                    "losing it would remove that share; diversification reduces the "
                    "exposure.".format(top_prod, top_share))
            # 'Bottom N' is only a meaningful insight if it's a genuine
            # minority subset — with 3 or fewer categories total, 'bottom 3'
            # trivially equals 100% of everything, which reads as a
            # nonsensical opportunity ('bottom 3 contribute only 100% of
            # revenue' when there IS no other 97%). Require at least 5
            # categories so the bottom slice excludes real top performers.
            if len(prod_perf) >= 5:
                bottom_n = 3
                bottom_share = prod_perf.iloc[-bottom_n:]["sum"].sum() / total_rev * 100
                opps.append(
                    "Bottom {} products/categories contribute only {:.0f}% of revenue — "
                    "review whether they justify their resource/shelf allocation".format(
                        bottom_n, bottom_share))

    # ── Profit Margin ──────────────────────────────────────
    if profit_col and profit_col in stats:
        st         = stats[profit_col]
        mean_profit= st.get("mean",0)
        neg_n      = int((df[profit_col].dropna()<0).sum()) if profit_col in df.columns else 0
        neg_pct    = round(neg_n/len(df)*100,1)

        if neg_n > 0:
            insights.append(build_insight(
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

    # ── Rep / Salesperson Performance (was detected but never analysed) ──
    if rep_col and rev_col and rev_col in df.columns:
        rep_perf = (df.groupby(rep_col)[rev_col]
                    .agg(["sum", "median", "count"]))
        rep_perf = rep_perf[rep_perf["count"] >= 3].sort_values("median", ascending=False)
        if len(rep_perf) >= 4:
            n_reps    = len(rep_perf)
            top_q     = rep_perf.head(max(1, n_reps // 4))
            bot_q     = rep_perf.tail(max(1, n_reps // 4))
            top_med   = float(top_q["median"].median())
            bot_med   = float(bot_q["median"].median())
            spread    = top_med / max(abs(bot_med), 1)
            # Revenue concentration among reps (Pareto): share from the top 20%.
            sorted_sum = rep_perf["sum"].sort_values(ascending=False)
            top20_n    = max(1, int(round(n_reps * 0.2)))
            top20_share = sorted_sum.head(top20_n).sum() / max(sorted_sum.sum(), 1) * 100
            # Quantified coaching opportunity: move the bottom quartile's records
            # to the team median.
            team_med  = float(df[rev_col].median())
            bot_recs  = int(bot_q["count"].sum())
            coach_val = max(0.0, (team_med - bot_med)) * bot_recs

            insights.append(build_insight(
                title="Rep Performance Spread: Top Quartile {:.1f}x the Bottom".format(spread),
                problem="Top-quartile reps post a median of {:,.0f} vs {:,.0f} for the "
                        "bottom quartile ({:.1f}x).".format(top_med, bot_med, spread),
                cause="Consistent spreads this wide usually reflect skill, territory "
                      "quality, or account mix differences — confirm which before acting.",
                evidence="{} reps with >=3 records. Top 20% of reps generate {:.0f}% of "
                         "revenue. Team median per record={:,.0f}.".format(
                             n_reps, top20_share, team_med),
                action="1. Shadow top-quartile reps — capture what they do differently  "
                       "2. Targeted coaching for the bottom quartile  "
                       "3. Check territory/account balance before attributing to skill",
                impact="Lifting the bottom quartile ({:,} records) to the team median is "
                       "worth about {:,.0f} in revenue — a coachable, measured upside.".format(
                           bot_recs, coach_val),
                severity="warning" if spread >= 2 else "info",
                category="rep_performance"
            ))
            if top20_share > 60:
                risks.append("Revenue concentrated in a few reps: top 20% drive {:.0f}% of "
                             "sales — key-person risk if any leave.".format(top20_share))
            if coach_val > 0:
                opps.append("Coach bottom-quartile reps to the team median: ~{:,.0f} upside "
                            "({:,} records).".format(coach_val, bot_recs))

    # ── Sales Cycle Length (derived from date columns) ──────
    _sales_cycle_insights(df, insights, findings, risks, opps)

    # ── Outcome-level analysis: rep win rates, cycle by outcome ─
    run_sales_performance(df, insights, findings, risks, opps)

    actions.extend([
        "Weekly revenue vs target review — per rep and per region",
        "Identify top 3 deals at risk in pipeline — intervention strategy",
        "Replicate top performer playbook — what do they do differently?",
        "Revenue concentration audit — reduce dependency on single customer/product",
        "Quarterly pricing review — ensure margins are healthy per product category",
    ])

    # ── Pipeline velocity ─────────────────────────────────
    # How long a deal takes to close is the number a sales leader
    # forecasts from, and the spread matters more than the average: a
    # pipeline averaging 60 days where a quarter take 150 cannot be
    # committed to a quarter-end.
    cycle_col = next((c for c in df.columns
                      if any(k in c.lower().replace("_", "")
                             for k in ["daysinpipeline", "daystoclose",
                                       "salescycle", "cycledays", "agedays",
                                       "daysopen"])
                      and pd.api.types.is_numeric_dtype(df[c])), None)
    if cycle_col:
        try:
            days = pd.to_numeric(df[cycle_col], errors="coerce").dropna()
            if len(days) >= 30:
                median_days = float(days.median())
                p90 = float(days.quantile(0.90))
                findings.append(
                    "Half of deals close within {:.0f} days; the slowest "
                    "tenth take {:.0f} or more.".format(median_days, p90))
                if p90 >= median_days * 2.5:
                    risks.append(
                        "The slowest tenth of the pipeline takes {:.0f} days "
                        "against a median of {:.0f}. Forecasting from the "
                        "average commits to dates that a quarter of the book "
                        "cannot meet.".format(p90, median_days))
                    insights.append(build_insight(
                        title="Pipeline Velocity: {:.0f} Days Median, "
                              "{:.0f} at P90".format(median_days, p90),
                        problem="The slowest tenth of deals takes {:.1f}x the "
                                "median time to close".format(
                                    p90 / max(median_days, 1)),
                        cause="Deal size, approval steps or a stage where "
                              "opportunities sit — the stage breakdown "
                              "narrows it",
                        evidence="Median {:.0f} days, P90 {:.0f} days across "
                                 "{:,} opportunities".format(
                                     median_days, p90, len(days)),
                        action="1. Age the open pipeline by stage  2. Find "
                               "the stage where deals stop moving  3. Set an "
                               "age limit that forces a decision  "
                               "4. Forecast from the P90, not the mean",
                        impact="Deals past {:.0f} days are the ones a "
                               "quarter-end commitment should exclude."
                               .format(p90),
                        severity="high", category="sales_velocity",
                    ))
        except Exception:
            logger.debug("sales velocity failed", exc_info=True)

    # ── Is the number resting on a few deals? ─────────────
    if rev_col:
        try:
            values = pd.to_numeric(df[rev_col], errors="coerce").dropna()
            values = values[values > 0].sort_values(ascending=False)
            if len(values) >= 30:
                top_decile = max(1, len(values) // 10)
                share = float(values.head(top_decile).sum()) / \
                    float(values.sum()) * 100
                if share >= 50:
                    findings.append(
                        "The largest {:.0f}% of deals carry {:.0f}% of "
                        "{}.".format(top_decile / len(values) * 100, share,
                                     rev_col))
                    risks.append(
                        "{:.0f}% of {} sits in {:,} deals. Losing two of "
                        "them moves the quarter more than the rest of the "
                        "pipeline combined.".format(
                            share, rev_col, top_decile))
                    insights.append(build_insight(
                        title="{:.0f}% of Value Sits in the Largest {:,} "
                              "Deals".format(share, top_decile),
                        problem="Revenue is concentrated in a small number "
                                "of opportunities",
                        cause="Enterprise mix, or a long tail of small deals "
                              "that costs as much to work as it returns",
                        evidence="Top {:,} of {:,} deals carry {:.0f}% of "
                                 "{}".format(top_decile, len(values), share,
                                             rev_col),
                        action="1. Name the top {:,} deals and their close "
                               "dates  2. Review each one weekly, not "
                               "monthly  3. Check whether the tail earns its "
                               "coverage cost  4. Re-weight territory "
                               "coverage accordingly".format(top_decile),
                        impact="{} sits in those deals.".format(
                            fmt(float(values.head(top_decile).sum()))),
                        severity="high", category="sales_concentration",
                    ))
        except Exception:
            logger.debug("sales concentration failed", exc_info=True)

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  GENERAL INSIGHTS
# ══════════════════════════════════════════════════════════

