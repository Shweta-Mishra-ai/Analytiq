"""
core/engines/sales.py — Sales Performance domain engine.
Single responsibility: revenue, quota, margin, and rep performance insights.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from app.engines.domains.base import Insight, build_insight, col_stats
from app.engines.domains.sales_performance import (
    find_outcome_col,
    run_sales_performance,
)
from app.engines.industry_benchmarks import lookup_benchmark, format_benchmark_context

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


def _outcome_column(df: pd.DataFrame):
    """(column, won mask, lost mask) for the outcome column, or None.

    Delegates the detection to `sales_performance.find_outcome_col` so
    there is one definition of "was this deal won" in the codebase, and
    unpacks its tri-state series — True won, False lost, NaN still open —
    into the two masks the callers here need. Open opportunities appear in
    neither: a deal that has not been decided is not a loss, and counting
    it as one halves every win rate on a live pipeline.
    """
    found = find_outcome_col(df)
    if not found:
        return None
    col, outcome = found
    won  = outcome.eq(True).reindex(df.index, fill_value=False)
    lost = outcome.eq(False).reindex(df.index, fill_value=False)
    return col, won, lost


def _win_rate_insights(df: pd.DataFrame, rev_col, rep_col, insights: List,
                       findings: List, risks: List, opps: List) -> None:
    """Win rate — the first number anyone running a sales team asks for.

    The engine computed revenue distribution, cycle length and rep spread
    and never computed this, on files that carried a `deal_stage` column
    with "Closed Won" and "Closed Lost" in it.
    """
    found = _outcome_column(df)
    if not found:
        return
    col, won, lost = found
    decided = int(won.sum() + lost.sum())
    if decided < 30:
        return
    rate = won.sum() / decided * 100
    open_n = len(df) - decided

    value_note = ""
    if rev_col and rev_col in df.columns:
        won_value  = float(df.loc[won, rev_col].sum())
        lost_value = float(df.loc[lost, rev_col].sum())
        if won_value + lost_value > 0:
            value_rate = won_value / (won_value + lost_value) * 100
            # Where the two diverge, the team is winning a different size
            # of deal than it is losing — which is the finding, not the
            # count-based rate on its own.
            value_note = (" By value the rate is {:.0f}% ({:,.0f} won against "
                          "{:,.0f} lost), so the deals being won are {} than "
                          "the deals being lost.".format(
                              value_rate, won_value, lost_value,
                              "larger" if value_rate > rate + 3 else
                              "smaller" if value_rate < rate - 3 else
                              "about the same size"))

    rep_note = ""
    if rep_col and rep_col in df.columns:
        try:
            per_rep = (df.assign(_won=won, _decided=won | lost)
                         .groupby(rep_col)[["_won", "_decided"]].sum())
            per_rep = per_rep[per_rep["_decided"] >= 10]
            if len(per_rep) >= 3:
                per_rep["_rate"] = per_rep["_won"] / per_rep["_decided"] * 100
                best  = per_rep["_rate"].idxmax()
                worst = per_rep["_rate"].idxmin()
                spread = per_rep.loc[best, "_rate"] - per_rep.loc[worst, "_rate"]
                if spread >= 10:
                    rep_note = (" Across {:,} people with at least ten decided "
                                "deals the rate runs from {:.0f}% ({}) to "
                                "{:.0f}% ({}).".format(
                                    len(per_rep), per_rep.loc[worst, "_rate"],
                                    worst, per_rep.loc[best, "_rate"], best))
        except Exception:
            logger.warning("per-rep win rate failed", exc_info=True)

    # `run_sales_performance` states the rate as a finding when it can
    # test it per rep; adding a second line here put the same 60% in the
    # findings list twice under slightly different wording.
    if not rep_note:
        findings.append(
            "Win rate is {:.0f}% — {:,} of {:,} decided opportunities closed "
            "won, with {:,} still open.".format(
                rate, int(won.sum()), decided, open_n))

    insights.append(build_insight(
        title="Win Rate {:.0f}% on {:,} Decided Opportunities".format(
            rate, decided),
        problem="{:,} of {:,} decided opportunities closed won ({:.0f}%). The "
                "remaining {:,} rows are still open and are excluded from both "
                "sides of the ratio.".format(
                    int(won.sum()), decided, rate, open_n),
        cause="Win rate is an outcome of qualification, competitive position "
              "and pricing together. This data records the outcome, not which "
              "of the three moved it.",
        evidence="Read from '{}'. Won={:,}, lost={:,}, open={:,}.{}{}".format(
            col, int(won.sum()), int(lost.sum()), open_n, value_note, rep_note),
        action="1. Split the rate by segment and deal size — a single "
               "team-level figure hides where it is actually lost  "
               "2. Compare the stage that lost deals reached against won "
               "deals, to locate the drop  "
               "3. Track the rate against the same quarter last year rather "
               "than against a published range",
        impact="At the current rate every {:.1f} decided opportunities produce "
               "one win, so pipeline volume and win rate trade off directly "
               "against each other in any coverage calculation.".format(
                   100 / rate if rate else 0),
        severity="info", category="conversion",
    ))

    if rate < 20 and decided >= 100:
        risks.append(
            "Win rate of {:.0f}% means roughly {:.0f} opportunities are worked "
            "for each one closed — pipeline coverage has to carry that ratio "
            "before any target is credible.".format(rate, 100 / rate))


def _quota_attainment_insights(df: pd.DataFrame, rev_col, target_col, rep_col,
                               insights: List, findings: List, risks: List,
                               opps: List) -> None:
    """Attainment measured per person, against the quota as recorded.

    The previous version divided the mean of the revenue column by the
    mean of the quota column. On a normal export that is a per-deal
    amount over a per-period-per-rep target: 18,420 against 250,000,
    reported as "7% achievement, 93pp below target" and pushed to the top
    of the report as the critical finding. It was not a low attainment
    figure; it was not an attainment figure at all.

    Attainment only means anything per quota-holder: bookings summed for
    that person over the period the data covers, divided by their quota.
    Without something to group by, the two columns cannot be compared and
    the honest output is to say so.
    """
    if not (target_col and rev_col
            and target_col in df.columns and rev_col in df.columns):
        return
    quota = pd.to_numeric(df[target_col], errors="coerce")
    revenue = pd.to_numeric(df[rev_col], errors="coerce")
    if quota.dropna().empty or (quota.dropna() <= 0).all():
        return

    if not rep_col or rep_col not in df.columns:
        findings.append(
            "'{}' is present but there is no column identifying who each "
            "target belongs to, so attainment cannot be calculated. A "
            "per-row amount and a per-period target are not comparable "
            "figures.".format(target_col))
        return

    # Quota repeats on every row belonging to the same person, so it is
    # taken once per person rather than summed.
    work = pd.DataFrame({"_rep": df[rep_col], "_rev": revenue,
                         "_quota": quota}).dropna(subset=["_rep"])
    found = _outcome_column(df)
    basis = "all opportunities in the data"
    if found:
        _col, won, _lost = found
        if won.sum() >= 10:
            work = work[won.reindex(work.index, fill_value=False)]
            basis = "closed-won opportunities only"

    per_rep = work.groupby("_rep").agg(bookings=("_rev", "sum"),
                                       quota=("_quota", "median"),
                                       deals=("_rev", "size"))
    per_rep = per_rep[per_rep["quota"] > 0]
    if len(per_rep) < 3:
        return
    per_rep["attainment"] = per_rep["bookings"] / per_rep["quota"] * 100

    median_att = float(per_rep["attainment"].median())
    at_quota = int((per_rep["attainment"] >= 100).sum())
    share_at = at_quota / len(per_rep) * 100
    best = per_rep["attainment"].idxmax()
    worst = per_rep["attainment"].idxmin()

    span = ""
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    if date_cols:
        try:
            dates = df[date_cols[0]].dropna()
            if len(dates):
                months = max(1, round((dates.max() - dates.min()).days / 30.44))
                span = (" Bookings are summed over the {:,} months the data "
                        "covers; if the recorded quota is annual rather than "
                        "for that window, the attainment figures scale "
                        "accordingly.".format(months))
        except Exception:
            logger.debug("could not describe the period covered", exc_info=True)

    findings.append(
        "Median quota attainment is {:.0f}%, with {:,} of {:,} quota-holders "
        "at or above target.".format(median_att, at_quota, len(per_rep)))

    severity = ("critical" if share_at < 25 else
                "warning" if share_at < 50 else "positive")
    insights.append(build_insight(
        title="{:,} of {:,} Quota-Holders at Target; Median Attainment "
              "{:.0f}%".format(at_quota, len(per_rep), median_att),
        problem=("{:.0f}% of quota-holders reached target and the median "
                 "attainment is {:.0f}%.".format(share_at, median_att)),
        cause="Attainment below target is either a target-setting problem or "
              "a performance one, and the two need separating before either "
              "is acted on. This data cannot distinguish them.",
        evidence="Measured per '{}' on {}: bookings summed per person against "
                 "their own recorded quota. Range {:.0f}% ({}) to {:.0f}% "
                 "({}) across {:,} quota-holders.{}".format(
                     rep_col, basis, per_rep.loc[worst, "attainment"], worst,
                     per_rep.loc[best, "attainment"], best, len(per_rep), span),
        action="1. Compare this distribution against the same period last "
               "year before changing any target  "
               "2. Where most of the team misses, treat it as a "
               "target-setting question, not a coaching one  "
               "3. Check territory and account allocation against attainment "
               "before attributing the spread to people",
        impact="Closing the gap between the median holder and target across "
               "all {:,} quota-holders is worth about {:,.0f} in bookings, "
               "measured against their own quotas.".format(
                   len(per_rep),
                   max(0.0, float((per_rep["quota"] - per_rep["bookings"])
                                  .clip(lower=0).sum()))),
        severity=severity, category="target",
    ))

    if share_at < 40:
        risks.append(
            "Only {:,} of {:,} quota-holders are at target — a plan built on "
            "the current quota set is unlikely to be met without either "
            "changing the targets or changing the coverage.".format(
                at_quota, len(per_rep)))


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
    region_col = next((c for c in df.select_dtypes(include=["object", "string"]).columns
                       if any(k in c.lower() for k in ["region","territory","zone","area"])
                       and df[c].nunique()<=25), None)
    # "category" and "segment" are used for things that are not products.
    # `forecast_category` holds Commit / Best Case / Pipeline — a
    # confidence band — and was read as the product line, so the report
    # recommended reviewing "revenue by forecast_category for
    # concentration and whether the long tail justifies its resource".
    _NOT_A_PRODUCT = ("forecast", "risk", "priority", "age", "size", "tier",
                      "credit", "confidence", "probability", "stage",
                      "customer_segment", "lead")
    product_col= next((c for c in df.select_dtypes(include=["object", "string"]).columns
                       if any(k in c.lower() for k in ["product","category","segment"])
                       and not any(b in c.lower() for b in _NOT_A_PRODUCT)
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

        # The title used to be "Revenue Overview: Mean 18420 | Median
        # 13450 | Range 1314-186265" — a spreadsheet header rather than a
        # finding, and the stated problem was "Revenue distribution
        # analysis", which is a description of the activity and not of
        # anything wrong.
        top_quartile_share = 0.0
        if rev_col in df.columns:
            try:
                s_rev = pd.to_numeric(df[rev_col], errors="coerce").dropna()
                if len(s_rev) >= 8 and s_rev.sum() > 0:
                    top_quartile_share = float(
                        s_rev.nlargest(max(1, len(s_rev) // 4)).sum()
                        / s_rev.sum() * 100)
            except Exception:
                logger.debug("top-quartile share failed", exc_info=True)

        insights.append(build_insight(
            title=("Top Quarter of Deals Carries {:.0f}% of Revenue".format(
                       top_quartile_share) if top_quartile_share >= 40 else
                   "Revenue Is Spread Evenly Across Deal Sizes"),
            problem=("The largest 25% of transactions account for {:.0f}% of "
                     "revenue, so the total moves with a small number of "
                     "deals.".format(top_quartile_share)
                     if top_quartile_share >= 40 else
                     "No small group of transactions dominates the total; "
                     "the largest 25% account for {:.0f}% of "
                     "revenue.".format(top_quartile_share)),
            cause="Skewness of {:.1f} on {} — {}".format(
                skew, rev_col,
                "a long right tail, which is normal in a deal-based business "
                "and matters mainly for how the average is read"
                if skew > 1 else
                "a broadly symmetric distribution"),
            evidence="Mean={:,.0f} against a median of {:,.0f} ({:.0f}% apart), "
                     "so the median is the fairer description of a typical "
                     "transaction. Upper quartile begins at {:,.0f}.".format(
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

    # ── Win rate and quota attainment ───────────────────────
    try:
        _win_rate_insights(df, rev_col, rep_col, insights, findings, risks, opps)
    except Exception:
        logger.warning("win rate analysis failed", exc_info=True)
    try:
        _quota_attainment_insights(df, rev_col, target_col, rep_col,
                                   insights, findings, risks, opps)
    except Exception:
        logger.warning("quota attainment analysis failed", exc_info=True)

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
                "'{}' contributes {:.0f}% of revenue across {} groups in "
                "'{}' — losing it would remove that share.".format(
                    top_r, top_share, len(reg_perf), region_col)
                if top_share > 50 else
                "Revenue is spread across {} groups in '{}', the largest "
                "holding {:.0f}%.".format(len(reg_perf), region_col, top_share))

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

    # These five ran on every sales file regardless of what was in it. On
    # an opportunity export with no product, customer or margin column,
    # the report still closed with "quarterly pricing review — ensure
    # margins are healthy per product category" and "reduce dependency on
    # a single customer/product". Neither could be acted on from this
    # data, and a recommendation the data cannot support is the thing a
    # reader spots first. Each is now conditional on the column it needs.
    if target_col and rep_col:
        actions.append(
            "Review attainment per {} against target on the same cadence the "
            "targets are set — the distribution across the team, not the "
            "average.".format(rep_col))
    if _outcome_column(df):
        actions.append(
            "Track win rate and the stage lost deals reached together; the "
            "rate on its own does not say where deals are lost.")
    if rep_col:
        actions.append(
            "Profile what separates the top quartile of {} from the bottom "
            "on this data before attributing the gap to capability.".format(
                rep_col))
    if product_col:
        actions.append(
            "Review revenue by {} for concentration, and whether the "
            "long tail justifies the resource it takes.".format(product_col))
    if profit_col:
        actions.append(
            "Review pricing where {} is thin or negative, by the segments "
            "available in this data.".format(profit_col))
    if not actions:
        actions.append(
            "Add the outcome, owner and target columns to this export — win "
            "rate, attainment and rep performance cannot be measured "
            "without them, and they are the questions this data is closest "
            "to answering.")

    return {"findings":findings, "risks":risks, "opportunities":opps,
            "actions":actions, "insights":insights}


# ══════════════════════════════════════════════════════════
#  GENERAL INSIGHTS
# ══════════════════════════════════════════════════════════

