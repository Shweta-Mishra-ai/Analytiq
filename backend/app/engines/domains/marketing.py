"""
engines/domains/marketing.py — Marketing & campaign analytics.

Marketing was detectable long before it was analysable: `detect_domain`
scored a marketing dataset at 0.91 and then the dispatch chain's `else`
handed it to the general engine, so the report carried a "marketing"
heading over generic column statistics. This is the engine that heading
always implied.

The questions a CMO actually asks, in the order they ask them: what did
the spend return, which channel is carrying it, where is budget being
wasted, and where does the funnel leak.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from app.engines.domains.base import Insight, build_insight
from app.engines.domains._common import (
    benchmark_note, concentration, find_measure, fmt, grouping_columns,
    segment_gap,
)

logger = logging.getLogger(__name__)

# A channel costing this many times the median per acquisition is not
# "underperforming" — it is spending money the other channels could use.
CPA_OUTLIER_RATIO = 2.0
# Below this, a spend gap between channels is a rounding difference.
MIN_EFFICIENCY_RATIO = 1.5
# One channel above this share of spend is a concentration risk worth
# naming, whether or not it is performing.
CONCENTRATION_PCT = 60.0


def _channel_col(df: pd.DataFrame):
    """The dimension a marketer reallocates budget across."""
    for kw in ("channel", "campaign", "medium", "source", "adgroup",
               "platform", "placement"):
        for c in grouping_columns(df):
            if kw in c.lower().replace("_", ""):
                return c
    groups = grouping_columns(df)
    return groups[0] if groups else None


def _insights_marketing(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights: List[Insight] = []

    spend = find_measure(df, ["spend", "cost", "budget", "investment"],
                         exclude=["costper", "cpa", "cpc", "cpm"])
    revenue = find_measure(df, ["revenue", "sales", "conversionvalue"])
    conversions = find_measure(df, ["conversions", "conversion", "purchases",
                                    "leads", "signups", "acquisitions"])
    clicks = find_measure(df, ["clicks", "click"], exclude=["clickrate", "ctr"])
    impressions = find_measure(df, ["impressions", "impression", "reach"])
    ctr_col = find_measure(df, ["ctr", "clickthroughrate", "clickrate"])
    channel = _channel_col(df)

    # ── Return on spend ───────────────────────────────────
    if spend and revenue:
        total_spend = float(df[spend].sum())
        total_rev = float(df[revenue].sum())
        if total_spend > 0:
            roas = total_rev / total_spend
            severity = ("critical" if roas < 1 else
                        "warning" if roas < 2 else "positive")
            verdict = ("returns less than it costs" if roas < 1
                       else "roughly breaks even" if roas < 1.5
                       else "returns {:.1f}x".format(roas))
            findings.append(
                "Total spend of {} returned {} — {:.2f}x on spend.".format(
                    fmt(total_spend), fmt(total_rev), roas))
            if roas < 1:
                risks.append(
                    "Marketing spend is running at a loss overall "
                    "({:.2f}x return). Every additional unit of budget "
                    "destroys value until the mix changes.".format(roas))
            insights.append(build_insight(
                title="Return on Spend: {:.2f}x".format(roas),
                problem="Marketing {} at {:.2f}x return on {} of spend".format(
                    verdict, roas, fmt(total_spend)),
                cause="Channel mix, audience targeting, or creative "
                      "performance — the channel breakdown below narrows it",
                evidence="Spend {} against revenue {} across {:,} rows".format(
                    fmt(total_spend), fmt(total_rev), len(df)),
                action="1. Rank channels by return  2. Cut or fix the "
                       "bottom quartile  3. Shift that budget to the top "
                       "two  4. Re-measure after one full cycle",
                impact="A 0.5x improvement on {} of spend is {} of "
                       "additional revenue".format(
                           fmt(total_spend), fmt(total_spend * 0.5)),
                severity=severity, category="marketing_efficiency",
            ))

    # ── Which channel carries the result ──────────────────
    if channel and (revenue or conversions):
        measure = revenue or conversions
        gap = segment_gap(df, channel, measure, agg="sum")
        if gap and gap["ratio"] and gap["ratio"] >= MIN_EFFICIENCY_RATIO:
            findings.append(
                "'{}' delivers {} of {} against '{}' at {} — a {:.1f}x "
                "spread across {} channels.".format(
                    gap["best"], fmt(gap["best_val"]), measure,
                    gap["worst"], fmt(gap["worst_val"]), gap["ratio"],
                    gap["n_groups"]))
            opps.append(
                "Reallocating budget from '{}' toward '{}' targets a "
                "{:.1f}x efficiency difference already visible in the "
                "data.".format(gap["worst"], gap["best"], gap["ratio"]))
            insights.append(build_insight(
                title="Channel Spread: '{}' outperforms '{}' by {:.1f}x".format(
                    gap["best"], gap["worst"], gap["ratio"]),
                problem="{} varies {:.1f}x across {} channels".format(
                    measure, gap["ratio"], gap["n_groups"]),
                cause="Audience fit, creative quality or bid strategy "
                      "differs by channel — not usually budget size alone",
                evidence="'{}' {} (n={}) vs '{}' {} (n={})".format(
                    gap["best"], fmt(gap["best_val"]), gap["n_best"],
                    gap["worst"], fmt(gap["worst_val"]), gap["n_worst"]),
                action="1. Confirm the gap holds after controlling for "
                       "spend  2. Move a test tranche of budget  "
                       "3. Hold the rest until the test reads",
                impact="Closing half the gap on '{}' is worth about {}".format(
                    gap["worst"], fmt(gap["gap"] * 0.5)),
                severity="high", category="marketing_channel",
            ))

    # ── Wasted spend: cost per acquisition outliers ───────
    if channel and spend and conversions:
        try:
            grouped = df.groupby(channel).agg(
                _spend=(spend, "sum"), _conv=(conversions, "sum"))
            grouped = grouped[grouped["_conv"] > 0]
            if len(grouped) >= 2:
                grouped["_cpa"] = grouped["_spend"] / grouped["_conv"]
                median_cpa = float(grouped["_cpa"].median())
                worst = grouped["_cpa"].idxmax()
                worst_cpa = float(grouped["_cpa"].max())
                if median_cpa > 0 and worst_cpa >= median_cpa * CPA_OUTLIER_RATIO:
                    wasted = float(grouped.loc[worst, "_spend"]) * (
                        1 - median_cpa / worst_cpa)
                    risks.append(
                        "'{}' acquires at {} against a median of {} — "
                        "roughly {} of its spend buys nothing the median "
                        "channel would not have bought cheaper.".format(
                            worst, fmt(worst_cpa), fmt(median_cpa),
                            fmt(wasted)))
                    actions.append(
                        "Cap or restructure '{}' — at median efficiency the "
                        "same acquisitions cost {} less.".format(
                            worst, fmt(wasted)))
                    insights.append(build_insight(
                        title="'{}' Costs {:.1f}x the Median to Acquire".format(
                            worst, worst_cpa / median_cpa),
                        problem="'{}' pays {} per acquisition against a "
                                "{} median".format(worst, fmt(worst_cpa),
                                                   fmt(median_cpa)),
                        cause="Audience saturation, weak creative, or bidding "
                              "into a more competitive auction than the "
                              "channel can justify",
                        evidence="{} spend, {:,.0f} conversions, {} CPA "
                                 "vs median {}".format(
                                     fmt(float(grouped.loc[worst, "_spend"])),
                                     float(grouped.loc[worst, "_conv"]),
                                     fmt(worst_cpa), fmt(median_cpa)),
                        action="1. Pause the worst-performing segments "
                               "within '{}'  2. Re-test with a lower bid "
                               "cap  3. Reallocate if it does not reach "
                               "median within a cycle".format(worst),
                        impact="About {} of recoverable spend".format(
                            fmt(wasted)),
                        severity="critical", category="marketing_waste",
                    ))
        except Exception:
            logger.debug("marketing CPA analysis failed", exc_info=True)

    # ── Funnel leakage ────────────────────────────────────
    if impressions and clicks:
        try:
            imp_total = float(df[impressions].sum())
            clk_total = float(df[clicks].sum())
            if imp_total > 0:
                ctr = clk_total / imp_total * 100
                note = benchmark_note("marketing", ctr_col or "ctr", ctr)
                findings.append(
                    "Click-through rate is {:.2f}% across {} impressions."
                    "{}".format(ctr, fmt(imp_total),
                                " " + note if note else ""))
                if ctr < 1.0:
                    insights.append(build_insight(
                        title="Click-Through Rate at {:.2f}%".format(ctr),
                        problem="Only {:.2f}% of {} impressions produced a "
                                "click".format(ctr, fmt(imp_total)),
                        cause="Creative or targeting mismatch — impressions "
                              "are being bought from an audience the message "
                              "does not land with",
                        evidence="{} clicks from {} impressions{}".format(
                            fmt(clk_total), fmt(imp_total),
                            ". " + note if note else ""),
                        action="1. Split-test creative on the largest "
                               "channel  2. Tighten audience definition  "
                               "3. Re-measure before adding budget",
                        impact="Doubling CTR at current spend doubles the "
                               "top of the funnel without new budget",
                        severity="high", category="marketing_funnel",
                    ))
        except Exception:
            logger.debug("marketing funnel analysis failed", exc_info=True)

    if clicks and conversions:
        try:
            clk_total = float(df[clicks].sum())
            cnv_total = float(df[conversions].sum())
            if clk_total > 0:
                cvr = cnv_total / clk_total * 100
                findings.append(
                    "{:.2f}% of clicks convert ({} of {}).".format(
                        cvr, fmt(cnv_total), fmt(clk_total)))
                if cvr < 2.0:
                    risks.append(
                        "Post-click conversion is {:.2f}% — traffic is being "
                        "paid for and then lost on the landing "
                        "experience.".format(cvr))
        except Exception:
            logger.debug("marketing conversion analysis failed", exc_info=True)

    # ── Concentration risk ────────────────────────────────
    if channel and spend:
        conc = concentration(df, channel, spend, top_n=1)
        if conc and conc["top_share"] >= CONCENTRATION_PCT and conc["n_groups"] > 1:
            risks.append(
                "'{}' carries {:.0f}% of total spend across {} channels — "
                "performance of the whole programme depends on one "
                "channel.".format(conc["top_names"][0], conc["top_share"],
                                  conc["n_groups"]))
            insights.append(build_insight(
                title="{:.0f}% of Spend Sits in One Channel".format(
                    conc["top_share"]),
                problem="'{}' holds {:.0f}% of spend".format(
                    conc["top_names"][0], conc["top_share"]),
                cause="Historic performance, or the path of least "
                      "resistance in budget planning",
                evidence="{} of {} total across {} channels".format(
                    conc["top_names"][0], fmt(conc["total"]),
                    conc["n_groups"]),
                action="1. Model the revenue impact of a 20% cut to this "
                       "channel  2. Fund a second channel to a testable "
                       "level  3. Set a maximum concentration policy",
                impact="An auction, policy or platform change in one "
                       "channel currently moves the whole programme",
                severity="warning", category="marketing_risk",
            ))

    if not findings and not insights:
        logger.info("marketing engine found no domain signal in %d columns",
                    len(df.columns))

    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}
