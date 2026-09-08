"""
engines/domains/logistics.py — supply chain, freight and delivery.

Distinct from `operations`, which is about making a thing: this is about
moving it. The measures barely overlap — a plant cares about cycle time
and first-pass yield, a network cares about on-time delivery, cost per
shipment and where the days actually go.

The three questions a logistics director opens a review with:

**Are we on time, and if not, where does the delay live?** An overall
on-time rate is a scoreboard. The lane, carrier or hub carrying the
failure is the thing you can renegotiate.

**What does a shipment cost, and why does it vary?** Cost per shipment
that swings three-fold across lanes is either mix or a contract nobody
has revisited.

**Is the variability worse than the average?** A network averaging three
days with a two-day spread is a network nobody can plan around, and it
is invisible to any report that prints only the mean. Promising on the
average is how a customer-facing date gets missed half the time, so the
engine reports the spread beside the mean everywhere it matters.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from app.engines.domains.base import Insight, build_insight
from app.engines.domains._common import (
    benchmark_note, binary_rate, find_col, find_measure, fmt,
    grouping_columns, segment_gap, variability,
)

logger = logging.getLogger(__name__)

# Below this, on-time performance is the headline problem whatever else
# the data says.
ON_TIME_TARGET = 95.0
# A lane costing this many times the median is not "more expensive" — it
# is a different commercial arrangement.
COST_OUTLIER_RATIO = 2.0
# Transit-time spread above this fraction of the mean means the promise
# date cannot be met reliably, whatever the average says.
UNRELIABLE_CV = 40.0
# One carrier above this share of volume is a single point of failure.
CARRIER_CONCENTRATION = 55.0


def _lane_col(df: pd.DataFrame):
    """The thing a logistics team renegotiates: a carrier, a lane, a hub."""
    for kw in ("carrier", "lane", "route", "hub", "warehouse", "origin",
               "destination", "region", "mode", "shipmenttype", "depot"):
        for c in grouping_columns(df):
            if kw in c.lower().replace("_", "").replace(" ", ""):
                return c
    groups = grouping_columns(df)
    return groups[0] if groups else None


def _insights_logistics(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights: List[Insight] = []

    transit = find_measure(df, ["transittime", "transitdays", "deliverydays",
                                "leadtime", "shippingdays", "daysintransit",
                                "deliverytime"])
    cost = find_measure(df, ["freightcost", "shippingcost", "costpershipment",
                             "transportcost", "cost", "charge"],
                        exclude=["costcode"])
    weight = find_measure(df, ["weight", "volume", "cbm", "pallets", "units"])
    distance = find_measure(df, ["distance", "km", "miles", "mileage"])
    lane = _lane_col(df)
    on_time = find_col(df, ["ontime", "ontimedelivery", "delivered",
                            "onschedule", "sla", "met"])
    delayed = find_col(df, ["delayed", "late", "missed", "failed"])

    # ── On time, or not ───────────────────────────────────
    rate = None
    if on_time:
        rate = binary_rate(df[on_time])
    elif delayed:
        late = binary_rate(df[delayed])
        rate = None if late is None else round(100 - late, 2)

    if rate is not None:
        severity = ("critical" if rate < 85 else
                    "high" if rate < ON_TIME_TARGET else "positive")
        findings.append(
            "On-time delivery is {:.1f}% across {:,} shipments.".format(
                rate, len(df)))
        missed = int(len(df) * (100 - rate) / 100)
        note = benchmark_note("logistics", on_time or delayed or "", rate)
        if rate < ON_TIME_TARGET:
            risks.append(
                "{:.1f}% on-time means about {:,} of {:,} shipments arrive "
                "late. Each one is a customer-facing promise that was made "
                "and missed.".format(rate, missed, len(df)))
        insights.append(build_insight(
            title="On-Time Delivery at {:.1f}%".format(rate),
            problem="{:,} of {:,} shipments miss their promised date".format(
                missed, len(df)),
            cause="Carrier performance, lane congestion, or a promise date "
                  "set from an average rather than from the spread",
            evidence="{:,} shipments{}".format(
                len(df), (" — " + note) if note else ""),
            action="1. Break the miss rate down by carrier and lane  "
                   "2. Separate the lanes that are slow from the ones that "
                   "are unpredictable  3. Renegotiate or re-promise on the "
                   "worst two  4. Re-measure over a full cycle",
            impact="Every point of on-time rate is about {:,} shipments "
                   "across this volume.".format(max(1, len(df) // 100)),
            severity=severity, category="logistics_service",
        ))

    # ── A network you cannot plan around ──────────────────
    if transit:
        cv = variability(df[transit])
        try:
            mean_days = float(df[transit].mean())
            p95 = float(df[transit].quantile(0.95))
        except Exception:
            mean_days = p95 = None
        if cv is not None and mean_days is not None and cv >= UNRELIABLE_CV:
            findings.append(
                "{} averages {} but the 95th percentile is {} — the spread "
                "is {:.0f}% of the mean.".format(
                    transit, fmt(mean_days), fmt(p95), cv))
            risks.append(
                "Promising on the average {} of {} means missing roughly one "
                "delivery in two. The number a customer should be given is "
                "closer to {}.".format(transit, fmt(mean_days), fmt(p95)))
            insights.append(build_insight(
                title="Transit Time Is Unpredictable: {} Mean, {} at P95"
                      .format(fmt(mean_days), fmt(p95)),
                problem="{} varies by {:.0f}% around its mean, so the "
                        "average is not a date anyone can commit to".format(
                            transit, cv),
                cause="Mixed lanes or modes inside one number, or a hub "
                      "whose dwell time swings — the lane breakdown "
                      "separates them",
                evidence="Mean {}, P95 {}, coefficient of variation "
                         "{:.0f}% across {:,} shipments".format(
                             fmt(mean_days), fmt(p95), cv, len(df)),
                action="1. Quote the 95th percentile, not the mean  "
                       "2. Split the measure by lane and find which one "
                       "carries the spread  3. Fix or reroute that lane  "
                       "4. Re-quote once the spread narrows",
                impact="Quoting {} instead of {} turns roughly half the "
                       "late deliveries into on-time ones without moving a "
                       "single box faster.".format(fmt(p95), fmt(mean_days)),
                severity="high", category="logistics_reliability",
            ))

    # ── Where the cost sits ───────────────────────────────
    if lane and cost:
        gap = segment_gap(df, lane, cost, agg="mean")
        if gap and gap["ratio"] and gap["ratio"] >= COST_OUTLIER_RATIO:
            findings.append(
                "'{}' costs {} per shipment against '{}' at {} — {:.1f}x "
                "across {} {}s.".format(
                    gap["best"], fmt(gap["best_val"]), gap["worst"],
                    fmt(gap["worst_val"]), gap["ratio"], gap["n_groups"],
                    lane))
            opps.append(
                "The {:.1f}x cost spread between '{}' and '{}' is either "
                "mix or a contract that has not been revisited. Both are "
                "answerable this quarter.".format(
                    gap["ratio"], gap["worst"], gap["best"]))
            insights.append(build_insight(
                title="Cost Spread: '{}' Costs {:.1f}x '{}'".format(
                    gap["best"], gap["ratio"], gap["worst"]),
                problem="Average {} runs {:.1f}x higher on '{}' than on "
                        "'{}'".format(cost, gap["ratio"], gap["best"],
                                      gap["worst"]),
                cause="Distance and weight explain some of it; the rest is "
                      "commercial terms, and the two need separating before "
                      "anyone renegotiates",
                evidence="{:,} shipments on '{}' against {:,} on '{}'".format(
                    gap["n_best"], gap["best"], gap["n_worst"], gap["worst"]),
                action="1. Normalise cost per {} before comparing  "
                       "2. Re-tender the expensive lane  3. Move a slice of "
                       "volume and measure the landed difference  "
                       "4. Hold the comparison for one full quarter".format(
                           weight or distance or "shipment"),
                impact="Closing half the gap on '{}' is about {} per "
                       "shipment across {:,} of them.".format(
                           gap["best"], fmt((gap["best_val"] -
                                             gap["worst_val"]) / 2),
                           gap["n_best"]),
                severity="high", category="logistics_cost",
            ))

    # ── One carrier holding the network ───────────────────
    if lane:
        # Share of SHIPMENTS, not share of spend. The exposure being
        # described is "if this carrier stops, how much of the book
        # stops", and that is a count of consignments — weighting it by
        # cost would understate a cheap carrier moving most of the volume.
        try:
            counts = df[lane].value_counts(dropna=True)
            share = (float(counts.iloc[0]) / float(counts.sum()) * 100
                     if len(counts) > 1 else None)
        except Exception:
            logger.debug("logistics concentration failed", exc_info=True)
            share = None
        if share and share >= CARRIER_CONCENTRATION:
            findings.append(
                "{:.0f}% of the book sits with a single {}.".format(
                    share, lane))
            risks.append(
                "One {} carrying {:.0f}% of volume is a single point of "
                "failure. A service interruption there is a network "
                "interruption.".format(lane, share))
            insights.append(build_insight(
                title="{:.0f}% of Volume Sits With One {}".format(
                    share, lane.replace("_", " ").title()),
                problem="A single {} carries {:.0f}% of the book".format(
                    lane, share),
                cause="Consolidation for rate, usually — the saving is real "
                      "and so is the exposure",
                evidence="{:.0f}% concentration across {:,} records".format(
                    share, len(df)),
                action="1. Price a second carrier on the top two lanes  "
                       "2. Move enough volume to keep the relationship live  "
                       "3. Agree a failover in the contract  4. Review the "
                       "split each quarter",
                impact="A week of disruption at that share touches roughly "
                       "{:,} shipments.".format(int(len(df) * share / 100 / 52)),
                severity="warning", category="logistics_risk",
            ))

    if distance and cost:
        actions.append(
            "'{}' and '{}' are both present — cost per {} is the comparable "
            "measure across lanes, and worth adding as a derived column."
            .format(cost, distance, distance))

    if not findings and not insights:
        logger.info("logistics engine found no domain signal in %d columns",
                    len(df.columns))

    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}
