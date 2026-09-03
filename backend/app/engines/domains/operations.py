"""
engines/domains/operations.py — Operations & process analytics.

Operations data used to detect as e-commerce (throughput and inventory
words overlap), so a plant's cycle-time and defect data was described in
the vocabulary of an online store.

Operations analysis is about the binding constraint: a process is only as
good as its slowest, least reliable step, and variability usually costs
more than the average does.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from app.engines.domains.base import Insight, build_insight
from app.engines.domains._common import (
    benchmark_note, binary_rate, find_measure, fmt, grouping_columns,
    segment_gap, variability,
)

logger = logging.getLogger(__name__)

# Coefficient of variation above this means the process is unpredictable
# enough that the average stops being a usable planning number.
UNSTABLE_CV = 50.0
# Sustained utilisation above this removes the slack needed to absorb
# demand variation — high is not the same as good.
UTILISATION_CEILING = 85.0
MIN_SITE_RATIO = 1.3


def _site_col(df: pd.DataFrame):
    for kw in ("plant", "site", "facility", "line", "machine", "location",
               "shift", "team", "warehouse", "depot"):
        for c in grouping_columns(df):
            if kw in c.lower().replace("_", ""):
                return c
    groups = grouping_columns(df)
    return groups[0] if groups else None


def _insights_operations(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights: List[Insight] = []

    cycle = find_measure(df, ["cycletime", "leadtime", "processtime",
                              "turnaround", "duration"])
    defects = find_measure(df, ["defectrate", "defects", "scraprate",
                                "rejectrate", "errorrate", "failurerate"])
    throughput = find_measure(df, ["throughput", "unitsproduced", "output",
                                   "volume", "quantity"])
    downtime = find_measure(df, ["downtime", "outage", "stoppage"])
    utilisation = find_measure(df, ["utilization", "utilisation", "oee",
                                    "capacityused"])
    turns = find_measure(df, ["inventoryturns", "inventoryturnover", "turns"])
    site = _site_col(df)

    otd_col = None
    for c in df.columns:
        n = c.lower().replace("_", "")
        if any(k in n for k in ("ontime", "delivered", "sla", "met")):
            if binary_rate(df[c]) is not None:
                otd_col = c
                break

    # ── Process stability ─────────────────────────────────
    if cycle:
        cv = variability(df[cycle])
        mean_cycle = float(pd.to_numeric(df[cycle], errors="coerce").mean())
        if cv is not None:
            findings.append(
                "'{}' averages {} with {:.0f}% variability.".format(
                    cycle, fmt(mean_cycle), cv))
            if cv > UNSTABLE_CV:
                risks.append(
                    "'{}' varies {:.0f}% around its mean — the average is "
                    "not a number the business can plan or promise "
                    "against.".format(cycle, cv))
                insights.append(build_insight(
                    title="'{}' Varies {:.0f}% Around Its Mean".format(
                        cycle, cv),
                    problem="Cycle time is unpredictable: {:.0f}% coefficient "
                            "of variation on a {} average".format(
                                cv, fmt(mean_cycle)),
                    cause="Mixed job types on one line, unplanned stoppages, "
                          "or a queue that builds and clears rather than "
                          "flowing",
                    evidence="'{}' mean {}, CV {:.0f}% across {:,} "
                             "records".format(cycle, fmt(mean_cycle), cv,
                                              len(df)),
                    action="1. Separate the measure by job type before "
                           "acting  2. Find the step with the widest spread  "
                           "3. Stabilise that step before optimising the "
                           "average",
                    impact="Predictability, not speed, is usually what "
                           "customers experience as reliability",
                    severity="high", category="ops_stability",
                ))

    # ── Quality ───────────────────────────────────────────
    if defects:
        try:
            raw = float(pd.to_numeric(df[defects], errors="coerce").mean())
            # A defect column arrives as a fraction or as a percentage;
            # reporting 0.03% when the truth is 3% understates by 100x.
            rate = raw * 100 if raw <= 1 else raw
            note = benchmark_note("operations", defects, rate)
            severity = ("critical" if rate > 5 else
                        "high" if rate > 3 else
                        "warning" if rate > 1 else "positive")
            findings.append(
                "Defect rate averages {:.2f}%.{}".format(
                    rate, " " + note if note else ""))
            if rate > 3:
                risks.append(
                    "Defect rate at {:.2f}% means roughly 1 in {:.0f} units "
                    "is reworked or scrapped.".format(rate, 100 / max(rate, .01)))
            insights.append(build_insight(
                title="Defect Rate at {:.2f}%".format(rate),
                problem="{:.2f}% of units fail quality".format(rate),
                cause="Process drift, material variation, or an inspection "
                      "step positioned too late to prevent the defect",
                evidence="Mean of '{}' across {:,} records.{}".format(
                    defects, len(df), " " + note if note else ""),
                action="1. Pareto the defects by type  2. Move inspection "
                       "upstream of the top cause  3. Track first-pass yield "
                       "as the headline measure",
                impact="Every point of defect rate is rework cost, delayed "
                       "delivery, and capacity spent twice",
                severity=severity, category="ops_quality",
            ))
        except Exception:
            logger.debug("ops defect analysis failed", exc_info=True)

    # ── Delivery reliability ──────────────────────────────
    if otd_col:
        rate = binary_rate(df[otd_col])
        if rate is not None:
            note = benchmark_note("operations", otd_col, rate)
            findings.append(
                "On-time performance is {:.1f}%.{}".format(
                    rate, " " + note if note else ""))
            if rate < 95:
                insights.append(build_insight(
                    title="On-Time Performance at {:.1f}%".format(rate),
                    problem="{:.1f}% of commitments are met on time".format(rate),
                    cause="Either the promise is set from an optimistic "
                          "average, or the constraint step is unstable",
                    evidence="'{}' across {:,} records.{}".format(
                        otd_col, len(df), " " + note if note else ""),
                    action="1. Set promises from the 85th percentile, not "
                           "the mean  2. Stabilise the constraint  "
                           "3. Re-measure before re-promising",
                    impact="{:.0f} in every 100 commitments currently "
                           "arrive late".format(100 - rate),
                    severity="high" if rate < 90 else "warning",
                    category="ops_delivery",
                ))

    # ── Utilisation ───────────────────────────────────────
    if utilisation:
        try:
            raw = float(pd.to_numeric(df[utilisation], errors="coerce").mean())
            util = raw * 100 if raw <= 1 else raw
            note = benchmark_note("operations", utilisation, util)
            findings.append(
                "Capacity utilisation averages {:.1f}%.{}".format(
                    util, " " + note if note else ""))
            if util > UTILISATION_CEILING:
                risks.append(
                    "Utilisation at {:.1f}% leaves no slack. Above roughly "
                    "{:.0f}%, queues grow non-linearly and any disruption "
                    "propagates.".format(util, UTILISATION_CEILING))
                insights.append(build_insight(
                    title="Utilisation at {:.1f}% — Above the Slack Line".format(
                        util),
                    problem="Sustained utilisation of {:.1f}% removes the "
                            "buffer needed to absorb variation".format(util),
                    cause="Capacity planned to the average rather than to "
                          "the variation around it",
                    evidence="Mean of '{}' is {:.1f}%.{}".format(
                        utilisation, util, " " + note if note else ""),
                    action="1. Plan to the demand peak, not the mean  "
                           "2. Hold a defined buffer on the constraint  "
                           "3. Re-check queue times after the change",
                    impact="High utilisation with no slack converts small "
                           "disruptions into missed delivery dates",
                    severity="high", category="ops_capacity",
                ))
            elif util < 50:
                opps.append(
                    "Utilisation at {:.1f}% suggests capacity is available "
                    "without new investment.".format(util))
        except Exception:
            logger.debug("ops utilisation failed", exc_info=True)

    # ── Site comparison ───────────────────────────────────
    if site:
        measure = cycle or defects or downtime or throughput
        if measure:
            gap = segment_gap(df, site, measure, agg="mean")
            if gap and gap["ratio"] and gap["ratio"] >= MIN_SITE_RATIO:
                findings.append(
                    "'{}' and '{}' differ {:.1f}x on '{}' — a practice gap, "
                    "not a capability gap, across {} sites.".format(
                        gap["best"], gap["worst"], gap["ratio"], measure,
                        gap["n_groups"]))
                opps.append(
                    "Whatever '{}' does differently on '{}' is already "
                    "working inside the business and can be copied.".format(
                        gap["best"] if measure == throughput else gap["worst"],
                        measure))
                insights.append(build_insight(
                    title="{:.1f}x Spread on '{}' Between Sites".format(
                        gap["ratio"], measure),
                    problem="'{}' at {} against '{}' at {}".format(
                        gap["best"], fmt(gap["best_val"]),
                        gap["worst"], fmt(gap["worst_val"])),
                    cause="Different local practice, equipment age or "
                          "scheduling discipline — same process on paper",
                    evidence="{} sites compared; best n={}, worst n={}".format(
                        gap["n_groups"], gap["n_best"], gap["n_worst"]),
                    action="1. Observe the better site's actual practice  "
                           "2. Document what differs  3. Pilot at the "
                           "weaker site before rolling out",
                    impact="Internal best practice needs no capital to "
                           "adopt and is already proven in this business",
                    severity="warning", category="ops_variance",
                ))

    if downtime:
        try:
            total_dt = float(pd.to_numeric(df[downtime], errors="coerce").sum())
            findings.append(
                "Recorded downtime totals {} across {:,} records.".format(
                    fmt(total_dt), len(df)))
        except Exception:
            logger.debug("ops downtime failed", exc_info=True)

    if turns:
        try:
            t = float(pd.to_numeric(df[turns], errors="coerce").mean())
            note = benchmark_note("operations", turns, t)
            findings.append("Inventory turns average {:.1f}.{}".format(
                t, " " + note if note else ""))
        except Exception:
            logger.debug("ops turns failed", exc_info=True)

    if not findings and not insights:
        logger.info("operations engine found no domain signal in %d columns",
                    len(df.columns))

    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}
