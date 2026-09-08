"""
engines/domains/energy.py — consumption, generation and emissions.

Energy data is the one domain here where the shape of demand matters more
than its total. A site using 100 MWh evenly and a site using 100 MWh in
four hours a day pay very different bills and need very different fixes,
and any report that prints only a total treats them as identical.

What an energy or sustainability manager asks:

**Where is the load, and is it peaky?** Peak against average is the
number that drives capacity charges. A peak-to-average ratio of four is
a tariff conversation before it is an efficiency one.

**Which site or asset is inefficient once size is taken out?**
Consumption per unit of floor area or output is the comparable measure;
raw consumption mostly ranks sites by how big they are.

**What is the carbon intensity, and is it falling?** Emissions per unit
of energy separates a cleaner supply from simply using less.

**Is anything running when nothing is happening?** Baseload that does not
drop overnight or at weekends is usually the cheapest saving available,
and it is invisible in a monthly total.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from app.engines.domains.base import Insight, build_insight
from app.engines.domains._common import (
    benchmark_note, find_measure, fmt, grouping_columns, segment_gap,
    variability,
)

logger = logging.getLogger(__name__)

# Peak this many times the average is a capacity-charge problem whatever
# the total consumption says.
PEAKY_RATIO = 2.5
# A site this much above the fleet median, per unit of size, is an
# efficiency outlier rather than simply a larger site.
INTENSITY_OUTLIER_RATIO = 1.5
# Overnight or minimum load above this share of the average is equipment
# left running rather than genuine base demand.
BASELOAD_SHARE = 40.0


def _site_col(df: pd.DataFrame):
    for kw in ("site", "meter", "building", "facility", "plant", "asset",
               "location", "region", "zone", "circuit"):
        for c in grouping_columns(df):
            if kw in c.lower().replace("_", "").replace(" ", ""):
                return c
    groups = grouping_columns(df)
    return groups[0] if groups else None


def _insights_energy(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights: List[Insight] = []

    consumption = find_measure(df, ["consumption", "kwh", "mwh", "energyused",
                                    "usage", "demand", "load", "units"],
                               exclude=["costperkwh"])
    demand = find_measure(df, ["peakdemand", "maxdemand", "peakload", "kw",
                               "peak"])
    emissions = find_measure(df, ["emissions", "co2", "carbon", "ghg",
                                  "co2e"])
    cost = find_measure(df, ["energycost", "billamount", "cost", "spend",
                             "charge"])
    size = find_measure(df, ["floorarea", "sqft", "sqm", "area", "output",
                             "production", "occupants", "headcount"])
    renewable = find_measure(df, ["renewable", "solar", "wind", "greenenergy"])
    site = _site_col(df)

    # ── Is the load peaky? ────────────────────────────────
    if consumption:
        try:
            series = pd.to_numeric(df[consumption], errors="coerce").dropna()
            if len(series) >= 30:
                mean_load = float(series.mean())
                peak = float(series.quantile(0.99))
                floor = float(series.quantile(0.05))
                ratio = peak / mean_load if mean_load > 0 else None

                if ratio and ratio >= PEAKY_RATIO:
                    findings.append(
                        "Peak {} is {} against an average of {} — {:.1f}x, "
                        "and capacity charges are set by the peak, not the "
                        "average.".format(
                            consumption, fmt(peak), fmt(mean_load), ratio))
                    opps.append(
                        "A {:.1f}x peak-to-average ratio is a tariff and "
                        "load-shifting opportunity before it is an "
                        "efficiency one — the same energy moved to a "
                        "different hour costs less.".format(ratio))
                    insights.append(build_insight(
                        title="Peak Load Is {:.1f}x Average".format(ratio),
                        problem="Peak {} of {} against an average of "
                                "{}".format(consumption, fmt(peak),
                                            fmt(mean_load)),
                        cause="Simultaneous start-up, a single large asset, "
                              "or no load management — a half-hourly profile "
                              "identifies which within a week",
                        evidence="99th percentile {} against mean {} across "
                                 "{:,} readings".format(
                                     fmt(peak), fmt(mean_load), len(series)),
                        action="1. Pull the half-hourly profile for the "
                               "worst days  2. Identify what starts at the "
                               "peak  3. Stagger or shift it  4. Re-check "
                               "the capacity charge on the next bill",
                        impact="Capacity charges scale with the peak, so a "
                               "{:.0f}% peak reduction is roughly the same "
                               "reduction in that part of the bill.".format(
                                   (1 - 1 / ratio) * 50),
                        severity="high", category="energy_demand",
                    ))

                # Baseload — what runs when nothing is happening.
                if mean_load > 0:
                    base_share = floor / mean_load * 100
                    if base_share >= BASELOAD_SHARE:
                        findings.append(
                            "Minimum {} sits at {:.0f}% of the average — "
                            "that share runs whether or not anything is "
                            "happening.".format(consumption, base_share))
                        opps.append(
                            "Baseload at {:.0f}% of average is usually the "
                            "cheapest saving available: it is equipment left "
                            "running, and it is found by walking the site "
                            "out of hours rather than by analysis.".format(
                                base_share))
                        insights.append(build_insight(
                            title="Baseload Is {:.0f}% of Average "
                                  "Consumption".format(base_share),
                            problem="The floor of consumption is {} against "
                                    "an average of {}".format(
                                        fmt(floor), fmt(mean_load)),
                            cause="Equipment running out of hours — HVAC "
                                  "schedules, compressors, lighting or IT "
                                  "load that never steps down",
                            evidence="5th percentile {} against mean {} "
                                     "across {:,} readings".format(
                                         fmt(floor), fmt(mean_load),
                                         len(series)),
                            action="1. Walk the site outside operating hours  "
                                   "2. List what is running and why  "
                                   "3. Put schedules on everything that does "
                                   "not need to run  4. Re-measure the "
                                   "overnight floor after two weeks",
                            impact="Every point off the baseload is a "
                                   "permanent reduction, not a behavioural "
                                   "one that drifts back.",
                            severity="warning", category="energy_efficiency",
                        ))
        except Exception:
            logger.debug("energy load profile failed", exc_info=True)

    # ── Efficiency, normalised for size ───────────────────
    if site and consumption and size:
        try:
            work = df[(pd.to_numeric(df[size], errors="coerce") > 0)].copy()
            work["_intensity"] = (pd.to_numeric(work[consumption],
                                                errors="coerce") /
                                  pd.to_numeric(work[size], errors="coerce"))
            gap = segment_gap(work, site, "_intensity", agg="mean")
            if gap and gap["ratio"] and gap["ratio"] >= INTENSITY_OUTLIER_RATIO:
                findings.append(
                    "'{}' uses {} per unit of {} against '{}' at {} — "
                    "{:.1f}x, with size already divided out.".format(
                        gap["best"], fmt(gap["best_val"]), size, gap["worst"],
                        fmt(gap["worst_val"]), gap["ratio"]))
                insights.append(build_insight(
                    title="'{}' Uses {:.1f}x the Energy per Unit of {}"
                          .format(gap["best"], gap["ratio"], size),
                    problem="Energy intensity runs {:.1f}x higher at '{}' "
                            "than at '{}'".format(
                                gap["ratio"], gap["best"], gap["worst"]),
                    cause="Plant age, control settings or operating hours — "
                          "the comparison holds because consumption is "
                          "already divided by {}".format(size),
                    evidence="{:,} readings at '{}' against {:,} at "
                             "'{}'".format(gap["n_best"], gap["best"],
                                           gap["n_worst"], gap["worst"]),
                    action="1. Confirm the two sites are comparable in use  "
                           "2. Compare control settings and operating hours  "
                           "3. Apply the better site's settings  "
                           "4. Re-measure intensity, not total",
                    impact="Bringing '{}' to the fleet best would be about "
                           "{:.0f}% of its consumption.".format(
                               gap["best"], (1 - 1 / gap["ratio"]) * 100),
                    severity="high", category="energy_intensity",
                ))
        except Exception:
            logger.debug("energy intensity failed", exc_info=True)
    elif consumption and site and not size:
        actions.append(
            "Consumption is recorded by {} but no floor area, output or "
            "headcount is present — without one, sites can only be ranked "
            "by size, not by efficiency.".format(site))

    # ── Carbon intensity ──────────────────────────────────
    if emissions and consumption:
        try:
            total_e = float(pd.to_numeric(df[emissions],
                                          errors="coerce").sum())
            total_c = float(pd.to_numeric(df[consumption],
                                          errors="coerce").sum())
            if total_c > 0:
                intensity = total_e / total_c
                note = benchmark_note("energy", emissions, intensity)
                findings.append(
                    "Carbon intensity is {} of {} per unit of {}.{}".format(
                        fmt(intensity), emissions, consumption,
                        (" " + note) if note else ""))
                actions.append(
                    "Track {} per unit of {} rather than total {} — a total "
                    "falls when output falls, and that is not a decarbonisation "
                    "result.".format(emissions, consumption, emissions))
        except Exception:
            logger.debug("energy carbon intensity failed", exc_info=True)
    elif emissions:
        try:
            findings.append("Total {} recorded is {}.".format(
                emissions, fmt(float(pd.to_numeric(df[emissions],
                                                   errors="coerce").sum()))))
        except Exception:
            logger.debug("energy emissions total failed", exc_info=True)

    if renewable and consumption:
        try:
            share = (float(pd.to_numeric(df[renewable],
                                         errors="coerce").sum()) /
                     float(pd.to_numeric(df[consumption],
                                         errors="coerce").sum()) * 100)
            if 0 < share <= 100:
                findings.append(
                    "{:.1f}% of consumption is met from {}.".format(
                        share, renewable))
        except Exception:
            logger.debug("energy renewable share failed", exc_info=True)

    if consumption:
        cv = variability(df[consumption])
        if cv is not None and cv > 60:
            findings.append(
                "{} varies by {:.0f}% around its mean — a single average "
                "describes very little of this profile.".format(
                    consumption, cv))

    if cost and consumption:
        actions.append(
            "'{}' and '{}' are both present — unit rate is the measure that "
            "separates a tariff problem from a consumption one, and it can "
            "be derived here.".format(cost, consumption))

    if not findings and not insights:
        logger.info("energy engine found no domain signal in %d columns",
                    len(df.columns))

    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}
