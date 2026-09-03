"""
engines/domains/healthcare.py — Healthcare operations analytics.

Patient records used to detect as HR at 0.13 confidence: "department",
"age" and "satisfaction" were enough for the HR engine to claim them,
which meant a hospital's case data was described in the vocabulary of
employee attrition.

SCOPE — this engine is deliberately administrative. It analyses capacity,
cost and operational risk across aggregate records. It does not produce
clinical guidance, diagnosis or treatment recommendations, and it does not
reason about any individual patient. Where a pattern could have a clinical
explanation, the output says clinical review is required rather than
offering one. The same limit is written into the healthcare LLM prompts.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from app.engines.domains.base import Insight, build_insight
from app.engines.domains._common import (
    benchmark_note, binary_rate, concentration, find_measure, fmt,
    grouping_columns, segment_gap, variability,
)

logger = logging.getLogger(__name__)

# Sustained occupancy above this is associated with rising delays and
# infection risk — higher is not better.
OCCUPANCY_CEILING = 85.0
LOS_UNSTABLE_CV = 60.0
MIN_DEPT_RATIO = 1.4

# Every insight this engine emits carries this boundary, so a reader never
# mistakes an operational observation for a clinical one.
_CLINICAL_NOTE = ("Operational observation only — any clinical "
                  "interpretation requires review by the responsible "
                  "clinical team.")


def _dept_col(df: pd.DataFrame):
    for kw in ("department", "specialty", "speciality", "ward", "unit",
               "service", "clinic", "facility"):
        for c in grouping_columns(df):
            if kw in c.lower().replace("_", ""):
                return c
    groups = grouping_columns(df)
    return groups[0] if groups else None


def _insights_healthcare(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights: List[Insight] = []

    los = find_measure(df, ["lengthofstay", "los", "staydays", "daysadmitted",
                            "bedhours"])
    occupancy = find_measure(df, ["bedoccupancy", "occupancy", "bedutilisation",
                                  "bedutilization"])
    cost = find_measure(df, ["costpercase", "casecost", "treatmentcost",
                             "cost", "charge"])
    wait = find_measure(df, ["waittime", "waitingtime", "timetotreatment",
                             "doortodoctor"])
    satisfaction = find_measure(df, ["satisfaction", "patientexperience",
                                     "hcahps"])
    dept = _dept_col(df)

    readmit_col = None
    for c in df.columns:
        n = c.lower().replace("_", "")
        if "readmi" in n and binary_rate(df[c]) is not None:
            readmit_col = c
            break

    # ── Readmission ───────────────────────────────────────
    if readmit_col:
        rate = binary_rate(df[readmit_col])
        if rate is not None:
            note = benchmark_note("healthcare", readmit_col, rate)
            severity = ("critical" if rate > 20 else
                        "high" if rate > 15 else
                        "warning" if rate > 10 else "positive")
            findings.append(
                "Readmission rate is {:.1f}%.{}".format(
                    rate, " " + note if note else ""))
            if rate > 15:
                risks.append(
                    "Readmission at {:.1f}% means roughly 1 in {:.0f} "
                    "discharges returns — an operational and cost burden "
                    "as well as a quality signal.".format(
                        rate, 100 / max(rate, .01)))
            insights.append(build_insight(
                title="Readmission Rate at {:.1f}%".format(rate),
                problem="{:.1f}% of discharges are followed by a "
                        "readmission".format(rate),
                cause="Discharge timing, follow-up availability, or case mix. "
                      + _CLINICAL_NOTE,
                evidence="'{}' across {:,} records.{}".format(
                    readmit_col, len(df), " " + note if note else ""),
                action="1. Break the rate down by department and case mix  "
                       "2. Review discharge and follow-up scheduling for the "
                       "highest departments  3. Refer clinical patterns to "
                       "the responsible clinical team",
                impact="Each readmission consumes a bed-day that was "
                       "planned for new admissions",
                severity=severity, category="healthcare_quality",
            ))

    # ── Length of stay ────────────────────────────────────
    if los:
        try:
            mean_los = float(pd.to_numeric(df[los], errors="coerce").mean())
            cv = variability(df[los])
            note = benchmark_note("healthcare", los, mean_los)
            findings.append(
                "Average length of stay is {} with {}% variability.{}".format(
                    fmt(mean_los), "{:.0f}".format(cv) if cv else "unknown",
                    " " + note if note else ""))
            if cv and cv > LOS_UNSTABLE_CV:
                insights.append(build_insight(
                    title="Length of Stay Varies {:.0f}% Around Its Mean".format(
                        cv),
                    problem="Stay length is highly variable: {:.0f}% "
                            "coefficient of variation on a {} average".format(
                                cv, fmt(mean_los)),
                    cause="Case-mix differences, discharge process delays, or "
                          "downstream capacity constraints. " + _CLINICAL_NOTE,
                    evidence="'{}' mean {}, CV {:.0f}% over {:,} "
                             "records".format(los, fmt(mean_los), cv, len(df)),
                    action="1. Separate by case mix before comparing  "
                           "2. Measure discharge-decision to discharge-actual "
                           "delay  3. Address non-clinical delay first",
                    impact="Non-clinical delay converts directly into "
                           "unavailable bed capacity",
                    severity="warning", category="healthcare_flow",
                ))
        except Exception:
            logger.debug("healthcare LOS failed", exc_info=True)

    # ── Capacity ──────────────────────────────────────────
    if occupancy:
        try:
            raw = float(pd.to_numeric(df[occupancy], errors="coerce").mean())
            occ = raw * 100 if raw <= 1 else raw
            note = benchmark_note("healthcare", occupancy, occ)
            findings.append(
                "Bed occupancy averages {:.1f}%.{}".format(
                    occ, " " + note if note else ""))
            if occ > OCCUPANCY_CEILING:
                risks.append(
                    "Occupancy at {:.1f}% sits above the level where delays "
                    "and infection risk rise. Higher occupancy is not better "
                    "performance here.".format(occ))
                insights.append(build_insight(
                    title="Bed Occupancy at {:.1f}%".format(occ),
                    problem="Sustained occupancy of {:.1f}% leaves no "
                            "capacity buffer".format(occ),
                    cause="Admission rate exceeding discharge rate, or "
                          "capacity planned to average rather than peak "
                          "demand",
                    evidence="Mean of '{}' is {:.1f}%.{}".format(
                        occupancy, occ, " " + note if note else ""),
                    action="1. Track admission and discharge rates by hour  "
                           "2. Target the discharge bottleneck  "
                           "3. Hold a defined escalation buffer",
                    impact="Above roughly {:.0f}%, small surges produce "
                           "queueing, cancellations and diverted "
                           "admissions".format(OCCUPANCY_CEILING),
                    severity="critical" if occ > 95 else "high",
                    category="healthcare_capacity",
                ))
        except Exception:
            logger.debug("healthcare occupancy failed", exc_info=True)

    # ── Cost variation by department ──────────────────────
    if cost and dept:
        gap = segment_gap(df, dept, cost, agg="mean")
        if gap and gap["ratio"] and gap["ratio"] >= MIN_DEPT_RATIO:
            findings.append(
                "Cost per case ranges {:.1f}x across {} departments — '{}' "
                "at {} against '{}' at {}.".format(
                    gap["ratio"], gap["n_groups"], gap["best"],
                    fmt(gap["best_val"]), gap["worst"], fmt(gap["worst_val"])))
            insights.append(build_insight(
                title="{:.1f}x Cost-per-Case Spread Across Departments".format(
                    gap["ratio"]),
                problem="'{}' averages {} per case against '{}' at {}".format(
                    gap["best"], fmt(gap["best_val"]), gap["worst"],
                    fmt(gap["worst_val"])),
                cause="Case mix and acuity differ legitimately between "
                      "departments; the comparison is only actionable "
                      "within a specialty. " + _CLINICAL_NOTE,
                evidence="{} departments; highest n={}, lowest n={}".format(
                    gap["n_groups"], gap["n_best"], gap["n_worst"]),
                action="1. Normalise by case mix before drawing conclusions  "
                       "2. Compare within specialty, never across  "
                       "3. Investigate only unexplained residual variation",
                impact="Cost variation that survives case-mix adjustment is "
                       "the part that can be managed",
                severity="warning", category="healthcare_cost",
            ))

    # ── Load concentration ────────────────────────────────
    if dept:
        measure = cost or los
        if measure:
            conc = concentration(df, dept, measure, top_n=1)
            if conc and conc["top_share"] >= 40 and conc["n_groups"] > 2:
                findings.append(
                    "'{}' accounts for {:.0f}% of total '{}' across {} "
                    "departments.".format(
                        conc["top_names"][0], conc["top_share"], measure,
                        conc["n_groups"]))

    if wait:
        try:
            mean_wait = float(pd.to_numeric(df[wait], errors="coerce").mean())
            findings.append("Average wait is {}.".format(fmt(mean_wait)))
        except Exception:
            logger.debug("healthcare wait failed", exc_info=True)

    if satisfaction:
        try:
            mean_sat = float(pd.to_numeric(df[satisfaction],
                                           errors="coerce").mean())
            note = benchmark_note("healthcare", satisfaction, mean_sat)
            findings.append(
                "Patient satisfaction averages {:.1f}.{}".format(
                    mean_sat, " " + note if note else ""))
        except Exception:
            logger.debug("healthcare satisfaction failed", exc_info=True)

    if not findings and not insights:
        logger.info("healthcare engine found no domain signal in %d columns",
                    len(df.columns))

    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}
