"""
engines/domains/education.py — student and course analytics.

An education dataset routed to the general engine came back as column
statistics: an average grade, a count of students, and nothing a head of
department could act on. The questions this engine answers are the ones
actually asked in a programme review.

**Who is failing, and is it the course or the cohort?** A pass rate that
varies four-fold across modules is a teaching or assessment problem; one
that varies across intake cohorts inside the same module is a selection
or preparation problem. The difference decides who owns the fix, so the
engine reports both rather than one average.

**Is attendance the driver everyone assumes it is?** It usually is, and
the size matters — "students below 60% attendance pass at half the rate"
funds an intervention, "attendance correlates with grades" does not.

**Where does the cohort leak?** Withdrawal concentrated in one term or
one module is a retention problem with an address.

Deliberately not here: anything about an individual student. This is a
programme-level analysis, and a report that ranks named students by
predicted failure is a different product with different obligations.
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

# Below this, a pass rate is a programme-level problem rather than a
# cohort of weak students.
PASS_RATE_CONCERN = 70.0
# A module whose pass rate sits this many points below the programme is
# an outlier worth a name.
MODULE_GAP_PP = 15.0
# Attendance below this is where the drop-off in outcomes usually starts.
ATTENDANCE_FLOOR = 75.0


def _module_col(df: pd.DataFrame):
    """The unit a programme lead can actually change: a module, a course,
    a subject — not a student."""
    for kw in ("module", "course", "subject", "unit", "class", "programme",
               "program", "department", "faculty"):
        for c in grouping_columns(df):
            if kw in c.lower().replace("_", "").replace(" ", ""):
                return c
    groups = grouping_columns(df)
    return groups[0] if groups else None


def _insights_education(df: pd.DataFrame, stats: Dict, corrs: List) -> Dict:
    findings, risks, opps, actions = [], [], [], []
    insights: List[Insight] = []

    grade = find_measure(df, ["grade", "score", "mark", "gpa", "result",
                              "percentage", "attainment"],
                         exclude=["passscore", "gradeid"])
    attendance = find_measure(df, ["attendance", "present", "absence",
                                   "engagement"])
    hours = find_measure(df, ["studyhours", "hoursstudied", "studytime",
                              "revision"])
    module = _module_col(df)
    pass_col = find_col(df, ["passed", "pass", "outcome", "completed",
                             "graduated", "achieved"])
    withdrawn = find_col(df, ["withdrawn", "dropout", "droppedout",
                              "discontinued"])

    # ── Does the cohort pass? ─────────────────────────────
    pass_rate = binary_rate(df[pass_col]) if pass_col else None
    if pass_rate is not None:
        severity = ("critical" if pass_rate < 60 else
                    "high" if pass_rate < PASS_RATE_CONCERN else "positive")
        findings.append(
            "Overall pass rate is {:.1f}% across {:,} enrolments.".format(
                pass_rate, len(df)))
        note = benchmark_note("education", pass_col, pass_rate)
        if pass_rate < PASS_RATE_CONCERN:
            risks.append(
                "A pass rate of {:.1f}% means roughly {:,} of {:,} enrolments "
                "do not complete successfully. At programme level that is a "
                "design question before it is a student question.".format(
                    pass_rate, int(len(df) * (100 - pass_rate) / 100), len(df)))
        insights.append(build_insight(
            title="Pass Rate at {:.1f}%".format(pass_rate),
            problem="{:.1f}% of enrolments pass across {:,} records".format(
                pass_rate, len(df)),
            cause="Assessment design, teaching hours, prior attainment or "
                  "cohort preparation — the module breakdown separates them",
            evidence="{:,} enrolments{}".format(
                len(df), (" — " + note) if note else ""),
            action="1. Rank modules by pass rate  2. Take the bottom two "
                   "and compare assessment design against the top two  "
                   "3. Check whether the gap follows the module or the "
                   "cohort  4. Re-measure at the next assessment point",
            impact="Each point of pass rate is about {:,} enrolments across "
                   "this cohort.".format(max(1, len(df) // 100)),
            severity=severity, category="education_outcomes",
        ))

    # ── Is it the module or the students? ─────────────────
    if module and grade:
        gap = segment_gap(df, module, grade, agg="mean")
        if gap and abs(gap["gap"]) >= 1:
            findings.append(
                "'{}' averages {} against '{}' at {} — a {} spread across "
                "{} {}s.".format(
                    gap["best"], fmt(gap["best_val"]), gap["worst"],
                    fmt(gap["worst_val"]), fmt(gap["gap"]), gap["n_groups"],
                    module))
            opps.append(
                "The gap between '{}' and '{}' is inside one programme, so "
                "it is comparable: whatever the stronger one does "
                "differently is available to the weaker one.".format(
                    gap["best"], gap["worst"]))
            insights.append(build_insight(
                title="{} Spread: '{}' to '{}'".format(
                    module.replace("_", " ").title(), gap["worst"],
                    gap["best"]),
                problem="Average {} runs from {} to {} across {} {}s".format(
                    grade, fmt(gap["worst_val"]), fmt(gap["best_val"]),
                    gap["n_groups"], module),
                cause="Teaching approach, assessment difficulty or cohort "
                      "intake — comparable within one programme, which is "
                      "what makes the gap actionable",
                evidence="{:,} records in '{}' against {:,} in '{}'".format(
                    gap["n_worst"], gap["worst"], gap["n_best"], gap["best"]),
                action="1. Sit in on both  2. Compare assessment rubrics "
                       "and contact hours  3. Move one practice from the "
                       "stronger to the weaker  4. Re-measure next term",
                impact="Closing half the gap would lift the weaker cohort "
                       "by about {}.".format(fmt(gap["gap"] / 2)),
                severity="high" if abs(gap["gap"]) >= MODULE_GAP_PP else "warning",
                category="education_delivery",
            ))

    # ── Attendance, with a number attached ────────────────
    if attendance and grade:
        try:
            low = df[df[attendance] < ATTENDANCE_FLOOR][grade].mean()
            high = df[df[attendance] >= ATTENDANCE_FLOOR][grade].mean()
            n_low = int((df[attendance] < ATTENDANCE_FLOOR).sum())
            if pd.notna(low) and pd.notna(high) and n_low >= 20 and high > low:
                findings.append(
                    "Students below {:.0f}% attendance average {} against {} "
                    "for the rest — {:,} students sit below that line.".format(
                        ATTENDANCE_FLOOR, fmt(low), fmt(high), n_low))
                opps.append(
                    "Attendance is measurable weekly, long before an "
                    "assessment. The {:,} students below {:.0f}% are an "
                    "identifiable list, not a statistical category.".format(
                        n_low, ATTENDANCE_FLOOR))
                insights.append(build_insight(
                    title="Attendance Below {:.0f}% Costs {} in {}".format(
                        ATTENDANCE_FLOOR, fmt(high - low), grade),
                    problem="{:,} students below {:.0f}% attendance average "
                            "{} against {}".format(
                                n_low, ATTENDANCE_FLOOR, fmt(low), fmt(high)),
                    cause="Attendance is usually a symptom as much as a "
                          "cause — timetable clashes, paid work and travel "
                          "all show up here",
                    evidence="{:,} students below the line against {:,} "
                             "above it".format(n_low, len(df) - n_low),
                    action="1. Flag at the third missed session, not at the "
                           "assessment  2. Ask why, once, and record the "
                           "reason  3. Fix the timetable causes centrally  "
                           "4. Track the flagged group's outcome",
                    impact="The gap is {} of {} across {:,} students.".format(
                        fmt(high - low), grade, n_low),
                    severity="high", category="education_engagement",
                ))
        except Exception:
            logger.debug("education attendance split failed", exc_info=True)

    # ── Where the cohort leaks ────────────────────────────
    if withdrawn:
        rate = binary_rate(df[withdrawn])
        if rate is not None and rate > 0:
            findings.append("Withdrawal rate is {:.1f}%.".format(rate))
            if rate > 10:
                risks.append(
                    "A {:.1f}% withdrawal rate is a retention problem before "
                    "it is an attainment one — those students are counted "
                    "nowhere in the grade distribution.".format(rate))

    # ── Consistency of marking ────────────────────────────
    if grade and module:
        cv = variability(df[grade])
        if cv is not None and cv > 35:
            findings.append(
                "{} varies by {:.0f}% around its mean — wide enough that a "
                "single programme average hides most of what is "
                "happening.".format(grade, cv))

    if hours and grade:
        actions.append(
            "'{}' is recorded alongside {} — worth testing whether it "
            "predicts attainment before it is used to advise students."
            .format(hours, grade))

    if not findings and not insights:
        logger.info("education engine found no domain signal in %d columns",
                    len(df.columns))

    return {"findings": findings, "risks": risks, "opportunities": opps,
            "actions": actions, "insights": insights}
