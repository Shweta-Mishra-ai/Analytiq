"""
engines/bi/insights.py — turning the analyses into sentences.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from app.engines.present import label as _L, value as _V

from typing import List, Tuple

from app.engines.bi.results import BIReport


#  KEY INSIGHTS GENERATOR
# ══════════════════════════════════════════════════════════

def _generate_key_insights(
    report: BIReport, df: pd.DataFrame
) -> Tuple[List[str], str]:
    insights = []

    # Pareto insights
    for p in report.pareto:
        if p.pareto_holds:
            insights.append(
                "Pareto holds for '{}' by '{}': top {:.0f}% of groups "
                "drive {:.0f}% of value. Focus resources on top performers.".format(
                    p.value_col, p.group_col,
                    20, p.top_groups_share)
            )
        # No concentration is the absence of a finding, not a finding.
        # Listing "Age is evenly distributed across Department" as a key
        # insight fills the section with things that are not news.

    # Root cause insights
    #
    # A driver relationship is symmetric, so running root cause on both
    # revenue and quantity yields the same fact twice, once each way
    # round: "what separates the low Revenue group most is Quantity"
    # followed by "what separates the low Quantity group most is
    # Revenue". Two lines, one finding — and the second reads as a
    # separate discovery. The pair is emitted once.
    seen_pairs = set()
    for rc in report.root_causes:
        if not rc.drivers:
            continue
        pair = frozenset((rc.target_col, rc.top_driver))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        lead = rc.drivers[0]
        gap = lead.get("diff_pct") or 0
        if gap:
            insights.append(
                "What separates the low {} group from the high one most is "
                "{}: {:.0f}% apart.".format(
                    _L(rc.target_col), _L(rc.top_driver), gap))
        else:
            # A categorical driver has no percentage gap, and printing a
            # zero read as "the top driver makes no difference".
            insights.append(
                "What separates the low {} group from the high one most is "
                "{}.".format(_L(rc.target_col), _L(rc.top_driver)))

    # Cohort insights
    # Significant and 1.9% apart is not a segmentation worth naming. On
    # 1,470 rows almost any split clears p<0.05.
    #
    # And a cohort built out of the metric is excluded outright, however
    # large the gap: "Revenue Band splits Revenue — 700+ averages 594.6%
    # more than 0-200" is the definition of the bands, and a gap that
    # large would top this list every time such a column appeared.
    sig_cohorts = [c for c in report.cohorts
                   if c.is_significant and c.gap_pct >= 10
                   and not c.is_definitional]
    for c in sig_cohorts[:2]:
        insights.append(
            "{} splits {}: {} averages {:.1f}% more than {}.".format(
                _L(c.cohort_col), _L(c.metric_col), _V(c.best_cohort),
                c.gap_pct, _V(c.worst_cohort))
        )

    # Segment health insights
    #
    # Only where health was actually scored. When no metric had a
    # direction its name settles, every segment holds the placeholder
    # 50 — and the "within 0 points of each other" branch below then
    # reported that as a finding: "none stands out as needing attention
    # before the others". Nothing was compared, so nothing is known.
    scored_segments = [s for s in report.segments if getattr(s, "scored", True)]
    if len(scored_segments) >= 2:
        best = scored_segments[0]
        worst = scored_segments[-1]
        spread = best.health_score - worst.health_score
        # A "healthiest" segment scoring 50 and a "needs most attention"
        # scoring 48 is a two-point gap dressed as a finding. Ranking
        # always produces a first and a last; only a real spread between
        # them is news.
        if spread >= 10:
            insights.append(
                "{} is the healthiest segment at {:.0f} of 100; {} is the "
                "weakest at {:.0f}, a {:.0f}-point spread.".format(
                    _V(best.segment_name), best.health_score,
                    _V(worst.segment_name), worst.health_score, spread))
        else:
            insights.append(
                "The {} segments score within {:.0f} points of each other "
                "on health ({:.0f} to {:.0f}), so none stands out as "
                "needing attention before the others.".format(
                    len(scored_segments), spread, worst.health_score,
                    best.health_score))

    # Executive brief
    brief = "Business Intelligence analysis completed. "
    if insights:
        brief += insights[0] + " "
    if sig_cohorts:
        brief += "{} significant cohort difference(s) identified. ".format(
            len(sig_cohorts))
    if report.root_causes:
        brief += "Root cause analysis run on {} metric(s).".format(
            len(report.root_causes))

    return insights, brief

