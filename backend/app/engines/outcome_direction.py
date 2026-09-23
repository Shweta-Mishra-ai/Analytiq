"""
engines/outcome_direction.py — which way the thing being predicted points.

A model predicts a column. Whether a high rate of that column is good
news or bad news is not a property of the model, and the report cannot
be written without knowing it.

It used to be assumed. Every predictive section was written for churn:
"highest-risk segment", "potentially avoidable", "a targeted
intervention here reaches the most affected records". On a sales
extract the target is `won`, so the report named the two best reps in
the company as the highest-risk segment and offered to help avoid
ninety-eight of their wins. The arithmetic was right and every sentence
around it was backwards.

`higher_is_better()` in domains/base already knew the answer from the
column name — the registry marks win_rate as higher-is-better and
attrition_rate as lower — and nothing in the report layer asked it.

Three cases, and the third earns its place: when the column name does
not say, the honest thing is to describe the gap and not call it either
an opportunity or a risk. Guessing produces exactly the failure above.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.engines.domains.base import higher_is_better

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Direction:
    """The words a report uses for one predicted outcome."""

    desirable: Optional[bool]

    #  Headings
    section_title: str
    section_sub: str

    #  What one occurrence is called: "win", "churn event", "event"
    event_noun: str
    event_noun_plural: str

    #  The concentrated group, and the model's top band
    cluster_label: str
    cluster_tail: str
    segment_label: str
    segment_tail: str

    #  The gap between that group and the base rate
    gap_label: str
    base_sub: str

    #  Chart and table captions
    heatmap_title: str
    heatmap_caption: str
    bands_intro: str

    @property
    def is_risk(self) -> bool:
        return self.desirable is False


_RISK = Direction(
    desirable=False,
    section_title="Predictive Risk Analysis",
    section_sub="A model trained to predict {} — drivers, accuracy, and "
                "the highest-risk segment",
    event_noun="event",
    event_noun_plural="events",
    cluster_label="Largest risk cluster",
    cluster_tail="This is the most concentrated addressable pocket of risk: "
                 "a targeted intervention here reaches the most affected "
                 "records for the least effort.",
    segment_label="Highest-risk segment",
    segment_tail="This is where intervention has the highest expected "
                 "return; pull this list from the source system and act on "
                 "it first.",
    gap_label="potentially avoidable",
    base_sub="overall event rate",
    heatmap_title="Risk Concentration Map",
    heatmap_caption="Darker cells carry a higher event rate. The hottest "
                    "cell is the segment to address first.",
    bands_intro="Records ranked by predicted risk. Each row is a different "
                "size of intervention: how many records it covers, how many "
                "of the events it would reach, and how much better that is "
                "than choosing at random.",
)

_OPPORTUNITY = Direction(
    desirable=True,
    section_title="Predictive Opportunity Analysis",
    section_sub="A model trained to predict {} — drivers, accuracy, and "
                "where it already happens most",
    event_noun="success",
    event_noun_plural="successes",
    cluster_label="Strongest cluster",
    cluster_tail="This is the most concentrated pocket of success in the "
                 "file: whatever is being done here is what the rest of the "
                 "book would need to copy.",
    segment_label="Best-performing segment",
    segment_tail="This is the group to study rather than to fix — the "
                 "question it answers is what the rest of the book is not "
                 "doing.",
    gap_label="the margin this group holds over the rest",
    base_sub="overall success rate",
    heatmap_title="Where the Outcome Concentrates",
    heatmap_caption="Darker cells succeed more often. The hottest cell is "
                    "the one to learn from first.",
    bands_intro="Records ranked by predicted likelihood of the outcome. Each "
                "row is a different size of shortlist: how many records it "
                "covers, how many of the successes it would capture, and how "
                "much better that is than choosing at random.",
)

_NEUTRAL = Direction(
    desirable=None,
    section_title="Predictive Analysis",
    section_sub="A model trained to predict {} — drivers, accuracy, and "
                "where the rate is highest",
    event_noun="occurrence",
    event_noun_plural="occurrences",
    cluster_label="Highest-rate cluster",
    cluster_tail="This is the most concentrated group in the file. Whether "
                 "that is worth encouraging or reducing is a judgement this "
                 "report does not make: the column name does not say which "
                 "direction is the good one.",
    segment_label="Highest-rate segment",
    segment_tail="Whether to act on this group, and in which direction, "
                 "depends on what the column means to you.",
    gap_label="the difference between this group and the rest",
    base_sub="overall rate",
    heatmap_title="Rate Concentration Map",
    heatmap_caption="Darker cells carry a higher rate.",
    bands_intro="Records ranked by predicted likelihood. Each row is a "
                "different size of shortlist: how many records it covers, "
                "how many of the cases it would reach, and how much better "
                "that is than choosing at random.",
)


def direction_for(target: str) -> Direction:
    """The vocabulary for a predicted column, from its name.

    Deliberately the column name and nothing else. Inferring direction
    from the data — treating whichever class is rarer as the bad one —
    would call a 9% win rate a risk and a 91% retention rate a problem.
    """
    answer = higher_is_better(target)
    if answer is True:
        return _OPPORTUNITY
    if answer is False:
        return _RISK
    return _NEUTRAL


__all__ = ["Direction", "direction_for"]
