"""
The report's Findings page comes from build_top_insights, and it had no
test — which is how it shipped with a crash on its very first branch:
a story whose top_insights arrived as dicts was passed category= to a
local Insight copy that never declared the field.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.engines.insights_builder import Insight, build_top_insights


class _Story:
    def __init__(self, top_insights):
        self.top_insights = top_insights


@pytest.fixture
def df():
    return pd.DataFrame({
        "department": ["Sales", "Eng", "Sales", "Eng", "Ops", "Ops"] * 8,
        "salary":     [70, 95, 68, 99, 55, 58] * 8,
        "tenure":     [2.0, 5.5, 1.0, 7.0, 3.0, 4.0] * 8,
    })


def test_a_story_whose_insights_are_dicts_does_not_crash(df):
    """The regression. Every field the builder passes must exist on the
    Insight it builds, or the whole Findings page raises."""
    story = _Story([{
        "severity": "high", "title": "Attrition is concentrated",
        "problem": "p", "cause": "c", "evidence": "e",
        "action": "a", "impact": "i", "category": "attrition",
    }])

    out = build_top_insights(df, domain="hr", story_obj=story)

    assert len(out) == 1
    assert out[0].category == "attrition", \
        "the category is what lets the blueprint file a finding under a heading"


def test_the_category_is_what_sorts_findings_under_headings(df):
    """Two findings of different kinds must stay distinguishable; without
    a category they all land under whichever heading comes first."""
    story = _Story([
        {"severity": "high", "title": "A", "problem": "", "cause": "",
         "evidence": "", "action": "", "impact": "", "category": "attrition"},
        {"severity": "warning", "title": "B", "problem": "", "cause": "",
         "evidence": "", "action": "", "impact": "", "category": "margin"},
    ])

    out = build_top_insights(df, story_obj=story)

    assert {i.category for i in out} == {"attrition", "margin"}


def test_a_dict_without_a_category_still_builds(df):
    story = _Story([{"severity": "info", "title": "A", "problem": "",
                     "cause": "", "evidence": "", "action": "", "impact": ""}])

    out = build_top_insights(df, story_obj=story)

    assert out[0].category == "general"


def test_there_is_one_insight_class_not_two():
    """The duplicate was not merely untidy — it was missing fields the
    callers already passed."""
    from app.engines.domains.base import Insight as Canonical

    assert Insight is Canonical


def test_it_builds_findings_with_no_story_at_all(df):
    """The fallback path: a report must not be short a Findings page
    because the story engine stayed quiet."""
    out = build_top_insights(df, domain="hr")

    assert isinstance(out, list)
    for item in out:
        assert item.title and item.severity
