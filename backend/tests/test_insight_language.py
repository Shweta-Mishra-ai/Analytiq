"""
How findings are worded — the difference between a report that reads as
written by an analyst and one that reads as generated.

Two failure modes are guarded:

1. Sensationalised headlines ("Attrition Crisis", "Rating Emergency").
   An analyst writes the number and the comparison; the reader decides
   whether it is a crisis.
2. Naming a cause the dataset cannot establish. "Department-specific
   issues: management quality, workload, or growth opportunities" was
   printed as CAUSE for a dataset containing no manager, workload or
   progression column. Stating it as fact is the kind of unfounded claim
   that gets a whole report dismissed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from app.engines.story_engine import generate_story


@pytest.fixture()
def hr_signal_df():
    rng = np.random.default_rng(5)
    n = 800
    dept = rng.choice(["Sales", "Eng", "HR"], n)
    left = np.where(dept == "Sales",
                    rng.choice(["Yes", "No"], n, p=[.55, .45]),
                    rng.choice(["Yes", "No"], n, p=[.12, .88]))
    return pd.DataFrame({
        "employee_id": range(n),
        "department": dept,
        "salary": rng.normal(60_000, 15_000, n).round(0),
        "tenure_years": rng.integers(0, 20, n),
        "satisfaction": rng.uniform(1, 5, n).round(1),
        "attrition": left,
    })


@pytest.fixture()
def ecommerce_df():
    rng = np.random.default_rng(4)
    n = 600
    cat = rng.choice(["Home", "Tech", "Toys"], n)
    return pd.DataFrame({
        "order_id": range(n),
        "category": cat,
        "price": (np.where(cat == "Tech", 420, 55) + rng.normal(0, 20, n)).round(2),
        "rating": np.where(cat == "Toys", 2.0, 4.3).round(1),
        "revenue": rng.gamma(2, 80, n).round(2),
    })


_SENSATIONAL = ("crisis", "emergency", "disaster", "catastroph", "alarming",
                "shocking", "dire", "devastating")


def test_titles_are_not_sensationalised(hr_signal_df, ecommerce_df):
    for df in (hr_signal_df, ecommerce_df):
        for ins in generate_story(df).top_insights:
            low = ins.title.lower()
            for word in _SENSATIONAL:
                assert word not in low, \
                    f"sensationalised headline: {ins.title!r}"


def test_titles_carry_a_concrete_figure(hr_signal_df):
    """A headline without a number is an opinion."""
    for ins in generate_story(hr_signal_df).top_insights:
        assert re.search(r"\d", ins.title), \
            f"headline states no figure: {ins.title!r}"


def test_cause_does_not_assert_unmeasured_drivers(hr_signal_df):
    """The dataset has no manager, workload or progression column, so
    naming any of them as the cause is unfounded. Mentioning them as
    something still to be established is fine — the distinction is
    whether the sentence hedges."""
    hedges = ("not identifiable", "not established", "cannot", "would narrow",
              "to confirm", "not yet proven", "confirm which", "requires",
              "is not,", "verify")
    for ins in generate_story(hr_signal_df).top_insights:
        cause = ins.cause.lower()
        names_unmeasured = any(w in cause for w in
                               ("management quality", "culture", "recognition",
                                "workload", "growth opportunities"))
        if names_unmeasured:
            assert any(h in cause for h in hedges), (
                "cause names a driver this dataset does not measure, without "
                f"flagging it as unestablished: {ins.cause!r}")


def test_cause_is_never_an_unqualified_equation(hr_signal_df):
    """"Score below 55% = systemic failure in culture, workload, ..." states
    a mechanism as arithmetic fact."""
    for ins in generate_story(hr_signal_df).top_insights:
        assert "= systemic failure" not in ins.cause, \
            f"cause asserts a mechanism as fact: {ins.cause!r}"


def test_every_insight_states_evidence(hr_signal_df, ecommerce_df):
    for df in (hr_signal_df, ecommerce_df):
        for ins in generate_story(df).top_insights:
            assert ins.evidence and ins.evidence.strip(), \
                f"insight with no evidence: {ins.title!r}"
            assert re.search(r"\d", ins.evidence), \
                f"evidence carries no figure: {ins.evidence!r}"


def test_department_gap_action_is_derived_from_the_data(hr_signal_df):
    """Generic advice ("Manager effectiveness review") is filler. The action
    should reference the actual departments compared."""
    story = generate_story(hr_signal_df)
    dept_insights = [i for i in story.top_insights
                     if "department" in i.title.lower()]
    for ins in dept_insights:
        assert "Sales" in ins.action or "Sales" in ins.cause, \
            f"action does not reference the department it found: {ins.action!r}"
