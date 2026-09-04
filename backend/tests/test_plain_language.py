"""The first sentences a reader sees, in words they already know.

The app is meant to work for someone who is not a statistician. That
does not mean hiding the statistics — the Deep EDA page exists for
readers who want them — it means the summary line at the top of the
report says something a business reader can act on, with the statistic
in brackets after it rather than instead of it.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.story_engine import generate_story


@pytest.fixture()
def correlated_frame():
    """Two columns that move together, and nothing else worth reporting —
    so the headline falls through to the correlation."""
    rng = np.random.default_rng(0)
    n = 2000
    revenue = rng.lognormal(6, 0.5, n).round(2)
    return pd.DataFrame({
        "revenue": revenue,
        "profit": (revenue * 0.28 + rng.normal(0, 12, n)).round(2),
        "region": rng.choice(["North", "South", "East"], n),
    })


def test_the_summary_explains_the_relationship_before_naming_it(
        correlated_frame):
    story = generate_story(correlated_frame)
    text = story.executive_summary
    assert "move up and down together" in text or \
           "move in opposite directions" in text, text[:200]
    # the statistic still there, for the reader who wants it
    assert "Spearman r=" in text


def test_the_summary_does_not_lead_with_shared_variance(correlated_frame):
    """"Explaining 91% of shared variance — the clearest non-trivial
    structural pattern in this dataset" was the opening line."""
    text = generate_story(correlated_frame).executive_summary
    assert not text.lstrip().startswith("'revenue' and 'profit' show a strong")
    assert "non-trivial structural pattern" not in text


def test_correlation_is_never_stated_as_cause(correlated_frame):
    text = generate_story(correlated_frame).executive_summary
    assert "does not show that either one causes the other" in text


def test_the_headline_is_never_a_status_word():
    """"Analysis complete" sat in the slot a reader looks at first and
    told them nothing about their data."""
    rng = np.random.default_rng(1)
    n = 600
    flat = pd.DataFrame({"a": rng.normal(50, 1, n),
                         "b": rng.normal(50, 1, n),
                         "g": rng.choice(["x", "y"], n)})
    headline = generate_story(flat).headline
    assert headline != "Analysis complete"
    assert len(headline) > 25, headline


def test_a_real_finding_still_leads_the_headline():
    """The fallback must not displace an actual finding."""
    rng = np.random.default_rng(2)
    n = 1200
    df = pd.DataFrame({
        "JobSatisfaction": rng.choice([1, 2, 3, 4], n, p=[.45, .3, .15, .1]),
        "Attrition": rng.choice(["Yes", "No"], n, p=[.3, .7]),
        "Department": rng.choice(["Sales", "R&D"], n),
        "MonthlyIncome": rng.normal(5000, 900, n).round(0),
    })
    headline = generate_story(df).headline
    assert "No single finding dominates" not in headline
    assert headline != "Analysis complete"
