"""
Telling a mixed population apart from a dirty column.

A single IQR fence across a whole file assumes one distribution. On a
retail export where Electronics averages 258 and Grocery averages 12, it
produced three separate warnings:

  - "'unit_price' has 21% outliers — verify before using it in any
    decision"
  - "'revenue' has 15% outliers — verify before using it in any decision"
  - "99% of 'Electronics' records are extreme values, against 21%
    overall", with a suggested cause of "a different unit of measure, or
    a data-entry route that differs for that group"

Every figure was correct and every conclusion was wrong. Three of the six
headline insights on that report were the same artefact, they were graded
as risks, and they pushed genuine findings — lapsed customers, cohort
retention, revenue concentration — off the front of the document.

The test for it is the one the code was missing: does the flag survive
computing the same fence within groups? If it does not, the column is
multi-modal and the honest output says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.domains.base import col_stats, outliers_explained_by_group
from app.engines.domains.general import _insights_general
from app.engines.story_engine import generate_story


def _general(df):
    stats = {c: col_stats(df[c])
             for c in df.select_dtypes(include="number").columns}
    return _insights_general(df, {c: s for c, s in stats.items() if s}, [])


@pytest.fixture()
def catalogue():
    """Prices that differ by category, and are correct within it."""
    rng = np.random.default_rng(9)
    n = 2000
    cat = rng.choice(["Home", "Beauty", "Electronics", "Apparel", "Grocery"],
                     n, p=[.28, .22, .2, .18, .12])
    level = {"Home": 45, "Beauty": 22, "Electronics": 260,
             "Apparel": 38, "Grocery": 12}
    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "customer_id": rng.integers(1, 700, n),
        "order_date": pd.to_datetime("2024-01-01")
                      + pd.to_timedelta(rng.integers(0, 400, n), "D"),
        "category": cat,
        "unit_price": [round(rng.normal(level[c], level[c] * .2), 2)
                       for c in cat],
        "quantity": rng.integers(1, 5, n),
        "rating": rng.choice([1, 2, 3, 4, 5], n, p=[.05, .07, .18, .35, .35]),
    })
    df["revenue"] = (df.unit_price * df.quantity).round(2)
    return df


@pytest.fixture()
def genuinely_dirty():
    """One distribution with real data-entry errors in it."""
    rng = np.random.default_rng(31)
    n = 800
    values = rng.normal(100, 12, n)
    values[::6] = rng.normal(100, 12, len(values[::6])) * 100   # decimal slips
    return pd.DataFrame({
        "branch": rng.choice(["North", "South", "East"], n),
        "reading": values.round(2),
    })


# ══════════════════════════════════════════════════════════
#  The detector
# ══════════════════════════════════════════════════════════

def test_a_multi_level_column_is_traced_to_its_grouping(catalogue):
    assert outliers_explained_by_group(catalogue, "unit_price") == "category"


def test_real_errors_are_not_explained_away(genuinely_dirty):
    """The check must not become a blanket excuse — errors scattered
    across every group stay flagged."""
    assert outliers_explained_by_group(genuinely_dirty, "reading") is None


def test_a_clean_column_returns_nothing():
    rng = np.random.default_rng(32)
    df = pd.DataFrame({"grp": rng.choice(["a", "b"], 300),
                       "value": rng.normal(50, 5, 300)})
    assert outliers_explained_by_group(df, "value") is None


def test_too_few_rows_to_judge():
    df = pd.DataFrame({"grp": ["a", "b"] * 5, "value": range(10)})
    assert outliers_explained_by_group(df, "value") is None


def test_an_identifier_grouping_is_not_accepted(catalogue):
    """Grouping by something with one row per group makes any value
    unremarkable within its group."""
    got = outliers_explained_by_group(catalogue, "unit_price")
    assert got != "order_id"


def test_it_does_not_raise_on_an_empty_frame():
    assert outliers_explained_by_group(pd.DataFrame({"a": []}), "a") is None


# ══════════════════════════════════════════════════════════
#  What the report says about it
# ══════════════════════════════════════════════════════════

def test_a_priced_catalogue_is_not_reported_as_dirty(catalogue):
    story = generate_story(catalogue)
    risks = " ".join(story.business_risks)
    assert "unit_price' has" not in risks, risks
    assert "verify before using it in any decision" not in risks, risks


def test_the_column_is_described_as_multi_modal_instead(catalogue):
    """The general engine is where the outlier check lives; the report's
    top-six list is a separate ranking question."""
    out = _general(catalogue)
    text = " ".join([i.title for i in out["insights"]] + out["findings"])
    assert "Splits by 'category'" in text or "per group" in text, text


def test_a_whole_category_is_not_called_a_collection_fault(catalogue):
    """"99% of 'Electronics' records are extreme values" with a cause of
    "a data-entry route that differs for that group"."""
    story = generate_story(catalogue)
    text = " ".join([i.title for i in story.top_insights]
                    + [i.cause for i in story.top_insights])
    assert "data-entry route" not in text, text
    assert "extreme values, against" not in text, text


def test_the_finding_says_nothing_needs_cleaning(catalogue):
    hits = [i for i in _general(catalogue)["insights"]
            if "Splits by" in i.title]
    assert hits
    assert "Nothing needs cleaning" in hits[0].impact
    assert hits[0].severity == "info", hits[0].severity


def test_genuine_errors_are_still_raised_as_a_risk(genuinely_dirty):
    """The fix must not silence the case it was built to distinguish."""
    story = generate_story(genuinely_dirty)
    joined = " ".join(story.business_risks
                      + [i.title for i in story.top_insights])
    assert "outlier" in joined.lower(), joined


def test_the_real_findings_are_no_longer_crowded_out(catalogue):
    """Three of six headline slots went to the same artefact."""
    story = generate_story(catalogue)
    quality = [i for i in story.top_insights
               if i.category in ("data_quality", "quality")
               and i.severity == "warning"]
    assert len(quality) <= 1, [i.title for i in story.top_insights]
