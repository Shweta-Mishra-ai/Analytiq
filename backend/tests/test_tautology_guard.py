"""Findings that restate their own definition.

"55+ and 18-25 differ on Age — HIGH priority" was on the summary page of
a client report, with an action attached: "decide which end of the Age
range is the desirable one". AgeGroup is Age, in buckets. A reader who
sees that at the top of a document stops trusting the rest of it, and
they are right to.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.domains.general import (_is_binned_from,
                                         _is_obvious_segment_pair)


# ══════════════════════════════════════════════════════════
#  CAUGHT BY THE NAMES
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("group_col,metric", [
    ("AgeGroup", "Age"),          # the one that reached a client report
    ("age_group", "age"),
    ("AGE_GROUP", "AGE"),
    ("TenureBand", "Tenure"),
    ("SalaryBand", "Salary"),
    ("IncomeBracket", "Income"),
    ("RevenueTier", "Revenue"),
    ("AgeBucket", "Age"),
])
def test_a_metric_grouped_by_its_own_buckets_is_not_a_finding(
        group_col, metric):
    assert _is_obvious_segment_pair(group_col, metric) is True


def test_the_three_letter_case_that_slipped_through():
    """The old rule compared a six-character prefix and required at least
    four characters, so "Age" was never tested at all."""
    assert _is_obvious_segment_pair("AgeGroup", "Age") is True


@pytest.mark.parametrize("group_col,metric", [
    ("Department", "Salary"),     # a real question
    ("Region", "Revenue"),
    ("Channel", "Revenue"),
    ("Segment", "Units"),
    ("AgeGroup", "MonthlyIncome"),  # different quantity — worth asking
    ("Gender", "Age"),
])
def test_a_real_comparison_is_not_suppressed(group_col, metric):
    assert _is_obvious_segment_pair(group_col, metric) is False


def test_the_older_rules_still_hold():
    """Pay by job role is definitional; age by seniority is mechanical."""
    assert _is_obvious_segment_pair("JobRole", "MonthlyIncome") is True
    assert _is_obvious_segment_pair("JobLevel", "YearsAtCompany") is True


# ══════════════════════════════════════════════════════════
#  CAUGHT BY THE DATA
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def banded():
    """`SalarySlab` is `MonthlyIncome` in bands, and the two names share
    no word — only the data can say so."""
    rng = np.random.default_rng(0)
    n = 800
    income = rng.integers(1000, 20_000, n)
    return pd.DataFrame({
        "MonthlyIncome": income,
        "SalarySlab": pd.cut(income, [0, 5_000, 10_000, 15_000, 20_001],
                             labels=["<5k", "5-10k", "10-15k", "15k+"]
                             ).astype(str),
        "Department": rng.choice(["Sales", "R&D", "HR"], n),
        "Revenue": rng.normal(500, 90, n),
    })


def test_a_binning_is_detected_from_the_values(banded):
    assert _is_binned_from(banded, "SalarySlab", "MonthlyIncome") is True


def test_the_same_bins_against_another_metric_are_a_real_comparison(banded):
    assert _is_binned_from(banded, "SalarySlab", "Revenue") is False


def test_an_ordinary_dimension_is_not_a_binning(banded):
    assert _is_binned_from(banded, "Department", "MonthlyIncome") is False
    assert _is_binned_from(banded, "Department", "Revenue") is False


def test_too_few_rows_to_judge_is_not_a_binning():
    """A handful of rows can look non-overlapping by chance."""
    df = pd.DataFrame({"band": ["a", "b"] * 5, "v": range(10)})
    assert _is_binned_from(df, "band", "v") is False


# ══════════════════════════════════════════════════════════
#  END TO END
# ══════════════════════════════════════════════════════════

def test_the_report_no_longer_headlines_a_definition():
    from app.engines.story_engine import generate_story

    rng = np.random.default_rng(1)
    n = 1200
    age = rng.integers(20, 60, n)

    def band(a):
        return ("18-25" if a <= 25 else "26-35" if a <= 35
                else "36-45" if a <= 45 else "46-55" if a <= 55 else "55+")

    df = pd.DataFrame({
        "Age": age,
        "AgeGroup": [band(a) for a in age],
        "Department": rng.choice(["Sales", "R&D"], n),
        "MonthlyIncome": rng.normal(5_000, 900, n).round(0),
    })
    story = generate_story(df)
    titles = " ".join(getattr(i, "title", str(i))
                      for i in (story.top_insights or []))
    assert "differ on Age" not in titles, (
        "a definition was reported as a finding: " + titles[:160])


# ══════════════════════════════════════════════════════════
#  A PERFECT SEPARATION IS EITHER A BINNING OR THE BEST
#  FINDING IN THE FILE
# ══════════════════════════════════════════════════════════

def test_two_populations_that_do_not_overlap_are_not_a_binning():
    """The first version of the data check asked only whether the groups
    overlapped. Two branches whose readings average 60 and 120 do not
    overlap either — and that is the most valuable finding in the file.
    It was suppressed.

    Bands tile a range, so consecutive bands touch. Two different
    populations leave a hole between them.
    """
    rng = np.random.default_rng(217)
    n = 400
    seg = rng.choice(["Alpha", "Beta"], n)
    val = np.where(seg == "Alpha", rng.normal(60, 8, n),
                   rng.normal(120, 8, n))
    df = pd.DataFrame({"branch": seg, "reading": val.round(2)})
    assert _is_binned_from(df, "branch", "reading") is False


def test_that_gap_still_reaches_the_report():
    from app.engines.domains.base import col_stats
    from app.engines.domains.general import _insights_general

    rng = np.random.default_rng(217)
    n = 400
    seg = rng.choice(["Alpha", "Beta"], n)
    val = np.where(seg == "Alpha", rng.normal(60, 8, n),
                   rng.normal(120, 8, n))
    df = pd.DataFrame({"branch": seg, "reading": val.round(2),
                       "other": rng.normal(0, 1, n)})
    stats = {c: col_stats(df[c]) for c in df.columns
             if pd.api.types.is_numeric_dtype(df[c])}
    out = _insights_general(df, stats, [])
    assert any("sits below" in r for r in out["risks"]), \
        "a two-fold gap between real groups was suppressed"


def test_bands_that_touch_are_still_a_binning():
    """The bands of a cut leave only rounding between them."""
    rng = np.random.default_rng(3)
    income = rng.integers(1_000, 20_000, 900)
    df = pd.DataFrame({
        "MonthlyIncome": income,
        "SalarySlab": pd.cut(income, [0, 5_000, 10_000, 15_000, 20_001],
                             labels=["<5k", "5-10k", "10-15k", "15k+"]
                             ).astype(str)})
    assert _is_binned_from(df, "SalarySlab", "MonthlyIncome") is True
