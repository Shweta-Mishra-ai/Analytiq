"""
Defects found by reading a real twenty-four-page report rather than
counting its pages.

Each of these shipped. A client reading the sales report saw one
finding listed twice as two separate risks, a quantity given as "55
more win rate", a sentence running into the next without a full stop,
a contents entry with no page number, and an HR report headed "Hr
Performance Analysis".
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def sales_story():
    from app.engines.story_engine import generate_story

    df = pd.DataFrame({
        "opportunity_id": range(1, 801),
        "sales_rep": (["Rep 01"] * 200 + ["Rep 02"] * 200
                      + ["Rep 07"] * 200 + ["Rep 08"] * 200),
        "region": ["East", "West", "North", "South"] * 200,
        "stage": ["Qualify", "Propose", "Negotiate", "Qualify"] * 200,
        "deal_size": [8000 + (i % 40) * 900 for i in range(800)],
        "days_in_pipeline": [30 + (i % 120) for i in range(800)],
        # Rep 01 wins 10% of the time, Rep 08 wins 60%.
        "won": ([1] * 20 + [0] * 180) + ([1] * 50 + [0] * 150)
               + ([1] * 60 + [0] * 140) + ([1] * 120 + [0] * 80),
    })
    return generate_story(df)


# ══════════════════════════════════════════════════════════
#  One finding, stated once
# ══════════════════════════════════════════════════════════

def test_the_same_finding_is_not_listed_as_two_risks(sales_story):
    """The domain engine and the general top-up both run an outcome
    pass over the same column under different names for it, so the
    Executive Summary carried the same gap twice — identical but for
    the noun, which is why comparing whole strings found nothing."""
    from app.engines.domains.registry import _fingerprint

    marks = [_fingerprint(r) for r in sales_story.business_risks]
    marks = [m for m in marks if m]

    assert len(marks) == len(set(marks)), (
        "the same group and rate appears in two risks:\n"
        + "\n".join(sales_story.business_risks))


def test_a_different_group_is_still_reported_separately():
    """The de-duplication must be narrow. Two findings about two
    different reps are two findings."""
    from app.engines.domains.registry import _fingerprint

    one = "Win rate in sales_rep 'Rep 01' is 9.5% against 25.3% overall."
    two = "Win rate in sales_rep 'Rep 04' is 18.0% against 25.3% overall."
    same = "Won in sales_rep 'Rep 01' is 9.5% against 25.3% overall."

    assert _fingerprint(one) != _fingerprint(two)
    assert _fingerprint(one) == _fingerprint(same)


# ══════════════════════════════════════════════════════════
#  Sentences that parse
# ══════════════════════════════════════════════════════════

def test_the_verdict_does_not_run_two_sentences_together(sales_story):
    """An insight's `problem` is stored as a clause with no terminal
    punctuation, so interpolating it produced "...in sales_rep 'Rep 08'
    This is the most urgent finding" on the first two pages."""
    headline = sales_story.headline or ""
    assert " This is the most urgent" not in headline, headline
    if "This is the most urgent" in headline:
        assert ". This is the most urgent" in headline


def test_a_quantity_is_never_given_in_units_of_a_rate(sales_story):
    """Several domains name their outcome as a rate — "win rate",
    "pass rate", "on-time delivery" — and the gap sentence read "about
    55 more win rate", which is not a quantity of anything."""
    for line in sales_story.opportunities:
        low = line.lower()
        for rate_noun in ("more win rate", "fewer win rate",
                          "more pass rate", "more conversion rate",
                          "more on-time delivery"):
            assert rate_noun not in low, line


def test_the_gap_is_counted_in_records(sales_story):
    hits = [o for o in sales_story.opportunities if "Moving" in o]
    if hits:
        assert "records" in hits[0], hits[0]


# ══════════════════════════════════════════════════════════
#  The contents page
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("domain,expected", [
    ("sales", "Sales Performance Analysis"),
    ("hr", "HR Performance Analysis"),
    ("saas", "SaaS Performance Analysis"),
    ("general", "Business Performance Analysis"),
])
def test_the_deep_page_title_reads_like_english(domain, expected):
    """`.title()` was applied to a label that already carried its own
    capitals, so every HR report was headed "Hr Performance Analysis"
    and every SaaS one "Saas"."""
    from app.engines.pdf.performance_page import deep_page_title

    assert deep_page_title(domain) == expected


def test_the_contents_entry_matches_the_heading_it_points_at():
    """The page-number map is keyed on the heading text. The contents
    built its own version of the string — "Sales Analysis" against the
    page's "Sales Performance Analysis" — so the entry matched nothing
    and printed with the number column blank."""
    from app.engines.pdf.performance_page import deep_page_title

    # One function, used by both. If the contents ever hard-codes its
    # own spelling again, this is what notices.
    import inspect
    from app.engines.pdf import builder

    source = inspect.getsource(builder.build_pdf)
    assert "deep_page_title(domain)" in source
    assert '"{} Analysis".format' not in source
    assert deep_page_title("sales") == "Sales Performance Analysis"


def test_the_predictive_contents_entry_follows_the_outcome():
    """It used to be the literal string "Predictive Risk Analysis", so
    a sales report listed that in its contents and then printed
    "Predictive Opportunity Analysis" on the page — and, because the
    page map is keyed on the heading, no page number either."""
    from app.engines.outcome_direction import direction_for

    assert direction_for("won").section_title == \
        "Predictive Opportunity Analysis"
    assert direction_for("churned").section_title == \
        "Predictive Risk Analysis"


# ══════════════════════════════════════════════════════════
#  A flag is a rate, not a shape
# ══════════════════════════════════════════════════════════

def test_a_two_value_column_gets_no_distribution_verdict():
    """`won` scores a skew of 1.13 because a quarter of its rows are 1,
    so the Distribution Summary called it "heavily right-skewed" and
    added that its average was "pulled up by a few unusually high
    values". Both true, both meaningless, and together they imply a
    tail that does not exist."""
    from app.engines.stats_engine import _numeric_stats

    won = pd.Series([1] * 253 + [0] * 747)
    cs = _numeric_stats(won, "won")

    assert cs.is_binary is True
    assert cs.binary_rate == pytest.approx(25.3, abs=0.1)
    assert cs.binary_values == (0.0, 1.0)
    assert cs.skew_label is None, "a flag has no skew worth naming"
    assert cs.is_normal is None, "normality of a two-value column is noise"
    assert cs.outlier_count_iqr == 0


def test_a_continuous_column_is_still_described_as_one():
    """The guard must not swallow the columns that do have a shape —
    a right-skewed money column is exactly where the median warning
    earns its place."""
    from app.engines.stats_engine import _numeric_stats

    # Genuinely continuous: many distinct values with a long right tail.
    deal = pd.Series([1000 + i * 7 for i in range(200)]
                     + [80000 + i * 500 for i in range(20)])
    cs = _numeric_stats(deal, "deal_size")

    assert cs.is_binary is False
    assert cs.skew_label and "skew" in cs.skew_label


def test_a_two_value_column_that_is_not_zero_one_is_still_a_flag():
    """Status recorded as 1/2, or a price at two levels, has no more
    distribution than 0/1 does."""
    from app.engines.stats_engine import _numeric_stats

    cs = _numeric_stats(pd.Series([10] * 60 + [20] * 40), "tier")

    assert cs.is_binary is True
    assert cs.binary_values == (10.0, 20.0)
    assert cs.binary_rate == pytest.approx(40.0, abs=0.1)
