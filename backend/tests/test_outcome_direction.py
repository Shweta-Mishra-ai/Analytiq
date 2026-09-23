"""
Which way the predicted column points, and what the report may say
about it.

A model predicts a column; whether a high rate of that column is good
news is not a property of the model. Every predictive section was
written for churn, so on a sales extract — where the target is `won` —
the report named the two best reps in the company as the highest-risk
segment and offered to help avoid ninety-eight of their wins. The
arithmetic was right and every sentence around it was backwards.
"""
from __future__ import annotations

import pytest

from app.engines.domains.base import higher_is_better
from app.engines.outcome_direction import direction_for


# ══════════════════════════════════════════════════════════
#  Reading the column name
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("column", [
    "won", "is_won", "closed_won", "converted", "renewed", "retained",
    "survived", "completed", "approved", "conversion", "retention",
])
def test_a_good_outcome_is_recognised(column):
    assert higher_is_better(column) is True
    assert direction_for(column).desirable is True


@pytest.mark.parametrize("column", [
    "churned", "attrition", "terminated", "resigned", "defaulted",
    "cancelled", "readmitted", "rejected", "is_fraud", "defect_flag",
])
def test_a_bad_outcome_is_recognised(column):
    assert higher_is_better(column) is False
    assert direction_for(column).desirable is False


@pytest.mark.parametrize("column", [
    "status_flag", "active", "flag", "turnover", "category_b", "outcome",
])
def test_a_column_that_does_not_say_is_left_alone(column):
    """The third case earns its place. Guessing is what produced the
    wrong report; a column whose name says nothing gets neutral wording
    and no opinion about which end is the good one."""
    assert higher_is_better(column) is None
    assert direction_for(column).desirable is None


def test_the_inflections_real_columns_actually_use_are_covered():
    """`won` is how a sales extract names its outcome, and it matched
    nothing: the token list held `win` and `wins`. Same for `churned`
    against `churn`, `renewed` against `renewals`."""
    for stem, inflected in [("win", "won"), ("churn", "churned"),
                            ("renewals", "renewed"),
                            ("conversion", "converted")]:
        assert higher_is_better(stem) == higher_is_better(inflected), stem


def test_turnover_still_refuses_to_guess():
    """It means revenue in finance and staff attrition in HR. Claiming
    either would be wrong half the time."""
    assert higher_is_better("turnover") is None


# ══════════════════════════════════════════════════════════
#  What the report is then allowed to say
# ══════════════════════════════════════════════════════════

def test_a_win_is_never_described_as_a_risk_to_avoid():
    """The headline regression."""
    D = direction_for("won")
    prose = " ".join([
        D.section_title, D.section_sub, D.cluster_label, D.cluster_tail,
        D.segment_label, D.segment_tail, D.heatmap_title,
        D.heatmap_caption, D.bands_intro, D.base_sub,
    ]).lower()

    for word in ("risk", "avoidable", "intervention", "affected"):
        assert word not in prose, \
            "a desirable outcome was described with '{}'".format(word)


def test_a_churn_model_still_speaks_of_risk():
    """The fix must not flatten everything into neutral wording — where
    the outcome really is a hazard, saying so is the useful thing."""
    D = direction_for("churned")

    assert "Risk" in D.section_title
    assert "risk" in D.cluster_label.lower()
    assert "avoidable" in D.gap_label


def test_the_neutral_case_claims_neither():
    D = direction_for("status_flag")
    prose = (D.cluster_tail + " " + D.segment_tail).lower()

    assert "avoidable" not in prose
    assert "opportunity" not in prose
    # It should say plainly that it does not know.
    assert "does not say" in prose or "depends on what" in prose


def test_every_direction_fills_every_field():
    """A half-populated vocabulary would print an empty heading."""
    for column in ("won", "churned", "status_flag"):
        D = direction_for(column)
        for field in D.__dataclass_fields__:
            if field == "desirable":
                continue
            value = getattr(D, field)
            assert isinstance(value, str) and value.strip(), (column, field)


def test_the_section_subtitle_takes_the_target_name():
    assert "Won" in direction_for("won").section_sub.format("Won")
