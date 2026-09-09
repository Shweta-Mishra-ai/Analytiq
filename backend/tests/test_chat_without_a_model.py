"""
Chat, with no API key set.

The Chat page generates four questions from the user's own columns and
shows them as buttons. Before this, every one of them returned 503 when
no model was configured — the page offered questions it could not answer,
which is worse than offering none.

All four are a group-by, a sort or a correlation: arithmetic the app
already does. The model was only ever mapping the sentence to a tool
call, so that mapping happens locally first.

These tests cover both halves: the questions it must answer, and the
questions it must refuse rather than guess at. The second half matters
more. A parser that reaches for the nearest column when unsure produces a
confident answer about the wrong thing, and the first version of this one
did exactly that.
"""
import numpy as np
import pandas as pd
import pytest

from app.ai.intent_parser import answerable_examples, parse


@pytest.fixture()
def hr():
    r = np.random.default_rng(4)
    n = 400
    return pd.DataFrame({
        "EmployeeID": range(1, n + 1),
        "Department": r.choice(["Sales", "Engineering", "Support"], n),
        "OverTime": r.choice(["Yes", "No"], n),
        "MonthlyIncome": r.normal(6500, 1500, n).round(0),
        "YearsAtCompany": r.integers(0, 20, n),
        "Age": r.integers(21, 60, n),
    })


# ── the four the page actually offers ─────────────────────

def test_which_group_has_the_highest_average(hr):
    got = parse("Which Department has the highest average Monthly Income?", hr)
    assert got["tool"] == "aggregate"
    assert got["params"] == {"group_col": "Department",
                             "value_col": "MonthlyIncome",
                             "agg_func": "mean"}


def test_show_the_top_rows(hr):
    got = parse("Show the 10 rows with the highest Monthly Income", hr)
    assert got["tool"] == "top_n"
    assert got["params"]["sort_col"] == "MonthlyIncome"
    assert got["params"]["n"] == 10
    assert got["params"]["ascending"] is False


def test_compare_a_measure_across_a_dimension(hr):
    got = parse("Compare Years at Company across Department", hr)
    assert got["tool"] == "aggregate"
    assert got["params"]["group_col"] == "Department"
    assert got["params"]["value_col"] == "YearsAtCompany"


def test_which_columns_move_together(hr):
    got = parse("Which columns move together?", hr)
    assert got["tool"] == "plot_heatmap"


# ── the regression that matters most ──────────────────────

def test_a_column_name_is_not_matched_inside_another_word(hr):
    """`Age` sits inside "aver-age-".

    The first version flattened the question to
    "whichdepartmenthasthehighestaveragemonthlyincome" and searched for
    substrings, so it found Age and answered "Mean Age by Department" —
    the right shape of answer about entirely the wrong column.
    """
    got = parse("Which Department has the highest average Monthly Income?", hr)
    assert got["params"]["value_col"] == "MonthlyIncome"
    assert got["params"]["value_col"] != "Age"


def test_a_shorter_column_does_not_claim_a_longer_one(hr):
    """With both `Income` and `MonthlyIncome` present, the longer name
    owns the words it covers."""
    df = hr.assign(Income=hr["MonthlyIncome"] / 12)
    got = parse("Compare Monthly Income across Department", df)
    assert got["params"]["value_col"] == "MonthlyIncome"


@pytest.mark.parametrize("written", [
    "Monthly Income", "monthly_income", "MonthlyIncome", "monthly income",
])
def test_a_column_is_recognised_however_it_is_written(hr, written):
    got = parse("Compare {} across Department".format(written), hr)
    assert got is not None
    assert got["params"]["value_col"] == "MonthlyIncome"


# ── what it must refuse ───────────────────────────────────

def test_a_question_it_cannot_map_returns_none(hr):
    """None means "hand it to a model", never a guess."""
    assert parse("What is the capital of France?", hr) is None
    assert parse("Why did revenue fall last quarter?", hr) is None


def test_a_question_naming_no_column_is_refused(hr):
    assert parse("Which one has the highest average?", hr) is None


def test_an_empty_frame_or_question_is_refused(hr):
    assert parse("", hr) is None
    assert parse("Compare Monthly Income across Department",
                 pd.DataFrame()) is None


def test_it_never_raises_on_odd_input(hr):
    for q in ("...", "SELECT * FROM x", "🙂", "a" * 5000, "highest lowest top"):
        parse(q, hr)          # must not raise


# ── the helpful refusal ───────────────────────────────────

def test_the_examples_offered_are_answerable_on_this_frame(hr):
    """The message shown when no model is configured lists what still
    works. Every one of those has to actually work, or it is the same
    broken promise in a different place."""
    examples = answerable_examples(hr)
    assert examples
    for question in examples:
        assert parse(question, hr) is not None, question


def test_the_examples_name_this_frames_columns(hr):
    joined = " ".join(answerable_examples(hr))
    assert any(c in joined for c in hr.columns)


# ── aggregates and directions ─────────────────────────────

@pytest.mark.parametrize("phrase,func", [
    ("total", "sum"), ("average", "mean"), ("median", "median"),
])
def test_the_aggregate_word_chooses_the_function(hr, phrase, func):
    got = parse("What is the {} Monthly Income by Department?".format(phrase),
                hr)
    assert got["params"]["agg_func"] == func


def test_lowest_sorts_the_other_way(hr):
    got = parse("Show the 5 rows with the lowest Age", hr)
    assert got["params"]["ascending"] is True
    assert got["params"]["n"] == 5


def test_a_silly_row_count_is_clamped(hr):
    got = parse("Show the 99999 rows with the highest Age", hr)
    assert got["params"]["n"] <= 200
