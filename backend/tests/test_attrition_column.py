"""
Reading the one column an HR report is about.

Four engines found it for themselves — `_find_left_mask` and
`_run_attrition` in the HR engine, and twice more in `insights_builder`
— with four different keyword lists. Worse, they disagreed about how to
read it. The HR engine normalised "Yes"/"No" to a boolean; both copies
in `insights_builder` called `.mean()` on the raw column, which is right
for a 0/1 spelling and raises on the Yes/No one that an HRIS export
normally produces. So the HR headline simply never appeared on half the
files it was written for.

That failure was hidden behind a second one: `insights_builder` declared
its own `Insight` dataclass, identical to the shared one except that it
had no `category` field — and all ten `Insight(...)` calls in the module
pass `category=`. Every one raised TypeError. Nothing noticed, because
the function returns early whenever the story engine produced insights
of its own, which it nearly always does. The fallback path the whole
module exists to provide was dead on arrival.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from app.engines.column_roles import left_mask, resolve


def _hr(spelling, n=500, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "employee_id": range(n),
        "department": rng.choice(["Sales", "Eng", "Ops"], n),
        "salary": rng.choice(["low", "medium", "high"], n),
        "tenure_years": rng.integers(0, 20, n),
    })
    left = df.salary.eq("low") & (rng.random(n) < 0.5)
    df["Attrition"] = [spelling[1] if v else spelling[0] for v in left]
    return df


# ══════════════════════════════════════════════════════════
#  Every spelling reads the same
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("spelling", [
    ("No", "Yes"),
    ("no", "yes"),
    (0, 1),
    (False, True),
    ("Active", "Terminated"),
    ("Stayed", "Resigned"),
])
def test_the_rate_is_the_same_however_it_is_stored(spelling):
    df = _hr(spelling)
    found = left_mask(df)
    assert found is not None, spelling
    col, mask = found
    assert col == "Attrition"
    # The same underlying employees, so the same rate every time.
    assert 0.10 < float(mask.mean()) < 0.25, (spelling, mask.mean())


def test_the_yes_no_spelling_used_to_raise():
    """`.mean()` on an object column of "Yes"/"No"."""
    df = _hr(("No", "Yes"))
    with pytest.raises(TypeError):
        float(df["Attrition"].mean())
    assert left_mask(df) is not None


@pytest.mark.parametrize("name", [
    "Attrition", "attrition_flag", "churned", "has_left",
    "employee_exited", "Terminated",
])
def test_the_column_is_found_under_any_name(name):
    rng = np.random.default_rng(4)
    n = 300
    df = pd.DataFrame({"dept": rng.choice(["a", "b"], n),
                       name: rng.choice(["Yes", "No"], n)})
    found = left_mask(df)
    assert found and found[0] == name


# ══════════════════════════════════════════════════════════
#  Refusing
# ══════════════════════════════════════════════════════════

def test_a_tenure_column_is_not_a_flag():
    """`months_to_exit` holding 0-40 averaged as a flag produced an
    attrition rate out of nowhere."""
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"months_to_exit": rng.integers(0, 40, 300),
                       "x": rng.normal(0, 1, 300)})
    assert left_mask(df) is None


def test_a_frame_with_no_attrition_column_returns_none():
    df = pd.DataFrame({"revenue": np.linspace(1, 100, 60),
                       "region": ["a", "b"] * 30})
    assert left_mask(df) is None
    assert resolve(df).attrition is None


def test_a_column_where_nobody_left_is_refused():
    df = pd.DataFrame({"attrition": ["No"] * 200,
                       "dept": ["a", "b"] * 100})
    assert left_mask(df) is None


def test_an_empty_frame_does_not_raise():
    assert left_mask(pd.DataFrame()) is None


# ══════════════════════════════════════════════════════════
#  One definition, used everywhere
# ══════════════════════════════════════════════════════════

def test_the_hr_engine_and_the_resolver_agree():
    from app.engines.domains.hr import _find_left_mask

    df = _hr(("No", "Yes"))
    assert _find_left_mask(df).equals(left_mask(df)[1])


def test_the_attrition_analysis_reads_a_yes_no_column():
    from app.engines.domains.hr import _run_attrition

    result = _run_attrition(_hr(("No", "Yes")))
    assert result is not None
    assert 10 < result.rate < 25, result.rate


def test_the_analysis_agrees_across_spellings():
    from app.engines.domains.hr import _run_attrition

    yes_no = _run_attrition(_hr(("No", "Yes")))
    zero_one = _run_attrition(_hr((0, 1)))
    assert abs(yes_no.rate - zero_one.rate) < 0.01


# ══════════════════════════════════════════════════════════
#  The duplicate dataclass
# ══════════════════════════════════════════════════════════

def test_there_is_one_insight_type():
    from app.engines.domains.base import Insight as BaseInsight
    from app.engines.insights_builder import Insight as BuilderInsight

    assert BuilderInsight is BaseInsight


def test_an_insight_can_carry_its_category():
    """Every `Insight(...)` in insights_builder passes `category=`, and
    the local dataclass had no such field."""
    from app.engines.insights_builder import Insight

    fields = {f.name for f in dataclasses.fields(Insight)}
    assert "category" in fields


def test_the_hr_fallback_path_actually_runs():
    """It raised TypeError on every call, behind an early return that
    almost always fired first."""
    from app.engines.insights_builder import build_top_insights

    insights = build_top_insights(df=_hr(("No", "Yes")), domain="hr")
    assert insights
    assert any(i.category == "attrition" for i in insights), \
        [i.category for i in insights]


def test_the_dict_conversion_path_does_not_raise():
    from app.engines.insights_builder import build_top_insights

    story = type("S", (), {"top_insights": [
        {"title": "t", "severity": "high", "category": "attrition"}]})()
    out = build_top_insights(df=_hr((0, 1)), domain="hr", story_obj=story)
    assert out[0].category == "attrition"


def test_the_salary_band_insight_groups_without_raising():
    """It indexed a groupby with a Series, which is not valid pandas."""
    from app.engines.insights_builder import build_top_insights

    insights = build_top_insights(df=_hr(("No", "Yes")), domain="hr")
    titles = " ".join(i.title for i in insights)
    assert "Low-Salary" in titles or "Attrition" in titles, titles
