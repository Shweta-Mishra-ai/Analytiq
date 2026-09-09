"""
Who does the outcome happen to?

Every domain in this app has a binary outcome sitting in a column —
returned, churned, attrition, readmission, won — and until this layer
existed, nothing computed the rate of that outcome across anything. The
engines reached for a comparison of the MEAN OF A NUMERIC COLUMN between
groups, so a dataset where apparel is returned four times as often came
back as "Apparel and Electronics differ on Discount": true, and not the
finding anybody needed.

These tests plant an effect and check it comes back — and, just as
importantly, check the things that must NOT come back: an effect too
small to matter, one too uncertain to claim, and a column that defines
the outcome rather than driving it.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.domains._common import (
    binary_mask, describe_gap, outcome_rates, rate_insights,
)


def _frame(n=1200, seed=7):
    r = np.random.default_rng(seed)
    plan = r.choice(["Free", "Pro", "Enterprise"], n, p=[0.5, 0.35, 0.15])
    churn_p = np.where(plan == "Free", 0.32,
                       np.where(plan == "Pro", 0.10, 0.045))
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "plan": plan,
        "region": r.choice(["North", "South"], n),
        "tenure_months": r.integers(1, 40, n),
        "churned": (r.random(n) < churn_p).astype(int),
    })


# ── what it must find ─────────────────────────────────────

def test_it_finds_the_group_the_outcome_concentrates_in():
    gaps = outcome_rates(_frame(), "churned")
    drivers = [g.driver for g in gaps]
    assert "plan" in drivers, drivers

    plan = next(g for g in gaps if g.driver == "plan")
    assert plan.high_label == "Free"
    assert plan.low_label == "Enterprise"
    assert plan.high_rate > plan.overall > plan.low_rate
    assert plan.lift and plan.lift > 3


def test_a_small_group_is_still_reported_when_the_effect_is_real():
    """The enterprise tier is the smallest group and churns least, and an
    earlier version dropped the whole comparison because one cell of the
    contingency table held four records. That rule discarded a 33-point
    spread — the largest effect in the data."""
    gaps = outcome_rates(_frame(), "churned")
    plan = next(g for g in gaps if g.driver == "plan")
    assert plan.low_n < 250, "expected the small tier to be the low group"
    assert plan.significant


def test_a_numeric_driver_comes_back_as_readable_bands():
    gaps = outcome_rates(_frame(), "churned")
    bands = [g for g in gaps if g.kind == "band"]
    if bands:
        label = bands[0].high_label
        assert "(" not in label and "]" not in label, \
            "a raw pandas interval reached a report title: " + label


# ── what it must NOT claim ────────────────────────────────

def test_an_evenly_spread_outcome_produces_nothing():
    """An empty result is a real answer. It says the outcome does not
    concentrate, which is worth knowing and is not the same as not
    having looked."""
    r = np.random.default_rng(3)
    df = pd.DataFrame({
        "group": r.choice(list("ABCD"), 2000),
        "value": r.normal(0, 1, 2000),
        "flag": (r.random(2000) < 0.2).astype(int),
    })
    assert outcome_rates(df, "flag") == []


def test_a_gap_too_small_to_act_on_is_not_reported():
    r = np.random.default_rng(5)
    group = r.choice(["A", "B"], 4000)
    # Two points apart on a huge sample: significant, and not worth a
    # paragraph.
    df = pd.DataFrame({
        "group": group,
        "flag": (r.random(4000) < np.where(group == "A", 0.21, 0.19))
                .astype(int),
    })
    assert [g for g in outcome_rates(df, "flag") if g.driver == "group"] == []


def test_the_extremes_of_many_groups_are_corrected_for_the_search():
    """With twenty groups of pure noise there are 190 possible pairs, and
    the widest of them looks striking on chance alone. Picking the best
    and worst after seeing the data is a search, and the p-value has to
    know that."""
    r = np.random.default_rng(11)
    df = pd.DataFrame({
        "rep": r.choice([f"R{i:02d}" for i in range(20)], 3000),
        "flag": (r.random(3000) < 0.3).astype(int),
    })
    assert outcome_rates(df, "flag") == []


def test_a_column_that_defines_the_outcome_is_not_called_a_driver():
    """A pass flag derived as `score >= 50` makes score a perfect
    predictor of passing. Reporting "pass rate is 72% below a score of 60
    against 100% above it" is a tautology dressed as an insight, and one
    of those in a client report costs more than the rest of the page
    earns."""
    r = np.random.default_rng(13)
    score = r.normal(58, 15, 1500)
    df = pd.DataFrame({
        "module": r.choice(["Stats", "Law", "Marketing"], 1500),
        "score": score.round(1),
        "passed": (score >= 50).astype(int),
    })
    drivers = [g.driver for g in outcome_rates(df, "passed")]
    assert "score" not in drivers, drivers


def test_an_identifier_is_never_a_driver():
    r = np.random.default_rng(17)
    df = pd.DataFrame({
        "customer_id": range(1, 1201),
        "plan": r.choice(["A", "B"], 1200),
        "churned": (r.random(1200) < 0.3).astype(int),
    })
    assert "customer_id" not in [g.driver for g in outcome_rates(df, "churned")]


def test_a_column_that_is_not_binary_is_refused():
    df = pd.DataFrame({"g": list("ABAB") * 50, "count": range(200)})
    assert outcome_rates(df, "count") == []


# ── how it reads ──────────────────────────────────────────

def test_the_sentence_names_the_column_not_just_the_value():
    """"concentrates in 'Yes'" is a finding nobody can act on: yes of
    what? The driver has to travel with the value."""
    gaps = outcome_rates(_frame(), "churned")
    plan = next(g for g in gaps if g.driver == "plan")
    assert "plan" in describe_gap(plan, "churn")


def test_a_bad_outcome_blames_the_high_group():
    out = rate_insights(_frame(), "churned", "churn", good=False)
    assert out["insights"]
    assert out["insights"][0].title.startswith("Highest churn")
    assert "Free" in out["insights"][0].title


def test_a_good_outcome_blames_the_low_group():
    """Marked good, a 42% win rate must not be reported as a critical
    problem — which is exactly what happened before direction was part of
    the model."""
    r = np.random.default_rng(19)
    rep = r.choice([f"Rep {i:02d}" for i in range(1, 5)], 1600)
    skill = {"Rep 01": 0.08, "Rep 02": 0.25, "Rep 03": 0.35, "Rep 04": 0.45}
    df = pd.DataFrame({
        "sales_rep": rep,
        "won": (r.random(1600) < np.array([skill[x] for x in rep]))
               .astype(int),
    })
    out = rate_insights(df, "won", "win rate", good=True)
    assert out["insights"]
    title = out["insights"][0].title
    assert title.startswith("Lowest win rate"), title
    assert "Rep 01" in title, title


def test_every_insight_carries_an_action_and_its_evidence():
    out = rate_insights(_frame(), "churned", "churn")
    for insight in out["insights"]:
        assert insight.action.strip()
        assert "p = " in insight.evidence
        assert insight.impact.strip()
    assert out["actions"] and out["opportunities"]


@pytest.mark.parametrize("column", ["churned", "not_a_column"])
def test_it_never_raises_on_an_odd_frame(column):
    assert isinstance(outcome_rates(pd.DataFrame(), column), list)
    assert isinstance(outcome_rates(_frame().head(3), column), list)


def test_binary_mask_and_binary_rate_agree_on_which_side_is_yes():
    """A report that says "18% churn" over a table where the churned
    group is the other one is worse than either number alone."""
    from app.engines.domains._common import binary_rate

    for values in ([0, 1, 1, 0] * 50, ["Yes", "No"] * 100,
                   [True, False] * 100):
        series = pd.Series(values)
        mask = binary_mask(series)
        assert mask is not None
        assert round(float(mask.mean() * 100), 2) == binary_rate(series)


def test_the_shared_pass_does_not_repeat_the_engines_own_finding():
    """A report on HR data listed the same fact twice:

        1. 'Support' Department: 28% Attrition vs 12% Best
        3. Highest attrition: Department 'Support' at 27.6% against 11.9%

    One finding, two roundings of the same number, from the domain
    engine and the shared outcome pass respectively. A reader does not
    read that as corroboration.
    """
    from app.engines.domains.registry import _drop_repeats
    from app.engines.domains.base import build_insight

    def made(title):
        return build_insight(title=title, problem="", cause="", evidence="",
                             action="", impact="", severity="high")

    engine = {"insights": [made("'Support' Department: 28% Attrition vs 12% Best")]}
    shared = {
        "insights": [
            made("Highest attrition: Department 'Support' at 27.6% against 11.9%"),
            made("Highest attrition: OverTime 'Yes' at 35.5% against 8.9%"),
        ],
        "findings": ["a"], "risks": [], "opportunities": [], "actions": ["b"],
    }
    kept = _drop_repeats(shared, engine)
    titles = [i.title for i in kept["insights"]]
    assert len(titles) == 1
    assert "OverTime" in titles[0], titles


def test_a_different_group_in_the_same_column_still_survives():
    """Dropping every finding that mentions a column the engine touched
    would throw away real information. Only the same column AND the same
    group is a repeat."""
    from app.engines.domains.registry import _drop_repeats
    from app.engines.domains.base import build_insight

    def made(title):
        return build_insight(title=title, problem="", cause="", evidence="",
                             action="", impact="", severity="high")

    engine = {"insights": [made("'Support' Department: 28% Attrition")]}
    shared = {"insights": [
        made("Highest attrition: Department 'Finance' at 22.0% against 9.0%")],
        "findings": [], "risks": [], "opportunities": [], "actions": []}
    assert len(_drop_repeats(shared, engine)["insights"]) == 1
