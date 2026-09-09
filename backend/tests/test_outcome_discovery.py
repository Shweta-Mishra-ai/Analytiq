"""
The outcome nobody declared.

A domain engine finds its outcome by name: the HR engine knows the word
`Attrition`, the SaaS engine knows `churned`. An ordinary upload matches
no domain, falls to the general engine, and used to get a row count, one
median difference and a correlation — even when a `rework` flag sat in
the file running at 31% for one site against 11% for the rest, which is
the only line in that report anyone would act on.

These tests cover the discovery: which columns count as an outcome, which
must not, what each one is called in a sentence, and which end of it the
report treats as the problem. The last one matters most — a win rate
reported by its best group is an alarm about the best salesperson in the
business, and a flag whose name says nothing must get no verdict at all.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.domains.outcome_discovery import (
    _direction, _noun_for, discover_outcomes,
)


def _ops(n=2000, seed=5):
    r = np.random.default_rng(seed)
    site = r.choice(["Alpha", "Bravo", "Delta"], n, p=[.5, .3, .2])
    return pd.DataFrame({
        "record_id": range(1, n + 1),
        "site": site,
        "duration_min": r.normal(45, 9, n).round(1),
        "rework": (r.random(n) < np.where(site == "Delta", .31, .11))
                  .astype(int),
    })


# ── what counts as an outcome ─────────────────────────────

def test_a_flag_with_no_domain_keyword_is_still_found():
    found = discover_outcomes(_ops())
    assert [o.column for o in found] == ["rework"]


@pytest.mark.parametrize("values", [
    [0, 1], ["Yes", "No"], [True, False], ["Y", "N"],
])
def test_a_flag_is_found_however_it_was_exported(values):
    r = np.random.default_rng(9)
    n = 1000
    df = pd.DataFrame({
        "group": r.choice(list("AB"), n),
        "outcome": r.choice(values, n),
    })
    assert [o.column for o in discover_outcomes(df)] == ["outcome"]


def test_the_more_balanced_flag_leads():
    """A flag at 30% has two populated sides to compare. One at 2% is a
    rare-event problem that a rate table answers badly."""
    r = np.random.default_rng(11)
    n = 3000
    df = pd.DataFrame({
        "group": r.choice(list("ABC"), n),
        "rare_defect": (r.random(n) < .03).astype(int),
        "balanced_defect": (r.random(n) < .35).astype(int),
    })
    assert discover_outcomes(df)[0].column == "balanced_defect"


# ── what must not count ───────────────────────────────────

def test_an_identifier_is_never_an_outcome():
    df = _ops().assign(customer_id=lambda d: range(len(d)))
    assert "customer_id" not in [o.column for o in discover_outcomes(df)]


def test_a_flag_that_almost_never_fires_is_left_alone():
    """Forty events across six segments is a story about sampling."""
    r = np.random.default_rng(13)
    n = 4000
    df = pd.DataFrame({
        "group": r.choice(list("ABCDEF"), n),
        "catastrophe": (r.random(n) < .004).astype(int),
    })
    assert discover_outcomes(df) == []


def test_a_constant_column_is_not_an_outcome():
    df = _ops().assign(active=1)
    assert "active" not in [o.column for o in discover_outcomes(df)]


def test_a_small_file_is_left_alone():
    """Under a couple of hundred rows there is nothing a group
    comparison can say that survives correction."""
    assert discover_outcomes(_ops(n=80)) == []


def test_a_measure_is_not_mistaken_for_a_flag():
    assert "duration_min" not in [o.column for o in discover_outcomes(_ops())]


def test_only_a_handful_are_reported():
    """A file can carry a dozen flags. A report that works through all of
    them is a data dictionary."""
    r = np.random.default_rng(17)
    n = 2000
    df = pd.DataFrame({"group": r.choice(list("AB"), n)})
    for i in range(9):
        df["flag_{}".format(i)] = (r.random(n) < .3).astype(int)
    assert len(discover_outcomes(df)) <= 2


# ── how it is named and judged ────────────────────────────

@pytest.mark.parametrize("column,noun", [
    ("rework", "rework"),
    ("is_fraud", "fraud"),
    ("OnTimeDelivery", "on time delivery"),
    ("churn_flag", "churn"),
    ("has_complaint", "complaint"),
])
def test_the_column_name_becomes_the_word_in_the_sentence(column, noun):
    assert _noun_for(column) == noun


@pytest.mark.parametrize("column", [
    "churned", "Attrition", "is_fraud", "rework", "returned",
    "no_show", "defect_flag", "cancelled",
])
def test_a_bad_outcome_is_recognised_as_bad(column):
    assert _direction(column) is False, column


@pytest.mark.parametrize("column", [
    "won", "converted", "renewed", "OnTimeDelivery", "approved",
    "is_active", "completed",
])
def test_a_good_outcome_is_recognised_as_good(column):
    assert _direction(column) is True, column


def test_a_negation_reads_as_the_problem_it_names():
    """`no_show` carries "show", and `not_delivered` carries "deliver".
    Read as good outcomes, the report would lead with the group those
    happen to LEAST — exactly backwards."""
    assert _direction("no_show") is False
    assert _direction("not_delivered") is False


def test_a_name_that_says_nothing_gets_no_verdict():
    assert _direction("variant_b") is None
    assert _direction("segment_x") is None


def test_an_unnamed_flag_is_analysed_without_being_judged():
    """The concentration is a fact. Calling it critical is a claim about
    which end of it the business wants, and nothing here supports one."""
    from app.engines.domains._common import rate_insights

    r = np.random.default_rng(19)
    n = 2500
    seg = r.choice(["North", "South", "East"], n)
    df = pd.DataFrame({
        "segment": seg,
        "variant_b": (r.random(n) < np.where(seg == "North", .45, .18))
                     .astype(int),
    })
    outcome = discover_outcomes(df)[0]
    assert outcome.named is False

    out = rate_insights(df, outcome.column, outcome.noun,
                        good=bool(outcome.good), verdict=outcome.named)
    assert out["insights"], "it must still be analysed"
    # A 29-point spread would otherwise be marked critical.
    assert all(i.severity != "critical" for i in out["insights"])
    assert out["risks"] == []
    assert out["opportunities"] == []


def test_a_named_flag_keeps_its_verdict():
    r = np.random.default_rng(19)
    n = 2500
    seg = r.choice(["North", "South", "East"], n)
    df = pd.DataFrame({
        "segment": seg,
        "churned": (r.random(n) < np.where(seg == "North", .45, .18))
                   .astype(int),
    })
    from app.engines.domains._common import rate_insights
    outcome = discover_outcomes(df)[0]
    assert outcome.named and outcome.good is False
    out = rate_insights(df, outcome.column, outcome.noun,
                        good=False, verdict=True)
    assert any(i.severity == "critical" for i in out["insights"])
    assert out["risks"]


# ── the whole reason this exists ──────────────────────────

def test_the_general_engine_now_leads_with_the_rate_gap():
    """The regression this module was written for: an ops export whose
    rework flag ran at three times the rate in one site produced a row
    count, a median difference and a correlation, and never mentioned it.
    """
    from app.engines.domains.general import _insights_general
    from app.engines.stats_engine import DatasetStats

    df = _ops(n=3000)
    stats = DatasetStats(df).all_stats() if hasattr(DatasetStats, "all_stats") \
        else {}
    out = _insights_general(df, stats, [])
    titles = " ".join(i.title for i in out["insights"])
    assert "rework" in titles.lower(), titles
    assert any("Delta" in f for f in out["findings"]), out["findings"]


def test_a_flag_is_not_described_as_a_skewed_measure():
    """"Rework is right-skewed (mean 0.13 against median 0), so the
    median is the fair summary for this column" — true, and about a
    column where the median is always 0 and neither number means
    anything."""
    from app.engines.domains.general import _insights_general

    df = _ops(n=3000)
    stats = {"rework": {"skew": 2.2, "outlier_pct": 12.0, "mean": 0.13,
                        "median": 0.0, "std": 0.34}}
    out = _insights_general(df, stats, [])
    said = " ".join(out["findings"]) + " ".join(
        i.problem for i in out["insights"])
    assert "skew" not in said.lower() or "rework" not in said.lower()


def test_it_never_raises_on_an_odd_frame():
    for df in (pd.DataFrame(), pd.DataFrame({"a": [1, 2, 3]}),
               pd.DataFrame({"a": [None] * 500})):
        assert isinstance(discover_outcomes(df), list)


# ── the report saying one thing once ──────────────────────

def test_the_health_report_states_a_finding_once():
    """Three engines found the same gap and the report printed all three:

        'Support' Department: 28% Attrition vs 12% Best
        Highest attrition: Department 'Support' at 27.6% against 11.9%
        'Support' has the highest attrition: 27.6%

    One fact, three roundings. Matching on the exact title — which is
    what this did — treats three spellings of one number as three
    findings.
    """
    from app.engines.health_engine import _drop_restated

    cards = [
        {"title": "'Support' Department: 28% Attrition vs 12% Best",
         "severity": "critical"},
        {"title": "Highest attrition: OverTime 'Yes' at 35.5% against 8.9%",
         "severity": "critical"},
        {"title": "Highest attrition: Department 'Support' at 27.6% "
                  "against 11.9%", "severity": "critical"},
        {"title": "'Support' has the highest attrition: 27.6%",
         "severity": "warning"},
    ]
    titles = [c["title"] for c in _drop_restated(cards)]
    assert len(titles) == 2, titles
    assert any("OverTime" in t for t in titles)
    assert sum("Support" in t for t in titles) == 1


def test_two_findings_that_merely_share_a_word_both_survive():
    """"attrition" appears in every card of an HR report. Dropping on a
    shared word alone would leave one card and call it a summary."""
    from app.engines.health_engine import _drop_restated

    cards = [
        {"title": "Attrition rate is 16.9% (planning threshold: <10%)",
         "severity": "warning"},
        {"title": "Senior employees (6+ yrs) attrition: 21.2%",
         "severity": "warning"},
        {"title": "Avg satisfaction: 3.04 / 5 (61%)", "severity": "info"},
    ]
    assert len(_drop_restated(cards)) == 3


def test_two_unrelated_findings_at_the_same_percentage_both_survive():
    """A file can hold two 28% figures about entirely different things."""
    from app.engines.health_engine import _drop_restated

    cards = [
        {"title": "'Support' Department: 28% Attrition", "severity": "critical"},
        {"title": "Electronics carries 28% of refunds", "severity": "warning"},
    ]
    assert len(_drop_restated(cards)) == 2


def test_a_finding_with_no_percentage_is_never_treated_as_a_restatement():
    from app.engines.health_engine import _drop_restated

    cards = [
        {"title": "Support has the longest handling times", "severity": "info"},
        {"title": "Support has the largest headcount", "severity": "info"},
    ]
    assert len(_drop_restated(cards)) == 2


def test_one_file_gets_one_quality_score():
    """The analysis PDF printed "100 / 100" on its cover, "99.7 / 100" two
    pages later, and the health card in the app said 100 again — one
    number, three spellings, in a deliverable whose whole claim is that
    every figure traces to a source. They all round the same way now."""
    import numpy as np
    from app.engines.data_profiler import profile_dataset
    from app.engines.health_engine import compute_health
    from app.engines.present import quality_score

    r = np.random.default_rng(23)
    df = pd.DataFrame({
        "site": r.choice(list("ABC"), 800),
        "amount": r.normal(100, 15, 800).round(2),
    })
    df.loc[df.index[:3], "amount"] = np.nan   # enough to move off a round 100

    profiled = float(profile_dataset(df).overall_quality_score)
    assert compute_health(df)["score"] == round(profiled, 1)
    assert quality_score(profiled) == quality_score(compute_health(df)["score"])


# ── the pair the survival page runs on arrival ────────────

def test_the_first_survival_pair_is_the_one_an_analyst_would_pick():
    """The page draws its first curve without asking, so the first pair
    has to be right. Returned in column order, it put MonthlyIncome
    against OverTime and reported "median survival time: 8952.0
    (MonthlyIncome units)" — arithmetically valid, and about nothing.
    """
    import numpy as np
    from app.api.advanced_analytics import (
        _duration_rank, _is_binary_series, _measure_columns,
    )
    from app.engines.ml.targets import _names_an_outcome

    r = np.random.default_rng(1)
    n = 2000
    overtime = r.choice(["Yes", "No"], n, p=[.3, .7])
    df = pd.DataFrame({
        "EmployeeID": range(n),
        "Department": r.choice(["Sales", "Support"], n),
        "OverTime": overtime,
        "MonthlyIncome": r.normal(6500, 1500, n),
        "YearsAtCompany": r.integers(0, 20, n),
        "Attrition": np.where(
            r.random(n) < np.where(overtime == "Yes", .35, .09), "Yes", "No"),
    })
    measures = set(_measure_columns(df))
    durations = sorted(
        (c for c in df.columns
         if c in measures and pd.api.types.is_numeric_dtype(df[c])
         and not _is_binary_series(df[c])),
        key=lambda c: (_duration_rank(c), list(df.columns).index(c)))
    events = sorted(
        (c for c in df.columns if _is_binary_series(df[c])),
        key=lambda c: (0 if _names_an_outcome(c) else 1,
                       list(df.columns).index(c)))

    assert durations[0] == "YearsAtCompany", durations
    assert events[0] == "Attrition", events


@pytest.mark.parametrize("column,elapsed", [
    ("YearsAtCompany", True), ("tenure_months", True), ("days_to_churn", True),
    ("duration_min", True), ("MonthlyIncome", False), ("cost_usd", False),
])
def test_a_duration_column_is_recognised_by_its_name(column, elapsed):
    from app.api.advanced_analytics import _duration_rank
    assert (_duration_rank(column) == 0) is elapsed


# ── what "what predicts this" will model ──────────────────

def test_a_flag_nobody_named_is_still_a_prediction_target():
    """"No binary outcome column detected. Pass ?target=<column> naming a
    two-value column such as churn/attrition/left" — returned for a file
    holding a `rework` flag, which is a two-value column the app was
    looking straight at. The whole Deep Analysis prediction panel was
    unreachable for any file outside the keyword list."""
    from app.engines.predictive import find_binary_target
    assert find_binary_target(_ops(n=1500)) == "rework"


def test_a_recognised_outcome_still_wins_over_an_anonymous_flag():
    from app.engines.predictive import find_binary_target
    df = _ops(n=1500).assign(
        is_remote=lambda d: (d["duration_min"] > 45).astype(int))
    assert find_binary_target(df) == "rework"


def test_a_demographic_is_never_modelled_as_an_outcome():
    """"What predicts Gender" is the kind of panel that loses a reader's
    trust in every number above it."""
    import numpy as np
    from app.engines.predictive import find_binary_target

    r = np.random.default_rng(29)
    df = pd.DataFrame({
        "gender": r.choice(["M", "F"], 600),
        "spend": r.normal(80, 20, 600),
    })
    assert find_binary_target(df) is None
