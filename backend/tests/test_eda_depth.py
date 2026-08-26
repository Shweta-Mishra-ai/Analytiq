"""
The analysis a senior analyst adds on top of the descriptive pass.

The existing EDA reports distributions, correlations, group comparisons
with effect sizes, VIF and trends, and corrects for multiple testing —
all sound, and none of it what makes a finding senior. Four things were
missing: uncertainty around the estimates, interactions, groups too thin
to carry a finding, and class imbalance stated plainly.
"""
import io

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader

from app.engines.eda_depth import (
    MIN_EFFECT_SD, describe_imbalance, find_interactions,
    find_rare_categories, key_estimates, mean_with_ci, proportion_with_ci,
)


# ── uncertainty ───────────────────────────────────────────

def test_a_mean_comes_with_an_interval():
    rng = np.random.default_rng(1)
    est = mean_with_ci(pd.Series(rng.normal(100, 15, 500), name="value"))
    assert est.ci_low < est.value < est.ci_high


def test_a_smaller_sample_gives_a_wider_interval():
    """Otherwise 12 observations claim the precision of 1,200."""
    rng = np.random.default_rng(1)
    small = mean_with_ci(pd.Series(rng.normal(100, 15, 20), name="v"))
    large = mean_with_ci(pd.Series(rng.normal(100, 15, 2000), name="v"))
    assert small.margin > large.margin * 3


def test_too_few_observations_produce_no_estimate():
    assert mean_with_ci(pd.Series([1.0, 2.0], name="v")) is None


def test_a_rate_interval_never_goes_below_zero():
    """The textbook normal interval gives a negative lower bound for a 0%
    churn rate, which is visibly wrong in a client report."""
    rate, low, high = proportion_with_ci(0, 50)
    assert rate == 0.0
    assert low >= 0.0
    assert high > 0.0


def test_a_rate_interval_never_exceeds_one_hundred():
    _rate, _low, high = proportion_with_ci(50, 50)
    assert high <= 100.0


def test_key_estimates_skip_identifiers():
    df = pd.DataFrame({"order_id": range(300),
                       "revenue": np.linspace(10, 900, 300)})
    cols = [e.column for e in key_estimates(df)]
    assert "order_id" not in cols


# ── interactions ──────────────────────────────────────────

@pytest.fixture
def moderated():
    """Overtime adds ~40 hours for juniors and slightly reduces them for
    seniors. A main-effects summary averages these into near-nothing."""
    rng = np.random.default_rng(5)
    n = 1600
    sen = rng.choice(["Junior", "Senior"], n)
    ot = rng.choice(["Yes", "No"], n)
    hours = (160 + 40 * ((sen == "Junior") & (ot == "Yes"))
             - 8 * ((sen == "Senior") & (ot == "Yes")) + rng.normal(0, 6, n))
    return pd.DataFrame({"seniority": sen, "overtime": ot,
                         "monthly_hours": hours.round(1)})


def test_an_effect_that_differs_by_group_is_found(moderated):
    found = find_interactions(moderated)
    assert found, "the planted interaction was missed"
    assert any(i.factor == "overtime" and i.moderator == "seniority"
               for i in found)


def test_the_interaction_says_why_an_average_would_mislead(moderated):
    text = " ".join(i.description for i in find_interactions(moderated))
    assert "average" in text.lower()


def test_a_real_but_trivial_interaction_is_not_reported():
    """A 0.27 gap on a 1-5 scale beat a 0.01 gap by 27x and led the report
    before a magnitude floor was added."""
    rng = np.random.default_rng(2)
    n = 1200
    a = rng.choice(["x", "y"], n)
    b = rng.choice(["p", "q"], n)
    # Effects differ by a large ratio but both are a fraction of an SD.
    metric = (rng.normal(50, 10, n) + 0.3 * ((a == "x") & (b == "p")))
    df = pd.DataFrame({"a": a, "b": b, "metric": metric})
    for i in find_interactions(df):
        assert i.effect_sd >= MIN_EFFECT_SD


def test_interactions_report_their_size_in_standard_deviations(moderated):
    """So a reader can judge it without knowing the units."""
    for i in find_interactions(moderated):
        assert i.effect_sd > 0


def test_no_interactions_when_effects_are_uniform():
    rng = np.random.default_rng(3)
    n = 1200
    a = rng.choice(["x", "y"], n)
    b = rng.choice(["p", "q"], n)
    metric = rng.normal(50, 10, n) + 8 * (a == "x")   # same effect in both b
    df = pd.DataFrame({"a": a, "b": b, "metric": metric})
    assert not [i for i in find_interactions(df)
                if i.factor == "a" and i.moderator == "b"]


# ── thin groups and imbalance ─────────────────────────────

def test_a_group_too_small_to_rank_is_named():
    df = pd.DataFrame({"region": ["North"] * 500 + ["South"] * 480
                                 + ["Antarctica"] * 4,
                       "v": np.random.default_rng(1).normal(0, 1, 984)})
    rare = find_rare_categories(df)
    assert "Antarctica" in [r.level for r in rare]
    assert rare[0].n == 4


def test_healthy_categories_are_not_flagged_as_rare():
    df = pd.DataFrame({"region": ["N"] * 500 + ["S"] * 500,
                       "v": np.random.default_rng(1).normal(0, 1, 1000)})
    assert find_rare_categories(df) == []


def test_imbalance_explains_the_accuracy_trap():
    df = pd.DataFrame({"fraud": ["No"] * 970 + ["Yes"] * 30,
                       "amt": np.random.default_rng(1).normal(0, 1, 1000)})
    notes = describe_imbalance(df)
    assert notes
    assert "without learning anything" in notes[0].note


def test_a_balanced_column_produces_no_imbalance_note():
    df = pd.DataFrame({"flag": ["A"] * 500 + ["B"] * 500,
                       "v": np.random.default_rng(1).normal(0, 1, 1000)})
    assert describe_imbalance(df) == []


# ── wired into the EDA report ─────────────────────────────

def test_the_eda_report_carries_the_depth_layer(moderated):
    from app.engines.eda_engine import run_eda
    report = run_eda(moderated)
    assert report.estimates
    assert report.interactions
    assert hasattr(report, "rare_categories")
    assert hasattr(report, "imbalance_notes")


def test_an_interaction_leads_the_key_findings(moderated):
    """An effect that differs across a second factor is the finding a
    main-effects summary reports as 'no effect'."""
    from app.engines.eda_engine import run_eda
    findings = run_eda(moderated).key_findings
    assert findings
    assert "not the same across" in findings[0]


def test_group_findings_must_clear_both_significance_and_effect_size():
    """On a large sample almost every comparison is significant."""
    from app.engines.eda_engine import run_eda
    rng = np.random.default_rng(9)
    n = 20_000
    grp = rng.choice(["a", "b"], n)
    # A real but negligible difference: significant at this sample size.
    df = pd.DataFrame({"grp": grp,
                       "v": rng.normal(0, 1, n) + 0.02 * (grp == "a")})
    for f in run_eda(df).key_findings:
        assert "large enough to act on" not in f or "differs by" not in f


# ── it reaches the report ─────────────────────────────────

def test_the_report_shows_intervals_around_its_averages():
    from app.engines.pdf_builder import build_pdf
    rng = np.random.default_rng(4)
    n = 400
    df = pd.DataFrame({"revenue": rng.normal(500, 90, n).round(2),
                       "units": rng.integers(1, 60, n),
                       "region": rng.choice(["N", "S"], n)})
    pdf = build_pdf(df=df, domain="sales", config={
        "title": "Intervals", "client_name": "T", "subtitle": "",
        "confidential": True, "theme_name": "", "logo_path": None,
        "prepared_by": "", "source_table": "src"})
    text = "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "95% confidence interval" in text
    assert "not evidence of a change" in text
