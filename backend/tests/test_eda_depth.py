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


# ══════════════════════════════════════════════════════════
#  Group comparison picks a test the data actually supports
# ══════════════════════════════════════════════════════════

def test_unequal_variance_uses_welch_not_plain_anova():
    """The bug this replaces: the code ran Levene's test for equal
    variance, stored the answer, and then called one-way ANOVA
    regardless. Testing an assumption and discarding the result is worse
    than not testing it — it produces exactly the false positives the
    check exists to prevent."""
    import numpy as np
    from app.engines.statistics import compare_groups

    rng = np.random.default_rng(0)
    groups = [rng.normal(50, 2, 40), rng.normal(50, 18, 8), rng.normal(50, 2, 40)]
    _stat, p, name = compare_groups(groups, is_normal=True)

    assert name == "Welch's ANOVA"
    # Same data, same means. The old path reported p < 0.0001 here.
    assert p > 0.05, "no group differs, so nothing significant may be reported"


def test_equal_variance_still_uses_the_classic_test():
    """Welch is the fallback for unequal spread, not a blanket
    replacement — where the assumptions hold, the familiar test is
    right and slightly more powerful."""
    import numpy as np
    from app.engines.statistics import compare_groups

    rng = np.random.default_rng(1)
    groups = [rng.normal(10, 2, 60) for _ in range(3)]
    assert compare_groups(groups, is_normal=True)[2] == "One-Way ANOVA"


def test_non_normal_data_is_compared_by_rank():
    import numpy as np
    from app.engines.statistics import compare_groups

    rng = np.random.default_rng(2)
    groups = [rng.exponential(3, 50) for _ in range(3)]
    assert compare_groups(groups, is_normal=False)[2] == "Kruskal-Wallis"


def test_a_real_difference_is_still_detected():
    """A test that never reports significance is not a fix."""
    import numpy as np
    from app.engines.statistics import compare_groups

    rng = np.random.default_rng(3)
    groups = [rng.normal(50, 2, 40), rng.normal(70, 18, 20), rng.normal(50, 2, 40)]
    _stat, p, name = compare_groups(groups, is_normal=True)
    assert name == "Welch's ANOVA"
    assert p < 0.05


def test_every_fallback_assumes_less_than_the_test_it_replaces(monkeypatch):
    """When Welch is unavailable the code must fall to Kruskal-Wallis,
    never back to the plain ANOVA it just ruled out."""
    import numpy as np
    import app.engines.statistics as stats_mod

    rng = np.random.default_rng(4)
    groups = [rng.normal(50, 2, 40), rng.normal(50, 18, 8), rng.normal(50, 2, 40)]

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def _no_statsmodels(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ImportError("statsmodels is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_statsmodels)
    assert stats_mod.compare_groups(groups, is_normal=True)[2] == "Kruskal-Wallis"


# ══════════════════════════════════════════════════════════
#  Distribution fitting: right answer, at any size
# ══════════════════════════════════════════════════════════

import pytest  # noqa: E402


@pytest.mark.parametrize("truth,build", [
    ("norm",    lambda rng, n: rng.normal(50, 12, n)),
    ("lognorm", lambda rng, n: rng.lognormal(3, 0.6, n)),
    ("expon",   lambda rng, n: rng.exponential(4, n)),
    ("gamma",   lambda rng, n: rng.gamma(2.5, 3, n)),
    ("uniform", lambda rng, n: rng.uniform(0, 100, n)),
])
def test_the_right_distribution_is_identified(truth, build):
    """Two failures this covers, both of which the original had.

    Ranking by goodness of fit alone favours whichever candidate has the
    most parameters — a gamma imitates a normal well enough to win — so
    textbook-normal data came back as "gamma". And exponential is gamma
    with the shape fixed at 1, so a weak penalty picks gamma there too.
    """
    import numpy as np
    import pandas as pd
    from app.engines.eda_engine import _fit_distribution

    rng = np.random.default_rng(0)
    assert _fit_distribution(pd.Series(build(rng, 20_000)))[0] == truth


def test_a_large_column_is_identified_correctly_not_defaulted():
    """The bug this replaces: candidates were ranked by KS p-value, which
    collapses to exactly 0.0 once the sample is large enough — real data
    is never exactly lognormal. With every p at zero and the running best
    initialised to zero, `p > best_p` was never true and the function
    returned its own untried default of "norm" for clearly lognormal
    data."""
    import numpy as np
    import pandas as pd
    from app.engines.eda_engine import _fit_distribution

    rng = np.random.default_rng(0)
    base = rng.lognormal(9, 0.5, 97_000)
    contamination = rng.normal(90_000, 5_000, 3_000)
    realistic = pd.Series(np.round(np.concatenate([base, contamination])))

    name, params = _fit_distribution(realistic)
    assert name == "lognorm"
    assert params.get("params"), "a named distribution must carry its fit"


def test_a_column_no_standard_distribution_fits_says_so():
    """Naming a distribution that does not fit is worse than admitting
    none does — the shape decides which summaries and tests are
    appropriate downstream."""
    import numpy as np
    import pandas as pd
    from app.engines.eda_engine import _fit_distribution

    rng = np.random.default_rng(1)
    bimodal = pd.Series(np.concatenate([rng.normal(10, 1, 4000),
                                        rng.normal(60, 1, 4000)]))
    name, params = _fit_distribution(bimodal)
    assert name == "none"
    assert "bimodal" in params["note"]


def test_fitting_is_bounded_by_sample_size_not_row_count():
    """Fitting five distributions by numerical maximum likelihood over a
    whole column took 2.9 seconds on 100,000 rows. The fitted parameters
    are indistinguishable from a sample's."""
    import time

    import numpy as np
    import pandas as pd
    from app.engines.eda_engine import FIT_SAMPLE_SIZE, _fit_distribution

    rng = np.random.default_rng(2)
    small = pd.Series(rng.lognormal(3, 0.6, 3_000))
    large = pd.Series(rng.lognormal(3, 0.6, 200_000))

    _fit_distribution(small)                       # warm the lazy imports
    t = time.perf_counter(); _fit_distribution(large)
    elapsed = time.perf_counter() - t

    assert FIT_SAMPLE_SIZE <= 5_000
    assert elapsed < 1.0, f"200k rows took {elapsed:.1f}s — it is not sampling"


def test_the_same_column_always_gets_the_same_answer():
    """Sampling with a fixed seed: two runs of one analysis must not
    disagree about the shape of the data."""
    import numpy as np
    import pandas as pd
    from app.engines.eda_engine import _fit_distribution

    rng = np.random.default_rng(3)
    series = pd.Series(rng.gamma(2, 3, 50_000))
    assert _fit_distribution(series)[0] == _fit_distribution(series)[0]


def test_too_few_values_to_fit_is_admitted():
    import pandas as pd
    from app.engines.eda_engine import _fit_distribution
    assert _fit_distribution(pd.Series([1.0, 2.0, 3.0]))[0] == "unknown"


# ══════════════════════════════════════════════════════════
#  VIF: the same numbers, without the cubic cost
# ══════════════════════════════════════════════════════════

def test_vif_matches_the_regression_definition():
    """VIF is the diagonal of the inverted correlation matrix — the
    replacement must agree with the definition it replaced, which fitted
    one least-squares regression per column."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    from app.engines.eda_engine import analyze_vif

    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (2_000, 3))
    X = np.hstack([base, base @ rng.normal(0, 1, (3, 5)) + rng.normal(0, 0.5, (2_000, 5))])
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(8)])

    got = {r.feature: r.vif for r in analyze_vif(df)}
    for col in df.columns:
        y = df[col].values
        rest = df.drop(columns=[col]).values
        r2 = LinearRegression().fit(rest, y).score(rest, y)
        expected = 1 / (1 - r2)
        assert abs(got[col] - expected) < 0.05, f"{col}: {got[col]} vs {expected}"


def test_vif_does_not_grow_cubically_with_column_count():
    """One regression per column, each over every other column, is cubic
    work. A 120-column dataset spent 14.7 seconds here."""
    import time

    import numpy as np
    import pandas as pd
    from app.engines.eda_engine import analyze_vif

    rng = np.random.default_rng(1)
    wide = pd.DataFrame(rng.normal(0, 1, (5_000, 60)),
                        columns=[f"c{i}" for i in range(60)])
    analyze_vif(wide.iloc[:100, :4])               # warm
    t = time.perf_counter(); analyze_vif(wide)
    assert time.perf_counter() - t < 1.0


def test_a_duplicated_column_is_reported_as_unbounded():
    """A pseudo-inverse does not blow up on a singular matrix — it hands
    back a modest-looking number, so a literal copy of another column
    was reported as VIF 14.5 ("High") when the true answer is infinite
    and it is the worst case there is."""
    import numpy as np
    import pandas as pd
    from app.engines.eda_engine import analyze_vif

    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(0, 1, (1_000, 4)), columns=list("abcd"))
    df["copy_of_a"] = df["a"] * 3

    results = {r.feature: r for r in analyze_vif(df)}
    assert results["copy_of_a"].vif == float("inf")
    assert results["copy_of_a"].verdict == "Severe"
    assert "copy" in results["copy_of_a"].interpretation.lower()


def test_a_constant_column_does_not_poison_the_whole_matrix():
    """Zero variance means undefined correlation, which would make every
    other column's VIF nan."""
    import numpy as np
    import pandas as pd
    from app.engines.eda_engine import analyze_vif

    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 1, (500, 4)), columns=list("abcd"))
    df["always_7"] = 7

    results = analyze_vif(df)
    assert "always_7" not in [r.feature for r in results]
    assert results and all(np.isfinite(r.vif) for r in results)


def test_vif_needs_at_least_two_varying_columns():
    import pandas as pd
    from app.engines.eda_engine import analyze_vif
    assert analyze_vif(pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})) == []
