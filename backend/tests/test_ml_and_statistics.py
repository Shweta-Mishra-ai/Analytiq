"""
tests/test_ml_and_statistics.py — the analysis being right, not just
running.

Each case here was found by putting a dataset with a known answer through
the engine and comparing. None of them raised; each returned a confident
wrong answer.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from app.engines import statistics as st


# ── normality ────────────────────────────────────────────

class TestNormality:
    def test_a_near_normal_column_is_not_rejected_for_being_large(self):
        """Shapiro-Wilk at n=1,470 rejected monthly income with a skew of
        0.48. The power of a normality test grows with n, so past a few
        hundred rows it answers "is n large", not "is this normal"."""
        rng = np.random.default_rng(1)
        big = pd.Series(rng.normal(8000, 2000, 5000))
        verdict = st.assess_normality(big)
        assert verdict.normal_enough, verdict
        assert verdict.basis == "shape"

    def test_a_genuinely_skewed_column_is_still_rejected(self):
        rng = np.random.default_rng(1)
        skewed = pd.Series(rng.gamma(1.2, 3.0, 4000))
        assert not st.assess_normality(skewed).normal_enough

    def test_a_small_sample_still_trusts_the_test(self):
        """Below a few hundred rows the shape statistics are themselves
        unstable, and the test is the better guide."""
        rng = np.random.default_rng(2)
        small = pd.Series(rng.normal(0, 1, 60))
        assert st.assess_normality(small).basis == "test"

    def test_a_two_value_column_has_no_normality_question(self):
        assert st.assess_normality(pd.Series([3, 4] * 200)) is None


# ── p-values ─────────────────────────────────────────────

class TestPValues:
    def test_a_p_value_never_rounds_to_zero(self):
        """"p=0.0000" appeared throughout the report. That is round(1e-40,
        6), not a p-value — no evidence makes a hypothesis impossible."""
        assert st.clamp_p(1e-40) > 0
        assert "0.0000" not in st.format_p(1e-40)
        assert st.format_p(1e-40).startswith("p <")

    def test_an_ordinary_p_value_is_printed_as_itself(self):
        assert st.format_p(0.032) == "p = 0.032"

    def test_a_missing_p_value_is_not_invented(self):
        assert st.format_p(None) == "—"


# ── intervals ────────────────────────────────────────────

class TestIntervals:
    def test_a_correlation_carries_its_interval_and_its_n(self):
        """r=0.66 on forty rows and r=0.66 on fourteen hundred were
        printed identically."""
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, 1500)
        y = 0.6 * x + rng.normal(0, 0.8, 1500)
        est = st.correlation_with_ci(x, y)
        assert est.n == 1500
        assert est.ci_low < est.r < est.ci_high

    def test_a_small_sample_gets_a_wider_interval(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, 1500)
        y = 0.6 * x + rng.normal(0, 0.8, 1500)
        big = st.correlation_with_ci(x, y)
        small = st.correlation_with_ci(x[:40], y[:40])
        assert (small.ci_high - small.ci_low) > (big.ci_high - big.ci_low) * 3

    def test_correlation_strength_is_named_one_way(self):
        """0.66 was 'moderate' in one place and 'strong' in another, in
        the same report."""
        assert st.correlation_strength(0.66) == "strong"
        assert st.correlation_strength(0.35) == "moderate"
        assert st.correlation_strength(0.05) == "negligible"


class TestEffectGate:
    def test_significant_but_negligible_is_not_a_finding(self):
        """At n=1,470 a one-year difference in average age clears p<0.05
        with an effect size of 0.002."""
        assert not st.is_worth_reporting(0.049, 0.0021, "eta")

    def test_a_real_effect_survives(self):
        assert st.is_worth_reporting(0.001, 0.08, "eta")

    def test_an_insignificant_result_never_passes(self):
        assert not st.is_worth_reporting(0.4, 0.9, "d")


# ── identifiers ──────────────────────────────────────────

def test_an_identifier_is_held_out_and_named(hr_df):
    """EmployeeNumber was getting a mean of 735.5, a skewness, a
    normality verdict and a place in the correlation matrix."""
    from app.engines.stats_engine import analyze
    from app.engines.eda_engine import run_eda
    stats = analyze(hr_df)
    assert "employee_id" in stats.identifier_cols
    assert "employee_id" not in stats.column_stats
    assert not any("employee_id" in (c.col_a, c.col_b)
                   for c in stats.correlations)
    eda = run_eda(hr_df)
    assert "employee_id" in eda.identifier_cols
    assert "employee_id" not in eda.univariate


# ── ML: encoding ─────────────────────────────────────────

@pytest.fixture()
def nominal_df() -> pd.DataFrame:
    """A category whose effect is not monotonic in any ordering of its
    levels — the case label encoding cannot represent."""
    rng = np.random.default_rng(3)
    n = 1200
    region = rng.choice(["North", "South", "East", "West", "Central"], n)
    risk = np.where(np.isin(region, ["North", "East"]), 0.65, 0.12)
    return pd.DataFrame({"region": region,
                         "noise": rng.normal(0, 1, n),
                         "churn": rng.random(n) < risk})


class TestEncoding:
    def test_a_nominal_category_is_one_hot_not_ranked(self, nominal_df):
        """LabelEncoder assigns Sales=0, Research=1, HR=2 — an ordering
        that exists nowhere in the data, which a linear model then reads
        as a magnitude."""
        from app.engines.ml_engine import prepare_features
        X, _y, _le, mapping = prepare_features(nominal_df, "churn")
        assert "region" not in X.columns
        assert "region=North" in X.columns
        assert mapping["region=North"] == ("region", "North")

    def test_the_model_finds_the_right_levels(self, nominal_df):
        from app.engines.ml_engine import run_ml_pipeline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = run_ml_pipeline(nominal_df, "churn")
        top = {f.feature for f in report.feature_importance[:2]}
        assert top & {"region=North", "region=East"}, report.feature_importance

    def test_what_if_takes_a_category_not_a_dummy(self, nominal_df):
        """The UI asks for "Region: North", not six one-hot columns."""
        from app.engines.ml_engine import predict_what_if, run_ml_pipeline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = run_ml_pipeline(nominal_df, "churn")
        high = predict_what_if(report, {"region": "North", "noise": 0.0})
        low = predict_what_if(report, {"region": "South", "noise": 0.0})
        assert high["probabilities"]["1"] > low["probabilities"]["1"] * 2


# ── ML: leakage without information loss ─────────────────

class TestLeakagePolicy:
    def test_a_confirmed_leak_is_removed(self):
        """A field populated for exactly one outcome class gives the
        answer away by its presence alone."""
        from app.engines.rigour import detect_leakage
        rng = np.random.default_rng(3)
        n = 700
        churn = rng.choice([0, 1], n, p=[.7, .3])
        df = pd.DataFrame({
            "churn": churn,
            "churn_date_days": np.where(churn == 1, rng.uniform(1, 30, n),
                                        np.nan),
            "tenure": rng.integers(1, 60, n),
        })
        found = {f.column: f for f in detect_leakage(df, "churn")}
        assert found["churn_date_days"].confidence == "confirmed"
        assert found["churn_date_days"].drop

    def test_a_strong_predictor_is_kept_and_flagged(self):
        """Separation alone cannot tell a leak from a real driver. On a
        dataset whose target was defined as x > 0, x itself was dropped
        as leakage and the pipeline reported "no usable feature columns
        found" — the signal destroyed rather than protected."""
        from app.engines.rigour import detect_leakage
        rng = np.random.default_rng(3)
        n = 1200
        df = pd.DataFrame({"x": rng.normal(0, 1, n)})
        df["target"] = (df.x > 0).astype(int)
        found = {f.column: f for f in detect_leakage(df, "target")}
        assert found["x"].confidence == "suspected"
        assert not found["x"].drop

    def test_a_post_outcome_name_is_confirmed(self):
        from app.engines.rigour import _names_a_post_outcome_field
        assert _names_a_post_outcome_field("settlement_amount") == "settlement"
        assert _names_a_post_outcome_field("ExitInterviewScore") == "exit"
        assert _names_a_post_outcome_field("tenure_months") == ""

    def test_the_pipeline_still_trains_on_a_strong_predictor(self):
        from app.engines.ml_engine import run_ml_pipeline
        rng = np.random.default_rng(3)
        n = 1200
        df = pd.DataFrame({"x": rng.normal(0, 1, n)})
        df["target"] = (df.x > 0).astype(int)
        df["other"] = rng.normal(0, 1, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = run_ml_pipeline(df, "target")
        assert report.best_model is not None, report.warnings
        assert "x" in report.feature_cols
        assert "x" in report.suspect_features

    def test_the_reader_is_shown_the_model_without_the_suspect(self):
        """Keeping a suspiciously strong feature is only honest if the
        reader can see how much of the result rests on it."""
        from app.engines.ml_engine import run_ml_pipeline
        rng = np.random.default_rng(3)
        n = 1200
        df = pd.DataFrame({"x": rng.normal(0, 1, n)})
        df["target"] = (df.x > 0).astype(int)
        df["weak"] = df.target * 0.2 + rng.normal(0, 1, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = run_ml_pipeline(df, "target")
        assert report.score_without_suspects is not None, report.warnings
        assert report.score_without_suspects < report.best_model.test_score


# ══════════════════════════════════════════════════════════
#  A date is not something to predict
# ══════════════════════════════════════════════════════════

def _frame_with_a_date():
    import random
    import pandas as pd
    random.seed(7)
    n = 300
    return pd.DataFrame({
        "employee_id": range(n),
        "department": [random.choice(["Sales", "Eng", "HR"]) for _ in range(n)],
        "order_date": pd.to_datetime(
            [f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
             for _ in range(n)]),
        "monthly_income": [random.randint(2, 20) * 1000 for _ in range(n)],
        "attrition": [random.choice(["Yes", "No"]) for _ in range(n)],
    })


def test_a_dataset_with_a_date_column_can_still_suggest_targets():
    """The bug this covers: a datetime column fell through to the
    "continuous numeric" branch, and the caller then divided a Timestamp
    standard deviation by a Timestamp mean. That crashed the target
    endpoint outright for any dataset containing a date — which is most
    business data. The whole feature was unreachable and no test noticed
    because every fixture happened to be dateless."""
    from app.engines.ml_engine import suggest_targets

    targets = suggest_targets(_frame_with_a_date())
    assert targets, "a frame with an obvious outcome must still offer targets"
    assert "order_date" not in [t["column"] for t in targets]
    assert "attrition" in [t["column"] for t in targets]


def test_a_date_is_not_reported_as_a_regression_target():
    from app.engines.ml_engine import detect_task
    task, reason = detect_task(_frame_with_a_date()["order_date"])
    assert task == "unsupported"
    assert "predicted" in reason


def test_training_on_a_date_refuses_with_a_reason():
    """Named at the point of refusal rather than surfacing as a type
    error from inside a scaler."""
    import pytest
    from app.engines.ml_engine import run_ml_pipeline

    with pytest.raises(ValueError) as excinfo:
        run_ml_pipeline(_frame_with_a_date(), target_col="order_date")
    assert "order_date" in str(excinfo.value)
