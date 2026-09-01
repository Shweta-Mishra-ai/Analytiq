"""
Cleaning must not destroy what it was asked to measure.

Each test here corresponds to a loss measured on the previous cleaner:

  * six transactional rows carrying 500 in revenue became three rows
    carrying 250 — half the turnover deleted by blanket deduplication;
  * a column 89% missing and perfectly predictive of churn was dropped by
    a missingness threshold before anything had looked at it;
  * a constant column recording the scope of the extract was dropped;
  * median imputation left no trace of which rows had been empty.

The policy is non-destructive by default: correct what is unambiguously
wrong, report what is a judgement call, and hand over the SQL.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.data_cleaner import (
    DIALECTS, CleaningPolicy, auto_clean, classify_duplicates,
    get_cleaning_summary,
)


@pytest.fixture
def transactions():
    """The same item sold twice on the same day is two sales, not one
    error."""
    return pd.DataFrame({
        "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 2,
        "sku": ["A", "A", "B", "B", "A", "A"],
        "amount": [100, 100, 50, 50, 100, 100],
    })


@pytest.fixture
def sparse_but_predictive():
    """A complaint score exists only for customers who complained: ~89%
    missing, and the strongest signal in the frame."""
    rng = np.random.default_rng(0)
    n = 500
    churn = rng.choice([0, 1], n, p=[.7, .3])
    score = np.where(churn == 1, rng.uniform(1, 10, n), np.nan)
    score = np.where(rng.random(n) < 0.30, score, np.nan)
    return pd.DataFrame({"churn": churn, "complaint_score": score,
                         "tenure": rng.integers(1, 60, n)})


# ── revenue ───────────────────────────────────────────────

def test_cleaning_a_transactional_table_preserves_every_row(transactions):
    cleaned, report = auto_clean(transactions)
    assert len(cleaned) == len(transactions)
    assert report.duplicates_removed == 0


def test_cleaning_a_transactional_table_preserves_the_total(transactions):
    """The number a client checks first."""
    cleaned, _ = auto_clean(transactions)
    assert cleaned["amount"].sum() == transactions["amount"].sum() == 500


def test_repeat_events_are_reported_with_an_honest_verdict(transactions):
    _, report = auto_clean(transactions)
    assert report.duplicates_flagged == 3
    assert report.duplicate_confidence == "low"
    assert "NOT been removed" in report.duplicate_verdict


def test_transactional_shape_is_recognised(transactions):
    v = classify_duplicates(transactions)
    assert v["likely_error"] is False
    assert v["confidence"] == "low"


def test_entity_table_duplicates_are_called_what_they_probably_are():
    """One row per customer repeated is usually a broken join."""
    df = pd.DataFrame({"customer_id": [1, 1, 2, 3],
                       "segment": ["A", "A", "B", "C"]})
    v = classify_duplicates(df)
    assert v["likely_error"] is True
    assert v["confidence"] == "high"
    assert "identity column" in v["verdict"]


def test_a_clean_table_reports_no_duplicate_finding():
    df = pd.DataFrame({"id": range(50), "v": range(50)})
    v = classify_duplicates(df)
    assert v["count"] == 0 and v["verdict"] == ""


# ── sparse columns ────────────────────────────────────────

def test_sparse_predictive_column_survives(sparse_but_predictive):
    cleaned, report = auto_clean(sparse_but_predictive)
    assert "complaint_score" in cleaned.columns
    assert "complaint_score" in report.retained_sparse


def test_sparse_column_is_not_imputed(sparse_but_predictive):
    """Filling 89% of a column invents most of it."""
    _, report = auto_clean(sparse_but_predictive)
    assert "complaint_score" not in report.imputed_columns


def test_sparse_column_keeps_its_missingness_pattern(sparse_but_predictive):
    cleaned, report = auto_clean(sparse_but_predictive)
    assert "complaint_score__was_missing" in cleaned.columns
    assert "complaint_score__was_missing" in report.missingness_indicators


# ── constant columns ──────────────────────────────────────

def test_constant_column_is_kept_as_scope():
    df = pd.DataFrame({"country": ["India"] * 100,
                       "revenue": np.linspace(1, 9, 100)})
    cleaned, report = auto_clean(df)
    assert "country" in cleaned.columns
    assert "country" in report.retained_constant


# ── missingness indicators ────────────────────────────────

def test_imputation_records_which_rows_were_empty():
    df = pd.DataFrame({"x": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10] * 10,
                       "y": range(100)})
    cleaned, report = auto_clean(df)
    assert "x__was_missing" in cleaned.columns
    assert report.imputed_columns.get("x") == 20.0
    assert int(cleaned["x__was_missing"].sum()) == 20


def test_no_indicator_for_a_trivial_number_of_gaps():
    """An indicator that is 99% False is noise, not signal."""
    vals = [1.0] * 199 + [np.nan]
    df = pd.DataFrame({"x": vals, "y": range(200)})
    _, report = auto_clean(df)
    assert report.missingness_indicators == []


def test_indicators_are_excluded_from_charts():
    """'Monthly Income Was Missing by Department' is not a finding."""
    from app.engines.chart_exporter import generate_all_charts
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(100, 20, n)
    x[rng.random(n) < 0.25] = np.nan
    df = pd.DataFrame({"revenue": x, "region": rng.choice(["N", "S"], n)})
    cleaned, report = auto_clean(df)
    assert report.missingness_indicators, "fixture produced no indicator"
    for title, _, _spec in generate_all_charts(cleaned, max_charts=6):
        assert "Was Missing" not in title


# ── SQL dialects ──────────────────────────────────────────

@pytest.mark.parametrize("dialect", DIALECTS)
def test_every_dialect_produces_a_script(dialect, transactions):
    _, report = auto_clean(transactions)
    assert report.sql_script("orders", dialect).strip()


def test_flagged_steps_are_marked_as_not_applied(transactions):
    _, report = auto_clean(transactions)
    script = report.sql_script("orders")
    assert "FLAGGED, NOT APPLIED" in script, \
        "the script does not distinguish what ran from what was only found"


def test_applied_and_flagged_actions_are_distinguishable(transactions):
    _, report = auto_clean(transactions)
    assert any(a.applied is False for a in report.actions)


# ── the report says what it did not do ────────────────────

def test_summary_carries_what_was_deliberately_left_alone(transactions):
    _, report = auto_clean(transactions)
    summary = get_cleaning_summary(report)
    assert summary["duplicates_flagged"] == 3
    assert summary["duplicate_verdict"]
    assert "retained_sparse" in summary
    assert "missingness_indicators" in summary


# ── the escape hatch still works ──────────────────────────

def test_aggressive_policy_still_removes(transactions):
    cleaned, report = auto_clean(transactions, CleaningPolicy.aggressive())
    assert len(cleaned) == 3
    assert report.duplicates_removed == 3


def test_default_policy_is_non_destructive():
    p = CleaningPolicy()
    assert p.drop_duplicates is False
    assert p.drop_sparse_columns is False
    assert p.drop_constant_columns is False


def test_statistical_tests_do_not_treat_repeats_as_independent():
    """Under the default policy duplicates stay in the data. A row
    appearing twice is not two observations, and letting it count twice
    manufactures effect sizes that are not there."""
    from app.engines.data_cleaner import _missingness_is_informative
    rng = np.random.default_rng(4)
    n = 120
    base = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    base.loc[:19, "a"] = np.nan
    # Repeating the missing rows many times would fabricate an association
    # if repeats were counted as independent.
    inflated = pd.concat([base] + [base.head(20)] * 8, ignore_index=True)
    assert _missingness_is_informative(inflated, "a") == \
        _missingness_is_informative(base, "a")
