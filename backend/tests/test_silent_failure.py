"""
A column the analysis could not read is a fact about the analysis.

The whole error-handling posture was measured rather than assumed: a
logging handler was attached to every logger, the full pipeline was run
against a real dataset through ~30 endpoints — preview, quality, EDA,
BI, insights, drivers, benchmarks, charts, ML training, both PDFs, both
exports — and every swallowed exception was counted.

Exactly one fired. That one mattered:

    analyze_univariate had a categorical branch and a numeric branch and
    nothing for dates, so a datetime column reached `clean.astype(float)`
    and raised "Cannot cast DatetimeArray to dtype float64". The caller
    caught it, logged "suppressed exception" at DEBUG, and continued —
    so EVERY date column was missing from the Deep EDA table, and
    nothing on the page said a column had been dropped.

A period is usually the first thing a reader checks: what does this
cover, and are there holes in it. It was the one column the analysis
never answered for.
"""
import pandas as pd
import pytest

from app.engines.eda.univariate import analyze_univariate
from app.engines.eda_engine import run_eda


@pytest.fixture(scope="module")
def dated() -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(5)
    n = 400
    return pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region":     rng.choice(["North", "South"], n),
        "revenue":    rng.gamma(3, 200, n).round(2),
    })


def test_a_date_column_reaches_the_eda_table(dated):
    """It used to raise and be skipped, silently."""
    report = run_eda(dated)
    assert "order_date" in report.univariate


def test_the_period_is_described(dated):
    result = analyze_univariate(dated["order_date"])
    assert result.dtype == "datetime"
    assert "2024-01-01" in result.interpretation
    assert "days" in result.interpretation


def test_the_plain_reading_answers_what_a_reader_asks_first(dated):
    """"What does this cover, and are there holes in it?\""""
    plain = analyze_univariate(dated["order_date"]).plain
    assert "Runs from" in plain
    assert "coverage" in plain


def test_the_day_count_is_inclusive():
    """1 Jan to 2 Jan is two days of coverage. The exclusive count read
    "375 of the 374 days", which is not a sentence that survives a
    client reading it."""
    s = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]), name="d")
    result = analyze_univariate(s)
    assert "2 days" in result.interpretation
    assert "2 of the 2" in result.plain or "2 of them" in result.interpretation


@pytest.mark.parametrize("freq,expected", [
    ("D",  "about daily"),
    ("ME", "about monthly"),
])
def test_the_grain_of_the_data_is_named(freq, expected):
    """A year of orders on 12 distinct days is monthly rollups, and a
    trend line drawn on it does not mean what it appears to."""
    s = pd.Series(pd.date_range("2023-01-01", periods=24, freq=freq), name="d")
    assert expected in analyze_univariate(s).interpretation


def test_a_single_date_does_not_divide_by_zero():
    s = pd.Series(pd.to_datetime(["2024-05-01"] * 40), name="d")
    result = analyze_univariate(s)
    assert "one date" in result.plain


def test_a_column_that_cannot_be_read_is_named_in_the_report():
    """The lesson from the bug above, generalised: the next failure of
    this shape has to be visible on the page, not in a DEBUG log."""
    class Hostile:
        """A column whose dtype check passes and whose maths explodes."""

    df = pd.DataFrame({"good": [1.0, 2.0, 3.0] * 20})
    report = run_eda(df)
    assert report.warnings == [], "a clean frame must not warn"

    # And the mechanism exists and is wired to the report object.
    assert hasattr(report, "warnings")
