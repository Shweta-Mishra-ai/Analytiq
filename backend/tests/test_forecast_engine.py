"""
Forecasting, and the discipline that makes one worth printing.

"What will revenue be next quarter" is among the first questions a
business asks, and the app did not attempt it — there was trend
detection, which says a line is going up, and nothing that says how far.

The rule here matches the one the rigour gate applies to classification:
a forecast is reported only when it beats the naive alternative — carry
the last value forward — measured on periods the method never saw. Most
business series are a level with noise around them, and for those the
honest planning figure is the current level.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.forecast_engine import (
    MIN_PERIODS, build_series, find_forecastable, forecast_series,
    run_forecast,
)


def _monthly(values, start="2021-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.DataFrame({"month": idx, "revenue": np.round(values, 2)})


@pytest.fixture
def trending():
    """A series with trend and an annual cycle — genuinely forecastable."""
    rng = np.random.default_rng(7)
    n = 48
    return _monthly(np.linspace(100, 260, n)
                    + 25 * np.sin(np.arange(n) / 12 * 2 * np.pi)
                    + rng.normal(0, 6, n))


@pytest.fixture
def noise():
    """A level with noise around it. Nothing should beat the naive."""
    rng = np.random.default_rng(7)
    return _monthly(200 + rng.normal(0, 20, 48))


# ── it forecasts what it can ──────────────────────────────

def test_a_trending_seasonal_series_is_forecast(trending):
    r = run_forecast(trending, horizon=4)
    assert r.usable is True
    assert len(r.points) == 4


def test_the_chosen_method_beats_carrying_the_last_value_forward(trending):
    r = run_forecast(trending, horizon=4)
    assert r.model_error < r.naive_error
    assert r.skill > 0.05


def test_every_point_carries_an_interval(trending):
    """A point estimate invites planning against a number that was never
    more than the middle of a range."""
    for p in run_forecast(trending, horizon=4).points:
        assert p.lower < p.value < p.upper


def test_the_interval_widens_with_distance(trending):
    """Uncertainty compounds: a projection four periods out carries the
    error of every step before it."""
    points = run_forecast(trending, horizon=4).points
    widths = [p.upper - p.lower for p in points]
    assert widths == sorted(widths)
    assert widths[-1] > widths[0]


def test_the_alternatives_and_their_errors_are_reported(trending):
    r = run_forecast(trending, horizon=4)
    assert len(r.candidates) >= 3
    names = [n for n, _e in r.candidates]
    assert "Last value carried forward" in names
    assert r.candidates[0][0] == r.method


# ── it refuses what it cannot ─────────────────────────────

def test_pure_noise_is_not_forecast(noise):
    r = run_forecast(noise, horizon=4)
    assert r.usable is False
    assert r.points == []


def test_refusing_says_what_to_plan_against_instead(noise):
    """"No forecast" is only useful with the number that replaces it."""
    r = run_forecast(noise, horizon=4)
    assert "current level" in r.verdict


def test_a_short_series_is_not_forecast():
    r = run_forecast(_monthly([10, 12, 11, 13, 12, 14]))
    assert r.usable is False
    assert str(MIN_PERIODS) in r.verdict


def test_a_dataset_with_no_dates_returns_nothing():
    df = pd.DataFrame({"a": range(50), "b": np.linspace(1, 9, 50)})
    assert run_forecast(df) is None


# ── series construction ───────────────────────────────────

def test_an_irregular_event_log_becomes_a_regular_series():
    """Real data arrives as events at arbitrary times, not as a tidy
    monthly index."""
    rng = np.random.default_rng(3)
    n = 900
    df = pd.DataFrame({
        "order_date": pd.to_datetime("2022-01-01")
        + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
        "amount": rng.uniform(10, 500, n).round(2),
    })
    s = build_series(df, "order_date", "amount", how="sum")
    assert s is not None and len(s) > 12
    assert s.index.is_monotonic_increasing


def test_the_grain_is_chosen_from_the_gap_between_observations():
    """A year of daily data and a decade of monthly data have similar row
    counts and want completely different treatment."""
    daily = pd.DataFrame({
        "d": pd.date_range("2024-01-01", periods=400, freq="D"),
        "v": np.linspace(1, 100, 400)})
    s = build_series(daily, "d", "v")
    assert s.attrs["freq"] == "D"


def test_measures_are_ranked_before_one_is_projected():
    """Forecasting the first numeric column would project an order id."""
    rng = np.random.default_rng(5)
    n = 40
    df = pd.DataFrame({
        "order_id": range(n),
        "month": pd.date_range("2021-01-31", periods=n, freq="ME"),
        "revenue": np.linspace(100, 400, n),
    })
    pairs = find_forecastable(df)
    assert pairs
    assert "order_id" not in [v for _d, v in pairs]


# ── degenerate input ──────────────────────────────────────

@pytest.mark.parametrize("frame", [
    pd.DataFrame(),
    pd.DataFrame({"month": pd.date_range("2024-01-31", periods=3, freq="ME")}),
    pd.DataFrame({"month": [None] * 20, "v": range(20)}),
])
def test_degenerate_frames_do_not_raise(frame):
    run_forecast(frame)


def test_a_constant_series_is_not_forecast():
    """Either answer is right and both mean the same to a reader: a column
    that never moves is dropped by measure ranking before it reaches the
    forecaster, so this returns None rather than an unusable result."""
    r = run_forecast(_monthly([50.0] * 40))
    assert r is None or r.usable is False


# ── it reaches the report ─────────────────────────────────

def test_the_report_carries_the_outlook(trending):
    import io

    from pypdf import PdfReader

    from app.engines.forecast_engine import run_forecast as rf
    from app.engines.pdf_builder import build_pdf

    pdf = build_pdf(
        df=trending, domain="sales", forecast=rf(trending, horizon=4),
        config={"title": "Outlook", "client_name": "T", "subtitle": "",
                "confidential": True, "theme_name": "", "logo_path": None,
                "prepared_by": "", "source_table": "src"})
    text = "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "Outlook" in text
    assert "BEATS NAIVE BY" in text
    assert "Plan against the range" in text


def test_the_report_says_why_when_it_does_not_forecast(noise):
    import io

    from pypdf import PdfReader

    from app.engines.forecast_engine import run_forecast as rf
    from app.engines.pdf_builder import build_pdf

    pdf = build_pdf(
        df=noise, domain="sales", forecast=rf(noise, horizon=4),
        config={"title": "Outlook", "client_name": "T", "subtitle": "",
                "confidential": True, "theme_name": "", "logo_path": None,
                "prepared_by": "", "source_table": "src"})
    text = "\n".join((p.extract_text() or "")
                     for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "No forecast is shown" in text


def test_the_endpoint_returns_a_forecast(client, uploaded_dataset_id):
    r = client.get(f"/api/analytics/{uploaded_dataset_id}/forecast")
    # The HR fixture has no usable date/measure pair; either a 422 with a
    # reason or a result whose `usable` is false is correct — an
    # unforecastable series is a finding, not a server error.
    assert r.status_code in (200, 422)
    if r.status_code == 200:
        assert "usable" in r.json()
