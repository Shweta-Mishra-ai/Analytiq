"""
engines/forecast.py — the next three months of the main measure, with a
range and a measured accuracy.

A forecast is only as useful as its track record, so every number here
comes with one. Each candidate method is tried on the recent past as if
it were the future: standing at each of the last six months, it forecasts
one to three months ahead from what came before, and its errors are
measured against what happened. The method with the smallest errors is
used, and

  - its accuracy is reported beside that of the simplest forecast there is
    (next month = this month), so a reader can see whether it earns its
    complexity;
  - the range is the spread of its own past errors (8 in 10 fell inside),
    not a textbook interval that assumes the method is right.

Methods: last month's level; same month last year (two years of data);
exponential smoothing with a damped trend; and the same with a yearly
season (two years of data). Months the data only partly covers are left
out, as on the trend chart.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZON = 3
MIN_MONTHS = 12          # below this there is no history to test against
BACKTEST_ORIGINS = 12     # test months, when the history allows


@dataclass
class ForecastPoint:
    month: pd.Timestamp
    value: float
    low: float
    high: float


@dataclass
class Forecast:
    measure: str
    method: str
    points: List[ForecastPoint] = field(default_factory=list)
    history: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)
    error_pct: float = 0.0           # mean absolute % error in the backtest
    naive_error_pct: float = 0.0     # the same for "next month = this month"
    n_tests: int = 0
    n_months: int = 0                # test months in the backtest
    aggregate: str = "total"         # "total" or "average" per month
    sentences: List[str] = field(default_factory=list)


# ── methods: each takes the history (np.ndarray) and returns HORIZON values ──

def _naive(y: np.ndarray) -> np.ndarray:
    return np.repeat(y[-1], HORIZON)


def _seasonal_naive(y: np.ndarray) -> np.ndarray:
    return np.array([y[len(y) - 12 + h] for h in range(HORIZON)])


def _ets(y: np.ndarray, seasonal: bool) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y, trend="add", damped_trend=True,
            seasonal="add" if seasonal else None,
            seasonal_periods=12 if seasonal else None,
            initialization_method="estimated").fit()
        return np.asarray(model.forecast(HORIZON), dtype=float)


def _methods(n: int) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    m = {"last month's level": _naive,
         "trend (damped exponential smoothing)": lambda y: _ets(y, False)}
    if n >= 24:
        m["same month last year"] = _seasonal_naive
    if n >= 30:
        # A season is estimated from at least two full years before the
        # first test month.
        m["trend and yearly season (Holt-Winters)"] = lambda y: _ets(y, True)
    return m


# History each method needs before its first test month: a season is
# learnt from two full years. Tested earlier, Holt-Winters failed and
# dropped out, and a worse method was chosen for seasonal data.
_MIN_HISTORY = {"same month last year": 12, "trend and yearly season (Holt-Winters)": 24}


def _first_origin(n: int, names) -> int:
    """The first test month, the same for every method so they are compared
    on the same months."""
    need = max([MIN_MONTHS - 3] + [_MIN_HISTORY.get(nm, 0) for nm in names])
    return max(n - BACKTEST_ORIGINS - HORIZON + 1, need)


def _backtest(y: np.ndarray, fn, start: int) -> Optional[List[Tuple[int, float, float]]]:
    """(horizon, forecast, actual) for each test month."""
    out = []
    n = len(y)
    origins = range(start, n - 1)
    for o in origins:
        try:
            f = fn(y[:o])
        except Exception:
            logger.debug("forecast method failed at origin %s", o, exc_info=True)
            return None
        for h in range(HORIZON):
            if o + h < n and np.isfinite(f[h]):
                out.append((h, float(f[h]), float(y[o + h])))
    return out or None


def _spreads(tests) -> List[Tuple[str, float]]:
    """For each step ahead, (kind, half-width) of an 80% range
    from the backtest errors.

    The width is a t-quantile times the standard deviation of the log
    errors, widened for the uncertainty of the estimate itself. The 10th
    to 90th percentile of six errors per step, used first, covered 60% of
    held-out months instead of 80%. Steps further ahead are never given a
    narrower range than nearer ones.
    """
    from scipy.stats import t as student
    out, prev = [], 0.0
    positive = all(f > 0 and a > 0 for _, f, a in tests)
    for h in range(HORIZON):
        errs = [(np.log(a / f) if positive else a - f) for hh, f, a in tests if hh <= h]
        errs = np.asarray(errs, dtype=float)
        n = len(errs)
        if n < 3:
            out.append(("ratio" if positive else "add", prev))
            continue
        sd = float(np.std(errs, ddof=1)) * np.sqrt(1 + 1 / n)
        width = float(student.ppf(0.9, df=n - 1)) * sd
        # Errors further ahead: the steps up to h are pooled, then the
        # spread is scaled up by how much the h-step errors exceed them.
        own = np.asarray([(np.log(a / f) if positive else a - f)
                          for hh, f, a in tests if hh == h], dtype=float)
        if len(own) >= 3:
            width *= max(1.0, float(np.std(own, ddof=1)) / max(float(np.std(errs, ddof=1)), 1e-12))
        width = max(width, prev)
        prev = width
        out.append(("ratio" if positive else "add", width))
    return out


def _mape(tests) -> float:
    errs = [abs(f - a) / abs(a) for _, f, a in tests if a]
    return float(np.mean(errs) * 100) if errs else float("inf")


def monthly_series(df: pd.DataFrame) -> Optional[Tuple[str, str, pd.Series]]:
    """(measure, aggregation, monthly series over complete months), or None."""
    from app.engines.chart_exporter import is_additive
    from app.engines.domains.general_depth import _complete_months
    from app.engines.scorecard import build_scorecard
    sc = build_scorecard(df)
    measure, date_col = sc.revenue_col, sc.date_col
    if not measure or not date_col:
        from app.engines.chart_exporter import _event_date, _measures
        date_col = _event_date(df)
        ms = _measures(df)
        measure = ms[0] if ms else None
    if not measure or not date_col:
        return None
    data = df[[date_col, measure]].copy()
    data[measure] = pd.to_numeric(data[measure], errors="coerce")
    monthly, rows = _complete_months(data, measure, date_col)
    # A gap month (no rows) would read as a zero sale; the series must be
    # unbroken to be forecast.
    if len(monthly) < MIN_MONTHS or (rows <= 0).any():
        return None
    idx = pd.date_range(monthly.index[0], monthly.index[-1], freq=monthly.index.freq or "ME")
    if len(idx) != len(monthly):
        return None
    return measure, ("total" if is_additive(measure) else "average"), monthly


def forecast(df: pd.DataFrame) -> Optional[Forecast]:
    found = monthly_series(df)
    if not found:
        return None
    measure, agg, monthly = found
    # Totals are forecast per day and multiplied back by the days in each
    # month: February's three fewer days are a 10% dip that is the
    # calendar, not a trend. Errors in % are the same either way.
    days = monthly.index.days_in_month.to_numpy(dtype=float) if agg == "total" \
        else np.ones(len(monthly))
    y = monthly.to_numpy(dtype=float) / days
    scores = {}
    methods = _methods(len(y))
    start = _first_origin(len(y), methods)
    for name, fn in methods.items():
        tests = _backtest(y, fn, start)
        if tests:
            scores[name] = (fn, tests, _mape(tests))
    if "last month's level" not in scores:
        return None
    naive_err = scores["last month's level"][2]
    name, (fn, tests, err) = min(scores.items(), key=lambda kv: kv[1][2])
    try:
        ahead = fn(y)
    except Exception:
        logger.warning("final forecast failed", exc_info=True)
        return None

    spreads = _spreads(tests)
    floor = 0.0 if (y >= 0).all() else None
    out = Forecast(measure=measure, method=name, error_pct=err, naive_error_pct=naive_err,
                   n_tests=len(tests), n_months=sum(1 for h, _, _ in tests if h == 0),
                   aggregate=agg,
                   history=[(m, float(v)) for m, v in monthly.items()])
    last = monthly.index[-1]
    for h in range(HORIZON):
        month = last + pd.offsets.MonthEnd(h + 1)
        k = float(month.days_in_month) if agg == "total" else 1.0
        kind, sd = spreads[h]
        if kind == "ratio" and ahead[h] > 0:
            lo, hi = ahead[h] * np.exp(-sd), ahead[h] * np.exp(sd)
        else:
            lo, hi = ahead[h] - sd, ahead[h] + sd
        value, lo, hi = float(ahead[h]) * k, float(lo) * k, float(hi) * k
        if floor is not None:
            value, lo, hi = max(value, floor), max(lo, floor), max(hi, floor)
        out.points.append(ForecastPoint(month, value, float(min(lo, value)),
                                        float(max(hi, value))))
    out.sentences = _sentences(out)
    return out


def _sentences(f: Forecast) -> List[str]:
    from app.engines.chart_exporter import fmt
    from app.i18n import month_year
    what = "Monthly total '{}'" if f.aggregate == "total" else "Monthly average '{}'"
    pts = " · ".join("{}: {} (range {} to {})".format(
        month_year(p.month), fmt(p.value), fmt(p.low), fmt(p.high)) for p in f.points)
    s = ["{}, next three months: {}.".format(what.format(f.measure), pts)]
    if f.method == "last month's level":
        s.append("No method beat holding last month's level: tested on the last {} months "
                 "of history, its forecasts were off by {:.1f}% on average. There is no "
                 "trend or season in the history steady enough to forecast.".format(
                     f.n_months, f.error_pct))
    else:
        better = f.naive_error_pct - f.error_pct
        s.append("Method: {}. Tested on the last {} months of history as if they were the "
                 "future, its forecasts were off by {:.1f}% on average, against {:.1f}% for "
                 "holding last month's level{}.".format(
                     f.method, f.n_months, f.error_pct, f.naive_error_pct,
                     "" if better >= 1 else " — little better, so treat the trend as tentative"))
    s.append("8 in 10 of its past forecasts fell within the range given; it assumes the "
             "next months behave like the history, so a promotion, price change or "
             "lost customer will not be in it.")
    return s


def forecast_message(df: pd.DataFrame):
    """The forecast as a report key message (scorecard.Message)."""
    from app.engines.chart_exporter import fmt
    from app.engines.scorecard import Message
    from app.i18n import month_year
    f = forecast(df)
    if f is None:
        return None
    total = sum(p.value for p in f.points)
    lo, hi = sum(p.low for p in f.points), sum(p.high for p in f.points)
    title = ("Next three months: about {} in '{}' ({} to {})".format(
        fmt(total), f.measure, month_year(f.points[0].month), month_year(f.points[-1].month))
        if f.aggregate == "total" else
        "Next three months: '{}' around {} a month".format(f.measure, fmt(total / 3)))
    detail = " ".join(f.sentences)
    return Message(
        title=title, detail=detail,
        action=("Plan against the range, not the single figure, and compare each month "
                "as it closes; a month outside the range means something has changed."),
        at_stake=0.0, kind="info")
