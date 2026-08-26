"""
engines/forecast_engine.py — what happens next, and how much to trust it.

"What will revenue be next quarter" is among the first questions any
business asks, and the app did not attempt it: there was trend detection,
which says a line is going up, and nothing that says how far.

The discipline here is the same one the rigour gate applies to
classification. A forecast is only reported when it beats the naive
alternative — carrying the last value forward, or repeating last season —
measured by backtest on data the model did not see. Most series in a
business dataset are noise around a level, and for those the honest
answer is that next period will look like this one, plus or minus. A
forecast that cannot beat that is a decoration.

Every forecast ships with an interval. A point estimate for next quarter
invites someone to plan against a number that was never more than the
middle of a range.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.services.dtypes import MONTH_END

logger = logging.getLogger(__name__)

# Below this many periods there is nothing to learn a pattern from.
MIN_PERIODS = 12
# Periods held out to score the candidates.
DEFAULT_HOLDOUT = 4
# A model must beat the naive baseline by at least this share of its error
# before it is worth the complexity and the false confidence.
MIN_SKILL = 0.05
# Seasonal methods need at least two full cycles to fit.
MIN_CYCLES = 2


@dataclass
class ForecastPoint:
    period: str
    value: float
    lower: float
    upper: float


@dataclass
class ForecastResult:
    column: str
    date_col: str
    freq: str
    method: str
    usable: bool
    n_periods: int
    horizon: int
    points: List[ForecastPoint] = field(default_factory=list)
    history: List[Tuple[str, float]] = field(default_factory=list)
    naive_error: float = 0.0
    model_error: float = 0.0
    skill: float = 0.0
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    seasonal_periods: Optional[int] = None
    verdict: str = ""
    reason: str = ""


# ══════════════════════════════════════════════════════════
#  SERIES PREPARATION
# ══════════════════════════════════════════════════════════

_FREQ_LABEL = {"D": "daily", "W": "weekly", "ME": "monthly",
               "M": "monthly", "QE": "quarterly", "Q": "quarterly"}
_SEASON = {"D": 7, "W": 52, "ME": 12, "M": 12, "QE": 4, "Q": 4}


def _pick_frequency(index: pd.DatetimeIndex) -> str:
    """The grain to forecast at.

    Chosen from the median gap between observations rather than from the
    row count: a year of daily data and a decade of monthly data have
    similar lengths and want completely different treatment.
    """
    if len(index) < 3:
        return MONTH_END
    gap = pd.Series(index.sort_values()).diff().dt.days.median()
    if gap is None or not np.isfinite(gap):
        return MONTH_END
    if gap <= 1.5:
        return "D"
    if gap <= 10:
        return "W"
    if gap <= 45:
        return MONTH_END
    return "QE"


def build_series(df: pd.DataFrame, date_col: str, value_col: str,
                 how: str = "sum") -> Optional[pd.Series]:
    """A regular, gap-free series from an irregular event table."""
    try:
        work = df[[date_col, value_col]].dropna()
        if work.empty:
            return None
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])
        if work.empty:
            return None
        freq = _pick_frequency(pd.DatetimeIndex(work[date_col]))
        series = (work.set_index(date_col)[value_col]
                      .resample(freq).agg(how))
        # Interior gaps are real zeros for a count and unknown for a level;
        # forward-filling a level is the lesser distortion, and either way
        # the gap is not evidence of a drop to nothing.
        series = series.interpolate(limit_direction="both")
        series.attrs["freq"] = freq
        return series.dropna()
    except Exception:
        logger.debug("could not build a series for %r", value_col,
                     exc_info=True)
        return None


# ══════════════════════════════════════════════════════════
#  CANDIDATES
# ══════════════════════════════════════════════════════════

def _naive(train: pd.Series, horizon: int) -> np.ndarray:
    """Carry the last value forward. The baseline every other method has
    to beat to justify itself."""
    return np.repeat(float(train.iloc[-1]), horizon)


def _seasonal_naive(train: pd.Series, horizon: int,
                    season: int) -> Optional[np.ndarray]:
    """Repeat the equivalent period from the previous cycle."""
    if len(train) < season:
        return None
    last_cycle = train.iloc[-season:].to_numpy(dtype=float)
    return np.array([last_cycle[i % season] for i in range(horizon)])


def _drift(train: pd.Series, horizon: int) -> np.ndarray:
    """Extend the average change per period — a straight line through the
    first and last observation."""
    n = len(train)
    if n < 2:
        return _naive(train, horizon)
    slope = (float(train.iloc[-1]) - float(train.iloc[0])) / (n - 1)
    last = float(train.iloc[-1])
    return np.array([last + slope * (i + 1) for i in range(horizon)])


def _holt_winters(train: pd.Series, horizon: int,
                  season: Optional[int]) -> Optional[np.ndarray]:
    """Exponential smoothing with trend, and seasonality where there is
    enough history to estimate it."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except Exception:
        logger.info("statsmodels unavailable — smoothing candidate skipped")
        return None
    try:
        use_season = (season is not None
                      and len(train) >= season * MIN_CYCLES
                      and (train > 0).all())
        model = ExponentialSmoothing(
            train.astype(float),
            trend="add",
            seasonal="add" if use_season else None,
            seasonal_periods=season if use_season else None,
            initialization_method="estimated")
        return np.asarray(model.fit().forecast(horizon), dtype=float)
    except Exception:
        logger.debug("holt-winters failed", exc_info=True)
        return None


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual, dtype=float)
                                - np.asarray(predicted, dtype=float))))


# ══════════════════════════════════════════════════════════
#  FORECAST
# ══════════════════════════════════════════════════════════

def forecast_series(series: pd.Series, horizon: int = 3,
                    holdout: int = DEFAULT_HOLDOUT,
                    column: str = "", date_col: str = "") -> ForecastResult:
    """Forecast, having first proved the method is worth using.

    Candidates are scored on periods held back from fitting, against the
    naive baseline scored the same way. Whichever wins is refitted on the
    whole series to produce the forecast.
    """
    freq = series.attrs.get("freq", MONTH_END)
    label = _FREQ_LABEL.get(freq, "period")
    season = _SEASON.get(freq)
    n = len(series)

    result = ForecastResult(
        column=column or str(series.name or "value"), date_col=date_col,
        freq=label, method="none", usable=False, n_periods=n,
        horizon=horizon,
        history=[(str(i.date()), round(float(v), 4))
                 for i, v in series.tail(36).items()],
        seasonal_periods=season)

    if n < MIN_PERIODS:
        result.reason = ("only {} {} periods — a pattern cannot be "
                         "separated from noise below {}".format(
                             n, label, MIN_PERIODS))
        result.verdict = ("Not enough history to forecast. " +
                          result.reason.capitalize() + ".")
        return result

    holdout = max(2, min(holdout, n // 4))
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    actual = test.to_numpy(dtype=float)

    naive_pred = _naive(train, holdout)
    naive_err = _mae(actual, naive_pred)
    result.naive_error = round(naive_err, 4)

    candidates: Dict[str, np.ndarray] = {"Last value carried forward": naive_pred}
    sn = _seasonal_naive(train, holdout, season) if season else None
    if sn is not None:
        candidates["Same period last cycle"] = sn
    candidates["Trend continued"] = _drift(train, holdout)
    hw = _holt_winters(train, holdout, season)
    if hw is not None:
        candidates["Exponential smoothing"] = hw

    scored = []
    for name, pred in candidates.items():
        try:
            scored.append((name, _mae(actual, pred)))
        except Exception:
            logger.debug("scoring %s failed", name, exc_info=True)
    scored.sort(key=lambda kv: kv[1])
    result.candidates = [(n_, round(e, 4)) for n_, e in scored]

    best_name, best_err = scored[0]
    result.method = best_name
    result.model_error = round(best_err, 4)
    result.skill = round((naive_err - best_err) / naive_err, 4) \
        if naive_err > 0 else 0.0

    if best_name == "Last value carried forward" or result.skill < MIN_SKILL:
        result.reason = (
            "no method beat carrying the last value forward on held-out "
            "periods (best improvement {:.0%})".format(max(result.skill, 0)))
        result.verdict = (
            "This series is best described as a level with noise around it. "
            "Nothing tested predicted it better than assuming next {} looks "
            "like this one, so no forecast is shown — the honest planning "
            "figure is the current level, {} the typical swing of {}."
            .format(label.rstrip("ly") if label.endswith("ly") else label,
                    "plus or minus", _fmt(float(series.diff().abs().median()))))
        return result

    # Refit the winner on everything and project forward.
    full_pred = _refit(series, best_name, horizon, season)
    if full_pred is None:
        result.reason = "the selected method could not be refitted"
        result.verdict = "Forecast unavailable: " + result.reason + "."
        return result

    # Interval from backtest error, widening with horizon — uncertainty
    # compounds the further out the projection runs, and a flat band
    # understates the risk at the far end.
    spread = max(best_err, float(np.std(actual - candidates[best_name])))
    future_index = pd.date_range(series.index[-1], periods=horizon + 1,
                                 freq=freq)[1:]
    for i, (idx, value) in enumerate(zip(future_index, full_pred), start=1):
        widen = spread * 1.96 * np.sqrt(i)
        result.points.append(ForecastPoint(
            period=str(idx.date()), value=round(float(value), 4),
            lower=round(float(value - widen), 4),
            upper=round(float(value + widen), 4)))

    result.usable = True
    last = result.points[-1]
    current = float(series.iloc[-1])
    direction = ("above" if last.value > current
                 else "below" if last.value < current else "level with")
    change = abs(last.value - current)
    result.verdict = (
        "Projected over the next {} {} periods to {}, which is {} {} the "
        "current {}. The range at that point is {} to {}. {} was selected "
        "because it predicted held-out periods {:.0%} more accurately than "
        "carrying the last value forward."
    ).format(horizon, label, _fmt(last.value),
             _fmt(change) if direction != "level with" else "",
             direction, _fmt(current), _fmt(last.lower), _fmt(last.upper),
             best_name, result.skill)
    return result


def _refit(series: pd.Series, method: str, horizon: int,
           season: Optional[int]) -> Optional[np.ndarray]:
    if method == "Same period last cycle" and season:
        return _seasonal_naive(series, horizon, season)
    if method == "Trend continued":
        return _drift(series, horizon)
    if method == "Exponential smoothing":
        return _holt_winters(series, horizon, season)
    return _naive(series, horizon)


def _fmt(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    if abs(v) >= 1_000_000:
        return "{:,.1f}M".format(v / 1_000_000)
    if abs(v) >= 1000:
        return "{:,.0f}".format(v)
    return "{:,.2f}".format(v)


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

def find_forecastable(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """(date column, measure) pairs worth projecting.

    Measures are ranked the way charts rank them, so the forecast leads
    with what the business actually tracks rather than the first numeric
    column in the frame.
    """
    dates = df.select_dtypes(include="datetime").columns.tolist()
    if not dates:
        for c in df.columns:
            if any(k in str(c).lower() for k in ("date", "month", "period",
                                                 "day", "time", "week")):
                try:
                    if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.8:
                        dates.append(c)
                except Exception:
                    logger.debug("date sniff failed for %r", c, exc_info=True)
    if not dates:
        return []
    try:
        from app.engines.chart_exporter import _rank_measures
        measures = _rank_measures(
            df, df.select_dtypes(include="number").columns.tolist())
    except Exception:
        logger.warning("measure ranking unavailable", exc_info=True)
        measures = df.select_dtypes(include="number").columns.tolist()
    return [(dates[0], m) for m in measures[:3]]


def run_forecast(df: pd.DataFrame, date_col: Optional[str] = None,
                 value_col: Optional[str] = None,
                 horizon: int = 3) -> Optional[ForecastResult]:
    """Forecast one measure, choosing sensible columns when none is named."""
    pairs = ([(date_col, value_col)] if date_col and value_col
             else find_forecastable(df))
    if not pairs:
        return None
    d, v = pairs[0]
    if d not in df.columns or v not in df.columns:
        return None
    how = "sum"
    try:
        from app.engines.chart_exporter import _agg_for_metric
        how, _is_score = _agg_for_metric(v)
    except Exception:
        logger.debug("aggregation lookup failed for %r", v, exc_info=True)
    series = build_series(df, d, v, how=how)
    if series is None or series.empty:
        return None
    return forecast_series(series, horizon=horizon, column=v, date_col=d)
