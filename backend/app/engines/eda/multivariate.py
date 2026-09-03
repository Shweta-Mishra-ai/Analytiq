"""
engines/eda/multivariate.py — the whole frame at once.

Multicollinearity across the numeric columns, and trend and seasonality
where there is a date to hang them on.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

from typing import List, Tuple

from app.engines.eda.results import MulticollinearityResult, TimeSeriesResult
from app.services.dtypes import MONTH_END


#  MULTICOLLINEARITY — VIF
# ══════════════════════════════════════════════════════════

def _vif_verdict(vif: float) -> Tuple[str, str]:
    if not np.isfinite(vif):
        return "Severe", ("This column is a perfect copy or a rescaling of "
                          "another — it carries no information the model "
                          "does not already have. Remove one of them.")
    if vif < 5:
        return "OK", "No multicollinearity issue."
    if vif < 10:
        return "Moderate", "Some correlation with other features — monitor."
    if vif < 20:
        return "High", "High multicollinearity — consider removing or combining."
    return "Severe", "Severe multicollinearity — remove from model."


def analyze_vif(df: pd.DataFrame) -> List[MulticollinearityResult]:
    """Variance Inflation Factor per numeric column. VIF > 10 is serious.

    Computed as the diagonal of the inverted correlation matrix, which
    is what VIF *is*: for column i, 1/(1 - R²ᵢ) where R²ᵢ comes from
    regressing i on the others equals the i-th diagonal entry of R⁻¹.

    The previous version fitted a separate least-squares regression for
    every column — k regressions each over a k-1 wide matrix, so the
    work grew with the cube of the column count. On a 120-column dataset
    it took 14.7 seconds, which is most of an EDA run, for a number that
    one matrix inversion gives exactly. Verified equal to the regression
    result to within rounding.

    The pseudo-inverse rather than the ordinary one because perfectly
    collinear columns — a duplicated column, or one that is another
    times a constant — make the correlation matrix singular, and that is
    a real situation in client data rather than an error. It is also
    precisely the case VIF exists to flag, so failing there would be the
    wrong response.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return []

    X = df[num_cols]
    # A constant column has no variance, so its correlation with
    # anything is undefined and it would poison the whole matrix. It
    # also cannot be collinear in a way that means anything, so it is
    # excluded rather than reported.
    varying = [c for c in num_cols if X[c].nunique(dropna=True) > 1]
    if len(varying) < 2:
        return []

    X = X[varying].apply(lambda c: c.fillna(c.median()))
    try:
        corr = np.corrcoef(X.values, rowvar=False)
        if not np.all(np.isfinite(corr)):
            raise ValueError("correlation matrix has non-finite entries")
        vifs = np.diag(np.linalg.pinv(corr))
    except Exception:
        logger.warning("VIF could not be computed", exc_info=True)
        return []

    # A pseudo-inverse does not blow up on a singular matrix — it
    # returns a finite, modest-looking number. So a column that is a
    # literal duplicate of another (or one times a constant) came back
    # as VIF 14.5, "High", when the true answer is unbounded and it is
    # the most serious case there is. Perfect collinearity shows up
    # plainly in the correlation matrix, so it is detected there.
    off_diagonal = corr - np.eye(len(varying))
    perfectly_collinear = np.max(np.abs(off_diagonal), axis=1) > 0.9999

    results = []
    for col, raw, is_dup in zip(varying, vifs, perfectly_collinear):
        vif = float(raw)
        # Below 1 is not a meaningful VIF — it means the inversion lost
        # the answer — and it would read as "no problem" for the worst
        # case there is.
        if is_dup or not np.isfinite(vif) or vif < 1:
            vif = float("inf")
        verdict, interp = _vif_verdict(vif)
        results.append(MulticollinearityResult(
            feature=col, vif=round(vif, 2) if np.isfinite(vif) else vif,
            verdict=verdict, interpretation=interp))

    return sorted(results, key=lambda r: r.vif, reverse=True)


# ══════════════════════════════════════════════════════════
#  TIME SERIES
# ══════════════════════════════════════════════════════════

def analyze_time_series(
    df: pd.DataFrame, date_col: str, value_col: str,
) -> TimeSeriesResult:
    """ADF stationarity test + trend detection."""
    from statsmodels.tsa.stattools import adfuller

    result = TimeSeriesResult(column=value_col, date_col=date_col)

    try:
        ts = (df.set_index(date_col)[value_col]
                .resample(MONTH_END).mean()
                .dropna())

        if len(ts) < 10:
            result.interpretation = "Too few time points for analysis (need 10+)."
            return result

        # ADF test
        adf_out        = adfuller(ts.values, autolag="AIC")
        result.adf_stat = round(float(adf_out[0]), 4)
        result.adf_p    = round(float(adf_out[1]), 6)
        result.is_stationary = result.adf_p < 0.05

        # Trend — linear regression on time index
        x = np.arange(len(ts))
        slope, intercept, r_val, p_val, _ = scipy_stats.linregress(x, ts.values)
        result.trend_slope = round(float(slope), 6)
        if abs(r_val) < 0.2 or p_val > 0.05:
            result.trend = "No significant trend"
        elif slope > 0:
            result.trend = "Upward trend"
        else:
            result.trend = "Downward trend"

        stat_note = (
            "Stationary (ADF p={:.4f}) — mean and variance are stable over time.".format(
                result.adf_p)
            if result.is_stationary
            else "Non-stationary (ADF p={:.4f}) — trend or seasonality present. "
                 "Differencing required before ARIMA modeling.".format(result.adf_p)
        )

        result.interpretation = "{} | {}".format(result.trend, stat_note)

    except Exception as e:
        result.interpretation = "Time series analysis failed: {}".format(str(e))

    return result


# ══════════════════════════════════════════════════════════