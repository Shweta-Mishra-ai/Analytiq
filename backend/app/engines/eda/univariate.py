"""
engines/eda/univariate.py — one column at a time.

Distribution shape, spread, outliers, entropy, and which named
distribution (if any) describes the column. The distribution fitting in
particular is the part with the sharp edges — see _fit_distribution for
why it samples and why it ranks candidates by BIC rather than by
goodness of fit.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

from typing import Dict, Tuple

from scipy.stats import (anderson, kstest, normaltest,
                         shapiro)

from app.engines.eda.results import UnivariateResult
from app.engines.statistics import assess_normality, clamp_p


#  UNIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════

def _modified_zscore_outliers(s: pd.Series) -> int:
    """Modified Z-score (Iglewicz & Hoaglin) — robust to non-normal data."""
    median = s.median()
    mad    = np.median(np.abs(s - median))
    if mad == 0:
        return 0
    mz = 0.6745 * (s - median) / mad
    return int((np.abs(mz) > 3.5).sum())


def _entropy(series: pd.Series) -> float:
    """Shannon entropy — measures diversity of categories."""
    vc   = series.value_counts(normalize=True)
    return float(-np.sum(vc * np.log2(vc + 1e-10)))


#: Distribution fitting is done on a sample this size. Two reasons, and
#: the second matters more than the first.
#:
#: Cost: fitting five distributions by numerical maximum likelihood over
#: a whole column took 2.9 seconds on 100,000 rows, which was most of the
#: EDA run. The fitted parameters are indistinguishable at this size.
#:
#: 2,000 rather than more: measured, the identification is stable at this
#: size and degrades at 1,000 (an exponential column started coming back
#: as gamma), while the cost falls with the sample.
#:
#: Meaning: the Kolmogorov-Smirnov test's power grows with n, so past a
#: few thousand rows it rejects every candidate — real data is never
#: exactly lognormal — and every p-value collapses to zero. Which is how
#: the previous version of this function came to report "norm" for
#: clearly lognormal data: with all p-values at 0.0 and the running best
#: initialised to 0, `p > best_p` was never true and it returned its own
#: untried default.
FIT_SAMPLE_SIZE = 2_000

#: A KS statistic (not p-value) above this means no candidate described
#: the data well. Reported as "no standard distribution fits" rather than
#: silently naming one — the honest answer for a bimodal column or a
#: mixture, which business data very often is.
MAX_FIT_DISTANCE = 0.15


def _fit_distribution(s: pd.Series) -> Tuple[str, Dict]:
    """Which common distribution best describes this column.

    Ranked by the KS *statistic* rather than its p-value. The statistic
    is a distance — how far the fitted curve sits from the data — and it
    stays meaningful at any sample size. The p-value answers a different
    question ("could this have come from that distribution"), and at
    scale the answer is always no, which makes it useless for choosing
    between candidates even though it looks like a quality score.
    """
    values = s.dropna()
    if len(values) < 20:
        return "unknown", {}

    if len(values) > FIT_SAMPLE_SIZE:
        # Deterministic seed: the same column must produce the same
        # answer on every report build, or two runs of the same analysis
        # disagree about the data's shape.
        values = values.sample(FIT_SAMPLE_SIZE, random_state=0)

    best_dist, best_bic, best_distance, best_params = "unknown", float("inf"), float("inf"), {}
    for dist_name in ("norm", "lognorm", "expon", "gamma", "uniform"):
        try:
            dist = getattr(scipy_stats, dist_name)
            params = dist.fit(values)
            distance, p = kstest(values, dist_name, args=params)
            # Bayesian information criterion: twice the negative
            # log-likelihood plus log(n) per fitted parameter.
            #
            # Ranking by goodness of fit alone hands the win to whichever
            # candidate has the most parameters — a three-parameter gamma
            # imitates a two-parameter normal closely enough to edge past
            # it, and textbook-normal data was being reported as "gamma"
            # for exactly that reason. A penalty is needed.
            #
            # BIC rather than AIC because these candidates are nested:
            # exponential *is* gamma with the shape fixed at 1, so gamma
            # can always match it. AIC's flat penalty of 2 per parameter
            # is too weak to prefer the simpler form, and exponential data
            # came back as gamma. BIC's log(n) penalty — about 7.6 at this
            # sample size — is enough, and it is the right criterion here
            # anyway: AIC optimises predictive accuracy, BIC identifies
            # which model generated the data, and identification is the
            # question being asked.
            nnlf = float(dist.nnlf(params, values))
            bic = math.log(len(values)) * len(params) + 2 * nnlf
            if not math.isfinite(bic):
                continue
        except Exception:
            logger.debug("could not fit %s", dist_name, exc_info=True)
            continue
        if bic < best_bic:
            best_dist, best_bic, best_distance = dist_name, bic, distance
            best_params = {
                "params": params,
                "bic": round(bic, 2),
                "ks_distance": round(float(distance), 4),
                "ks_p": round(float(p), 4),
                "fitted_on": int(len(values)),
            }

    if best_distance > MAX_FIT_DISTANCE:
        # Naming a distribution that does not fit is worse than saying
        # so: the shape drives which summary statistics and which tests
        # are appropriate downstream.
        return "none", {"closest": best_dist,
                        "ks_distance": round(float(best_distance), 4),
                        "note": "No standard distribution describes this "
                                "column well — it may be bimodal, mixed, "
                                "or heavily rounded."}
    return best_dist, best_params


def analyze_univariate(series: pd.Series) -> UnivariateResult:
    """Full univariate analysis for one column."""
    name  = str(series.name)
    clean = series.dropna()
    n     = len(clean)

    result = UnivariateResult(
        column=name, dtype=str(series.dtype),
        n=n, missing=int(series.isna().sum()),
        missing_pct=round(series.isna().mean() * 100, 2),
        unique_count=int(clean.nunique()),
    )

    if n < 3:
        result.interpretation = "Too few values for analysis."
        return result

    # ── Categorical ───────────────────────────────────────
    if series.dtype == object or str(series.dtype) == "str":
        vc = clean.value_counts()
        result.top_value   = str(vc.index[0])[:40] if len(vc) > 0 else None
        result.top_pct     = round(vc.iloc[0] / n * 100, 2) if len(vc) > 0 else None
        result.entropy     = round(_entropy(clean), 4)
        uniq_pct           = clean.nunique() / n
        if uniq_pct > 0.8:
            result.interpretation = (
                "High cardinality ({} unique / {} rows = {:.0f}%) — "
                "likely an ID or free-text column. Not suitable for grouping.".format(
                    clean.nunique(), n, uniq_pct * 100))
        elif result.top_pct and result.top_pct > 80:
            result.interpretation = (
                "Dominated by '{}' ({:.0f}%) — "
                "low variance, limited analytical value.".format(
                    result.top_value, result.top_pct))
        else:
            result.interpretation = (
                "{} categories. Top: '{}' ({:.0f}%). "
                "Entropy={:.2f} (higher = more diverse).".format(
                    clean.nunique(), result.top_value,
                    result.top_pct or 0, result.entropy or 0))
        return result

    # ── Numeric ───────────────────────────────────────────
    s = clean.astype(float)

    # Descriptive
    result.mean     = round(float(s.mean()), 6)
    result.median   = round(float(s.median()), 6)
    result.std      = round(float(s.std()), 6)
    result.variance = round(float(s.var()), 6)
    result.min_val  = round(float(s.min()), 6)
    result.max_val  = round(float(s.max()), 6)
    result.range_val = round(result.max_val - result.min_val, 6)
    result.q1       = round(float(s.quantile(0.25)), 6)
    result.q3       = round(float(s.quantile(0.75)), 6)
    result.iqr      = round(result.q3 - result.q1, 6)
    result.p5       = round(float(s.quantile(0.05)), 6)
    result.p95      = round(float(s.quantile(0.95)), 6)
    result.cv       = round(result.std / abs(result.mean), 4) if result.mean != 0 else 0

    try:
        mode_val   = float(s.mode().iloc[0])
        result.mode = round(mode_val, 6)
    except Exception:
        logger.debug("analyze_univariate: suppressed exception", exc_info=True)

    # Distribution shape
    skew = float(s.skew())
    kurt = float(s.kurtosis())
    result.skewness = round(skew, 4)
    result.kurtosis = round(kurt, 4)

    if abs(skew) < 0.5:
        result.skew_label = "Approximately symmetric"
    elif 0.5 <= abs(skew) < 1:
        result.skew_label = "Moderately {}".format(
            "right-skewed" if skew > 0 else "left-skewed")
    else:
        result.skew_label = "Heavily {}".format(
            "right-skewed" if skew > 0 else "left-skewed")

    if kurt > 3:
        result.kurtosis_label = "Leptokurtic — heavy tails, extreme values likely"
    elif kurt < -1:
        result.kurtosis_label = "Platykurtic — light tails, few extremes"
    else:
        result.kurtosis_label = "Mesokurtic — normal-like tails"

    # Normality tests
    sample = s.sample(min(n, 5000), random_state=42)
    try:
        sw_stat, sw_p = shapiro(sample)
        result.shapiro_stat = round(float(sw_stat), 6)
        result.shapiro_p    = round(float(sw_p), 6)
    except Exception:
        logger.debug("analyze_univariate: suppressed exception", exc_info=True)

    try:
        da_stat, da_p = normaltest(s)
        result.dagostino_stat = round(float(da_stat), 6)
        result.dagostino_p    = round(float(da_p), 6)
    except Exception:
        logger.debug("analyze_univariate: suppressed exception", exc_info=True)

    # SciPy 1.17 warns on every call that leaves `method` unset, and from
    # 1.19 `critical_values` is gone entirely — at which point this block
    # would have raised into the bare `except` below and the statistic
    # would have quietly stopped appearing, with nothing to say why.
    # Asking for the p-value directly is the supported form and is the
    # number a reader can actually use; the 5% critical value it replaces
    # was written to the result and never read by anything.
    try:
        try:
            ad_result = anderson(sample, dist="norm", method="interpolate")
            result.anderson_p = round(float(ad_result.pvalue), 6)
        except TypeError:       # SciPy < 1.17 has no `method` argument
            ad_result = anderson(sample, dist="norm")
            result.anderson_critical = round(
                float(ad_result.critical_values[2]), 6)
        result.anderson_stat = round(float(ad_result.statistic), 6)
    except Exception:
        logger.debug("analyze_univariate: suppressed exception", exc_info=True)

    # Normality decided on shape, not on a vote between three
    # significance tests. All three answer the same over-powered
    # question at scale — at n=1,470 each of them rejects a column with
    # a skew of 0.48, so a majority of three is a majority of three wrong
    # answers. The test statistics stay in the output for a reader who
    # wants them; they no longer decide anything on a large sample.
    verdict = assess_normality(s)
    if verdict is None:
        result.is_normal = None
        result.normality_verdict = "Not assessed"
    else:
        result.is_normal = verdict.normal_enough
        result.normality_basis = verdict.basis
        result.normality_verdict = (
            "Normal enough for parametric tests (skew {:.2f}, excess "
            "kurtosis {:.2f})".format(verdict.skew, verdict.excess_kurtosis)
            if verdict.normal_enough else
            "Too skewed or heavy-tailed for parametric tests (skew {:.2f}, "
            "excess kurtosis {:.2f})".format(verdict.skew,
                                             verdict.excess_kurtosis))
    # p-values are floored rather than rounded to a literal zero.
    result.shapiro_p   = clamp_p(result.shapiro_p)
    result.dagostino_p = clamp_p(result.dagostino_p)

    # Outliers — 3 methods
    if result.iqr and result.iqr > 0:
        lo = result.q1 - 1.5 * result.iqr
        hi = result.q3 + 1.5 * result.iqr
        result.iqr_lower       = round(lo, 4)
        result.iqr_upper       = round(hi, 4)
        result.outliers_iqr    = int(((s < lo) | (s > hi)).sum())

    if result.std and result.std > 0:
        z = np.abs((s - result.mean) / result.std)
        result.outliers_zscore = int((z > 3).sum())

    result.outliers_modz = _modified_zscore_outliers(s)
    result.outlier_pct   = round(result.outliers_iqr / n * 100, 2)

    # Method recommendation
    result.recommended_method = (
        "Z-Score (data is normal)" if result.is_normal
        else "Modified Z-Score (robust, non-normal data)"
    )

    # Distribution fit
    if n >= 30:
        try:
            result.best_fit_dist, result.best_fit_params = _fit_distribution(s)
        except Exception:
            logger.debug("analyze_univariate: suppressed exception", exc_info=True)

    # Plain English interpretation
    mean_vs_median = abs(result.mean - result.median)
    rel_diff = mean_vs_median / abs(result.median) if result.median != 0 else 0

    if rel_diff > 0.1:
        central = "Mean ({:.2f}) and median ({:.2f}) differ by {:.0f}% — use median for reporting.".format(
            result.mean, result.median, rel_diff * 100)
    else:
        central = "Mean ({:.2f}) and median ({:.2f}) are close — distribution is symmetric.".format(
            result.mean, result.median)

    outlier_note = ""
    if result.outlier_pct > 5:
        outlier_note = " {:.1f}% outliers detected — validate before analysis.".format(
            result.outlier_pct)

    result.interpretation = "{} {} {} {}".format(
        result.skew_label + ".",
        central,
        result.kurtosis_label + ".",
        outlier_note
    ).strip()

    return result


# ══════════════════════════════════════════════════════════