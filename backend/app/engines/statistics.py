"""
engines/statistics.py — the statistical decisions the rest of the app
makes, made once and made correctly.

Three mistakes were spread across the engines, each of them the kind that
looks rigorous and is not.

**A significance test answering a magnitude question.** Normality was
decided by Shapiro-Wilk at p<0.05. At n=1,470 that test rejects
essentially any real data: monthly income with a skew of 0.48 and a
kurtosis of 0.65 — normal enough for any practical purpose — came back
"Non-Normal", and so did every other column in the file. The report then
recommended non-parametric tests throughout on the strength of it. The
power of a normality test grows with n, so at large n it stops answering
"is this normal enough to use a t-test" and starts answering "is n
large". Shape has to be judged on shape.

**A p-value printed as zero.** `round(p, 6)` on a p-value of 1e-40 gives
0.0, and the report said "p=0.0000". No evidence makes a hypothesis
impossible; a floor and a "<" are the honest rendering.

**A point estimate with no interval.** A correlation of 0.66 on 1,470
rows and one on 40 rows were printed identically.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# Below this a p-value is reported as a bound rather than a number: it is
# past the point where floating point carries meaning, and far past the
# point where the difference between 1e-17 and 1e-40 changes a decision.
P_FLOOR = 1e-16

# Shape thresholds for "normal enough to use a parametric test". These are
# the conventional applied limits (Kline; Curran, West & Finch): well
# inside them, t-tests and Pearson correlations behave.
MAX_SKEW = 1.0
MAX_EXCESS_KURTOSIS = 2.0

# Above this many rows a normality test's p-value is ignored in favour of
# shape, because its power makes rejection near-certain.
TEST_TRUSTWORTHY_UP_TO = 300


def format_p(p: Optional[float]) -> str:
    """A p-value as it should appear in a document."""
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "—"
    p = float(p)
    if p < P_FLOOR:
        return "p < 0.0001"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return "p = {:.3f}".format(p)
    return "p = {:.3f}".format(p)


def clamp_p(p: Optional[float]) -> Optional[float]:
    """A p-value never rounded to a literal zero."""
    if p is None:
        return None
    try:
        value = float(p)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return max(value, P_FLOOR)


@dataclass
class NormalityVerdict:
    normal_enough: bool
    basis: str              # "shape" or "test"
    skew: float
    excess_kurtosis: float
    test_name: str = ""
    p_value: Optional[float] = None
    note: str = ""


def assess_normality(series) -> Optional[NormalityVerdict]:
    """Is this column normal enough to analyse with parametric methods?

    Not "would a normality test reject it" — at any useful sample size it
    will. Skew and excess kurtosis answer the question that actually
    determines whether a t-test or a Pearson correlation is safe, and
    they do not get more damning as the data gets bigger.

    The test still runs and is still reported, because a reader who wants
    it should have it; it just does not decide anything on a large
    sample.
    """
    try:
        clean = pd.Series(series).dropna().astype(float)
    except (TypeError, ValueError):
        return None
    n = len(clean)
    if n < 8 or clean.nunique() < 3:
        return None

    skew = float(clean.skew())
    kurt = float(clean.kurtosis())          # pandas returns excess kurtosis
    if not (math.isfinite(skew) and math.isfinite(kurt)):
        return None

    test_name, p_value = "", None
    try:
        if n <= 5000:
            _stat, p_value = scipy_stats.shapiro(clean)
            test_name = "Shapiro-Wilk"
        else:
            _stat, p_value = scipy_stats.normaltest(clean)
            test_name = "D'Agostino-Pearson"
        p_value = clamp_p(p_value)
    except Exception:
        logger.debug("normality test failed", exc_info=True)

    shape_ok = abs(skew) <= MAX_SKEW and abs(kurt) <= MAX_EXCESS_KURTOSIS

    if n <= TEST_TRUSTWORTHY_UP_TO and p_value is not None:
        # Small sample: the test is the better guide, because shape
        # statistics are themselves unstable here.
        normal = bool(p_value > 0.05)
        basis = "test"
        note = ("{} on {:,} rows, {}.".format(test_name, n, format_p(p_value)))
    else:
        normal = shape_ok
        basis = "shape"
        note = ("Skew {:.2f}, excess kurtosis {:.2f} on {:,} rows. At this "
                "sample size a normality test rejects almost any real "
                "column, so the decision rests on shape; {} returned {} "
                "for the record.".format(skew, kurt, n, test_name,
                                         format_p(p_value)))
    return NormalityVerdict(normal_enough=normal, basis=basis, skew=round(skew, 4),
                            excess_kurtosis=round(kurt, 4), test_name=test_name,
                            p_value=p_value, note=note)


@dataclass
class CorrelationEstimate:
    r: float
    ci_low: float
    ci_high: float
    p_value: Optional[float]
    n: int
    method: str
    strength: str
    significant: bool


def correlation_with_ci(x, y, method: str = "pearson",
                        confidence: float = 0.95) -> Optional[CorrelationEstimate]:
    """A correlation with the interval that says how firm it is.

    Fisher's z transform: r is bounded and its sampling distribution is
    skewed near ±1, so the interval is built in z space and mapped back.
    A correlation printed without one invites the reader to treat 0.66 on
    forty rows the same as 0.66 on fourteen hundred.
    """
    try:
        a = pd.Series(x).astype(float)
        b = pd.Series(y).astype(float)
        mask = a.notna() & b.notna()
        a, b = a[mask], b[mask]
        n = int(len(a))
        if n < 10 or a.nunique() < 3 or b.nunique() < 3:
            return None
        if method == "spearman":
            r, p = scipy_stats.spearmanr(a, b)
        else:
            r, p = scipy_stats.pearsonr(a, b)
        r = float(r)
        if not math.isfinite(r):
            return None

        # Fisher z, with the Spearman standard error correction.
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        se = (1.06 / math.sqrt(n - 3)) if method == "spearman" \
            else (1.0 / math.sqrt(n - 3))
        crit = float(scipy_stats.norm.ppf(1 - (1 - confidence) / 2))
        lo, hi = np.tanh(z - crit * se), np.tanh(z + crit * se)
        p = clamp_p(p)
        return CorrelationEstimate(
            r=round(r, 4), ci_low=round(float(lo), 4),
            ci_high=round(float(hi), 4), p_value=p, n=n, method=method,
            strength=correlation_strength(r),
            significant=bool(p is not None and p < 0.05),
        )
    except Exception:
        logger.debug("correlation estimate failed", exc_info=True)
        return None


def correlation_strength(r: float) -> str:
    """Cohen's conventional bands, named consistently everywhere.

    0.66 was being called "moderate" in one place and "strong" in
    another, in the same report.
    """
    a = abs(float(r))
    if a >= 0.7:
        return "very strong"
    if a >= 0.5:
        return "strong"
    if a >= 0.3:
        return "moderate"
    if a >= 0.1:
        return "weak"
    return "negligible"


def mean_difference_ci(a, b, confidence: float = 0.95):
    """(difference, low, high) for two independent group means.

    Welch, so unequal variances and unequal group sizes are handled — the
    common case when one department has 142 people and another has 701.
    """
    try:
        x = pd.Series(a).dropna().astype(float)
        y = pd.Series(b).dropna().astype(float)
        if len(x) < 3 or len(y) < 3:
            return None
        diff = float(x.mean() - y.mean())
        se = math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        if se == 0:
            return None
        dof = (x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)) ** 2 / (
            (x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1)
            + (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1))
        crit = float(scipy_stats.t.ppf(1 - (1 - confidence) / 2, dof))
        return diff, diff - crit * se, diff + crit * se
    except Exception:
        logger.debug("mean difference interval failed", exc_info=True)
        return None


def cohens_d(a, b) -> Optional[float]:
    """Standardised difference between two group means (pooled SD)."""
    try:
        x = pd.Series(a).dropna().astype(float)
        y = pd.Series(b).dropna().astype(float)
        if len(x) < 3 or len(y) < 3:
            return None
        pooled = math.sqrt(((len(x) - 1) * x.var(ddof=1)
                            + (len(y) - 1) * y.var(ddof=1))
                           / (len(x) + len(y) - 2))
        if pooled == 0:
            return None
        return float((x.mean() - y.mean()) / pooled)
    except Exception:
        logger.debug("cohen's d failed", exc_info=True)
        return None


def effect_label(value: float, kind: str = "d") -> str:
    """Plain words for an effect size, by its own conventional bands."""
    v = abs(float(value))
    if kind == "eta":            # eta-squared / epsilon-squared
        if v >= 0.14:
            return "large"
        if v >= 0.06:
            return "moderate"
        if v >= 0.01:
            return "small"
        return "negligible"
    if kind == "v":              # Cramer's V
        if v >= 0.5:
            return "large"
        if v >= 0.3:
            return "moderate"
        if v >= 0.1:
            return "small"
        return "negligible"
    if v >= 0.8:                 # Cohen's d
        return "large"
    if v >= 0.5:
        return "moderate"
    if v >= 0.2:
        return "small"
    return "negligible"


def is_worth_reporting(p_value: Optional[float], effect: Optional[float],
                       kind: str = "d") -> bool:
    """Significant AND large enough to change a decision.

    At n=1,470 a difference in average age of one year clears p<0.05 with
    an effect size of 0.002. Reporting it as a finding invites someone to
    act on nothing.
    """
    if p_value is None or p_value >= 0.05:
        return False
    if effect is None:
        return True
    return effect_label(effect, kind) != "negligible"
