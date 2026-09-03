"""
engines/ab_test_engine.py — A/B Test significance calculator.

Supports two modes:
  1. Conversion rate test (binary outcome: converted / not converted) —
     uses a two-proportion z-test (or Fisher's exact test for small samples).
  2. Continuous metric test (revenue, time-on-page, order value, etc.) —
     uses Welch's t-test (unequal variance) with a Mann-Whitney U fallback
     when the data is heavily non-normal.

Computes: p-value, confidence interval, statistical power, minimum
detectable effect, and a plain-English verdict. All computation is done
directly from the submitted data — no external significance tables.

Ported from dataforge-ai's core/ab_test_engine.py (pure scipy/numpy/pandas,
no framework coupling).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class ABTestResult:
    test_type:         str    # "conversion" | "continuous"
    metric_name:       str
    variant_a_name:    str
    variant_b_name:    str
    n_a:               int
    n_b:               int
    # Conversion-specific
    conversions_a:     Optional[int] = None
    conversions_b:     Optional[int] = None
    rate_a:            Optional[float] = None
    rate_b:            Optional[float] = None
    # Continuous-specific
    mean_a:            Optional[float] = None
    mean_b:            Optional[float] = None
    std_a:             Optional[float] = None
    std_b:             Optional[float] = None
    median_a:          Optional[float] = None
    median_b:          Optional[float] = None
    # Shared stats
    test_used:         str = ""
    statistic:         float = 0.0
    p_value:           float = 1.0
    is_significant:    bool = False
    confidence_level:  float = 0.95
    relative_uplift:   float = 0.0     # % change B vs A
    absolute_diff:     float = 0.0
    ci_lower:          float = 0.0     # CI on the difference
    ci_upper:          float = 0.0
    power:             Optional[float] = None
    min_detectable_effect: Optional[float] = None
    sample_size_adequate:  bool = True
    verdict:           str = ""
    recommendation:    str = ""
    warnings:          list = field(default_factory=list)


# ══════════════════════════════════════════════════════════
#  SAMPLE SIZE / POWER
# ══════════════════════════════════════════════════════════

def required_sample_size(
    baseline_rate: float,
    min_detectable_effect_pct: float = 10.0,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Minimum sample size PER VARIANT for a two-proportion test to detect
    a relative effect of `min_detectable_effect_pct` with the given
    alpha/power. Standard formula using normal approximation.
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be between 0 and 1")
    p1 = baseline_rate
    p2 = baseline_rate * (1 + min_detectable_effect_pct / 100)
    p2 = min(p2, 0.999)

    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
    z_beta  = scipy_stats.norm.ppf(power)
    p_bar   = (p1 + p2) / 2

    numerator = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
                 z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    if denominator == 0:
        return 0
    n = numerator / denominator
    return int(np.ceil(n))


def _post_hoc_power_proportions(n_a: int, n_b: int, p_a: float, p_b: float,
                                 alpha: float = 0.05) -> float:
    """Approximate post-hoc statistical power for a two-proportion test."""
    try:
        p_pool = (p_a * n_a + p_b * n_b) / (n_a + n_b)
        se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        se_diff = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
        if se_pool == 0 or se_diff == 0:
            return 0.0
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
        effect  = abs(p_b - p_a)
        z_power = (effect - z_alpha * se_pool) / se_diff
        return float(scipy_stats.norm.cdf(z_power))
    except Exception:
        logger.warning("Post-hoc power calc failed", exc_info=True)
        return 0.0


def _post_hoc_power_means(n_a: int, n_b: int, std_a: float, std_b: float,
                          mean_diff: float, alpha: float = 0.05) -> float:
    """Approximate post-hoc statistical power for a two-sample mean test."""
    try:
        pooled_std = np.sqrt(((n_a-1)*std_a**2 + (n_b-1)*std_b**2) / max(n_a+n_b-2, 1))
        if pooled_std == 0:
            return 0.0
        se = pooled_std * np.sqrt(1/n_a + 1/n_b)
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
        z_power = (abs(mean_diff) - z_alpha * se) / se if se > 0 else 0
        return float(scipy_stats.norm.cdf(z_power))
    except Exception:
        logger.warning("Post-hoc power calc (means) failed", exc_info=True)
        return 0.0


# ══════════════════════════════════════════════════════════
#  CONVERSION RATE TEST  (two-proportion z-test)
# ══════════════════════════════════════════════════════════

def run_conversion_test(
    conversions_a: int, n_a: int,
    conversions_b: int, n_b: int,
    variant_a_name: str = "Control (A)",
    variant_b_name: str = "Variant (B)",
    metric_name: str = "Conversion Rate",
    confidence_level: float = 0.95,
) -> ABTestResult:
    """
    Two-proportion significance test. Uses z-test for adequate sample
    sizes, falls back to Fisher's exact test when any cell count < 5
    (z-test's normal approximation breaks down for tiny samples).

    Raises:
        ValueError — if n_a or n_b is 0, or conversions exceed n.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Sample sizes must be positive")
    if conversions_a > n_a or conversions_b > n_b:
        raise ValueError("Conversions cannot exceed sample size")

    warnings_list = []
    rate_a = conversions_a / n_a
    rate_b = conversions_b / n_b
    alpha = 1 - confidence_level

    # Cell count check — z-test normal approximation needs ≥5 per cell
    cells = [conversions_a, n_a - conversions_a, conversions_b, n_b - conversions_b]
    use_fisher = any(c < 5 for c in cells)

    if use_fisher:
        table = [[conversions_a, n_a - conversions_a],
                 [conversions_b, n_b - conversions_b]]
        odds_ratio, p_value = scipy_stats.fisher_exact(table)
        test_used = "Fisher's Exact Test"
        statistic = float(odds_ratio)
        warnings_list.append(
            "Sample size is small (a cell count < 5) — using Fisher's Exact "
            "Test instead of z-test for accuracy. Results should be treated "
            "as directional until more data is collected."
        )
    else:
        p_pool = (conversions_a + conversions_b) / (n_a + n_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        if se == 0:
            z_stat, p_value = 0.0, 1.0
        else:
            z_stat = (rate_b - rate_a) / se
            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
        test_used = "Two-Proportion Z-Test"
        statistic = float(z_stat)

    # CI on the difference (Wald CI)
    se_diff = np.sqrt(rate_a*(1-rate_a)/n_a + rate_b*(1-rate_b)/n_b)
    z_crit  = scipy_stats.norm.ppf(1 - alpha/2)
    diff    = rate_b - rate_a
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    is_sig = p_value < alpha
    rate_a_is_zero = (rate_a == 0)
    # Relative % change is undefined when the baseline is 0% — '(b-0)/0'
    # is not '0.0%', it's undefined/infinite. Defaulting to 0.0 silently
    # hid genuine differences (e.g. 0% vs 1% conversion) behind a verdict
    # that claimed 'no difference'. Percentage-point framing is used
    # instead in that case, which is always well-defined.
    relative_uplift = ((rate_b - rate_a) / rate_a * 100) if not rate_a_is_zero else 0.0
    pp_diff = (rate_b - rate_a) * 100   # percentage points, well-defined always

    power = _post_hoc_power_proportions(n_a, n_b, rate_a, rate_b, alpha)
    if 0 < rate_a < 1:
        min_n = required_sample_size(rate_a, abs(relative_uplift) or 5.0, alpha, 0.8)
        sample_adequate = (n_a >= min_n and n_b >= min_n) if min_n > 0 else True
    else:
        # Baseline rate of exactly 0% or 100% — minimum-detectable-effect
        # sample size formula is undefined at these boundaries (division
        # by zero in the effect-size denominator).
        min_n = 0
        sample_adequate = True
        warnings_list.append(
            f"Variant A's conversion rate is {rate_a:.0%} — sample size "
            "recommendation cannot be computed at this boundary. Interpret "
            "results with caution regardless of significance."
        )

    if not sample_adequate:
        warnings_list.append(
            f"Current sample size ({n_a}/{n_b} per variant) may be underpowered "
            f"to reliably detect this effect size. Recommended minimum: ~{min_n:,} "
            f"per variant for 80% power at this effect size."
        )

    uplift_phrase = (
        f"{pp_diff:+.1f} percentage points (baseline was 0%, so a relative "
        f"% change is undefined)" if rate_a_is_zero else
        f"{relative_uplift:+.1f}% relative"
    )

    if is_sig:
        direction = "outperforms" if rate_b > rate_a else "underperforms vs"
        verdict = (
            f"Statistically significant at {confidence_level:.0%} confidence "
            f"(p={p_value:.4f}). '{variant_b_name}' {direction} '{variant_a_name}' "
            f"by {uplift_phrase}."
        )
        recommendation = (
            f"Ship '{variant_b_name}'." if rate_b > rate_a else
            f"Keep '{variant_a_name}' — the tested variant performed worse."
        )
    else:
        verdict = (
            f"NOT statistically significant (p={p_value:.4f}, threshold={alpha:.2f}). "
            f"Observed difference of {uplift_phrase} could be due to chance."
        )
        recommendation = (
            "Do not ship based on this result. Either collect more data to "
            "reach adequate power, or the variants perform equivalently."
        )

    return ABTestResult(
        test_type="conversion", metric_name=metric_name,
        variant_a_name=variant_a_name, variant_b_name=variant_b_name,
        n_a=n_a, n_b=n_b,
        conversions_a=conversions_a, conversions_b=conversions_b,
        rate_a=round(rate_a, 4), rate_b=round(rate_b, 4),
        test_used=test_used, statistic=round(statistic, 4),
        p_value=round(float(p_value), 6), is_significant=is_sig,
        confidence_level=confidence_level,
        relative_uplift=round(relative_uplift, 2),
        absolute_diff=round(diff, 4),
        ci_lower=round(ci_lower, 4), ci_upper=round(ci_upper, 4),
        power=round(power, 3) if power else None,
        min_detectable_effect=min_n,
        sample_size_adequate=sample_adequate,
        verdict=verdict, recommendation=recommendation,
        warnings=warnings_list,
    )


# ══════════════════════════════════════════════════════════
#  CONTINUOUS METRIC TEST  (Welch's t-test / Mann-Whitney U)
# ══════════════════════════════════════════════════════════

def run_continuous_test(
    values_a: pd.Series, values_b: pd.Series,
    variant_a_name: str = "Control (A)",
    variant_b_name: str = "Variant (B)",
    metric_name: str = "Metric",
    confidence_level: float = 0.95,
) -> ABTestResult:
    """
    Two-sample test for a continuous metric (revenue, session time, etc.).
    Uses Welch's t-test by default (robust to unequal variance). Falls back
    to Mann-Whitney U (non-parametric) when either group is heavily
    non-normal (Shapiro p < 0.01) or has fewer than 20 observations.

    Raises:
        ValueError — if either sample has fewer than 2 valid observations.
    """
    a = pd.to_numeric(values_a, errors="coerce").dropna()
    b = pd.to_numeric(values_b, errors="coerce").dropna()

    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            f"Need at least 2 valid observations per variant. "
            f"Got A={len(a)}, B={len(b)}."
        )

    warnings_list = []
    alpha = 1 - confidence_level

    mean_a, mean_b = float(a.mean()), float(b.mean())
    std_a, std_b   = float(a.std()), float(b.std())
    med_a, med_b   = float(a.median()), float(b.median())

    # Normality check to decide test
    use_nonparametric = len(a) < 20 or len(b) < 20
    if not use_nonparametric:
        try:
            _, p_norm_a = scipy_stats.shapiro(a.sample(min(len(a), 5000), random_state=42))
            _, p_norm_b = scipy_stats.shapiro(b.sample(min(len(b), 5000), random_state=42))
            if p_norm_a < 0.01 or p_norm_b < 0.01:
                use_nonparametric = True
        except Exception:
            logger.debug("Normality check failed — defaulting to t-test", exc_info=True)

    if use_nonparametric:
        stat, p_value = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        test_used = "Mann-Whitney U Test (non-parametric)"
        if len(a) < 20 or len(b) < 20:
            warnings_list.append(
                f"Sample size is small (A={len(a)}, B={len(b)}) — using the "
                "non-parametric Mann-Whitney U test instead of t-test."
            )
        else:
            warnings_list.append(
                "Data is not normally distributed — using the non-parametric "
                "Mann-Whitney U test (compares distributions/medians, not means)."
            )
    else:
        stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=False)  # Welch's
        test_used = "Welch's T-Test (unequal variance)"

    diff = mean_b - mean_a
    # Relative % change is only well-defined and sign-correct when the
    # baseline (A) is positive. A non-positive baseline (zero or negative)
    # either divides by zero or arbitrarily flips the percentage's sign
    # regardless of the real direction of change, producing verdicts like
    # "'B' has higher metric than 'A' (-182.4% relative difference)" that
    # directly contradict the stated direction. Use well-defined absolute-
    # difference framing in that case instead.
    uplift_is_meaningful = mean_a > 0
    relative_uplift = (diff / mean_a * 100) if uplift_is_meaningful else 0.0

    # CI on the mean difference (Welch-Satterthwaite)
    se_diff = np.sqrt(std_a**2/len(a) + std_b**2/len(b))
    if std_a == 0 and std_b == 0:
        # Both variants are constant (zero variance) — no meaningful df to
        # compute via Welch-Satterthwaite; fall back to normal approximation.
        df_welch = len(a) + len(b) - 2
        warnings_list.append(
            "Both variants have zero variance (all values identical) — "
            "confidence interval uses a normal approximation."
        )
    elif len(a) > 1 and len(b) > 1:
        denom = (std_a**2/len(a))**2/(len(a)-1) + (std_b**2/len(b))**2/(len(b)-1)
        df_welch = (std_a**2/len(a) + std_b**2/len(b))**2 / denom if denom > 0 else len(a) + len(b) - 2
    else:
        df_welch = len(a) + len(b) - 2
    t_crit = scipy_stats.t.ppf(1 - alpha/2, df_welch) if df_welch > 0 else 1.96
    ci_lower = diff - t_crit * se_diff
    ci_upper = diff + t_crit * se_diff

    is_sig = p_value < alpha
    power = _post_hoc_power_means(len(a), len(b), std_a, std_b, diff, alpha)

    diff_phrase = (
        f"({relative_uplift:+.1f}% relative difference)" if uplift_is_meaningful else
        f"(absolute difference of {diff:+.2f} — relative % isn't meaningful "
        f"when the baseline is zero or negative)"
    )

    if is_sig:
        direction = "higher" if mean_b > mean_a else "lower"
        verdict = (
            f"Statistically significant at {confidence_level:.0%} confidence "
            f"(p={p_value:.4f}). '{variant_b_name}' has {direction} "
            f"{metric_name.lower()} than '{variant_a_name}' "
            f"{diff_phrase}."
        )
        recommendation = (
            f"Ship '{variant_b_name}'." if mean_b > mean_a else
            f"Keep '{variant_a_name}' — the tested variant performed worse."
        )
    else:
        verdict = (
            f"NOT statistically significant (p={p_value:.4f}, threshold={alpha:.2f}). "
            f"Observed difference {diff_phrase} could be due to chance."
        )
        recommendation = (
            "Do not ship based on this result. Collect more data or consider "
            "the variants statistically equivalent on this metric."
        )

    if power is not None and power < 0.8:
        warnings_list.append(
            f"Statistical power is {power:.0%} (target: 80%+). This test may "
            "be underpowered to reliably detect the observed effect — "
            "consider collecting more data before making a final decision."
        )

    return ABTestResult(
        test_type="continuous", metric_name=metric_name,
        variant_a_name=variant_a_name, variant_b_name=variant_b_name,
        n_a=len(a), n_b=len(b),
        mean_a=round(mean_a, 4), mean_b=round(mean_b, 4),
        std_a=round(std_a, 4), std_b=round(std_b, 4),
        median_a=round(med_a, 4), median_b=round(med_b, 4),
        test_used=test_used, statistic=round(float(stat), 4),
        p_value=round(float(p_value), 6), is_significant=is_sig,
        confidence_level=confidence_level,
        relative_uplift=round(relative_uplift, 2),
        absolute_diff=round(diff, 4),
        ci_lower=round(ci_lower, 4), ci_upper=round(ci_upper, 4),
        power=round(power, 3) if power else None,
        sample_size_adequate=(power or 0) >= 0.8,
        verdict=verdict, recommendation=recommendation,
        warnings=warnings_list,
    )
