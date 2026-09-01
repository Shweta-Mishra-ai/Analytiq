"""
eda_engine.py — Senior analyst level EDA.
Proper statistical tests, not just describe().
"""
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from scipy import stats as scipy_stats
from scipy.stats import (
    shapiro, normaltest, kstest, anderson,
    ttest_ind, mannwhitneyu, f_oneway, kruskal,
    chi2_contingency, pointbiserialr, spearmanr, pearsonr,
    levene, bartlett
)

from app.services.dtypes import MONTH_END, text_columns

logger = logging.getLogger(__name__)

from app.engines.domains.base import is_id_column
from app.engines.present import label as _L
from app.engines.statistics import (assess_normality, clamp_p,
                                    correlation_with_ci, effect_label,
                                    format_p, is_worth_reporting)


# ══════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class UnivariateResult:
    column:          str
    dtype:           str
    n:               int
    missing:         int
    missing_pct:     float
    # Descriptive
    mean:            Optional[float] = None
    median:          Optional[float] = None
    mode:            Optional[float] = None
    std:             Optional[float] = None
    variance:        Optional[float] = None
    cv:              Optional[float] = None   # coefficient of variation
    min_val:         Optional[float] = None
    max_val:         Optional[float] = None
    range_val:       Optional[float] = None
    q1:              Optional[float] = None
    q3:              Optional[float] = None
    iqr:             Optional[float] = None
    p5:              Optional[float] = None   # 5th percentile
    p95:             Optional[float] = None   # 95th percentile
    # Distribution shape
    skewness:        Optional[float] = None
    skew_label:      Optional[str]   = None
    kurtosis:        Optional[float] = None
    kurtosis_label:  Optional[str]   = None
    # Normality tests
    shapiro_stat:    Optional[float] = None
    shapiro_p:       Optional[float] = None
    dagostino_stat:  Optional[float] = None
    dagostino_p:     Optional[float] = None
    anderson_stat:   Optional[float] = None
    anderson_critical: Optional[float] = None
    is_normal:       Optional[bool]  = None
    # Whether the verdict came from the column's shape or from a
    # test — only the first means anything at a useful sample size.
    normality_basis: Optional[str]   = None
    normality_verdict: Optional[str] = None
    # Outliers — multiple methods
    outliers_iqr:    int = 0
    outliers_zscore: int = 0
    outliers_modz:   int = 0   # modified z-score
    outlier_pct:     float = 0.0
    recommended_method: str = "IQR"
    iqr_lower:       Optional[float] = None
    iqr_upper:       Optional[float] = None
    # Distribution fit
    best_fit_dist:   Optional[str]   = None
    best_fit_params: Optional[Dict]  = None
    # Categorical
    unique_count:    int = 0
    top_value:       Optional[str]   = None
    top_pct:         Optional[float] = None
    entropy:         Optional[float] = None   # information entropy
    interpretation:  str = ""


@dataclass
class BivariateResult:
    col_a:           str
    col_b:           str
    test_name:       str
    statistic:       float
    p_value:         float
    is_significant:  bool
    effect_size:     Optional[float] = None
    effect_label:    Optional[str]   = None   # small/medium/large
    interpretation:  str = ""
    recommendation:  str = ""


@dataclass
class GroupComparisonResult:
    numeric_col:     str
    group_col:       str
    n_groups:        int
    test_used:       str   # ANOVA or Kruskal-Wallis
    statistic:       float
    p_value:         float
    is_significant:  bool
    effect_size:     Optional[float] = None
    effect_label:    Optional[str]   = None
    group_stats:     Dict = field(default_factory=dict)
    interpretation:  str = ""
    post_hoc:        List[str] = field(default_factory=list)


@dataclass
class MulticollinearityResult:
    feature:         str
    vif:             float
    verdict:         str   # "OK", "Moderate", "High", "Severe"
    interpretation:  str


@dataclass
class TimeSeriesResult:
    column:          str
    date_col:        str
    adf_stat:        Optional[float] = None
    adf_p:           Optional[float] = None
    is_stationary:   Optional[bool]  = None
    trend:           Optional[str]   = None   # "upward", "downward", "flat"
    trend_slope:     Optional[float] = None
    seasonality:     Optional[str]   = None
    interpretation:  str = ""


@dataclass
class EDAReport:
    n_rows:          int
    n_cols:          int
    numeric_cols:    List[str]
    categorical_cols: List[str]
    datetime_cols:   List[str]
    univariate:      Dict[str, UnivariateResult] = field(default_factory=dict)
    correlations:    List[BivariateResult]       = field(default_factory=list)
    group_comparisons: List[GroupComparisonResult] = field(default_factory=list)
    multicollinearity: List[MulticollinearityResult] = field(default_factory=list)
    # Named rather than dropped in silence.
    identifier_cols: List[str] = field(default_factory=list)
    time_series:     List[TimeSeriesResult]      = field(default_factory=list)
    key_findings:    List[str]                   = field(default_factory=list)
    warnings:        List[str]                   = field(default_factory=list)


# ══════════════════════════════════════════════════════════
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


def _fit_distribution(s: pd.Series) -> Tuple[str, Dict]:
    """Try fitting common distributions, return best fit."""
    distributions = ["norm", "lognorm", "expon", "gamma", "uniform"]
    best_dist, best_p, best_params = "norm", 0, {}
    for dist_name in distributions:
        try:
            dist   = getattr(scipy_stats, dist_name)
            params = dist.fit(s)
            _, p   = kstest(s, dist_name, args=params)
            if p > best_p:
                best_p      = p
                best_dist   = dist_name
                best_params = {"params": params, "ks_p": round(p, 4)}
        except Exception:
            logger.debug("_fit_distribution: suppressed exception", exc_info=True)
            continue
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

    try:
        ad_result = anderson(sample, dist="norm")
        result.anderson_stat     = round(float(ad_result.statistic), 6)
        result.anderson_critical = round(float(ad_result.critical_values[2]), 6)
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
#  BIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for two groups."""
    pooled_std = np.sqrt(
        ((len(a) - 1) * a.std()**2 + (len(b) - 1) * b.std()**2)
        / (len(a) + len(b) - 2)
    )
    return float(abs(a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def _effect_label(d: float) -> str:
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def analyze_bivariate_numeric(
    df: pd.DataFrame, col_a: str, col_b: str,
    is_normal_a: bool, is_normal_b: bool,
) -> BivariateResult:
    """
    Numeric vs Numeric bivariate test.
    Picks Pearson (normal) or Spearman (non-normal).
    """
    common = df[[col_a, col_b]].dropna()
    a, b   = common[col_a].values, common[col_b].values

    if len(common) < 10:
        return BivariateResult(
            col_a=col_a, col_b=col_b,
            test_name="N/A", statistic=0, p_value=1,
            is_significant=False,
            interpretation="Insufficient data.")

    if is_normal_a and is_normal_b:
        r, p       = pearsonr(a, b)
        test_name  = "Pearson Correlation"
        effect_s   = abs(r)
        effect_lbl = ("Negligible" if effect_s < 0.1 else
                      "Small" if effect_s < 0.3 else
                      "Medium" if effect_s < 0.5 else "Large")
    else:
        r, p       = spearmanr(a, b)
        test_name  = "Spearman Rank Correlation"
        effect_s   = abs(r)
        effect_lbl = ("Negligible" if effect_s < 0.1 else
                      "Small" if effect_s < 0.3 else
                      "Medium" if effect_s < 0.5 else "Large")

    sig  = p < 0.05
    dirn = "positive" if r > 0 else "negative"

    interp = (
        "{} {} correlation between '{}' and '{}' "
        "(r={:.3f}, p={:.4f}, effect={}).".format(
            effect_lbl, dirn, col_a, col_b,
            round(r, 3), round(p, 4), effect_lbl)
    )
    if sig and effect_lbl in ("Medium", "Large"):
        rec = "Significant relationship — consider including both in models or investigating causation."
    elif sig:
        rec = "Statistically significant but small effect — may not be practically important."
    else:
        rec = "No significant relationship detected at p=0.05."

    return BivariateResult(
        col_a=col_a, col_b=col_b,
        test_name=test_name,
        statistic=round(float(r), 4),
        p_value=round(float(p), 6),
        is_significant=sig,
        effect_size=round(float(effect_s), 4),
        effect_label=effect_lbl,
        interpretation=interp,
        recommendation=rec,
    )


def analyze_group_comparison(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    is_normal: bool,
) -> GroupComparisonResult:
    """
    Compare numeric column across groups.
    Uses ANOVA (normal) or Kruskal-Wallis (non-normal).
    With eta-squared effect size.
    """
    groups     = df.groupby(group_col)[numeric_col].apply(
        lambda x: x.dropna().values
    )
    groups     = {k: v for k, v in groups.items() if len(v) >= 3}
    n_groups   = len(groups)

    if n_groups < 2:
        return GroupComparisonResult(
            numeric_col=numeric_col, group_col=group_col,
            n_groups=n_groups, test_used="N/A",
            statistic=0, p_value=1, is_significant=False,
            interpretation="Need at least 2 groups with 3+ samples.")

    group_arrays = list(groups.values())

    # Choose test
    if is_normal and n_groups >= 2:
        # Check variance homogeneity first
        try:
            _, lev_p = levene(*group_arrays)
            equal_var = lev_p > 0.05
        except Exception:
            equal_var = True

        try:
            stat, p   = f_oneway(*group_arrays)
            test_name = "One-Way ANOVA"
        except Exception:
            stat, p   = kruskal(*group_arrays)
            test_name = "Kruskal-Wallis"
    else:
        stat, p   = kruskal(*group_arrays)
        test_name = "Kruskal-Wallis"

    # Eta-squared effect size
    all_values = np.concatenate(group_arrays)
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_arrays)
    ss_total   = sum((all_values - grand_mean)**2)
    eta_sq     = ss_between / ss_total if ss_total > 0 else 0
    eta_lbl    = ("Negligible" if eta_sq < 0.01 else
                  "Small" if eta_sq < 0.06 else
                  "Medium" if eta_sq < 0.14 else "Large")

    # Group stats
    group_stats = {}
    for grp, vals in groups.items():
        group_stats[str(grp)] = {
            "n":      len(vals),
            "mean":   round(float(vals.mean()), 4),
            "median": round(float(np.median(vals)), 4),
            "std":    round(float(vals.std()), 4),
        }

    sig = p < 0.05
    interp = (
        "{} test: {} difference in '{}' across {} groups of '{}' "
        "(F/H={:.3f}, p={:.4f}, eta²={:.3f} — {} effect).".format(
            test_name,
            "Significant" if sig else "No significant",
            numeric_col, n_groups, group_col,
            round(float(stat), 3), round(float(p), 4),
            round(eta_sq, 4), eta_lbl)
    )

    # Post-hoc hint
    post_hoc = []
    if sig and n_groups > 2:
        post_hoc.append(
            "Significant difference detected — run post-hoc pairwise comparisons "
            "(Tukey HSD or Dunn test) to identify which groups differ."
        )
    elif sig and n_groups == 2:
        grp_names = list(group_stats.keys())
        m1 = group_stats[grp_names[0]]["mean"]
        m2 = group_stats[grp_names[1]]["mean"]
        higher = grp_names[0] if m1 > m2 else grp_names[1]
        lower  = grp_names[1] if m1 > m2 else grp_names[0]
        post_hoc.append(
            "'{}' has significantly higher '{}' than '{}' "
            "({:.2f} vs {:.2f}).".format(
                higher, numeric_col, lower,
                max(m1, m2), min(m1, m2))
        )

    return GroupComparisonResult(
        numeric_col=numeric_col, group_col=group_col,
        n_groups=n_groups, test_used=test_name,
        statistic=round(float(stat), 4),
        p_value=round(float(p), 6),
        is_significant=sig,
        effect_size=round(float(eta_sq), 4),
        effect_label=eta_lbl,
        group_stats=group_stats,
        interpretation=interp,
        post_hoc=post_hoc,
    )


# ══════════════════════════════════════════════════════════
#  MULTICOLLINEARITY — VIF
# ══════════════════════════════════════════════════════════

def analyze_vif(df: pd.DataFrame) -> List[MulticollinearityResult]:
    """
    Variance Inflation Factor for numeric columns.
    VIF > 10 = serious multicollinearity.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return []

    # Prep
    X = df[num_cols].copy()
    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=num_cols
    )

    results = []
    for i, col in enumerate(num_cols):
        try:
            y      = X[col].values
            X_rest = X.drop(columns=[col]).values
            r2     = LinearRegression().fit(X_rest, y).score(X_rest, y)
            vif    = 1 / (1 - r2) if r2 < 1 else float("inf")
            vif    = round(float(vif), 2)

            if vif < 5:
                verdict = "OK"
                interp  = "No multicollinearity issue."
            elif vif < 10:
                verdict = "Moderate"
                interp  = "Some correlation with other features — monitor."
            elif vif < 20:
                verdict = "High"
                interp  = "High multicollinearity — consider removing or combining."
            else:
                verdict = "Severe"
                interp  = "Severe multicollinearity — remove from model."

            results.append(MulticollinearityResult(
                feature=col, vif=vif,
                verdict=verdict, interpretation=interp,
            ))
        except Exception:
            logger.debug("analyze_vif: suppressed exception", exc_info=True)
            continue

    return sorted(results, key=lambda x: x.vif, reverse=True)


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
#  KEY FINDINGS GENERATOR
# ══════════════════════════════════════════════════════════

def _generate_key_findings(report: "EDAReport") -> List[str]:
    findings = []

    # Non-normal columns
    non_normal = [
        col for col, r in report.univariate.items()
        if r.is_normal is False and r.mean is not None
    ]
    if non_normal:
        findings.append(
            "{} column(s) are non-normally distributed: {}. "
            "Use non-parametric tests (Mann-Whitney, Kruskal-Wallis).".format(
                len(non_normal), ", ".join(non_normal[:4]))
        )

    # High outlier columns
    outlier_cols = [
        (col, r.outlier_pct) for col, r in report.univariate.items()
        if r.outlier_pct and r.outlier_pct > 5
    ]
    if outlier_cols:
        worst = max(outlier_cols, key=lambda x: x[1])
        findings.append(
            "'{}' has {:.1f}% outliers — highest in dataset. "
            "Validate these values before modeling.".format(*worst)
        )

    # Skewed columns
    skewed = [
        col for col, r in report.univariate.items()
        if r.skewness and abs(r.skewness) > 2
    ]
    if skewed:
        findings.append(
            "{} column(s) heavily skewed (|skew|>2): {}. "
            "Log-transform recommended before regression.".format(
                len(skewed), ", ".join(skewed[:3]))
        )

    # Strong correlations
    strong_corr = [
        r for r in report.correlations
        if r.is_significant and r.effect_size and r.effect_size >= 0.5
    ]
    if strong_corr:
        top = strong_corr[0]
        # Whether a correlation actually causes a multicollinearity
        # problem is what VIF answers, and VIF has already run. The
        # report used to hedge "may indicate multicollinearity" directly
        # above a table stating there was none.
        flagged = [m for m in report.multicollinearity
                   if m.verdict not in ("OK", "")]
        if flagged:
            verdict = ("This does inflate the variance of {}: VIF {:.1f}. "
                       "Drop one of the pair, or combine them, before "
                       "fitting a model that assumes independent "
                       "predictors.".format(_L(flagged[0].feature),
                                            flagged[0].vif))
        else:
            verdict = ("It does not amount to a multicollinearity problem: "
                       "every variance inflation factor is below the "
                       "conventional threshold, the highest being "
                       "{:.1f}.".format(max((m.vif for m in
                                             report.multicollinearity),
                                            default=0.0)))
        findings.append(
            "{} and {} move together ({}, r={:.2f}, n={:,}). {}".format(
                _L(top.col_a), _L(top.col_b),
                format_p(top.p_value), top.statistic,
                report.n_rows, verdict))

    # Group differences. Significance alone is not enough: on a large
    # sample almost every comparison is significant, and a difference too
    # small to act on is not a finding however certain it is.
    try:
        from app.engines.rigour import assess_finding
        sig_groups = []
        for r in report.group_comparisons:
            if not r.is_significant:
                continue
            n = sum(g.get("n", 0) for g in (r.group_stats or {}).values()) \
                if isinstance(r.group_stats, dict) else None
            if assess_finding(p_value=r.p_value, effect_size=r.effect_size,
                              n=n or None).reportable:
                sig_groups.append(r)
    except Exception:
        logger.warning("finding gate unavailable", exc_info=True)
        sig_groups = [r for r in report.group_comparisons if r.is_significant]
    if sig_groups:
        top = sig_groups[0]
        findings.append(
            "'{}' differs by '{}' ({}, p={:.4f}, effect {}). The gap is "
            "both statistically reliable and large enough to act on.".format(
                top.numeric_col, top.group_col,
                top.test_used, top.p_value, top.effect_label)
        )

    # Interactions lead, because an effect that reverses across a second
    # factor is the finding a main-effects summary reports as "no effect".
    for inter in list(getattr(report, "interactions", None) or [])[:2]:
        findings.insert(0, inter.description)

    # Class imbalance changes how every subsequent number reads.
    for note in list(getattr(report, "imbalance_notes", None) or [])[:1]:
        findings.append(note.note)

    # Groups too thin to carry a comparison.
    rare = list(getattr(report, "rare_categories", None) or [])
    if rare:
        findings.append(
            "{} category level(s) hold too few rows to support a finding "
            "on their own — smallest is '{}' in '{}' at {} row(s). These "
            "are excluded from group comparisons rather than ranked "
            "alongside groups many times their size.".format(
                len(rare), rare[0].level, rare[0].column, rare[0].n))

    # VIF issues
    severe_vif = [r for r in report.multicollinearity if r.vif >= 10]
    if severe_vif:
        findings.append(
            "{} feature(s) have high VIF (multicollinearity): {}. "
            "Remove or combine before regression modeling.".format(
                len(severe_vif),
                ", ".join([r.feature for r in severe_vif[:3]]))
        )

    return findings


# ══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, max_rows: int = 50_000) -> EDAReport:
    """
    Full EDA pipeline.
    Returns EDAReport with all analyses.
    """
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    # Identifiers are not measures. EmployeeNumber was being profiled,
    # correlated, VIF-tested and compared across departments — a full
    # statistical treatment of a row number, printed beside the findings
    # about salary.
    all_numeric = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in all_numeric if not is_id_column(c, df[c])]
    id_cols  = [c for c in all_numeric if c not in num_cols]
    cat_cols  = text_columns(df)
    dt_cols   = df.select_dtypes(include="datetime").columns.tolist()

    report = EDAReport(
        n_rows=len(df), n_cols=len(df.columns),
        numeric_cols=num_cols, categorical_cols=cat_cols,
        datetime_cols=dt_cols,
    )
    report.identifier_cols = id_cols

    # 1. Univariate — every column that is a measure
    analysable = [c for c in df.columns if c not in id_cols]
    for col in analysable[:30]:
        try:
            report.univariate[col] = analyze_univariate(df[col])
        except Exception:
            logger.debug("run_eda: suppressed exception", exc_info=True)
            continue

    # 2. Correlations — numeric pairs
    normality = {
        col: report.univariate[col].is_normal or False
        for col in num_cols if col in report.univariate
    }
    for i in range(len(num_cols)):
        for j in range(i+1, min(len(num_cols), i+8)):
            col_a, col_b = num_cols[i], num_cols[j]
            try:
                res = analyze_bivariate_numeric(
                    df, col_a, col_b,
                    normality.get(col_a, False),
                    normality.get(col_b, False),
                )
                report.correlations.append(res)
            except Exception:
                logger.debug("run_eda: suppressed exception", exc_info=True)
                continue

    # Benjamini-Hochberg across the whole family of pairwise tests:
    # is_significant is re-decided on adjusted p, so testing many pairs
    # doesn't manufacture "significant" findings.
    if report.correlations:
        from app.services.stat_guards import FDR_Q, bh_adjust
        qvals = bh_adjust([c.p_value for c in report.correlations])
        for c, q in zip(report.correlations, qvals):
            c.is_significant = bool(q < FDR_Q)
            c.p_value = float(q)   # store adjusted p (q-value)

    report.correlations.sort(
        key=lambda x: abs(x.statistic), reverse=True
    )

    # 3. Group comparisons — top numeric vs top categorical
    useful_cats = [c for c in cat_cols if 2 <= df[c].nunique() <= 15]
    for cat in useful_cats[:3]:
        for num in num_cols[:3]:
            try:
                res = analyze_group_comparison(
                    df, num, cat, normality.get(num, False)
                )
                report.group_comparisons.append(res)
            except Exception:
                logger.debug("run_eda: suppressed exception", exc_info=True)
                continue

    # 4. Multicollinearity
    if len(num_cols) >= 2:
        try:
            report.multicollinearity = analyze_vif(df)
        except Exception:
            logger.debug("run_eda: suppressed exception", exc_info=True)

    # 5. Time series
    if dt_cols and num_cols:
        for dt_col in dt_cols[:1]:
            for num_col in num_cols[:2]:
                try:
                    res = analyze_time_series(df, dt_col, num_col)
                    report.time_series.append(res)
                except Exception:
                    logger.debug("run_eda: suppressed exception", exc_info=True)
                    continue

    # 6. Depth layer
    try:
        from app.engines.eda_depth import (
            describe_imbalance, find_interactions, find_rare_categories,
            key_estimates,
        )
        report.estimates = key_estimates(df)
        report.interactions = find_interactions(df)
        report.rare_categories = find_rare_categories(df)
        report.imbalance_notes = describe_imbalance(df)
    except Exception:
        logger.warning("EDA depth layer failed", exc_info=True)

    # 7. Key findings
    report.key_findings = _generate_key_findings(report)

    return report

