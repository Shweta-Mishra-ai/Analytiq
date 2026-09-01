import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

from app.engines.domains.base import is_id_column
from app.engines.present import label as _L
from app.engines.statistics import (assess_normality, clamp_p,
                                    correlation_with_ci, format_p,
                                    correlation_strength)


from app.services.dtypes import text_columns


@dataclass
class ColumnStats:
    name: str
    dtype: str
    # Descriptive
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    range_val: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    # Distribution shape
    skewness: Optional[float] = None
    skew_label: Optional[str] = None   # "right-skewed", "left-skewed", "symmetric"
    kurtosis: Optional[float] = None
    kurtosis_label: Optional[str] = None  # "leptokurtic", "platykurtic", "mesokurtic"
    # Normality
    is_normal: Optional[bool] = None
    normality_test: Optional[str] = None   # "Shapiro-Wilk" or "D'Agostino"
    normality_pvalue: Optional[float] = None
    normality_label: Optional[str] = None
    # Whether the verdict came from the column's shape or from the test —
    # at any useful sample size only the first of those means anything.
    normality_basis: Optional[str] = None
    normality_note: Optional[str] = None
    # Outliers
    outlier_count_iqr: int = 0
    outlier_count_zscore: int = 0
    outlier_pct: float = 0.0
    outlier_method_recommended: str = "IQR"
    # Missing
    missing_count: int = 0
    missing_pct: float = 0.0
    # Categorical specific
    unique_count: int = 0
    top_value: Optional[str] = None
    top_value_pct: Optional[float] = None
    cardinality_label: Optional[str] = None  # "low", "medium", "high"


@dataclass
class CorrelationInsight:
    col_a: str
    col_b: str
    pearson_r: float
    spearman_r: float
    p_value: float
    is_significant: bool
    strength: str      # "strong", "moderate", "weak"
    direction: str     # "positive", "negative"
    label: str         # human-readable
    # The interval and the sample it rests on, so a correlation is never
    # read as firmer than its evidence.
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    n: Optional[int] = None
    method: str = "pearson"


@dataclass
class DatasetStats:
    rows: int
    cols: int
    numeric_cols: List[str]
    categorical_cols: List[str]
    datetime_cols: List[str]
    column_stats: Dict[str, ColumnStats] = field(default_factory=dict)
    correlations: List[CorrelationInsight] = field(default_factory=list)
    top_correlations: List[CorrelationInsight] = field(default_factory=list)
    dataset_insights: List[str] = field(default_factory=list)
    # Overall flags
    has_skewed_cols: bool = False
    has_non_normal_cols: bool = False
    has_strong_correlations: bool = False
    recommended_analysis: List[str] = field(default_factory=list)
    # Named rather than dropped in silence, so a reader can see
    # which columns were held out of the analysis and why.
    identifier_cols: List[str] = field(default_factory=list)


def analyze(df: pd.DataFrame) -> DatasetStats:
    """
    Full statistical analysis of a DataFrame.
    Runs proper stats — not just describe().
    """
    # Identifiers are not measures. EmployeeNumber was being given a
    # mean of 735.5, a skewness, a normality verdict and a place in the
    # correlation matrix — four statements about a row number, printed
    # with the same authority as the ones about salary.
    all_numeric = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in all_numeric if not is_id_column(c, df[c])]
    id_cols  = [c for c in all_numeric if c not in num_cols]
    cat_cols  = text_columns(df)
    dt_cols   = df.select_dtypes(include="datetime").columns.tolist()

    ds = DatasetStats(
        rows=len(df), cols=len(df.columns),
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        datetime_cols=dt_cols,
    )
    ds.identifier_cols = id_cols
    if id_cols:
        logger.info("excluded %d identifier column(s) from analysis: %s",
                    len(id_cols), ", ".join(id_cols))

    # ── Per-column stats ───────────────────────────────────
    for col in num_cols:
        ds.column_stats[col] = _numeric_stats(df[col], col)

    for col in cat_cols:
        ds.column_stats[col] = _categorical_stats(df[col], col)

    # ── Correlations with significance ─────────────────────
    if len(num_cols) >= 2:
        ds.correlations = _correlation_analysis(df, num_cols)
        ds.top_correlations = [
            c for c in ds.correlations
            if c.is_significant and c.strength in ("strong", "moderate")
        ][:8]
        ds.has_strong_correlations = any(
            c.strength == "strong" for c in ds.correlations if c.is_significant
        )

    # ── Dataset-level flags ────────────────────────────────
    skewed = [c for c in num_cols
              if ds.column_stats[c].skewness is not None
              and abs(ds.column_stats[c].skewness) > 1]
    non_normal = [c for c in num_cols
                  if ds.column_stats[c].is_normal is False]

    ds.has_skewed_cols = len(skewed) > 0
    ds.has_non_normal_cols = len(non_normal) > 0

    # ── Plain-English dataset insights ────────────────────
    ds.dataset_insights = _generate_insights(df, ds, num_cols, cat_cols)

    # ── Recommended analysis types ─────────────────────────
    ds.recommended_analysis = _recommend_analysis(ds, num_cols, cat_cols, dt_cols)

    return ds


def _numeric_stats(s: pd.Series, name: str) -> ColumnStats:
    cs = ColumnStats(name=name, dtype=str(s.dtype))
    clean = s.dropna()
    n = len(clean)

    cs.missing_count = int(s.isna().sum())
    cs.missing_pct   = round(cs.missing_count / max(len(s), 1) * 100, 1)
    cs.unique_count  = int(s.nunique())

    if n < 3:
        return cs

    # ── Descriptive ───────────────────────────────────────
    cs.mean     = round(float(clean.mean()), 4)
    cs.median   = round(float(clean.median()), 4)
    cs.std      = round(float(clean.std()), 4)
    cs.variance = round(float(clean.var()), 4)
    cs.min_val  = round(float(clean.min()), 4)
    cs.max_val  = round(float(clean.max()), 4)
    cs.range_val = round(cs.max_val - cs.min_val, 4)
    cs.q1       = round(float(clean.quantile(0.25)), 4)
    cs.q3       = round(float(clean.quantile(0.75)), 4)
    cs.iqr      = round(cs.q3 - cs.q1, 4)

    # ── Skewness ──────────────────────────────────────────
    skew = float(clean.skew())
    cs.skewness = round(skew, 4)
    if skew > 1:
        cs.skew_label = "heavily right-skewed"
    elif skew > 0.5:
        cs.skew_label = "moderately right-skewed"
    elif skew < -1:
        cs.skew_label = "heavily left-skewed"
    elif skew < -0.5:
        cs.skew_label = "moderately left-skewed"
    else:
        cs.skew_label = "approximately symmetric"

    # ── Kurtosis ──────────────────────────────────────────
    kurt = float(clean.kurtosis())  # excess kurtosis
    cs.kurtosis = round(kurt, 4)
    if kurt > 1:
        cs.kurtosis_label = "leptokurtic (heavy tails)"
    elif kurt < -1:
        cs.kurtosis_label = "platykurtic (light tails)"
    else:
        cs.kurtosis_label = "mesokurtic (normal-like tails)"

    # ── Normality ─────────────────────────────────────────
    # Decided on shape, not on a p-value. Shapiro-Wilk at n=1,470
    # rejected every column in a perfectly ordinary HR file, including
    # one with a skew of 0.48, and the report then recommended
    # non-parametric tests throughout on that basis.
    verdict = assess_normality(clean)
    if verdict is None:
        cs.is_normal = None
        cs.normality_label = "Not assessed"
    else:
        cs.normality_test   = verdict.test_name
        cs.normality_pvalue = verdict.p_value
        cs.is_normal        = verdict.normal_enough
        cs.normality_basis  = verdict.basis
        cs.normality_note   = verdict.note
        cs.normality_label  = ("Normal enough for parametric tests"
                               if verdict.normal_enough
                               else "Too skewed or heavy-tailed for "
                                    "parametric tests")

    # ── Outliers — IQR (1.5x) ─────────────────────────────
    if cs.iqr and cs.iqr > 0:
        lo_iqr = cs.q1 - 1.5 * cs.iqr
        hi_iqr = cs.q3 + 1.5 * cs.iqr
        cs.outlier_count_iqr = int(((clean < lo_iqr) | (clean > hi_iqr)).sum())

    # ── Outliers — Z-score (|z| > 3) ──────────────────────
    if cs.std and cs.std > 0:
        z_scores = np.abs((clean - cs.mean) / cs.std)
        cs.outlier_count_zscore = int((z_scores > 3).sum())

    cs.outlier_pct = round(cs.outlier_count_iqr / max(n, 1) * 100, 1)

    # Recommend method based on normality
    cs.outlier_method_recommended = (
        "Z-Score (normal distribution)" if cs.is_normal
        else "IQR (non-normal distribution)"
    )

    return cs


def _categorical_stats(s: pd.Series, name: str) -> ColumnStats:
    cs = ColumnStats(name=name, dtype=str(s.dtype))
    n  = len(s)

    cs.missing_count = int(s.isna().sum())
    cs.missing_pct   = round(cs.missing_count / max(n, 1) * 100, 1)
    cs.unique_count  = int(s.nunique())

    vc = s.value_counts()
    if len(vc) > 0:
        cs.top_value     = str(vc.index[0])[:40]
        cs.top_value_pct = round(vc.iloc[0] / max(n, 1) * 100, 1)

    uniq_pct = cs.unique_count / max(n, 1)
    if uniq_pct > 0.8:
        cs.cardinality_label = "high (likely ID/free text)"
    elif cs.unique_count <= 10:
        cs.cardinality_label = "low (good for grouping)"
    else:
        cs.cardinality_label = "medium"

    return cs


def _correlation_analysis(
    df: pd.DataFrame, num_cols: List[str]
) -> List[CorrelationInsight]:
    insights = []
    cols = num_cols[:12]  # max 12 columns

    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a, b = cols[i], cols[j]
            s_a = df[a].dropna()
            s_b = df[b].dropna()
            common = s_a.index.intersection(s_b.index)

            if len(common) < 10:
                continue

            x = s_a[common].values
            y = s_b[common].values

            # Pearson or Spearman decided by the shape of both columns
            # rather than assumed. A rank correlation on two well-behaved
            # columns throws away information; a Pearson on a skewed pair
            # reports a relationship the data does not have.
            va, vb = assess_normality(x), assess_normality(y)
            parametric = bool(va and vb and va.normal_enough
                              and vb.normal_enough)
            method = "pearson" if parametric else "spearman"
            est = correlation_with_ci(x, y, method=method)
            if est is None:
                continue
            other = correlation_with_ci(
                x, y, method="spearman" if parametric else "pearson")

            direction = "positive" if est.r > 0 else "negative"
            # The interval, and the n it rests on. r=0.66 on 40 rows and
            # r=0.66 on 1,470 were printed identically.
            label = ("{} {} relationship between {} and {}: r={:.2f}, "
                     "95% CI {:.2f} to {:.2f}, n={:,} ({}, {})").format(
                est.strength.title(), direction, _L(a), _L(b), est.r,
                est.ci_low, est.ci_high, est.n,
                "Pearson" if parametric else "Spearman",
                format_p(est.p_value))

            insights.append(CorrelationInsight(
                col_a=a, col_b=b,
                pearson_r=(est.r if parametric
                           else (other.r if other else est.r)),
                spearman_r=((other.r if other else est.r) if parametric
                            else est.r),
                p_value=est.p_value,
                is_significant=est.significant,
                strength=est.strength,
                direction=direction,
                label=label,
                ci_low=est.ci_low, ci_high=est.ci_high, n=est.n,
                method=method,
            ))

    # re-decide significance on BH-adjusted p across all tested pairs
    if insights:
        from app.services.stat_guards import FDR_Q, bh_adjust
        qvals = bh_adjust([c.p_value for c in insights])
        for c, q in zip(insights, qvals):
            c.is_significant = bool(q < FDR_Q)
            c.p_value = round(float(q), 6)

    return sorted(insights, key=lambda x: abs(x.pearson_r), reverse=True)


def _generate_insights(
    df: pd.DataFrame, ds: DatasetStats,
    num_cols: List[str], cat_cols: List[str]
) -> List[str]:
    insights = []

    # Distribution insights
    for col in num_cols[:6]:
        cs = ds.column_stats.get(col)
        if not cs:
            continue
        if cs.skewness and abs(cs.skewness) > 1:
            insights.append(
                "'{}' is {} (skew={:.2f}) — median ({:.2f}) is a better central measure than mean ({:.2f}).".format(
                    col, cs.skew_label, cs.skewness, cs.median, cs.mean)
            )
        if cs.is_normal is False:
            # Says why, and says it in the terms the decision was made in.
            # The old line quoted a p-value of 0.0000 — which is not a
            # p-value — and attributed the verdict to a test that had no
            # part in it.
            insights.append(
                "{} is too skewed or heavy-tailed for parametric tests "
                "(skew {:.2f}, excess kurtosis {:.2f}), so rank-based "
                "methods are used for it: Mann-Whitney or Kruskal-Wallis "
                "rather than a t-test or ANOVA.".format(
                    _L(col), cs.skewness or 0.0, cs.kurtosis or 0.0))

    # Correlation insights
    for c in ds.top_correlations[:3]:
        if c.is_significant:
            insights.append(
                "{} — consider this relationship in modeling.".format(c.label)
            )

    # High cardinality warning
    high_card = [c for c in cat_cols
                 if ds.column_stats.get(c) and
                 ds.column_stats[c].cardinality_label and
                 "high" in ds.column_stats[c].cardinality_label]
    if high_card:
        insights.append(
            "{} column(s) have very high cardinality ({}) — likely ID fields, exclude from grouping.".format(
                len(high_card), ", ".join(high_card[:3]))
        )

    # Missing data
    cols_with_missing = [
        col for col in df.columns if df[col].isna().sum() > 0
    ]
    if cols_with_missing:
        insights.append(
            "{} column(s) have missing values — imputation method depends on distribution shape.".format(
                len(cols_with_missing))
        )

    return insights


def _recommend_analysis(
    ds: DatasetStats,
    num_cols, cat_cols, dt_cols
) -> List[str]:
    recs = []

    if len(num_cols) >= 2 and ds.has_strong_correlations:
        recs.append("Linear/Logistic Regression — strong correlations detected")

    if ds.has_non_normal_cols:
        recs.append("Mann-Whitney U / Kruskal-Wallis — non-normal distributions present")

    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        recs.append("ANOVA / Group comparison — categorical + numeric columns available")

    if dt_cols:
        recs.append("Time Series Analysis — datetime columns detected")

    if len(num_cols) >= 3:
        recs.append("PCA / Dimensionality Reduction — multiple numeric features")

    if not recs:
        recs.append("Exploratory Data Analysis (EDA) — start with distributions and correlations")

    return recs

