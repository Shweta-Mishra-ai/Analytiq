"""
engines/eda/results.py — what an EDA run returns.

The dataclasses only. They sit in their own module because every other
module in this package produces one and the runner assembles them, so
putting them anywhere else would make two of those import a third for
its types alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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