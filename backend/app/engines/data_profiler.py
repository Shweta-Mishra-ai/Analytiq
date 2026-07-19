"""
core/data_profiler.py
=====================
Production-grade data profiling engine.
Handles ALL dataset types: HR, Ecommerce, Finance, Healthcare, General.
Robust against: mixed types, inf values, large files, dirty data.
NO Streamlit imports.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class ColumnProfile:
    name:             str
    dtype:            str
    inferred_type:    str      # "numeric", "categorical", "datetime", "id", "constant", "text"

    # Completeness
    missing_count:    int
    missing_pct:      float

    # Uniqueness
    unique_count:     int
    unique_pct:       float
    is_id_like:       bool
    is_constant:      bool

    # Numeric stats (None for non-numeric)
    mean:             Optional[float] = None
    median:           Optional[float] = None
    std:              Optional[float] = None
    min_val:          Optional[float] = None
    max_val:          Optional[float] = None
    q25:              Optional[float] = None
    q75:              Optional[float] = None
    skewness:         Optional[float] = None
    has_outliers:     bool = False
    outlier_count:    int  = 0
    outlier_pct:      float = 0.0

    # Categorical stats
    top_value:        Optional[str]  = None
    top_value_pct:    Optional[float] = None
    value_counts:     Dict = field(default_factory=dict)

    # Samples
    sample_values:    List = field(default_factory=list)
    stats:            Dict = field(default_factory=dict)

    # Quality
    quality_score:    float = 100.0
    quality_issues:   List[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    rows:                  int
    cols:                  int
    total_cells:           int
    missing_cells:         int
    missing_pct:           float
    duplicate_rows:        int
    duplicate_pct:         float
    overall_quality_score: float
    column_profiles:       List[ColumnProfile]
    numeric_cols:          List[str]
    categorical_cols:      List[str]
    datetime_cols:         List[str]
    id_cols:               List[str]
    constant_cols:         List[str]
    high_missing_cols:     List[str]    # >30% missing
    high_outlier_cols:     List[str]    # >10% outliers
    recommendations:       List[str]
    # Summary for quick access
    n_issues:              int = 0
    data_quality_grade:    str = "A"    # A/B/C/D/F


# ══════════════════════════════════════════════════════════
#  SAFE NUMERIC EXTRACTION
# ══════════════════════════════════════════════════════════

def _safe_numeric_array(s: pd.Series) -> np.ndarray:
    """
    Extract clean float64 numpy array from any series.
    Removes: NaN, inf, -inf, non-numeric.
    Safe for ALL data types including mixed columns.
    """
    try:
        # Force numeric conversion
        arr = pd.to_numeric(s, errors="coerce").values.astype(float)
        # Remove NaN and infinite
        arr = arr[np.isfinite(arr)]
        return arr
    except Exception:
        return np.array([], dtype=float)


def _safe_percentile(arr: np.ndarray, q: float) -> Optional[float]:
    """Safe percentile — returns None if array too small."""
    try:
        if len(arr) < 4:
            return None
        return float(np.percentile(arr, q))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
#  COLUMN TYPE INFERENCE
# ══════════════════════════════════════════════════════════

def _infer_column_type(s: pd.Series) -> str:
    """
    Infer semantic column type beyond dtype.
    Returns: "numeric", "categorical", "datetime", "id", "constant", "text", "binary"
    """
    n      = len(s)
    unique = s.nunique(dropna=True)

    if unique <= 1:
        return "constant"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    if pd.api.types.is_bool_dtype(s):
        return "binary"

    if pd.api.types.is_numeric_dtype(s):
        if unique == 2:
            return "binary"
        # Numeric ID detection — strict rules
        if unique / max(n, 1) > 0.95 and n > 50:
            col_lower = str(s.name).lower() if s.name else ""
            # Explicit ID keywords
            id_keywords = ["_id", "id_", "userid", "orderid", "custid",
                           "empid", "itemid", "skuid", "uuid", "key"]
            is_id_name = any(kw in col_lower for kw in id_keywords)
            # Numeric values should NOT be flagged as ID unless name says so
            # Prices, ratings, amounts are NOT IDs even if all unique
            non_id_keywords = ["price", "amount", "cost", "rate", "count",
                                "score", "salary", "revenue", "sales", "qty",
                                "quantity", "hours", "age", "percent", "pct",
                                "evaluation", "satisfaction", "project"]
            is_value_col = any(kw in col_lower for kw in non_id_keywords)
            if is_id_name and not is_value_col:
                return "id"
        return "numeric"

    # Object/string columns
    if unique / max(n, 1) > 0.85 and n > 50:
        col_lower = str(s.name).lower() if s.name else ""
        # URL/link columns are always ID-like
        url_keywords = ["url", "link", "href", "img", "image", "photo",
                        "thumbnail", "src", "path", "uri"]
        if any(kw in col_lower for kw in url_keywords):
            return "id"
        sample  = s.dropna().astype(str).head(30)
        avg_len = float(sample.str.len().mean())
        # Short strings with high uniqueness → ID codes
        if avg_len < 20:
            return "id"
        # Long strings → free text
        return "text"

    if unique <= 50:
        return "categorical"

    return "text"


# ══════════════════════════════════════════════════════════
#  COLUMN PROFILER
# ══════════════════════════════════════════════════════════

def _profile_column(df: pd.DataFrame, col: str) -> ColumnProfile:
    """
    Full profile of one column.
    Robust: never crashes regardless of data type or quality.
    """
    s = df[col]
    n = len(s)

    missing     = int(s.isnull().sum())
    missing_pct = round(missing / max(n, 1) * 100, 1)
    unique      = int(s.nunique(dropna=True))
    unique_pct  = round(unique / max(n, 1) * 100, 1)
    inferred    = _infer_column_type(s)

    # ── Default values ─────────────────────────────────────
    mean = median = std = min_val = max_val = None
    q25 = q75 = skewness = None
    has_outliers  = False
    outlier_count = 0
    outlier_pct   = 0.0
    top_value     = None
    top_value_pct = None
    value_counts  = {}
    stats         = {}
    quality_issues = []

    # ── Numeric analysis ───────────────────────────────────
    if inferred in ("numeric", "binary") and pd.api.types.is_numeric_dtype(s):
        arr = _safe_numeric_array(s)

        if len(arr) >= 4:
            mean    = round(float(np.mean(arr)), 6)
            median  = round(float(np.median(arr)), 6)
            std     = round(float(np.std(arr, ddof=1)), 6) if len(arr) > 1 else 0.0
            min_val = round(float(np.min(arr)), 6)
            max_val = round(float(np.max(arr)), 6)

            q25_v = _safe_percentile(arr, 25)
            q75_v = _safe_percentile(arr, 75)

            if q25_v is not None and q75_v is not None:
                q25 = round(q25_v, 6)
                q75 = round(q75_v, 6)
                iqr = q75_v - q25_v

                # Outlier detection — IQR method
                if iqr > 0:
                    lo = q25_v - 1.5 * iqr
                    hi = q75_v + 1.5 * iqr
                    outlier_mask  = (arr < lo) | (arr > hi)
                    outlier_count = int(outlier_mask.sum())
                    has_outliers  = outlier_count > 0
                    outlier_pct   = round(outlier_count / max(n, 1) * 100, 1)

            # Skewness
            try:
                if std and std > 0:
                    skewness = round(float(pd.Series(arr).skew()), 4)
            except Exception:
                pass

            stats = {
                "mean":   mean,   "median": median,
                "std":    std,    "min":    min_val,
                "max":    max_val,"q25":    q25,
                "q75":    q75,    "skew":   skewness,
            }

        # Quality issues for numeric
        if missing_pct > 0:
            quality_issues.append("{:.1f}% missing".format(missing_pct))
        if outlier_pct > 10:
            quality_issues.append("{:.1f}% outliers".format(outlier_pct))
        if skewness and abs(skewness) > 2:
            quality_issues.append("heavily skewed ({:.1f})".format(skewness))

    # ── Categorical analysis ───────────────────────────────
    elif inferred in ("categorical", "id", "text", "constant"):
        try:
            vc = s.value_counts(dropna=True)
            if len(vc) > 0:
                top_value     = str(vc.index[0])[:50]
                top_value_pct = round(vc.iloc[0] / max(n - missing, 1) * 100, 1)
                # Store value counts for low-cardinality columns
                if unique <= 20:
                    value_counts = {
                        str(k): int(v)
                        for k, v in vc.head(20).items()
                    }
        except Exception:
            pass

        if missing_pct > 0:
            quality_issues.append("{:.1f}% missing".format(missing_pct))
        if inferred == "constant":
            quality_issues.append("constant value — no variance")
        if inferred == "id":
            quality_issues.append("ID column — exclude from analysis")

    # ── Datetime analysis ──────────────────────────────────
    elif inferred == "datetime":
        try:
            clean_dt = s.dropna()
            if len(clean_dt) > 0:
                min_val = None  # datetimes handled differently
                stats["min_date"] = str(clean_dt.min())
                stats["max_date"] = str(clean_dt.max())
                stats["date_range_days"] = (clean_dt.max() - clean_dt.min()).days
        except Exception:
            pass

    # ── Quality score ──────────────────────────────────────
    score = 100.0
    score -= (missing_pct / 100) * 50          # missing: up to -50
    if has_outliers:
        score -= min(outlier_pct * 0.5, 15)    # outliers: up to -15
    if unique <= 1 and n > 1:
        score -= 20                             # constant: -20
    if inferred == "id":
        score = min(score, 80)                  # ID cols capped at 80
    score = max(0.0, round(score, 1))

    # Safe sample values
    try:
        sample_raw = s.dropna().unique()[:5]
        sample_values = [str(v)[:50] for v in sample_raw.tolist()]
    except Exception:
        sample_values = []

    return ColumnProfile(
        name=col,
        dtype=str(s.dtype),
        inferred_type=inferred,
        missing_count=missing,
        missing_pct=missing_pct,
        unique_count=unique,
        unique_pct=unique_pct,
        is_id_like=(inferred == "id"),
        is_constant=(inferred == "constant"),
        mean=mean, median=median, std=std,
        min_val=min_val, max_val=max_val,
        q25=q25, q75=q75, skewness=skewness,
        has_outliers=has_outliers,
        outlier_count=outlier_count,
        outlier_pct=outlier_pct,
        top_value=top_value,
        top_value_pct=top_value_pct,
        value_counts=value_counts,
        sample_values=sample_values,
        stats=stats,
        quality_score=score,
        quality_issues=quality_issues,
    )


# ══════════════════════════════════════════════════════════
#  RECOMMENDATIONS ENGINE
# ══════════════════════════════════════════════════════════

def _generate_recommendations(
    profiles:     List[ColumnProfile],
    dup_rows:     int,
    missing:      int,
    total:        int,
    n_rows:       int,
) -> List[str]:
    """
    Senior analyst level recommendations.
    Plain English, no jargon, no emojis (PDF safe).
    Prioritized: Critical → Warning → Info.
    """
    recs = []

    # Critical: Duplicates
    if dup_rows > 0:
        dup_pct = dup_rows / max(n_rows, 1) * 100
        recs.append(
            "CRITICAL: {:,} duplicate rows detected ({:.1f}% of data). "
            "Remove duplicates before any analysis — they inflate counts "
            "and distort aggregations.".format(dup_rows, dup_pct)
        )

    # Critical: High missing overall
    miss_pct_overall = missing / max(total, 1) * 100
    if miss_pct_overall > 20:
        recs.append(
            "CRITICAL: {:.1f}% of all data is missing. "
            "Analysis reliability is compromised. "
            "Review data collection process and imputation strategy.".format(miss_pct_overall)
        )

    # Per-column issues — sorted by severity
    drop_cols   = []
    impute_cols = []
    outlier_cols= []
    skew_cols   = []
    id_cols     = []
    const_cols  = []

    for p in profiles:
        if p.is_constant:
            const_cols.append(p.name)
        elif p.is_id_like:
            id_cols.append(p.name)
        elif p.missing_pct > 50:
            drop_cols.append((p.name, p.missing_pct))
        elif p.missing_pct > 5:
            impute_cols.append((p.name, p.missing_pct, p.inferred_type))
        if p.has_outliers and p.outlier_pct > 5:
            outlier_cols.append((p.name, p.outlier_count, p.outlier_pct))
        if p.skewness and abs(p.skewness) > 2:
            skew_cols.append((p.name, p.skewness))

    # Constant columns
    if const_cols:
        recs.append(
            "DROP COLUMNS: {} column(s) have zero variance ({}) — "
            "they add no analytical value and should be removed.".format(
                len(const_cols), ", ".join(const_cols[:3]))
        )

    # High missing — recommend drop
    for col, pct in drop_cols[:3]:
        recs.append(
            "CONSIDER DROPPING '{}': {:.1f}% missing values. "
            "Imputing more than 50% of a column introduces more noise than signal. "
            "Drop unless the column is critical.".format(col, pct)
        )

    # Missing — recommend imputation
    numeric_impute = [(c,p) for c,p,t in impute_cols if t == "numeric"]
    cat_impute     = [(c,p) for c,p,t in impute_cols if t != "numeric"]

    if numeric_impute:
        recs.append(
            "IMPUTE (Numeric): {} column(s) have missing values — {}. "
            "Use median imputation (robust to outliers) not mean.".format(
                len(numeric_impute),
                ", ".join(["'{}' ({:.0f}%)".format(c,p) for c,p in numeric_impute[:3]]))
        )

    if cat_impute:
        recs.append(
            "IMPUTE (Categorical): {} column(s) have missing values — {}. "
            "Fill with mode (most frequent value) or a dedicated 'Unknown' category.".format(
                len(cat_impute),
                ", ".join(["'{}' ({:.0f}%)".format(c,p) for c,p in cat_impute[:3]]))
        )

    # Outliers
    if outlier_cols:
        top_out = max(outlier_cols, key=lambda x: x[2])
        recs.append(
            "OUTLIERS DETECTED: {} column(s) have significant outliers. "
            "Worst: '{}' ({:,} values, {:.1f}%). "
            "Investigate: are these data entry errors or genuine extremes? "
            "Do NOT remove without understanding root cause.".format(
                len(outlier_cols), top_out[0], top_out[1], top_out[2])
        )

    # Skewness
    if skew_cols:
        recs.append(
            "SKEWED DISTRIBUTIONS: {} column(s) are heavily skewed — {}. "
            "Use MEDIAN not mean for reporting these columns. "
            "Apply log-transform if using in regression models.".format(
                len(skew_cols),
                ", ".join(["'{}' (skew={:.1f})".format(c,s) for c,s in skew_cols[:3]]))
        )

    # ID columns
    if id_cols:
        recs.append(
            "ID COLUMNS DETECTED: {} ({}) — exclude from statistical analysis, "
            "grouping, and ML features. Use only as row identifiers.".format(
                len(id_cols), ", ".join(id_cols[:4]))
        )

    # Positive finding
    good_cols = [p for p in profiles if p.quality_score >= 95]
    if len(good_cols) >= len(profiles) * 0.7:
        recs.append(
            "GOOD: {:.0f}% of columns ({}/{}) have high data quality scores (95+). "
            "Dataset is suitable for reliable analysis.".format(
                len(good_cols)/len(profiles)*100, len(good_cols), len(profiles))
        )

    return recs


# ══════════════════════════════════════════════════════════
#  MAIN PROFILER
# ══════════════════════════════════════════════════════════

def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """
    Run complete data quality profiling.
    Handles any dataset — HR, Ecommerce, Finance, Healthcare, General.
    Robust to: mixed types, inf values, large files, dirty data.
    """
    # ── Pre-clean: replace infinities globally ─────────────
    try:
        df = df.copy()
        num_cols_all = df.select_dtypes(include="number").columns
        if len(num_cols_all) > 0:
            df[num_cols_all] = df[num_cols_all].replace(
                [np.inf, -np.inf], np.nan)
    except Exception:
        pass

    # ── Dataset-level stats ────────────────────────────────
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    datetime_cols    = df.select_dtypes(include="datetime").columns.tolist()

    total_cells   = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    dup_rows      = int(df.duplicated().sum())

    # ── Profile each column ────────────────────────────────
    col_profiles = []
    for col in df.columns:
        try:
            col_profiles.append(_profile_column(df, col))
        except Exception:
            # Absolute fallback — never crash
            col_profiles.append(ColumnProfile(
                name=col,
                dtype=str(df[col].dtype),
                inferred_type="unknown",
                missing_count=int(df[col].isna().sum()),
                missing_pct=round(df[col].isna().mean()*100, 1),
                unique_count=0, unique_pct=0.0,
                is_id_like=False, is_constant=False,
                quality_score=50.0,
            ))

    # ── Composite quality score ────────────────────────────
    completeness = (1 - missing_cells / max(total_cells, 1)) * 100
    dedup_score  = (1 - dup_rows / max(len(df), 1)) * 100
    col_health   = sum(p.quality_score for p in col_profiles) / max(len(col_profiles), 1)
    overall      = round(completeness*0.60 + dedup_score*0.30 + col_health*0.10, 1)

    # ── Grade ─────────────────────────────────────────────
    if overall >= 90:
        grade = "A"
    elif overall >= 75:
        grade = "B"
    elif overall >= 60:
        grade = "C"
    elif overall >= 45:
        grade = "D"
    else:
        grade = "F"

    # ── Categorize columns ────────────────────────────────
    id_cols          = [p.name for p in col_profiles if p.is_id_like]
    constant_cols    = [p.name for p in col_profiles if p.is_constant]
    high_missing     = [p.name for p in col_profiles if p.missing_pct > 30]
    high_outlier     = [p.name for p in col_profiles if p.outlier_pct > 10]

    # ── Recommendations ───────────────────────────────────
    recs     = _generate_recommendations(
        col_profiles, dup_rows, missing_cells, total_cells, len(df))
    n_issues = len([p for p in col_profiles if p.quality_issues])

    return DatasetProfile(
        rows=len(df),
        cols=len(df.columns),
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_pct=round(missing_cells / max(total_cells, 1) * 100, 1),
        duplicate_rows=dup_rows,
        duplicate_pct=round(dup_rows / max(len(df), 1) * 100, 1),
        overall_quality_score=overall,
        column_profiles=col_profiles,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        datetime_cols=datetime_cols,
        id_cols=id_cols,
        constant_cols=constant_cols,
        high_missing_cols=high_missing,
        high_outlier_cols=high_outlier,
        recommendations=recs,
        n_issues=n_issues,
        data_quality_grade=grade,
    )
