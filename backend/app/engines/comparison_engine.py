"""
core/comparison_engine.py — Multi-file dataset comparison.

Compares two datasets (e.g. "this month vs last month", "before vs after
an intervention") across three dimensions:
  1. Schema diff — columns added, removed, or changed dtype
  2. Row-level changes — record count delta, duplicate rate shift
  3. Column-level statistical comparison — for every common numeric column,
     computes the shift in mean/median/std and runs a statistical
     significance test (reusing core.ab_test_engine's Welch's t-test /
     Mann-Whitney U) so the user knows if a shift is real or just noise.
     For common categorical columns, compares category distribution shift
     via a chi-square test where applicable.

All computation happens directly on the two submitted datasets — no
external benchmarks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.engines.ab_test_engine import run_continuous_test, ABTestResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class SchemaDiff:
    added_columns:   List[str] = field(default_factory=list)
    removed_columns: List[str] = field(default_factory=list)
    common_columns:  List[str] = field(default_factory=list)
    dtype_changes:   Dict[str, str] = field(default_factory=dict)  # col -> "int64 → object"


@dataclass
class ColumnComparison:
    column:         str
    dtype:          str
    mean_a:         Optional[float] = None
    mean_b:         Optional[float] = None
    median_a:       Optional[float] = None
    median_b:       Optional[float] = None
    std_a:          Optional[float] = None
    std_b:          Optional[float] = None
    pct_change_mean:   Optional[float] = None
    pct_change_median: Optional[float] = None
    missing_pct_a:  float = 0.0
    missing_pct_b:  float = 0.0
    is_significant: Optional[bool] = None
    p_value:        Optional[float] = None
    test_used:      Optional[str] = None
    verdict:        str = ""
    # Categorical-specific
    top_category_a: Optional[str] = None
    top_category_b: Optional[str] = None
    category_shift_detected: Optional[bool] = None


@dataclass
class ComparisonReport:
    label_a:            str
    label_b:            str
    n_rows_a:            int
    n_rows_b:            int
    row_delta_pct:       float
    dup_pct_a:           float
    dup_pct_b:           float
    schema_diff:         SchemaDiff
    column_comparisons:  List[ColumnComparison]
    most_changed:        List[ColumnComparison]  # sorted by |pct_change_mean| desc
    significant_changes: List[ColumnComparison]  # only statistically significant
    summary:             str
    warnings:            List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
#  SCHEMA DIFF
# ══════════════════════════════════════════════════════════

def _compute_schema_diff(df_a: pd.DataFrame, df_b: pd.DataFrame) -> SchemaDiff:
    cols_a, cols_b = set(df_a.columns), set(df_b.columns)
    common = sorted(cols_a & cols_b)
    dtype_changes = {}
    for col in common:
        da, db = str(df_a[col].dtype), str(df_b[col].dtype)
        if da != db:
            dtype_changes[col] = f"{da} → {db}"
    return SchemaDiff(
        added_columns=sorted(cols_b - cols_a),
        removed_columns=sorted(cols_a - cols_b),
        common_columns=common,
        dtype_changes=dtype_changes,
    )


# ══════════════════════════════════════════════════════════
#  COLUMN-LEVEL COMPARISON
# ══════════════════════════════════════════════════════════

def _compare_numeric_column(col: str, s_a: pd.Series, s_b: pd.Series) -> ColumnComparison:
    a = pd.to_numeric(s_a, errors="coerce").dropna()
    b = pd.to_numeric(s_b, errors="coerce").dropna()

    cc = ColumnComparison(
        column=col, dtype="numeric",
        missing_pct_a=round(s_a.isna().mean() * 100, 1),
        missing_pct_b=round(s_b.isna().mean() * 100, 1),
    )

    if len(a) < 2 or len(b) < 2:
        cc.verdict = "Not enough data in one or both periods to compare."
        return cc

    cc.mean_a, cc.mean_b = float(a.mean()), float(b.mean())
    cc.median_a, cc.median_b = float(a.median()), float(b.median())
    cc.std_a, cc.std_b = float(a.std()), float(b.std())

    if cc.mean_a != 0:
        cc.pct_change_mean = round((cc.mean_b - cc.mean_a) / abs(cc.mean_a) * 100, 2)
    if cc.median_a != 0:
        cc.pct_change_median = round((cc.median_b - cc.median_a) / abs(cc.median_a) * 100, 2)

    # Statistical significance via the shared A/B test engine — reuses the
    # same Welch's t-test / Mann-Whitney U logic used for A/B tests, since
    # "did this metric really change between periods" is the same question
    # as "did this metric really differ between variants".
    try:
        ab_result: ABTestResult = run_continuous_test(
            a, b, variant_a_name="Period A", variant_b_name="Period B",
            metric_name=col,
        )
        cc.is_significant = ab_result.is_significant
        cc.p_value = ab_result.p_value
        cc.test_used = ab_result.test_used
        direction = "increased" if cc.mean_b > cc.mean_a else "decreased"
        change_phrase = (
            f"{cc.pct_change_mean:+.1f}% mean change" if cc.pct_change_mean is not None
            else f"absolute mean change of {cc.mean_b - cc.mean_a:+.2f} "
                 f"(relative % undefined — baseline mean was 0)"
        )
        if ab_result.is_significant:
            cc.verdict = (
                f"'{col}' {direction} significantly "
                f"({change_phrase}, p={cc.p_value:.4f})."
            )
        else:
            cc.verdict = (
                f"'{col}' shows a {change_phrase}, but this is "
                f"NOT statistically significant (p={cc.p_value:.4f}) — likely noise."
            )
    except ValueError as e:
        cc.verdict = f"Significance test skipped: {e}"
    except Exception:
        logger.warning("Comparison significance test failed for '%s'", col, exc_info=True)
        cc.verdict = "Significance test failed — showing descriptive stats only."

    return cc


def _compare_categorical_column(col: str, s_a: pd.Series, s_b: pd.Series) -> ColumnComparison:
    cc = ColumnComparison(
        column=col, dtype="categorical",
        missing_pct_a=round(s_a.isna().mean() * 100, 1),
        missing_pct_b=round(s_b.isna().mean() * 100, 1),
    )
    a_counts = s_a.dropna().value_counts(normalize=True)
    b_counts = s_b.dropna().value_counts(normalize=True)

    if a_counts.empty or b_counts.empty:
        cc.verdict = "Not enough data in one or both periods to compare."
        return cc

    cc.top_category_a = str(a_counts.index[0])
    cc.top_category_b = str(b_counts.index[0])
    cc.category_shift_detected = cc.top_category_a != cc.top_category_b

    # Chi-square test on shared categories (if both have enough categories/data)
    try:
        shared_cats = set(a_counts.index) & set(b_counts.index)
        if len(shared_cats) >= 2:
            obs_a = s_a.dropna().value_counts().reindex(shared_cats, fill_value=0)
            obs_b = s_b.dropna().value_counts().reindex(shared_cats, fill_value=0)
            contingency = np.array([obs_a.values, obs_b.values])
            if contingency.sum() > 0 and (contingency > 0).any():
                chi2, p, dof, _ = scipy_stats.chi2_contingency(contingency)
                cc.p_value = round(float(p), 6)
                cc.is_significant = p < 0.05
                cc.test_used = "Chi-Square Test of Independence"
                if cc.is_significant:
                    cc.verdict = (
                        f"'{col}' category distribution changed significantly "
                        f"(p={cc.p_value:.4f}). Top category: "
                        f"'{cc.top_category_a}' → '{cc.top_category_b}'."
                    )
                else:
                    cc.verdict = (
                        f"'{col}' category distribution is stable (p={cc.p_value:.4f})."
                    )
    except Exception:
        logger.warning("Chi-square test failed for categorical column '%s'", col, exc_info=True)

    if not cc.verdict:
        if cc.category_shift_detected:
            cc.verdict = f"Top category changed: '{cc.top_category_a}' → '{cc.top_category_b}'."
        else:
            cc.verdict = f"Top category unchanged: '{cc.top_category_a}'."

    return cc


# ══════════════════════════════════════════════════════════
#  MAIN COMPARISON
# ══════════════════════════════════════════════════════════

def run_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "Period A",
    label_b: str = "Period B",
    max_columns: int = 30,
) -> ComparisonReport:
    """
    Compare two datasets across schema, row counts, and per-column stats.

    Raises:
        TypeError — if either input is not a DataFrame.
        ValueError — if there are no common columns to compare, or either
        dataset is empty.
    """
    if not isinstance(df_a, pd.DataFrame) or not isinstance(df_b, pd.DataFrame):
        raise TypeError("run_comparison expects two pd.DataFrame objects")
    if df_a.empty or df_b.empty:
        raise ValueError("Both datasets must be non-empty to compare")

    warnings_list: List[str] = []
    schema = _compute_schema_diff(df_a, df_b)

    if not schema.common_columns:
        raise ValueError(
            "No common columns found between the two datasets — "
            "cannot run a comparison. Ensure both files have matching "
            "column names for the metrics you want to compare."
        )

    if schema.added_columns:
        warnings_list.append(
            f"{len(schema.added_columns)} column(s) only in {label_b}: "
            f"{', '.join(schema.added_columns[:5])}"
            + (f" (+{len(schema.added_columns)-5} more)" if len(schema.added_columns) > 5 else "")
        )
    if schema.removed_columns:
        warnings_list.append(
            f"{len(schema.removed_columns)} column(s) only in {label_a}: "
            f"{', '.join(schema.removed_columns[:5])}"
            + (f" (+{len(schema.removed_columns)-5} more)" if len(schema.removed_columns) > 5 else "")
        )
    if schema.dtype_changes:
        warnings_list.append(
            f"{len(schema.dtype_changes)} column(s) changed data type between periods — "
            "comparison for these may be less reliable."
        )

    n_rows_a, n_rows_b = len(df_a), len(df_b)
    row_delta_pct = round((n_rows_b - n_rows_a) / n_rows_a * 100, 1) if n_rows_a > 0 else 0.0
    dup_pct_a = round(df_a.duplicated().mean() * 100, 1)
    dup_pct_b = round(df_b.duplicated().mean() * 100, 1)

    columns_to_compare = schema.common_columns[:max_columns]
    if len(schema.common_columns) > max_columns:
        warnings_list.append(
            f"Comparing the first {max_columns} of {len(schema.common_columns)} "
            "common columns for performance. Reduce dataset width or contact "
            "support to raise this limit."
        )

    comparisons: List[ColumnComparison] = []
    for col in columns_to_compare:
        try:
            if pd.api.types.is_bool_dtype(df_a[col]) or pd.api.types.is_bool_dtype(df_b[col]):
                # NOTE: this check must come BEFORE is_numeric_dtype() below —
                # pandas' is_numeric_dtype() returns True for bool dtype too,
                # so checking numeric first would make this branch
                # unreachable (same root cause as the earlier col_stats()
                # boolean-column crash). Booleans are treated as categorical
                # (True/False are labels, not a meaningful mean/median).
                comparisons.append(_compare_categorical_column(
                    col, df_a[col].astype(str), df_b[col].astype(str)))
            elif pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
                comparisons.append(_compare_numeric_column(col, df_a[col], df_b[col]))
            else:
                comparisons.append(_compare_categorical_column(col, df_a[col], df_b[col]))
        except Exception:
            logger.warning("Column comparison failed for '%s'", col, exc_info=True)

    # Most-changed (by absolute % change in mean, numeric only)
    numeric_changes = [c for c in comparisons if c.pct_change_mean is not None]
    most_changed = sorted(numeric_changes, key=lambda c: abs(c.pct_change_mean), reverse=True)[:10]

    significant_changes = [c for c in comparisons if c.is_significant]

    # ── Narrative summary ────────────────────────────────────────────────────
    n_sig = len(significant_changes)
    if n_sig > 0:
        top = max(significant_changes,
                 key=lambda c: abs(c.pct_change_mean) if c.pct_change_mean is not None else 0)
        headline = (
            f"{n_sig} of {len(comparisons)} compared column(s) show a statistically "
            f"significant change between {label_a} and {label_b}. "
            f"Most notable: '{top.column}' "
            + (f"({top.pct_change_mean:+.1f}% change in mean)." if top.pct_change_mean is not None
               else f"({top.verdict})")
        )
    else:
        headline = (
            f"No statistically significant changes detected across {len(comparisons)} "
            f"compared columns between {label_a} and {label_b}. Observed differences "
            "are likely within normal variation."
        )

    if abs(row_delta_pct) > 20:
        headline += f" Record count changed {row_delta_pct:+.1f}% ({n_rows_a:,} → {n_rows_b:,})."

    return ComparisonReport(
        label_a=label_a, label_b=label_b,
        n_rows_a=n_rows_a, n_rows_b=n_rows_b,
        row_delta_pct=row_delta_pct,
        dup_pct_a=dup_pct_a, dup_pct_b=dup_pct_b,
        schema_diff=schema,
        column_comparisons=comparisons,
        most_changed=most_changed,
        significant_changes=significant_changes,
        summary=headline,
        warnings=warnings_list,
    )
