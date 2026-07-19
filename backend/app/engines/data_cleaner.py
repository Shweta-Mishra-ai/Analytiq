import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CleanAction:
    """Records a single cleaning action taken."""
    column: str
    issue: str
    action: str
    before: Any
    after: Any
    rows_affected: int


@dataclass
class CleaningReport:
    """Full before/after cleaning report."""
    original_shape: tuple
    cleaned_shape: tuple
    actions: List[CleanAction] = field(default_factory=list)
    duplicates_removed: int = 0
    rows_dropped: int = 0

    def add(self, col, issue, action, before, after, rows=0):
        self.actions.append(CleanAction(col, issue, action, before, after, rows))

    @property
    def total_changes(self):
        return len(self.actions) + self.duplicates_removed + self.rows_dropped


def auto_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Full auto-cleaning pipeline.
    Returns (cleaned_df, CleaningReport).
    Every action is logged — nothing silent.
    """
    df = df.copy()
    report = CleaningReport(
        original_shape=df.shape,
        cleaned_shape=df.shape,
    )

    # ── 1. Strip whitespace from column names ──────────────
    old_cols = df.columns.tolist()
    df.columns = df.columns.str.strip()
    renamed = [(o, n) for o, n in zip(old_cols, df.columns) if o != n]
    for old, new in renamed:
        report.add("column_name", "leading/trailing whitespace",
                   "stripped", old, new, 0)

    # ── 2. Drop fully empty columns ────────────────────────
    fully_empty = [c for c in df.columns if df[c].isna().all()]
    if fully_empty:
        df.drop(columns=fully_empty, inplace=True)
        for c in fully_empty:
            report.add(c, "100% empty", "column dropped", "all null", "removed", len(df))

    # ── 3. Drop constant columns (1 unique non-null value) ─
    constant_cols = []
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 1 and len(df) > 1:
            constant_cols.append(c)
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        for c in constant_cols:
            report.add(c, "constant value (no variance)",
                       "column dropped", "1 unique value", "removed", len(df))

    # ── 4. Remove duplicate rows ───────────────────────────
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_after = len(df)
    dupes_removed = n_before - n_after
    report.duplicates_removed = dupes_removed
    if dupes_removed > 0:
        report.add("all_columns", "{} duplicate rows".format(dupes_removed),
                   "rows removed", n_before, n_after, dupes_removed)

    # ── 5. Per-column cleaning ─────────────────────────────
    for col in df.columns:
        _clean_column(df, col, report)

    report.cleaned_shape = df.shape
    return df, report


def _clean_column(df: pd.DataFrame, col: str, report: CleaningReport):
    """Apply all relevant cleaning to one column."""
    s = df[col]

    # ── 5a. Strip string whitespace ────────────────────────
    if s.dtype == object:
        stripped = s.str.strip() if hasattr(s.str, "strip") else s
        ws_count = (s != stripped).sum()
        if ws_count > 0:
            df[col] = stripped
            s = df[col]
            report.add(col, "{} values had leading/trailing spaces".format(ws_count),
                       "whitespace stripped", ws_count, 0, ws_count)

    # ── 5b. Handle missing values ──────────────────────────
    missing = s.isna().sum()
    if missing > 0:
        missing_pct = missing / max(len(df), 1) * 100

        if missing_pct > 60:
            # Too many missing — drop the column
            df.drop(columns=[col], inplace=True)
            report.add(col,
                       "{:.1f}% missing ({} cells)".format(missing_pct, missing),
                       "column dropped — too sparse",
                       "{} missing".format(missing), "removed", missing)
            return

        elif pd.api.types.is_numeric_dtype(s):
            # Numeric → fill with median
            median_val = s.median()
            df[col] = s.fillna(median_val)
            report.add(col,
                       "{} missing values ({:.1f}%)".format(missing, missing_pct),
                       "filled with median ({:.4g})".format(median_val),
                       "{} nulls".format(missing), 0, missing)

        else:
            # Categorical → fill with mode or "Unknown"
            mode_vals = s.mode()
            if len(mode_vals) > 0 and missing_pct < 20:
                fill_val = mode_vals[0]
                df[col] = s.fillna(fill_val)
                report.add(col,
                           "{} missing values ({:.1f}%)".format(missing, missing_pct),
                           "filled with mode ('{}')".format(str(fill_val)[:30]),
                           "{} nulls".format(missing), 0, missing)
            else:
                df[col] = s.fillna("Unknown")
                report.add(col,
                           "{} missing values ({:.1f}%)".format(missing, missing_pct),
                           "filled with 'Unknown'",
                           "{} nulls".format(missing), 0, missing)

    # ── 5c. Numeric outlier flagging (NOT auto-removed) ────
    if pd.api.types.is_numeric_dtype(df[col]) and col in df.columns:
        s2 = df[col].dropna()
        if len(s2) > 10:
            q1, q3 = s2.quantile(0.25), s2.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo = q1 - 3.0 * iqr   # 3x IQR = extreme outliers only
                hi = q3 + 3.0 * iqr
                extreme = ((df[col] < lo) | (df[col] > hi)).sum()
                if extreme > 0:
                    report.add(col,
                               "{} extreme outliers (3x IQR: {:.2g} – {:.2g})".format(
                                   extreme, lo, hi),
                               "flagged — not removed (review recommended)",
                               extreme, extreme, extreme)

    # ── 5d. Normalize boolean-like text columns ────────────
    if df[col].dtype == object and col in df.columns:
        s3 = df[col].dropna().str.lower().str.strip()
        bool_vals = {"yes", "no", "true", "false", "1", "0", "y", "n"}
        if set(s3.unique()) <= bool_vals and s3.nunique() <= 4:
            mapping = {"yes": True, "true": True, "1": True, "y": True,
                       "no": False, "false": False, "0": False, "n": False}
            converted = df[col].str.lower().str.strip().map(mapping)
            if converted.notna().mean() > 0.95:
                df[col] = converted
                report.add(col,
                           "boolean-like text values",
                           "converted to bool (True/False)",
                           "text", "bool", len(df))


def get_cleaning_summary(report: CleaningReport) -> dict:
    """
    Returns structured summary for display in Streamlit.
    Grouped by issue type for easy rendering.
    """
    groups = {
        "duplicates":  [],
        "missing":     [],
        "type_fix":    [],
        "dropped_col": [],
        "flagged":     [],
        "whitespace":  [],
        "other":       [],
    }

    for a in report.actions:
        issue_l = a.issue.lower()
        action_l = a.action.lower()
        if "duplicate" in issue_l:
            groups["duplicates"].append(a)
        elif "missing" in issue_l or "null" in issue_l or "empty" in action_l:
            groups["missing"].append(a)
        elif "dropped" in action_l and "column" in action_l:
            groups["dropped_col"].append(a)
        elif "flag" in action_l:
            groups["flagged"].append(a)
        elif "whitespace" in issue_l or "spaces" in issue_l:
            groups["whitespace"].append(a)
        elif "bool" in action_l or "convert" in action_l:
            groups["type_fix"].append(a)
        else:
            groups["other"].append(a)

    return {
        "original_rows":    report.original_shape[0],
        "original_cols":    report.original_shape[1],
        "cleaned_rows":     report.cleaned_shape[0],
        "cleaned_cols":     report.cleaned_shape[1],
        "duplicates_removed": report.duplicates_removed,
        "rows_dropped":     report.rows_dropped,
        "total_actions":    report.total_changes,
        "groups":           groups,
    }
