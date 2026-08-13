import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from app.services.dtypes import is_text_dtype

logger = logging.getLogger(__name__)


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
    # Columns where values were substituted for missing data, with the
    # share imputed. Any statistic derived from these is partly computed
    # on fabricated values, so the report must be able to say so rather
    # than presenting an imputed mean as an observed one.
    imputed_columns: Dict[str, float] = field(default_factory=dict)
    # Columns whose missingness is NOT random — it predicts another
    # column. Imputing these erases a real signal.
    informative_missingness: List[str] = field(default_factory=list)
    # Duplicate identity keys (same entity appearing more than once with
    # differing values), which exact-row deduplication cannot catch.
    key_duplicate_note: str = ""

    def add(self, col, issue, action, before, after, rows=0):
        self.actions.append(CleanAction(col, issue, action, before, after, rows))

    @property
    def total_changes(self):
        return len(self.actions) + self.duplicates_removed + self.rows_dropped

    @property
    def heavily_imputed(self) -> List[str]:
        """Columns imputed beyond the point where derived statistics can be
        reported without a caveat."""
        return [c for c, pct in self.imputed_columns.items() if pct >= 20.0]


def _missingness_is_informative(df: pd.DataFrame, col: str) -> bool:
    """Does whether `col` is missing predict anything else in the frame?

    Missing-at-random can be imputed. Missingness that carries signal —
    a satisfaction score absent precisely for the people who left, an
    income blank only for one segment — must not be, because filling it
    with the median erases the very pattern worth reporting.

    Tested by comparing each other numeric column between the
    missing and non-missing groups; a large standardised difference means
    the missingness is not random.
    """
    mask = df[col].isna()
    n_missing = int(mask.sum())
    if n_missing < 15 or (len(df) - n_missing) < 15:
        return False
    for other in df.select_dtypes(include="number").columns:
        if other == col:
            continue
        a = df.loc[mask, other].dropna()
        b = df.loc[~mask, other].dropna()
        if len(a) < 15 or len(b) < 15:
            continue
        try:
            pooled = np.sqrt((a.var() + b.var()) / 2)
            if pooled == 0:
                continue
            # Cohen's d >= 0.5 is a medium effect — clearly not random.
            if abs(float(a.mean() - b.mean()) / pooled) >= 0.5:
                return True
        except Exception:
            logger.debug("informative-missingness check failed for %s vs %s",
                         col, other, exc_info=True)
    return False


def _describe_key_duplicates(df: pd.DataFrame) -> str:
    """Exact-row deduplication misses the commoner real problem: the same
    entity recorded twice with different values. Detect and describe it
    rather than silently keeping both rows."""
    from app.engines.domains.base import is_id_column
    for col in df.columns:
        try:
            if not is_id_column(col, df[col]):
                continue
            non_null = df[col].dropna()
            if len(non_null) < 10:
                continue
            n_dupe_keys = int(non_null.duplicated().sum())
            if n_dupe_keys > 0:
                return (
                    f"'{col}' repeats for {n_dupe_keys:,} row(s). Exact-duplicate "
                    "rows were removed, but these share an identifier while "
                    "differing elsewhere — confirm whether they are genuine "
                    "repeat events or an unresolved join before aggregating."
                )
        except Exception:
            logger.debug("key-duplicate check failed for %s", col, exc_info=True)
    return ""


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

    # ── 6. Identity duplicates ─────────────────────────────
    # drop_duplicates above removes only byte-identical rows. The commoner
    # real defect is one entity appearing twice with differing values,
    # which silently double-counts in every downstream sum.
    report.key_duplicate_note = _describe_key_duplicates(df)
    if report.key_duplicate_note:
        report.add("identity", "repeated identifier values",
                   report.key_duplicate_note, "review", "flagged", 0)

    report.cleaned_shape = df.shape
    return df, report


def _clean_column(df: pd.DataFrame, col: str, report: CleaningReport):
    """Apply all relevant cleaning to one column."""
    s = df[col]

    # ── 5a. Strip string whitespace ────────────────────────
    if is_text_dtype(s):
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

        elif _missingness_is_informative(df, col):
            # Missingness that predicts another column is itself a finding.
            # Median-filling it would erase the pattern and report a mean
            # computed partly from invented values. Left as NA — pandas
            # excludes NA from statistics natively — and recorded so the
            # report can say why.
            report.informative_missingness.append(col)
            report.add(col,
                       "{} missing values ({:.1f}%) — not missing at random".format(
                           missing, missing_pct),
                       "left as missing deliberately; the pattern of absence "
                       "predicts other columns and is reported as a finding",
                       "{} nulls".format(missing), "{} nulls".format(missing), 0)

        elif pd.api.types.is_numeric_dtype(s):
            # Numeric → fill with median
            median_val = s.median()
            df[col] = s.fillna(median_val)
            report.imputed_columns[col] = round(missing_pct, 1)
            caveat = (" — over a fifth of this column is imputed, so its mean, "
                      "spread and correlations are partly synthetic"
                      if missing_pct >= 20 else "")
            report.add(col,
                       "{} missing values ({:.1f}%)".format(missing, missing_pct),
                       "filled with median ({:.4g}){}".format(median_val, caveat),
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
    if is_text_dtype(df[col]) and col in df.columns:
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
