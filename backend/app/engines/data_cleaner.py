import logging
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from app.services.dtypes import is_text_dtype

logger = logging.getLogger(__name__)


def _q(identifier: str) -> str:
    """Quote a SQL identifier, escaping embedded quotes.

    Column names arriving from a client's export routinely contain spaces,
    punctuation or reserved words, so nothing may be interpolated bare.
    """
    return '"{}"'.format(str(identifier).replace('"', '""'))


def _lit(value: Any) -> str:
    """Render a Python value as a SQL literal, with strings escaped."""
    if value is None:
        return "NULL"
    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return "NULL"
        return repr(float(value)) if isinstance(value, (float, np.floating)) else str(int(value))
    return "'{}'".format(str(value).replace("'", "''"))


def table_name_from_filename(filename: str) -> str:
    """Guess the warehouse table a source file corresponds to.

    "Q3 Sales Export (final).csv" becomes "q3_sales_export_final" — close
    enough that the reader recognises their own table, and quoted at the
    point of use so an odd result is still valid SQL.
    """
    stem = re.split(r"[\\/]", str(filename or ""))[-1]
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", stem)
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", stem).strip("_").lower()
    slug = re.sub(r"_+", "_", slug)
    if not slug or slug[0].isdigit():
        slug = "t_" + slug if slug else "your_table"
    return slug


@dataclass
class CleanAction:
    """Records a single cleaning action taken.

    `sql` carries the equivalent statement against the source table. The
    analysis itself runs in pandas, but a client's data team works in the
    warehouse — handing them the SQL makes each step auditable and lets
    them apply the same cleaning upstream instead of taking the result on
    trust. It is documentation, not something this app executes.
    """
    column: str
    issue: str
    action: str
    before: Any
    after: Any
    rows_affected: int
    sql: str = ""


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

    def add(self, col, issue, action, before, after, rows=0, sql=""):
        self.actions.append(
            CleanAction(col, issue, action, before, after, rows, sql))

    def sql_script(self, table: str = "your_table") -> str:
        """The whole cleaning pass as a runnable SQL script.

        Statements appear in the order they were applied, because the order
        matters: dropping duplicates before imputing gives a different
        result from imputing first.
        """
        lines = [
            "-- Cleaning steps equivalent to the transformations applied.",
            "-- Order is significant: applying these out of sequence will not",
            "-- reproduce the same table.",
            "-- Review before running; this has not been executed anywhere.",
            "",
        ]
        for a in self.actions:
            if not a.sql:
                continue
            lines.append("-- {}: {}".format(a.column, a.issue))
            lines.append(a.sql.replace("{table}", _q(table)))
            lines.append("")
        if len(lines) <= 5:
            return ("-- No transformations were required; the source data "
                    "needed no cleaning.")
        return "\n".join(lines).rstrip() + "\n"

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


def _describe_key_duplicates(df: pd.DataFrame) -> tuple[str, str]:
    """Exact-row deduplication misses the commoner real problem: the same
    entity recorded twice with different values. Detect and describe it
    rather than silently keeping both rows.

    Returns (column, description); ("", "") when no identifier repeats.
    """
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
                return col, (
                    f"'{col}' repeats for {n_dupe_keys:,} row(s). Exact-duplicate "
                    "rows were removed, but these share an identifier while "
                    "differing elsewhere — confirm whether they are genuine "
                    "repeat events or an unresolved join before aggregating."
                )
        except Exception:
            logger.debug("key-duplicate check failed for %s", col, exc_info=True)
    return "", ""


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
                   "stripped", old, new, 0,
                   sql="ALTER TABLE {{table}} RENAME COLUMN {} TO {};".format(
                       _q(old), _q(new)))

    # ── 2. Drop fully empty columns ────────────────────────
    fully_empty = [c for c in df.columns if df[c].isna().all()]
    if fully_empty:
        df.drop(columns=fully_empty, inplace=True)
        for c in fully_empty:
            report.add(c, "100% empty", "column dropped", "all null", "removed", len(df),
                       sql="ALTER TABLE {{table}} DROP COLUMN {};".format(_q(c)))

    # ── 3. Drop constant columns (1 unique non-null value) ─
    constant_cols = []
    for c in df.columns:
        if df[c].nunique(dropna=True) <= 1 and len(df) > 1:
            constant_cols.append(c)
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        for c in constant_cols:
            report.add(c, "constant value (no variance)",
                       "column dropped", "1 unique value", "removed", len(df),
                       sql="ALTER TABLE {{table}} DROP COLUMN {};".format(_q(c)))

    # ── 4. Remove duplicate rows ───────────────────────────
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_after = len(df)
    dupes_removed = n_before - n_after
    report.duplicates_removed = dupes_removed
    if dupes_removed > 0:
        _cols = ", ".join(_q(c) for c in df.columns)
        report.add("all_columns", "{} duplicate rows".format(dupes_removed),
                   "rows removed", n_before, n_after, dupes_removed,
                   sql=("DELETE FROM {table} a USING (\n"
                        "  SELECT MIN(ctid) AS keep_ctid, " + _cols + "\n"
                        "  FROM {table} GROUP BY " + _cols + " HAVING COUNT(*) > 1\n"
                        ") d\nWHERE a.ctid <> d.keep_ctid;\n"
                        "-- Postgres form (ctid). On other engines use\n"
                        "-- ROW_NUMBER() OVER (PARTITION BY <all columns>) and delete rn > 1."))

    # ── 5. Per-column cleaning ─────────────────────────────
    for col in df.columns:
        _clean_column(df, col, report)

    # ── 6. Identity duplicates ─────────────────────────────
    # drop_duplicates above removes only byte-identical rows. The commoner
    # real defect is one entity appearing twice with differing values,
    # which silently double-counts in every downstream sum.
    key_col, report.key_duplicate_note = _describe_key_duplicates(df)
    if report.key_duplicate_note:
        report.add("identity", "repeated identifier values",
                   report.key_duplicate_note, "review", "flagged", 0,
                   sql=("-- Flagged, not resolved: whether a repeated key is a\n"
                        "-- genuine repeat event or a broken join is a question\n"
                        "-- about the source system, not about this table.\n"
                        "SELECT {c}, COUNT(*) AS n\n"
                        "FROM {{table}} GROUP BY {c} HAVING COUNT(*) > 1\n"
                        "ORDER BY n DESC;").format(c=_q(key_col)))

    report.cleaned_shape = df.shape
    return df, report


def _clean_column(df: pd.DataFrame, col: str, report: CleaningReport):
    """Apply all relevant cleaning to one column."""
    s = df[col]

    # ── 5a. Strip string whitespace ────────────────────────
    if is_text_dtype(s):
        stripped = s.str.strip() if hasattr(s.str, "strip") else s
        # NA != NA is True, so comparing the two series directly counts every
        # null as a whitespace fix and reports a number the reader can check
        # and find wrong. Only non-null values can have been trimmed.
        ws_count = int((s.notna() & (s != stripped)).sum())
        if ws_count > 0:
            df[col] = stripped
            s = df[col]
            report.add(col, "{} values had leading/trailing spaces".format(ws_count),
                       "whitespace stripped", ws_count, 0, ws_count,
                       sql="UPDATE {{table}} SET {c} = TRIM({c}) WHERE {c} <> TRIM({c});".format(
                           c=_q(col)))

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
                       "{} missing".format(missing), "removed", missing,
                       sql="ALTER TABLE {{table}} DROP COLUMN {};".format(_q(col)))
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
                       "{} nulls".format(missing), "{} nulls".format(missing), 0,
                       sql=("-- {c} is deliberately NOT imputed: rows where it is\n"
                            "-- NULL differ systematically from rows where it is not.\n"
                            "-- Inspect the pattern before deciding on a fill rule:\n"
                            "SELECT {c} IS NULL AS is_missing, COUNT(*)\n"
                            "FROM {{table}} GROUP BY 1;").format(c=_q(col)))

        elif (pd.api.types.is_numeric_dtype(s)
                and not pd.api.types.is_bool_dtype(s)):
            # Numeric → fill with median. Booleans are excluded: pandas
            # counts them as numeric, and the median of a nullable boolean
            # column can be 0.5 — a value the column cannot hold.
            median_val = s.median()
            df[col] = s.fillna(median_val)
            report.imputed_columns[col] = round(missing_pct, 1)
            caveat = (" — over a fifth of this column is imputed, so its mean, "
                      "spread and correlations are partly synthetic"
                      if missing_pct >= 20 else "")
            report.add(col,
                       "{} missing values ({:.1f}%)".format(missing, missing_pct),
                       "filled with median ({:.4g}){}".format(median_val, caveat),
                       "{} nulls".format(missing), 0, missing,
                       sql=("-- median of the non-null values, computed once and\n"
                            "-- written as a literal so the result is reproducible:\n"
                            "--   SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY {c})\n"
                            "--   FROM {{table}};\n"
                            "UPDATE {{table}} SET {c} = {v} WHERE {c} IS NULL;").format(
                                c=_q(col), v=_lit(median_val)))

        else:
            # Categorical → fill with mode or "Unknown"
            mode_vals = s.mode()
            if len(mode_vals) > 0 and missing_pct < 20:
                fill_val = mode_vals[0]
                df[col] = s.fillna(fill_val)
                report.add(col,
                           "{} missing values ({:.1f}%)".format(missing, missing_pct),
                           "filled with mode ('{}')".format(str(fill_val)[:30]),
                           "{} nulls".format(missing), 0, missing,
                           sql=("-- most frequent value of {c}, written as a literal:\n"
                                "--   SELECT {c} FROM {{table}} WHERE {c} IS NOT NULL\n"
                                "--   GROUP BY {c} ORDER BY COUNT(*) DESC LIMIT 1;\n"
                                "UPDATE {{table}} SET {c} = {v} WHERE {c} IS NULL;").format(
                                    c=_q(col), v=_lit(fill_val)))
            else:
                df[col] = s.fillna("Unknown")
                report.add(col,
                           "{} missing values ({:.1f}%)".format(missing, missing_pct),
                           "filled with 'Unknown'",
                           "{} nulls".format(missing), 0, missing,
                           sql=("UPDATE {{table}} SET {c} = 'Unknown' "
                                "WHERE {c} IS NULL;").format(c=_q(col)))

    # ── 5c. Numeric outlier flagging (NOT auto-removed) ────
    # Booleans are numeric to pandas, and np.quantile on a boolean array
    # raises "numpy boolean subtract ... is not supported". This is not
    # hypothetical: step 5d below converts "Yes"/"no" columns to bool, so
    # cleaning the same dataset twice — which the Data Quality page allows
    # with one click — turned the second call into a 500. An outlier in a
    # two-valued column is meaningless in any case.
    if (pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
            and col in df.columns):
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
                               extreme, extreme, extreme,
                               sql=("-- Flagged only. Nothing is deleted: an extreme value\n"
                                    "-- is as often a real event as a data-entry error.\n"
                                    "-- Bounds are Q1 - 3*IQR and Q3 + 3*IQR.\n"
                                    "SELECT * FROM {{table}}\n"
                                    "WHERE {c} < {lo} OR {c} > {hi};").format(
                                        c=_q(col),
                                        lo=_lit(round(float(lo), 4)),
                                        hi=_lit(round(float(hi), 4))))

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
                           "text", "bool", len(df),
                           sql=("ALTER TABLE {{table}} ADD COLUMN {n} BOOLEAN;\n"
                                "UPDATE {{table}} SET {n} = CASE\n"
                                "    WHEN lower(trim({c})) IN ('yes','true','1','y') THEN TRUE\n"
                                "    WHEN lower(trim({c})) IN ('no','false','0','n') THEN FALSE\n"
                                "END;\n"
                                "-- verify {n} before dropping the original:\n"
                                "-- ALTER TABLE {{table}} DROP COLUMN {c};\n"
                                "-- ALTER TABLE {{table}} RENAME COLUMN {n} TO {c};").format(
                                    c=_q(col), n=_q(col + "_bool")))


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
        # Grouped for display, but the report also needs the steps in the
        # order they were applied: deduplicating after imputing does not
        # produce the same table, so a grouped listing would misstate the
        # method.
        "actions":          list(report.actions),
        "original_rows":    report.original_shape[0],
        "original_cols":    report.original_shape[1],
        "cleaned_rows":     report.cleaned_shape[0],
        "cleaned_cols":     report.cleaned_shape[1],
        "duplicates_removed": report.duplicates_removed,
        "rows_dropped":     report.rows_dropped,
        "total_actions":    report.total_changes,
        "groups":           groups,
    }
