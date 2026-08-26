import logging
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from app.services.dtypes import is_text_dtype

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  SQL DIALECTS
# ══════════════════════════════════════════════════════════
# The analysis runs in pandas, but a client's data team works in the
# warehouse. The script is handed over so each step is auditable and can
# be applied upstream — which only helps if it runs on the warehouse they
# actually have.

DIALECTS = ("ansi", "postgres", "mysql", "snowflake", "bigquery")

# Identifier quoting. ANSI, Postgres and Snowflake use double quotes;
# MySQL and BigQuery use backticks. Only identifiers are double-quoted in
# generated SQL (_lit renders string literals with single quotes), so a
# render-time swap is unambiguous.
_BACKTICK_DIALECTS = ("mysql", "bigquery")


def _requote(sql: str, dialect: str) -> str:
    """Re-quote identifiers for dialects that do not use double quotes."""
    if dialect not in _BACKTICK_DIALECTS:
        return sql
    # Identifier quotes are the only double quotes present; an embedded
    # quote inside an identifier was doubled by _q, so collapse that first.
    return sql.replace('""', "\u0000").replace('"', "`").replace("\u0000", "``")


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
class CleaningPolicy:
    """What cleaning is allowed to change.

    The defaults are non-destructive, and deliberately so. Cleaning that
    silently removes rows or columns can destroy the very thing being
    measured: blanket deduplication of a transactional table deleted three
    of six rows and half the revenue in testing, and the >60%-missing rule
    dropped a column that was 89% empty and perfectly predictive of churn.

    Under these defaults Analytiq fixes what is unambiguously wrong
    (whitespace, type coercion, missing values it can justify imputing)
    and *reports* what is ambiguous, with the SQL to act on it. Removing
    rows is then the user's decision, made with the numbers in front of
    them.

    `aggressive()` restores the older behaviour for callers that want it.
    """
    # Remove exact duplicate rows. Off by default: identical rows in a
    # transactional table are often two genuine events, not one error.
    drop_duplicates: bool = False
    # Drop columns above `sparse_threshold` missing. Off by default: a
    # mostly-empty column can carry the strongest signal in the data.
    drop_sparse_columns: bool = False
    # Drop columns with a single distinct value. Off by default: a
    # constant column records the scope of the extract ("country = India")
    # and is worth keeping even though it carries no variance.
    drop_constant_columns: bool = False
    # Substitute a central value for ordinary (missing-at-random) gaps.
    impute_numeric: bool = True
    # Record a boolean companion column when a column is imputed, so the
    # fact of absence survives the fill. Standard practice: whether a value
    # was missing is frequently predictive in its own right.
    add_missingness_indicators: bool = True
    # Above this share missing, a column is reported as too sparse to rely
    # on (and dropped only if drop_sparse_columns is set).
    sparse_threshold: float = 60.0
    # Below this share missing, an indicator column would be noise.
    indicator_min_pct: float = 10.0

    @classmethod
    def aggressive(cls) -> "CleaningPolicy":
        """The older destructive behaviour, kept for callers that want it."""
        return cls(drop_duplicates=True, drop_sparse_columns=True,
                   drop_constant_columns=True)


# Suffix marking a generated missingness indicator. Downstream selection
# checks this so indicators are available to models without appearing as
# metrics in charts and KPI panels.
MISSING_INDICATOR_SUFFIX = "__was_missing"


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
    # Dialect-specific replacements for `sql`, keyed by dialect name. Only
    # populated where a statement genuinely differs (deduplication, mainly);
    # everything else is portable as written.
    sql_variants: Dict[str, str] = field(default_factory=dict)
    # True when the action changed data rather than only describing it.
    # A flagged finding and an applied fix must not read the same.
    applied: bool = True


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
    # Duplicates found but deliberately not removed, with the verdict on
    # what they probably are. Under the default policy nothing is deleted:
    # identical rows in a transactional table are commonly two genuine
    # events, and deduplicating them destroys revenue.
    duplicates_flagged: int = 0
    duplicate_verdict: str = ""
    duplicate_confidence: str = ""
    # Columns kept despite crossing a threshold that would once have
    # dropped them, with the reason. Retained so nothing is lost silently.
    retained_sparse: Dict[str, float] = field(default_factory=dict)
    retained_constant: List[str] = field(default_factory=list)
    # Generated <col>__was_missing companions.
    missingness_indicators: List[str] = field(default_factory=list)
    # Set when the policy in force allowed destructive steps.
    policy_note: str = ""

    def add(self, col, issue, action, before, after, rows=0, sql="",
            sql_variants=None, applied=True):
        self.actions.append(
            CleanAction(col, issue, action, before, after, rows, sql,
                        sql_variants or {}, applied))

    def sql_script(self, table: str = "your_table",
                   dialect: str = "ansi") -> str:
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
        dialect = str(dialect or "ansi").lower()
        if dialect not in DIALECTS:
            logger.warning("unknown SQL dialect %r — emitting portable ANSI",
                           dialect)
            dialect = "ansi"
        if dialect != "ansi":
            lines.insert(3, "-- Dialect: {}".format(dialect))
        for a in self.actions:
            sql = a.sql_variants.get(dialect) or a.sql
            if not sql:
                continue
            prefix = "" if a.applied else "FLAGGED, NOT APPLIED \u2014 "
            lines.append("-- {}{}: {}".format(prefix, a.column, a.issue))
            lines.append(_requote(sql.replace("{table}", _q(table)), dialect))
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

    Repeated rows are collapsed first. Under the default non-destructive
    policy duplicates are kept in the data, and a row appearing twice is
    not two independent observations — leaving them in lets a handful of
    repeated rows manufacture an effect size that is not there. The test
    is about the pattern, so it runs on distinct rows; the data itself is
    untouched.
    """
    try:
        df = df.drop_duplicates()
    except Exception:
        logger.debug("could not de-duplicate for the missingness test",
                     exc_info=True)
    mask = df[col].isna()
    n_missing = int(mask.sum())
    if n_missing < 15 or (len(df) - n_missing) < 15:
        return False
    for other in df.select_dtypes(include="number").columns:
        if other == col or str(other).endswith(MISSING_INDICATOR_SUFFIX):
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


_TRANSACTION_HINTS = ("amount", "revenue", "sales", "price", "qty",
                      "quantity", "units", "total", "value", "spend",
                      "cost", "payment", "charge", "order")


def classify_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """What repeated rows in this frame probably are.

    Identical rows are not automatically an error. In a transactional
    table two identical rows are usually two genuine events — the same SKU
    sold twice on the same day at the same price — and deleting one
    deletes real revenue. In an entity table (one row per customer) the
    same row twice is almost always a broken join or a double export.

    Returns the count, a verdict in plain language, and a confidence, so
    the report can say what it found without pretending to know more than
    it does.
    """
    n_dupes = int(df.duplicated().sum())
    if n_dupes == 0:
        return {"count": 0, "verdict": "", "confidence": "",
                "likely_error": False}

    cols_lower = [str(c).lower() for c in df.columns]
    has_money = any(any(h in c for h in _TRANSACTION_HINTS) for c in cols_lower)
    has_date = bool(df.select_dtypes(include="datetime").columns.tolist()) or \
        any("date" in c or "time" in c or "period" in c for c in cols_lower)

    # An identity column that is unique across the frame means each row is
    # meant to be one entity — so repeats are a defect, not an event.
    id_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl.endswith("_id") or cl == "id" or cl.endswith("id") \
                or "code" in cl or "key" in cl:
            id_col = c
            break

    share = n_dupes / max(len(df), 1) * 100

    if has_money and has_date:
        return {
            "count": n_dupes, "likely_error": False, "confidence": "low",
            "verdict": (
                "{:,} rows ({:.1f}%) are identical to another row. This table "
                "carries both a value column and a date, so identical rows are "
                "plausibly genuine repeat events — the same item sold twice on "
                "the same day. They have NOT been removed: deduplicating a "
                "transactional table deletes real turnover. Confirm against the "
                "source system before removing any."
            ).format(n_dupes, share),
        }
    if id_col is not None:
        return {
            "count": n_dupes, "likely_error": True, "confidence": "high",
            "verdict": (
                "{:,} rows ({:.1f}%) are identical to another row, and this "
                "table has an identity column ('{}'), so each row should be "
                "one entity. Repeats at this shape are usually a broken join "
                "or a re-export rather than real data. They have NOT been "
                "removed — the SQL to do so is included below."
            ).format(n_dupes, share, id_col),
        }
    return {
        "count": n_dupes, "likely_error": False, "confidence": "medium",
        "verdict": (
            "{:,} rows ({:.1f}%) are identical to another row. Whether these "
            "are duplicates or genuine repeats depends on what one row is "
            "meant to represent, which the data alone cannot say. They have "
            "NOT been removed."
        ).format(n_dupes, share),
    }


def auto_clean(df: pd.DataFrame,
               policy: "CleaningPolicy" = None) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Full auto-cleaning pipeline.
    Returns (cleaned_df, CleaningReport).
    Every action is logged — nothing silent.
    """
    policy = policy or CleaningPolicy()
    df = df.copy()
    report = CleaningReport(
        original_shape=df.shape,
        cleaned_shape=df.shape,
    )
    if policy.drop_duplicates or policy.drop_sparse_columns \
            or policy.drop_constant_columns:
        report.policy_note = (
            "Run under a destructive cleaning policy: rows and columns were "
            "removed as well as corrected.")

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
        if policy.drop_constant_columns:
            df.drop(columns=constant_cols, inplace=True)
            for c in constant_cols:
                report.add(c, "constant value (no variance)",
                           "column dropped", "1 unique value", "removed",
                           len(df),
                           sql="ALTER TABLE {{table}} DROP COLUMN {};".format(
                               _q(c)))
        else:
            # A constant column carries no variance but does carry scope:
            # "country = India" tells the reader what this extract covers.
            # Correlation and modelling ignore it anyway, so dropping it
            # buys nothing and loses a fact about the data.
            for c in constant_cols:
                try:
                    only = df[c].dropna().iloc[0] if df[c].notna().any() else None
                except Exception:
                    only = None
                report.retained_constant.append(c)
                report.add(c, "single distinct value ({})".format(
                               "all {}".format(only) if only is not None
                               else "no non-null values"),
                           "kept — records the scope of this extract; carries "
                           "no variance so it is excluded from correlation "
                           "and modelling automatically",
                           "1 unique value", "retained", 0,
                           applied=False,
                           sql=("-- Constant column, kept deliberately. To "
                                "confirm it is constant at source:\n"
                                "SELECT {c}, COUNT(*) FROM {{table}} "
                                "GROUP BY {c};").format(c=_q(c)))

    # ── 4. Duplicate rows ─────────────────────────────────
    verdict = classify_duplicates(df)
    if verdict["count"]:
        _cols = ", ".join(_q(c) for c in df.columns)
        dedupe_sql = {
            "ansi": ("-- Portable form: keep the first row of each identical "
                     "group.\nDELETE FROM {table}\nWHERE ctid IN (\n"
                     "  SELECT ctid FROM (\n"
                     "    SELECT ctid, ROW_NUMBER() OVER (PARTITION BY " + _cols +
                     " ORDER BY ctid) AS rn\n    FROM {table}\n  ) t "
                     "WHERE rn > 1\n);"),
            "postgres": ("DELETE FROM {table} a USING (\n"
                         "  SELECT MIN(ctid) AS keep_ctid, " + _cols + "\n"
                         "  FROM {table} GROUP BY " + _cols +
                         " HAVING COUNT(*) > 1\n) d\n"
                         "WHERE a.ctid <> d.keep_ctid;"),
            "mysql": ("DELETE t FROM {table} t\nJOIN (\n"
                      "  SELECT MIN(id) AS keep_id, " + _cols + "\n"
                      "  FROM {table} GROUP BY " + _cols +
                      " HAVING COUNT(*) > 1\n) d\n  ON t.id <> d.keep_id\n"
                      "-- Requires a unique row id. Without one, create a "
                      "deduplicated table with SELECT DISTINCT instead."),
            "snowflake": ("CREATE OR REPLACE TABLE {table} AS\n"
                          "SELECT * FROM {table}\n"
                          "QUALIFY ROW_NUMBER() OVER (PARTITION BY " + _cols +
                          " ORDER BY 1) = 1;"),
            "bigquery": ("CREATE OR REPLACE TABLE {table} AS\n"
                         "SELECT * EXCEPT(_rn) FROM (\n"
                         "  SELECT *, ROW_NUMBER() OVER (PARTITION BY " + _cols +
                         " ORDER BY 1) AS _rn\n  FROM {table}\n) "
                         "WHERE _rn = 1;"),
        }
        if policy.drop_duplicates:
            n_before = len(df)
            df.drop_duplicates(inplace=True)
            report.duplicates_removed = n_before - len(df)
            report.add("all_columns",
                       "{} duplicate rows".format(report.duplicates_removed),
                       "rows removed", n_before, len(df),
                       report.duplicates_removed,
                       sql=dedupe_sql["postgres"], sql_variants=dedupe_sql)
        else:
            report.duplicates_flagged = verdict["count"]
            report.duplicate_verdict = verdict["verdict"]
            report.duplicate_confidence = verdict["confidence"]
            report.add("all_columns",
                       "{} rows identical to another row".format(
                           verdict["count"]),
                       verdict["verdict"],
                       len(df), len(df), 0, applied=False,
                       sql=("-- NOT APPLIED. Removing rows is a decision about "
                            "what one row means,\n-- which the data alone "
                            "cannot settle. Inspect first:\n"
                            "SELECT " + _cols + ", COUNT(*) AS n\n"
                            "FROM {table} GROUP BY " + _cols +
                            " HAVING COUNT(*) > 1\nORDER BY n DESC;\n\n"
                            "-- Then, only if they are genuinely duplicates:\n"
                            + dedupe_sql["postgres"]),
                       sql_variants={})

    # ── 5. Per-column cleaning ─────────────────────────────
    for col in list(df.columns):
        _clean_column(df, col, report, policy)

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


def _add_missing_indicator(df: pd.DataFrame, col: str,
                           report: CleaningReport, policy: "CleaningPolicy",
                           missing_pct: float) -> None:
    """Add a boolean `<col>__was_missing` companion column.

    Only above `indicator_min_pct` — below that the indicator is nearly all
    False and adds noise rather than signal. Named by suffix so chart and
    KPI selection can exclude these while models still see them.
    """
    if not policy.add_missingness_indicators:
        return
    if missing_pct < policy.indicator_min_pct:
        return
    name = "{}{}".format(col, MISSING_INDICATOR_SUFFIX)
    if name in df.columns:
        return
    try:
        df[name] = df[col].isna()
    except Exception:
        logger.warning("could not add a missingness indicator for %r", col,
                       exc_info=True)
        return
    report.missingness_indicators.append(name)
    report.add(name,
               "companion column generated for {}".format(col),
               "records which rows were missing before any fill, so the "
               "absence itself stays available to modelling",
               "—", "{} true".format(int(df[name].sum())), 0,
               applied=True,
               sql=("ALTER TABLE {{table}} ADD COLUMN {n} BOOLEAN;\n"
                    "UPDATE {{table}} SET {n} = ({c} IS NULL);").format(
                        n=_q(name), c=_q(col)))


def _clean_column(df: pd.DataFrame, col: str, report: CleaningReport,
                  policy: 'CleaningPolicy' = None):
    """Apply all relevant cleaning to one column."""
    policy = policy or CleaningPolicy()
    if col not in df.columns:
        return
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

        if missing_pct > policy.sparse_threshold:
            if policy.drop_sparse_columns:
                df.drop(columns=[col], inplace=True)
                report.add(col,
                           "{:.1f}% missing ({} cells)".format(
                               missing_pct, missing),
                           "column dropped — too sparse",
                           "{} missing".format(missing), "removed", missing,
                           sql="ALTER TABLE {{table}} DROP COLUMN {};".format(
                               _q(col)))
                return
            # Kept. A mostly-empty column can be the strongest signal in
            # the frame — a complaint score recorded only for customers who
            # complained is 90% missing and near-perfectly predictive of
            # churn. Dropping it on a missingness threshold alone throws
            # that away before anything has looked at it.
            report.retained_sparse[col] = round(missing_pct, 1)
            report.add(col,
                       "{:.1f}% missing ({} cells)".format(missing_pct, missing),
                       "kept — too sparse for a reliable average, but the "
                       "pattern of presence may itself carry signal. Not "
                       "imputed: filling {:.0f}% of a column invents most of "
                       "it.".format(missing_pct),
                       "{} missing".format(missing), "retained", 0,
                       applied=False,
                       sql=("-- Sparse column, kept deliberately and NOT "
                            "imputed.\n-- Check whether presence itself "
                            "predicts anything before discarding:\n"
                            "SELECT {c} IS NULL AS is_missing, COUNT(*)\n"
                            "FROM {{table}} GROUP BY 1;").format(c=_q(col)))
            _add_missing_indicator(df, col, report, policy, missing_pct)
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
            # Record the fact of absence before erasing it. Whether a value
            # was missing is often predictive in its own right, and once the
            # median is written in there is no way to recover it.
            _add_missing_indicator(df, col, report, policy, missing_pct)
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
        # What was found but deliberately left alone. A report that shows
        # only what changed reads as though nothing else was wrong.
        "duplicates_flagged":    report.duplicates_flagged,
        "duplicate_verdict":     report.duplicate_verdict,
        "duplicate_confidence":  report.duplicate_confidence,
        "retained_sparse":       dict(report.retained_sparse),
        "retained_constant":     list(report.retained_constant),
        "missingness_indicators": list(report.missingness_indicators),
        "imputed_columns":       dict(report.imputed_columns),
        "informative_missingness": list(report.informative_missingness),
        "policy_note":           report.policy_note,
    }
