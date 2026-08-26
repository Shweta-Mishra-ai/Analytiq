"""
Every cleaning step has to be shown in SQL, not only performed in pandas.

A client's data team cannot audit a dataframe. They can audit a script.
The value of this feature is entirely in it being *correct* — a wrong
UPDATE handed to someone who runs it against their warehouse is worse
than no script at all — so these tests check the statements themselves,
not merely that a string was produced.

The app never executes this SQL. It is documentation of what was done.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from app.engines.data_cleaner import (
    _lit,
    _q,
    auto_clean,
    table_name_from_filename,
)


@pytest.fixture()
def messy_df():
    """One frame exercising every branch of the cleaner: whitespace in a
    column name and in values, a fully empty column, a constant column,
    exact duplicate rows, numeric gaps, categorical gaps, boolean-like
    text, and extreme outliers."""
    rng = np.random.default_rng(21)
    n = 300
    amount = rng.normal(500, 90, n)
    amount[:8] = np.nan
    amount[8:11] = 50_000.0                      # extreme outliers
    region = rng.choice(["North", "South", "East"], n).astype(object)
    region[:12] = None
    df = pd.DataFrame({
        "  order id  ": range(n),                # whitespace in the name
        "amount": amount,
        "region": region,
        "notes": [None] * n,                     # fully empty
        "currency": ["GBP"] * n,                 # constant
        "is_paid": rng.choice(["Yes", "no"], n),
        "channel": [" web " if i % 3 else "store" for i in range(n)],
        "tier": rng.choice(["a", "b"], n),
    })
    return pd.concat([df, df.head(6)], ignore_index=True)   # exact dupes


def _script(df) -> str:
    _cleaned, report = auto_clean(df)
    return report.sql_script("orders")


# ══════════════════════════════════════════════════════════
#  Identifier and literal escaping
# ══════════════════════════════════════════════════════════

def test_identifiers_are_quoted_and_escaped():
    assert _q("order id") == '"order id"'
    assert _q('we"ird') == '"we""ird"'
    assert _q("select") == '"select"'


def test_string_literals_escape_embedded_quotes():
    """A region called "O'Brien's" must not terminate the literal."""
    assert _lit("O'Brien's") == "'O''Brien''s'"
    assert _lit(None) == "NULL"
    assert _lit(True) == "TRUE"
    assert _lit(3) == "3"
    assert _lit(float("nan")) == "NULL"


def test_column_names_with_spaces_appear_quoted_in_the_script(messy_df):
    """`RENAME COLUMN   order id   TO order id` is a syntax error, and a
    column called "order" would be a reserved word besides."""
    _cleaned, report = auto_clean(messy_df)
    renames = [a for a in report.actions if "RENAME COLUMN" in a.sql]
    assert renames, "the whitespace-in-column-name rename produced no SQL"
    stmt = renames[0].sql
    assert '"  order id  "' in stmt and '"order id"' in stmt, \
        "a column name containing spaces was interpolated bare: {!r}".format(stmt)


# ══════════════════════════════════════════════════════════
#  Every action carries its SQL
# ══════════════════════════════════════════════════════════

def test_every_transforming_action_has_sql(messy_df):
    """A step shown in the UI without its SQL is the gap this closes."""
    _cleaned, report = auto_clean(messy_df)
    assert report.actions, "the fixture triggered no cleaning at all"
    missing = [a for a in report.actions if not a.sql.strip()]
    assert not missing, \
        "actions with no SQL: {}".format([(a.column, a.action) for a in missing])


def test_the_script_covers_each_kind_of_step(messy_df):
    sql = _script(messy_df).upper()
    for stmt in ("RENAME COLUMN", "DROP COLUMN", "DELETE FROM", "TRIM(",
                 "UPDATE", "CASE", "SELECT"):
        assert stmt in sql, "no {} step in the script".format(stmt)


def test_script_uses_the_table_name_it_is_given(messy_df):
    sql = _script(messy_df)
    assert '"orders"' in sql
    assert "{table}" not in sql, "an unsubstituted placeholder reached the output"
    # A doubled placeholder in a concatenated (rather than formatted) string
    # substituted to `DELETE FROM {"orders"}` — valid-looking, unrunnable.
    assert "{" not in sql and "}" not in sql, \
        "brace left in the emitted SQL: {!r}".format(
            [ln for ln in sql.splitlines() if "{" in ln or "}" in ln])


def test_nulls_are_not_counted_as_whitespace_fixes():
    """`s != s.str.strip()` is True for every NA, so a column of 12 nulls
    was reported as "12 values had leading/trailing spaces" — a figure the
    reader can check and find wrong."""
    df = pd.DataFrame({
        "region": [None] * 12 + ["North"] * 88,
        "keep": list(range(100)),
        "other": ["x", "y"] * 50,
    })
    _cleaned, report = auto_clean(df)
    ws = [a for a in report.actions
          if a.column == "region" and "spaces" in a.issue]
    assert not ws, "nulls were reported as whitespace: {!r}".format(
        [a.issue for a in ws])


def test_script_says_it_has_not_been_run(messy_df):
    """The reader must never assume this was executed against anything."""
    assert "has not been executed" in _script(messy_df).lower()


def test_clean_data_produces_no_misleading_script():
    """An empty script implying work was done would be a lie."""
    rng = np.random.default_rng(22)
    df = pd.DataFrame({
        "id": range(120),
        "value": rng.normal(10, 1, 120).round(3),
        "label": rng.choice(["x", "y"], 120),
    })
    _cleaned, report = auto_clean(df)
    sql = report.sql_script("t")
    if not any(a.sql for a in report.actions):
        assert "no transformations were required" in sql.lower()


# ══════════════════════════════════════════════════════════
#  The statements have to be right
# ══════════════════════════════════════════════════════════

def test_median_fill_states_the_value_actually_used(messy_df):
    """Emitting `SET amount = median(...)` would be a different statement
    from the one that ran — the fill value was computed once, before the
    nulls were replaced. It has to appear as a literal."""
    _cleaned, report = auto_clean(messy_df)
    fills = [a for a in report.actions
             if a.column == "amount" and "median" in a.action]
    assert fills, "amount was not median-filled"
    sql = fills[0].sql
    stmt = [ln for ln in sql.splitlines() if ln.strip().upper().startswith("UPDATE")]
    assert stmt, "median fill has no UPDATE statement"
    assert "WHERE" in stmt[0].upper() and "IS NULL" in stmt[0].upper(), \
        "median fill would overwrite non-null rows: {!r}".format(stmt[0])
    value = re.search(r"=\s*([0-9.eE+-]+)\s+WHERE", stmt[0])
    assert value, "no literal fill value in {!r}".format(stmt[0])
    reported = re.search(r"median \(([0-9.eE+-]+)\)", fills[0].action)
    assert reported, "the action text does not state the median it used"
    assert float(value.group(1)) == pytest.approx(float(reported.group(1)), rel=1e-3), \
        "the SQL fills a different value from the one pandas used"


def test_outlier_bounds_are_not_printed_with_float_noise(messy_df):
    """`WHERE "revenue" < 106.02249999999992` is binary representation
    leaking into a client-facing script. Rounding to 4 decimal places
    keeps the magnitude exact and cannot plausibly change which rows
    match."""
    _cleaned, report = auto_clean(messy_df)
    flags = [a for a in report.actions if "outlier" in a.issue.lower()]
    assert flags, "no outliers flagged"
    for a in flags:
        for number in re.findall(r"-?\d+\.(\d+)", a.sql):
            assert len(number) <= 4, \
                "float noise in the emitted bound: {!r}".format(a.sql)


def test_categorical_fill_targets_only_nulls(messy_df):
    _cleaned, report = auto_clean(messy_df)
    fills = [a for a in report.actions
             if a.column == "region" and "filled" in a.action]
    assert fills, "region was not filled"
    assert "IS NULL" in fills[0].sql.upper(), \
        "categorical fill would rewrite every row"


def test_whitespace_trim_touches_only_affected_rows(messy_df):
    _cleaned, report = auto_clean(messy_df)
    trims = [a for a in report.actions if "whitespace stripped" in a.action]
    assert trims, "no whitespace trim ran"
    sql = trims[0].sql.upper()
    assert "TRIM(" in sql and "WHERE" in sql


def test_outliers_are_flagged_by_a_select_never_deleted(messy_df):
    """The cleaner deliberately does not remove extreme values, so the SQL
    must not either — a DELETE here would destroy real data."""
    _cleaned, report = auto_clean(messy_df)
    flags = [a for a in report.actions if "outlier" in a.issue.lower()]
    assert flags, "no outliers flagged"
    for a in flags:
        assert a.sql.strip().upper().startswith("--") or \
            a.sql.strip().upper().startswith("SELECT")
        statements = [ln for ln in a.sql.splitlines()
                      if not ln.strip().startswith("--")]
        body = " ".join(statements).upper()
        assert "DELETE" not in body, "outlier step deletes rows"
        assert "UPDATE" not in body, "outlier step rewrites values"
        assert "SELECT" in body


def test_informative_missingness_sql_does_not_impute(leavers_frame):
    """The whole point of leaving it as NULL is that filling it destroys
    the finding. The SQL must not quietly do what pandas refused to."""
    _cleaned, report = auto_clean(leavers_frame)
    acts = [a for a in report.actions
            if a.column == "satisfaction" and "not missing at random" in a.issue]
    assert acts, "informative missingness was not detected"
    sql = acts[0].sql.upper()
    assert "UPDATE" not in sql, "SQL imputes a column pandas deliberately left NULL"
    assert "SELECT" in sql


@pytest.fixture()
def leavers_frame():
    rng = np.random.default_rng(23)
    n = 400
    left = rng.choice([True, False], n, p=[.3, .7])
    return pd.DataFrame({
        "satisfaction": np.where(left, np.nan, rng.uniform(3, 5, n)),
        "tenure_years": np.where(left, rng.uniform(0, 2, n),
                                 rng.uniform(4, 12, n)).round(1),
        "dept": rng.choice(["A", "B", "C"], n),
        "grade": rng.choice(["G1", "G2", "G3"], n),
        "hours": rng.normal(38, 4, n).round(1),
    })


def test_boolean_conversion_does_not_drop_the_source_column(messy_df):
    """Dropping the original in the same breath as the conversion leaves
    no way back if the mapping was wrong. The drop stays commented."""
    _cleaned, report = auto_clean(messy_df)
    conv = [a for a in report.actions if "converted to bool" in a.action]
    assert conv, "boolean-like column was not converted"
    for line in conv[0].sql.splitlines():
        if "DROP COLUMN" in line.upper():
            assert line.strip().startswith("--"), \
                "boolean conversion drops the source column outright"


def test_dedupe_sql_is_offered_but_marked_not_applied(messy_df):
    """Duplicates are reported, not deleted. The script still hands over
    the statement to remove them, clearly labelled as not run — removing
    rows is a decision about what one row means, and blanket deduplication
    of a transactional table deletes real turnover."""
    _cleaned, report = auto_clean(messy_df)
    dupes = [a for a in report.actions if "identical to another row" in a.issue]
    assert dupes, "duplicates were not reported at all"
    action = dupes[0]
    assert action.applied is False
    sql = action.sql
    assert "NOT APPLIED" in sql
    assert "HAVING COUNT(*) > 1" in sql, "no query to inspect the duplicates"
    assert "MIN(ctid)" in sql, "no statement offered to remove them"


def test_dedupe_sql_is_available_for_every_dialect(messy_df):
    """A client's data team works in their own warehouse. The script is
    only auditable if it runs there."""
    from app.engines.data_cleaner import DIALECTS, CleaningPolicy
    _cleaned, report = auto_clean(messy_df, CleaningPolicy.aggressive())
    for dialect in DIALECTS:
        script = report.sql_script("orders", dialect)
        assert script.strip(), f"{dialect} produced an empty script"
    assert "ctid" in report.sql_script("orders", "postgres")
    assert "QUALIFY" in report.sql_script("orders", "snowflake")
    assert "EXCEPT(_rn)" in report.sql_script("orders", "bigquery")


def test_backtick_dialects_requote_identifiers(messy_df):
    """MySQL and BigQuery do not accept double-quoted identifiers, and the
    column names in a client export routinely need quoting."""
    _cleaned, report = auto_clean(messy_df)
    for dialect in ("mysql", "bigquery"):
        script = report.sql_script("orders", dialect)
        assert "`orders`" in script, f"{dialect} did not requote the table"
        assert '"orders"' not in script, f"{dialect} left ANSI quoting in place"


def test_unknown_dialect_falls_back_to_portable_sql(messy_df):
    _cleaned, report = auto_clean(messy_df)
    assert report.sql_script("orders", "oracle_9i").strip()


# ══════════════════════════════════════════════════════════
#  Ordering, and the table name
# ══════════════════════════════════════════════════════════

def test_statements_appear_in_execution_order(messy_df):
    """Imputing before deduplicating gives a different table. The script
    is only reproducible if it preserves the order the steps ran in."""
    from app.engines.data_cleaner import CleaningPolicy
    _cleaned, report = auto_clean(messy_df, CleaningPolicy.aggressive())
    sql = report.sql_script("orders")
    dedupe_at = sql.find("DELETE FROM")
    fill_at = sql.find("IS NULL;")
    assert dedupe_at != -1 and fill_at != -1
    assert dedupe_at < fill_at, "imputation is emitted before deduplication"


# ══════════════════════════════════════════════════════════
#  The report has to show the preparation, not just the result
# ══════════════════════════════════════════════════════════

def _report_text(df) -> str:
    """Build the client PDF and return its text."""
    import io

    import pypdf

    from app.engines.data_cleaner import auto_clean, get_cleaning_summary
    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf_builder import build_pdf
    from app.engines.story_engine import detect_domain, generate_story

    cleaned, report = auto_clean(df)
    domain, _ = detect_domain(cleaned)
    story = generate_story(cleaned)
    pdf = build_pdf(
        df=cleaned,
        config={"title": "Q3 Review", "subtitle": "", "client_name": "Acme",
                "confidential": True, "theme_name": "", "logo_path": None,
                "source_table": "orders"},
        profile=profile_dataset(cleaned),
        cleaning_summary=get_cleaning_summary(report),
        stats_report=None, bi_report=None, ml_report=None, chart_data=[],
        executive_summary=story.executive_summary, findings=story.key_findings,
        risks=story.business_risks, opportunities=story.opportunities,
        recommendations=story.recommended_actions, top_insights=[],
        attrition=None, domain=domain,
    )
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def test_report_shows_what_was_changed_before_analysis(messy_df):
    """cleaning_summary was accepted by build_pdf and then never rendered,
    so the client saw figures with no account of how the file they sent
    became the table those figures came from."""
    text = _report_text(messy_df)
    assert "Data Preparation" in text, "no data preparation section in the report"
    assert "Equivalent SQL" in text, "the report shows no SQL for the cleaning"


def test_report_sql_survives_the_pdf_text_engine(messy_df):
    """`WHERE "amount" < 108` is a legal comparison and an illegal tag —
    unescaped, ReportLab drops the rest of the line."""
    text = _report_text(messy_df)
    assert "UPDATE" in text and "IS NULL" in text, \
        "SQL statements did not reach the page"
    assert "&lt;" not in text and "&amp;" not in text, \
        "escaping leaked into the rendered text"


def test_report_lists_steps_in_execution_order(messy_df):
    """The section tells the reader order matters, so it had better print
    them in the order they ran — the display grouping does not."""
    text = _report_text(messy_df)
    section = text[text.find("Data Preparation"):]
    # Empty and constant columns are dropped before deduplication, and the
    # display grouping puts duplicates first — so this ordering can only
    # hold if the section is using execution order.
    drop_at = section.find("100% empty")
    dedupe_at = section.find("identical to another row")
    fill_at = section.find("filled with median")
    flag_at = section.find("extreme outliers")
    assert min(drop_at, dedupe_at, fill_at, flag_at) > -1, \
        "a step is missing from the section entirely"
    assert drop_at < dedupe_at < fill_at < flag_at, \
        "steps are listed grouped by kind, not in the order they ran"


def test_long_sql_lines_are_wrapped_not_clipped():
    """A GROUP BY over many columns is one long line; unwrapped it runs
    off the page and the reader silently loses the tail."""
    from app.engines.pdf_builder import _SQL_COLS, _wrap_sql_line
    long_line = "  SELECT MIN(ctid) AS keep_ctid, " + ", ".join(
        '"column_number_{}"'.format(i) for i in range(30))
    wrapped = _wrap_sql_line(long_line)
    assert len(wrapped) > 1, "long line was not wrapped"
    assert all(len(ln) <= _SQL_COLS for ln in wrapped), \
        "a wrapped line is still over the page width"
    assert "".join(w.strip() for w in wrapped).replace(" ", "") == \
        long_line.replace(" ", ""), "wrapping lost or altered characters"
    assert _wrap_sql_line("SELECT 1;") == ["SELECT 1;"]


def test_report_names_the_table_the_sql_targets(messy_df):
    assert "orders" in _report_text(messy_df)


def test_report_states_the_sql_was_not_run(messy_df):
    """A client's DBA must not think this already touched their warehouse."""
    text = _report_text(messy_df).lower()
    assert "has been executed" in text or "not been executed" in text
    assert "none of it has been executed" in text


@pytest.mark.parametrize("filename,expected", [
    ("Q3 Sales Export (final).csv", "q3_sales_export_final"),
    ("employees.xlsx", "employees"),
    ("/tmp/uploads/hr-data_2024.CSV", "hr_data_2024"),
    ("2024report.csv", "t_2024report"),
    ("", "your_table"),
])
def test_table_name_is_derived_from_the_source_file(filename, expected):
    assert table_name_from_filename(filename) == expected
