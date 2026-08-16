"""
engines/readiness.py — is this data fit to analyse yet?

The app would happily run correlations, segmentation and a client PDF
over a file where the revenue column is text, half the rows are a
duplicated export, and the date column is unparsed strings. Every figure
that comes out is arithmetically correct and describes something other
than the client's business. Nothing anywhere said so.

This module answers one question before any analysis runs: **is the data
ready, and if not, exactly what has to happen first.** It draws a hard
line between:

  - **Blockers** — analysis run now would mislead. A numeric column
    stored as text is silently excluded from every statistic; duplicate
    identity keys double every sum. These are not warnings, they are
    reasons the output would be wrong.
  - **Advisories** — worth fixing, but the analysis is still valid.

It also records the governance facts a consultant has to know before
sending anything anywhere: what personal data is in the file, whether
values have been imputed, and how much of the file is actually
observed rather than filled in.

Deliberately not a cleaner. It reports; `data_cleaner` fixes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from app.engines.domains.base import is_id_column
from app.services.dtypes import is_text_dtype, text_columns

logger = logging.getLogger(__name__)

# A text column where this share of values parse as numbers is a number
# column that lost its type somewhere — an export with thousands
# separators, a currency symbol, or a stray "N/A".
NUMERIC_TEXT_SHARE = 0.80
DATE_TEXT_SHARE = 0.80
# Above this, a statistic is describing imputed values as much as
# observed ones.
HEAVY_MISSING_PCT = 40.0
MIN_ROWS_FOR_ANALYSIS = 30


@dataclass
class ReadinessIssue:
    column: str
    issue: str
    consequence: str      # what goes wrong in the analysis if this stands
    fix: str
    blocking: bool


@dataclass
class ReadinessReport:
    ready: bool = True
    rows: int = 0
    columns: int = 0
    issues: List[ReadinessIssue] = field(default_factory=list)
    personal_data_columns: List[str] = field(default_factory=list)
    observed_pct: float = 100.0
    summary: str = ""

    @property
    def blockers(self) -> List[ReadinessIssue]:
        return [i for i in self.issues if i.blocking]

    @property
    def advisories(self) -> List[ReadinessIssue]:
        return [i for i in self.issues if not i.blocking]


# ══════════════════════════════════════════════════════════
#  Personal data
# ══════════════════════════════════════════════════════════

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)
# Deliberately strict. A looser pattern matched "2024-01-01": a date
# stored as text is digits, dashes and the right length, and flagging
# every date column as a phone number would bury the disclosure that
# matters under noise nobody reads.
_PHONE_RE = re.compile(r"^\+?\d[\d\s()-]{7,17}$")
_DATEISH_RE = re.compile(r"^\d{4}-\d{2}-\d{2}|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")


def _looks_like_phone(sample: pd.Series) -> bool:
    """Phone numbers as they are actually written.

    Bare digit runs are deliberately not enough. "2024000012" is a
    ten-digit order number and a plausible phone number, and there is
    nothing in the value to separate them — so detection requires the
    formatting real contact data carries (+, spaces, dashes, parens).
    A phone column stored as bare digits is caught by its header
    instead, which is where that information actually lives.
    """
    candidates = sample[~sample.str.match(_DATEISH_RE, na=False)]
    if candidates.empty:
        return False
    matches = candidates.str.match(_PHONE_RE, na=False)
    # 8-15 digits is the international range for a subscriber number.
    digits = candidates.str.count(r"\d")
    formatted = candidates.str.contains(r"[+\s()-]", regex=True, na=False)
    plausible = matches & digits.between(8, 15) & formatted
    return float(plausible.mean()) > 0.5
_PII_NAME_HINTS = ("email", "e_mail", "mail", "phone", "mobile", "contact",
                   "address", "postcode", "zip", "passport", "aadhaar",
                   "aadhar", "ssn", "national_id", "nino", "pan_number",
                   "dob", "date_of_birth", "first_name", "last_name",
                   "full_name", "customer_name", "employee_name")


def find_personal_data(df: pd.DataFrame) -> List[str]:
    """Columns that look like they identify a person.

    Named by content where possible, not only by header — an "identifier"
    column full of email addresses is personal data whatever it is
    called. This drives a disclosure, not a block: it is the client's
    data and their decision, but a report that leaves the building with
    3,000 email addresses in an appendix is the kind of thing that ends
    an engagement.
    """
    found: List[str] = []
    for col in df.columns:
        name = re.sub(r"[^a-z0-9]+", "_", str(col).lower())
        if any(h in name for h in _PII_NAME_HINTS):
            found.append(str(col))
            continue
        if not is_text_dtype(df[col]):
            continue
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty:
            continue
        if (float(sample.str.match(_EMAIL_RE, na=False).mean()) > 0.5
                or _looks_like_phone(sample)):
            found.append(str(col))
    return found


# ══════════════════════════════════════════════════════════
#  Checks
# ══════════════════════════════════════════════════════════

def _numeric_share(s: pd.Series) -> float:
    sample = s.dropna().astype(str).str.strip()
    sample = sample[sample != ""]
    if sample.empty:
        return 0.0
    cleaned = (sample.str.replace(r"[,\s]", "", regex=True)
                     .str.replace(r"^[\$₹€£]", "", regex=True)
                     .str.replace(r"%$", "", regex=True))
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def _date_share(s: pd.Series) -> float:
    sample = s.dropna().astype(str).head(500)
    if sample.empty:
        return 0.0
    # A bare integer year or an id like "20240001" parses as a date and
    # would otherwise turn every numeric-looking id column into a
    # "date stored as text" finding.
    if sample.str.fullmatch(r"\d+").mean() > 0.5:
        return 0.0
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return 0.0
    return float(parsed.notna().mean())


def _check_types(df: pd.DataFrame, issues: List[ReadinessIssue]) -> None:
    for col in text_columns(df):
        s = df[col]
        if is_id_column(col, s):
            continue
        if _numeric_share(s) >= NUMERIC_TEXT_SHARE:
            issues.append(ReadinessIssue(
                column=str(col),
                issue="numbers are stored as text",
                consequence=(
                    "This column is excluded from every mean, correlation, "
                    "chart and model — the analysis proceeds without it and "
                    "says nothing about the omission."),
                fix=("Convert to a numeric type after stripping separators "
                     "and currency symbols (auto-clean does this)."),
                blocking=True))
        elif _date_share(s) >= DATE_TEXT_SHARE:
            issues.append(ReadinessIssue(
                column=str(col),
                issue="dates are stored as text",
                consequence=(
                    "No trend, seasonality or time-series analysis can run "
                    "on this column, and it sorts alphabetically — "
                    "'10 Jan' before '2 Feb'."),
                fix="Parse to a datetime type.",
                blocking=True))


def _check_missing(df: pd.DataFrame, issues: List[ReadinessIssue]) -> None:
    for col in df.columns:
        pct = float(df[col].isna().mean() * 100)
        if pct >= 100:
            issues.append(ReadinessIssue(
                column=str(col), issue="entirely empty",
                consequence="Carries no information and clutters every table.",
                fix="Drop the column.", blocking=False))
        elif pct >= HEAVY_MISSING_PCT:
            issues.append(ReadinessIssue(
                column=str(col),
                issue=f"{pct:.0f}% missing",
                consequence=(
                    "Any average or correlation using this column is "
                    "computed on a minority of the rows, and if it is "
                    "imputed the result is largely invented."),
                fix=("Source the missing values, or accept the column as "
                     "descriptive only and keep it out of conclusions."),
                blocking=True))


def _check_duplicates(df: pd.DataFrame, issues: List[ReadinessIssue]) -> None:
    exact = int(df.duplicated().sum())
    if exact:
        issues.append(ReadinessIssue(
            column="(all columns)",
            issue=f"{exact:,} duplicate rows ({exact / max(len(df), 1) * 100:.1f}%)",
            consequence=(
                "Every count, sum and average is inflated by the repeated "
                "rows, and a segment that happens to be duplicated looks "
                "larger than it is."),
            fix="Remove exact duplicates (auto-clean does this).",
            blocking=True))

    for col in df.columns:
        try:
            if not is_id_column(col, df[col]):
                continue
            non_null = df[col].dropna()
            if len(non_null) < 10:
                continue
            repeats = int(non_null.duplicated().sum())
            if repeats:
                issues.append(ReadinessIssue(
                    column=str(col),
                    issue=f"identifier repeats for {repeats:,} row(s)",
                    consequence=(
                        "One entity appears more than once, so it is counted "
                        "more than once in every total. Exact-duplicate "
                        "removal does not catch this because the other "
                        "columns differ."),
                    fix=("Confirm whether these are genuine repeat events or "
                         "an unresolved join, and aggregate to one row per "
                         "entity if they are not."),
                    blocking=True))
        except Exception:
            logger.debug("duplicate-key check failed for %s", col, exc_info=True)


def _check_shape(df: pd.DataFrame, issues: List[ReadinessIssue]) -> None:
    if len(df) < MIN_ROWS_FOR_ANALYSIS:
        issues.append(ReadinessIssue(
            column="(dataset)",
            issue=f"only {len(df)} rows",
            consequence=(
                f"Below about {MIN_ROWS_FOR_ANALYSIS} rows, a difference "
                "between groups cannot be distinguished from chance, so any "
                "finding is a description of these rows rather than "
                "evidence about the business."),
            fix="Analyse a fuller extract, or read the figures as indicative.",
            blocking=True))

    # An all-empty column is already reported by the missing-data check;
    # listing it again as "holds a single value" is the same defect twice.
    constant = [str(c) for c in df.columns
                if len(df) > 1 and df[c].nunique(dropna=True) <= 1
                and not df[c].isna().all()]
    if constant:
        issues.append(ReadinessIssue(
            column=", ".join(constant[:6]),
            issue=f"{len(constant)} column(s) hold a single value",
            consequence=(
                "They cannot explain any difference and will appear in "
                "tables as though they might."),
            fix="Drop them, or confirm the extract was filtered as intended.",
            blocking=False))

    numeric = df.select_dtypes(include="number").columns
    measures = [c for c in numeric if not is_id_column(c, df[c])]
    if not measures:
        issues.append(ReadinessIssue(
            column="(dataset)",
            issue="no measurable quantity",
            consequence=(
                "There is nothing to average, trend or compare — only "
                "labels and identifiers. Analysis can count rows and "
                "nothing else."),
            fix=("Include the value column (revenue, quantity, score, "
                 "duration) this data is about."),
            blocking=True))


def _check_mixed_types(df: pd.DataFrame, issues: List[ReadinessIssue]) -> None:
    for col in text_columns(df):
        sample = df[col].dropna().head(500)
        if len(sample) < 20:
            continue
        share = _numeric_share(sample)
        if 0.25 < share < NUMERIC_TEXT_SHARE:
            issues.append(ReadinessIssue(
                column=str(col),
                issue=f"mixed numbers and text ({share * 100:.0f}% numeric)",
                consequence=(
                    "Sorting, grouping and any threshold applied to this "
                    "column treat '1,200' and 'not recorded' as two labels "
                    "of equal standing."),
                fix=("Separate the real values from the placeholder text, "
                     "then convert."),
                blocking=False))


# ══════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════

def assess_readiness(df: pd.DataFrame) -> ReadinessReport:
    """Everything that has to be true before analysis means anything."""
    report = ReadinessReport(rows=len(df), columns=int(df.shape[1]))
    if df.empty:
        report.ready = False
        report.issues.append(ReadinessIssue(
            column="(dataset)", issue="no rows",
            consequence="There is nothing to analyse.",
            fix="Upload a file with data in it.", blocking=True))
        report.summary = "This dataset is empty."
        return report

    from app.services.dtypes import dedupe_columns

    df = dedupe_columns(df)

    issues: List[ReadinessIssue] = []
    failed: List[str] = []
    for check in (_check_shape, _check_types, _check_missing,
                  _check_duplicates, _check_mixed_types):
        try:
            check(df, issues)
        except Exception:
            # A check that raised was silently dropped, and the report
            # still came back saying the data was ready — a verdict
            # reached by not running the tests that would have
            # contradicted it. If a check cannot run, the reader is told,
            # and "ready" is withheld.
            logger.warning("readiness check %s failed", check.__name__,
                           exc_info=True)
            failed.append(check.__name__.lstrip("_").replace("_", " "))

    if failed:
        issues.append(ReadinessIssue(
            column="(whole dataset)",
            issue="{} readiness check(s) could not run: {}".format(
                len(failed), ", ".join(failed)),
            consequence="The checks those cover have not been made, so "
                        "nothing here rules out the problems they look "
                        "for. A pass from the remaining checks is not a "
                        "pass overall.",
            fix="Re-upload the file, or send it as CSV — this usually "
                "means a column the parser could not read consistently.",
            blocking=True,
        ))

    report.issues = issues
    report.ready = not any(i.blocking for i in issues)
    report.personal_data_columns = find_personal_data(df)
    total_cells = max(int(df.size), 1)
    report.observed_pct = float((1 - df.isna().sum().sum() / total_cells) * 100)
    report.summary = _summarise(report)
    return report


def _summarise(report: ReadinessReport) -> str:
    n_block = len(report.blockers)
    if report.ready:
        base = (
            "This dataset is ready to analyse: {:,} rows across {} columns, "
            "{:.1f}% of cells populated.".format(
                report.rows, report.columns, report.observed_pct))
        if report.advisories:
            base += (" {} point(s) are worth tidying but do not affect the "
                     "validity of the analysis.".format(len(report.advisories)))
    else:
        first = report.blockers[0]
        base = (
            "This dataset is not ready to analyse. {} issue(s) would make the "
            "output wrong rather than merely untidy — starting with '{}': {}"
            .format(n_block, first.column, first.issue))
    if report.personal_data_columns:
        base += (" It contains personal data in {}: {}. Confirm the client "
                 "has agreed to it being processed and remove it from "
                 "anything shared.".format(
                     "{} column(s)".format(len(report.personal_data_columns)),
                     ", ".join(report.personal_data_columns[:5])))
    return base


def readiness_payload(report: ReadinessReport) -> dict:
    """JSON-friendly shape for the API and the report builders."""
    def _issue(i: ReadinessIssue) -> dict:
        return {"column": i.column, "issue": i.issue,
                "consequence": i.consequence, "fix": i.fix,
                "blocking": i.blocking}

    return {
        "ready": report.ready,
        "rows": report.rows,
        "columns": report.columns,
        "observed_pct": round(report.observed_pct, 2),
        "summary": report.summary,
        "blockers": [_issue(i) for i in report.blockers],
        "advisories": [_issue(i) for i in report.advisories],
        "personal_data_columns": report.personal_data_columns,
    }
