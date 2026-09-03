"""
engines/governance.py — what this data is, where it came from, what is
sensitive in it, and who it could identify.

An analysis report says what the data means. A governance record says
what the data *is*, and it is the half a client's legal, security or
compliance function asks for — usually after the analysis has already
been circulated.

The part that is genuinely hard, and that almost nothing computes, is
re-identification risk. Removing names does not anonymise a dataset: a
postcode, a date of birth and a gender identify most people in a
population on their own. So this measures k-anonymity over the
quasi-identifiers it can find — how many rows share each combination —
and reports the share of people who are unique or near-unique, which is
the number that decides whether a file can be shared.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from app.engines.present import count as _C, label as _L
from app.services.dtypes import is_categorical_like, is_text_dtype

logger = logging.getLogger(__name__)


# ── classification ───────────────────────────────────────

# Fields that identify a person outright. Removing these is the first
# thing anyone does and the least of what is needed.
DIRECT_IDENTIFIERS = (
    "email", "e_mail", "phone", "mobile", "telephone", "passport",
    "aadhaar", "aadhar", "ssn", "national_id", "nino", "pan_number",
    "account_number", "card_number", "iban", "licence", "license_number",
    "full_name", "first_name", "last_name", "surname", "customer_name",
    "employee_name", "contact_name", "address", "street",
)

# Fields that identify nobody alone and most people in combination. This
# is the set k-anonymity is measured over.
QUASI_IDENTIFIERS = (
    "postcode", "postal_code", "zip", "zipcode", "pincode", "city",
    "region", "district", "birth", "dob", "age", "gender", "sex",
    "marital", "nationality", "job_title", "jobrole", "job_role",
    "department", "hire_date", "start_date", "joining_date",
)

# GDPR Article 9 and equivalents: processing these needs a lawful basis
# beyond legitimate interest, and they must never be a chart dimension.
SPECIAL_CATEGORY = (
    "race", "ethnic", "religion", "religious", "political", "union",
    "trade_union", "health", "medical", "diagnosis", "disability",
    "biometric", "genetic", "sexual", "orientation", "criminal",
    "conviction", "pregnan",
)

# Below this a combination of quasi-identifiers is treated as
# identifying. k=5 is the conventional floor for releasing data; k<5 is
# where a person becomes findable.
K_ANONYMITY_FLOOR = 5


@dataclass
class ColumnRecord:
    """One row of the data dictionary."""
    name: str
    label: str
    dtype: str
    role: str                 # identifier | measure | dimension | outcome | date
    sensitivity: str          # none | quasi-identifier | direct | special
    completeness_pct: float
    distinct: int
    example: str = ""
    note: str = ""


@dataclass
class ReidentificationRisk:
    quasi_identifiers: List[str]
    k_min: int                       # smallest group size
    unique_rows: int                 # rows in a group of one
    unique_pct: float
    below_floor_rows: int            # rows in a group smaller than the floor
    below_floor_pct: float
    verdict: str
    explanation: str


@dataclass
class GovernanceRecord:
    source_file: str = ""
    ingested_at: str = ""
    rows: int = 0
    columns: int = 0
    retention_days: Optional[int] = None
    retention_note: str = ""
    dictionary: List[ColumnRecord] = field(default_factory=list)
    direct_identifiers: List[str] = field(default_factory=list)
    quasi_identifiers: List[str] = field(default_factory=list)
    special_category: List[str] = field(default_factory=list)
    reidentification: Optional[ReidentificationRisk] = None
    lineage: List[str] = field(default_factory=list)
    obligations: List[str] = field(default_factory=list)


def _normalised(name) -> str:
    return re.sub(r"[^a-z0-9]+", "_",
                  re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name)).lower())


def classify_column(name, series=None) -> str:
    """The sensitivity class of one column."""
    flat = _normalised(name)
    if any(token in flat for token in SPECIAL_CATEGORY):
        return "special"
    if any(token in flat for token in DIRECT_IDENTIFIERS):
        return "direct"
    if any(token in flat for token in QUASI_IDENTIFIERS):
        return "quasi-identifier"
    # Content beats the header: a column called "reference" full of email
    # addresses is personal data whatever it is named.
    if series is not None and is_text_dtype(series):
        try:
            sample = series.dropna().astype(str).head(200)
            if not sample.empty and float(
                    sample.str.contains(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$",
                                        regex=True, na=False).mean()) > 0.5:
                return "direct"
        except Exception:
            logger.debug("content classification failed for %r", name,
                         exc_info=True)
    return "none"


def _role_of(name, series, target: Optional[str]) -> str:
    from app.engines.domains.base import is_id_column
    if target is not None and str(name) == str(target):
        return "outcome"
    if series is not None and pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if is_id_column(name, series):
        return "identifier"
    if series is not None and pd.api.types.is_numeric_dtype(series) \
            and not is_categorical_like(series):
        return "measure"
    return "dimension"


def build_dictionary(df: pd.DataFrame,
                     target: Optional[str] = None) -> List[ColumnRecord]:
    """A row per column: what it is, how complete, and how sensitive.

    The thing a client's data owner asks for and that nobody writes,
    because writing it by hand for forty columns is a day's work that
    goes stale the moment the schema changes.
    """
    records: List[ColumnRecord] = []
    n = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        present = int(series.notna().sum())
        example = ""
        try:
            non_null = series.dropna()
            if len(non_null):
                example = str(non_null.iloc[0])[:40]
        except Exception:
            logger.debug("example extraction failed for %r", col,
                         exc_info=True)
        sensitivity = classify_column(col, series)
        # An example value from a personal-data column does not belong in
        # a governance document that will itself be circulated.
        if sensitivity in ("direct", "special"):
            example = "(withheld)"
        records.append(ColumnRecord(
            name=str(col), label=_L(col), dtype=str(series.dtype),
            role=_role_of(col, series, target),
            sensitivity=sensitivity,
            completeness_pct=round(present / n * 100, 1),
            distinct=int(series.nunique(dropna=True)),
            example=example,
        ))
    return records


def assess_reidentification(df: pd.DataFrame,
                            quasi: Optional[List[str]] = None
                            ) -> Optional[ReidentificationRisk]:
    """How many people this file identifies, once the names are removed.

    Deleting the name column is what most teams mean by anonymising, and
    it is not anonymising. A postcode, a date of birth and a gender pick
    out most individuals in a population; so does department plus job
    title plus tenure in a company of any size. k-anonymity counts how
    many rows share each combination — if the smallest group is one, that
    person is identifiable to anyone holding the same three facts about
    them.
    """
    if quasi is None:
        quasi = [c for c in df.columns
                 if classify_column(c, df[c]) == "quasi-identifier"]
    quasi = [c for c in quasi if c in df.columns][:6]
    if len(quasi) < 2 or df.empty:
        return None
    try:
        # A continuous quasi-identifier is banded first: exact ages make
        # everyone unique and say nothing about real risk.
        work = pd.DataFrame(index=df.index)
        for col in quasi:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 20:
                work[col] = pd.qcut(s, q=10, duplicates="drop").astype(str)
            elif pd.api.types.is_datetime64_any_dtype(s):
                work[col] = s.dt.to_period("M").astype(str)
            else:
                work[col] = s.astype(str)

        sizes = work.groupby(list(quasi), observed=True, dropna=False).size()
        if sizes.empty:
            return None
        row_group_size = work.merge(
            sizes.rename("k").reset_index(), on=list(quasi), how="left")["k"]

        k_min = int(sizes.min())
        unique_rows = int((row_group_size == 1).sum())
        below = int((row_group_size < K_ANONYMITY_FLOOR).sum())
        n = len(df)
        unique_pct = round(unique_rows / n * 100, 1)
        below_pct = round(below / n * 100, 1)

        if k_min >= K_ANONYMITY_FLOOR:
            verdict = "Low"
            explanation = (
                "Every combination of {} is shared by at least {} records, "
                "so no individual is singled out by those fields alone."
                .format(_join(quasi), k_min))
        elif unique_pct >= 20:
            verdict = "High"
            explanation = (
                "{} of {} records ({}%) are unique on {} — those people are "
                "identifiable to anyone who knows those facts about them, "
                "whether or not a name column is present. Removing names "
                "does not address this; banding or suppressing the fields "
                "above does."
                .format(_C(unique_rows), _C(n), unique_pct, _join(quasi)))
        else:
            verdict = "Moderate"
            explanation = (
                "{} of {} records ({}%) sit in a group smaller than {} on "
                "{}, and {} are unique. Those rows are re-identifiable by "
                "someone holding the same facts; the rest are not."
                .format(_C(below), _C(n), below_pct, K_ANONYMITY_FLOOR,
                        _join(quasi), _C(unique_rows)))
        return ReidentificationRisk(
            quasi_identifiers=list(quasi), k_min=k_min,
            unique_rows=unique_rows, unique_pct=unique_pct,
            below_floor_rows=below, below_floor_pct=below_pct,
            verdict=verdict, explanation=explanation)
    except Exception:
        logger.warning("re-identification assessment failed", exc_info=True)
        return None


def _join(cols) -> str:
    from app.engines import present
    return present.join_and([_L(c) for c in cols], limit=4)


def build_governance(df: pd.DataFrame, meta=None,
                     cleaning_summary=None,
                     target: Optional[str] = None,
                     retention_days: Optional[int] = None) -> GovernanceRecord:
    """The whole governance record for one dataset."""
    record = GovernanceRecord(rows=len(df), columns=df.shape[1])

    if meta is not None:
        record.source_file = str(getattr(meta, "filename", "") or "")
        uploaded = getattr(meta, "uploaded_at", None)
        if uploaded:
            try:
                from datetime import datetime, timezone
                record.ingested_at = datetime.fromtimestamp(
                    float(uploaded), tz=timezone.utc).strftime(
                        "%Y-%m-%d %H:%M UTC")
            except Exception:
                logger.debug("ingest timestamp unreadable", exc_info=True)

    record.dictionary = build_dictionary(df, target=target)
    record.direct_identifiers = [c.name for c in record.dictionary
                                 if c.sensitivity == "direct"]
    record.quasi_identifiers = [c.name for c in record.dictionary
                                if c.sensitivity == "quasi-identifier"]
    record.special_category = [c.name for c in record.dictionary
                               if c.sensitivity == "special"]
    record.reidentification = assess_reidentification(
        df, record.quasi_identifiers)

    # Lineage: what happened to the data between the file and this report.
    record.lineage.append(
        "Loaded {} rows and {} columns{}.".format(
            _C(len(df)), df.shape[1],
            " from " + record.source_file if record.source_file else ""))
    if cleaning_summary:
        for step in (getattr(cleaning_summary, "steps", None)
                     or (cleaning_summary.get("steps")
                         if isinstance(cleaning_summary, dict) else []) or []):
            record.lineage.append(str(step))
    record.lineage.append(
        "No values were estimated into the findings; every figure in the "
        "report is computed from the rows above.")

    if retention_days:
        record.retention_days = int(retention_days)
        record.retention_note = (
            "This dataset and everything derived from it are deleted "
            "automatically {} days after upload. Nothing is retained beyond "
            "that without a new upload.".format(int(retention_days)))

    record.obligations = _obligations(record)
    return record


def _obligations(record: GovernanceRecord) -> List[str]:
    """What the classification actually requires of the holder.

    Stated as the specific consequence, not as a citation. "Contains
    special category data" tells a reader nothing they can act on.
    """
    out: List[str] = []
    if record.special_category:
        out.append(
            "{} {} special category data under GDPR Article 9 (and its "
            "equivalents elsewhere). Processing needs a lawful basis beyond "
            "legitimate interest, and these fields must not be used as a "
            "reporting dimension or a model feature without one."
            .format(_join(record.special_category),
                    "is" if len(record.special_category) == 1 else "are"))
    if record.direct_identifiers:
        out.append(
            "{} {} an individual outright. Remove or hash {} before this "
            "file leaves the analysis environment; nothing in the report "
            "requires {}."
            .format(_join(record.direct_identifiers),
                    "identifies" if len(record.direct_identifiers) == 1
                    else "identify",
                    "it" if len(record.direct_identifiers) == 1 else "them",
                    "it" if len(record.direct_identifiers) == 1 else "them"))
    risk = record.reidentification
    if risk is not None and risk.verdict in ("High", "Moderate"):
        out.append(
            "Removing the name columns would not anonymise this file: "
            + risk.explanation)
    if not out:
        out.append(
            "No column identifies an individual, and no combination of "
            "columns singles one out. This dataset can be shared as it "
            "stands.")
    return out
