"""
core/data_loader.py
===================
Production-grade file loader.
Handles ALL dirty data — original files, not just clean ones.
Supports: CSV, Excel (multi-sheet), JSON up to 200MB.
NO Streamlit imports. Always returns LoadResult.
"""
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


from app.services.dtypes import text_columns


@dataclass
class LoadResult:
    df:             Optional[pd.DataFrame]
    success:        bool
    error:          Optional[str]  = None
    file_size_mb:   float          = 0.0
    sheet_names:    List[str]      = field(default_factory=list)
    row_count:      int            = 0
    col_count:      int            = 0
    filename:       str            = ""
    warnings:       List[str]      = field(default_factory=list)
    was_sampled:    bool           = False
    original_rows:  int            = 0


MAX_FILE_MB       = 200
MAX_ROWS_FULL     = 500_000   # keep full data up to 500k rows

# The hard stop while READING. data_validator rejects anything past its
# own MAX_ROWS afterwards; this exists so the frame that gets rejected
# was never built at full size in the first place. One row of headroom
# so "exactly at the limit" is still accepted and "one over" is still
# detectable.
ROW_CEILING       = 1_000_000
SAMPLE_THRESHOLD  = 100_000   # sample for heavy analysis above this


def load_file(uploaded_file, sheet_name=0) -> LoadResult:
    """
    Load any CSV/Excel/JSON — dirty or clean.
    Never crops data. Returns full dataset with warnings.
    """
    warnings_list = []

    try:
        # ── Size check ─────────────────────────────────────
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            return LoadResult(
                df=None, success=False,
                error="File is {:.1f} MB. Maximum allowed is {} MB.".format(
                    size_mb, MAX_FILE_MB))

        fname  = uploaded_file.name.lower()
        sheets = []
        df     = None

        # ── Read by type ───────────────────────────────────
        if fname.endswith(".csv"):
            df = _load_csv(uploaded_file, warnings_list)

        elif fname.endswith((".xlsx", ".xls")):
            df, sheets = _load_excel(uploaded_file, sheet_name, warnings_list)

        elif fname.endswith(".json"):
            df = _load_json(uploaded_file, warnings_list)

        else:
            return LoadResult(df=None, success=False,
                error="Unsupported format. Upload CSV, Excel (.xlsx/.xls), or JSON.")

        if df is None or len(df) == 0:
            return LoadResult(df=None, success=False,
                error="File is empty or could not be parsed.")

        # ── Clean column names ─────────────────────────────
        df, col_warnings = _clean_columns(df)
        warnings_list.extend(col_warnings)

        # ── Basic sanitization (keep original data) ────────
        df = _sanitize(df, warnings_list)

        # ── Warn if very large ─────────────────────────────
        original_rows = len(df)
        was_sampled   = False
        if original_rows > SAMPLE_THRESHOLD:
            warnings_list.append(
                "Dataset has {:,} rows. Heavy analysis operations will use "
                "a representative sample of {:,} rows for performance.".format(
                    original_rows, SAMPLE_THRESHOLD))

        return LoadResult(
            df=df, success=True,
            file_size_mb=round(size_mb, 2),
            sheet_names=[str(s) for s in sheets],
            row_count=len(df),
            col_count=len(df.columns),
            filename=uploaded_file.name,
            warnings=warnings_list,
            was_sampled=was_sampled,
            original_rows=original_rows,
        )

    except Exception as e:
        return LoadResult(df=None, success=False,
            error="Failed to load file: {}".format(str(e)))


# ══════════════════════════════════════════════════════════
#  CSV LOADER — handles encoding, separators, dirty headers
# ══════════════════════════════════════════════════════════

# pandas types "00123" as the integer 123 while parsing, and the zeros
# are gone before any of our code sees the frame. That is silent data
# loss on exactly the columns a business cares about identifying rows
# by: order numbers, SKUs, cost centres, and US zip codes, where 01234
# is Massachusetts and 1234 is nowhere. The only place to catch it is at
# read time, so the file is sampled as text first and any column whose
# values carry leading zeros is read back as text.
def _leading_zero_columns(f, enc: str, sep: str) -> list:
    """Columns whose values would lose a leading zero to type inference."""
    import re as _re
    try:
        f.seek(0)
        head = pd.read_csv(f, encoding=enc, sep=sep, dtype=str,
                           nrows=200, on_bad_lines="skip",
                           encoding_errors="replace")
    except Exception:
        logger.debug("leading-zero probe failed", exc_info=True)
        return []

    zero_pat = _re.compile(r"^0\d+$")
    keep = []
    for col in head.columns:
        vals = head[col].dropna().astype(str).str.strip()
        vals = vals[vals != ""]
        if len(vals) == 0:
            continue
        # A single "0" is the number zero; "007" is an identifier.
        if any(zero_pat.match(v) for v in vals):
            keep.append(col)
    return keep


def _preserve_leading_zeros(f, df: pd.DataFrame, enc: str, sep: str,
                            warnings: list) -> pd.DataFrame:
    """Re-read the affected columns as text, keeping their zeros."""
    cols = _leading_zero_columns(f, enc, sep)
    cols = [c for c in cols if c in df.columns
            and not pd.api.types.is_object_dtype(df[c])]
    if not cols:
        return df
    try:
        f.seek(0)
        fixed = pd.read_csv(f, encoding=enc, sep=sep, low_memory=False,
                            on_bad_lines="warn", encoding_errors="replace",
                            dtype={c: str for c in cols})
        for c in cols:
            if c in fixed.columns and len(fixed) == len(df):
                df[c] = fixed[c]
        # Inference runs later and would convert "01234" straight back
        # to 1234, undoing the re-read. It needs to know these columns
        # were deliberately kept as text.
        df.attrs["_leading_zero_cols"] = sorted(
            set(df.attrs.get("_leading_zero_cols", [])) | set(cols))
        warnings.append(
            "Kept {} as text to preserve leading zeros — read as numbers "
            "they would have become {} instead.".format(
                ", ".join("'{}'".format(c) for c in cols[:6]),
                "different values" if len(cols) > 1 else "a different value"))
    except Exception:
        logger.warning("could not re-read leading-zero columns %s",
                       cols, exc_info=True)
    return df


def _load_csv(f, warnings: list) -> Optional[pd.DataFrame]:
    """Load CSV — tries multiple encodings and separators."""
    encodings  = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    separators = [",", ";", "\t", "|"]

    for enc in encodings:
        for sep in separators:
            try:
                f.seek(0)
                df = pd.read_csv(
                    f, encoding=enc, sep=sep,
                    low_memory=False,
                    on_bad_lines="warn",
                    encoding_errors="replace",
                    # One row past the limit is enough to know the file
                    # is over it, and stops pandas building the rest.
                    # The row check used to run on the finished frame,
                    # which is far too late: a 79 MB CSV took the
                    # process from 203 MB to 1,704 MB of RSS and was
                    # THEN rejected for having too many rows. Reading
                    # the file is what has to be bounded, not the
                    # verdict on it.
                    nrows=ROW_CEILING + 1,
                )
                if df.shape[1] >= 2:  # at least 2 columns = real CSV
                    if enc != "utf-8":
                        warnings.append("Encoding detected: {}".format(enc))
                    if sep != ",":
                        warnings.append("Separator detected: '{}'".format(sep))
                    df = _preserve_leading_zeros(f, df, enc, sep, warnings)
                    return df
            except Exception:
                logger.debug("_load_csv: suppressed exception", exc_info=True)
                continue

    # Last resort — no separator detection
    f.seek(0)
    return pd.read_csv(f, encoding_errors="replace", low_memory=False,
                       nrows=ROW_CEILING + 1)


# ══════════════════════════════════════════════════════════
#  EXCEL LOADER
# ══════════════════════════════════════════════════════════

def _load_excel(f, sheet_name, warnings: list):
    """Load Excel — returns (df, sheet_names)."""
    try:
        xl     = pd.ExcelFile(f)
        sheets = xl.sheet_names

        # Validate sheet_name
        if isinstance(sheet_name, str) and sheet_name not in sheets:
            sheet_name = 0
        elif isinstance(sheet_name, int) and sheet_name >= len(sheets):
            sheet_name = 0

        df = xl.parse(sheet_name, na_values=["", "NA", "N/A", "null",
                                               "NULL", "None", "none", "#N/A"])
        return df, sheets
    except Exception as e:
        raise Exception("Excel read error: {}".format(str(e)))


# ══════════════════════════════════════════════════════════
#  JSON LOADER
# ══════════════════════════════════════════════════════════

def _load_json(f, warnings: list) -> Optional[pd.DataFrame]:
    """Load JSON — handles records, list, and nested formats."""
    import json
    try:
        f.seek(0)
        data = json.load(f)

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try records format first
            if any(isinstance(v, list) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        else:
            df = pd.read_json(f)

        return df
    except Exception:
        try:
            f.seek(0)
            return pd.read_json(f)
        except Exception:
            return None


# ══════════════════════════════════════════════════════════
#  COLUMN CLEANING
# ══════════════════════════════════════════════════════════

def _clean_columns(df: pd.DataFrame):
    """
    Clean column names — strip whitespace, remove blank names.
    Does NOT rename or modify column values.
    """
    warnings = []
    original = list(df.columns)

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Find blank/unnamed columns
    blank_cols = [c for c in df.columns
                  if not c or c.startswith("Unnamed:") or c.strip() == ""]
    if blank_cols:
        warnings.append(
            "{} column(s) have blank, unnamed, or whitespace-padded names "
            "— will be cleaned.".format(len(blank_cols)))
        # Rename blank columns
        new_cols = list(df.columns)
        counter  = 1
        for i, c in enumerate(new_cols):
            if not c or c.startswith("Unnamed:") or c.strip() == "":
                new_cols[i] = "Column_{}".format(counter)
                counter += 1
        df.columns = new_cols

    return df, warnings


# ══════════════════════════════════════════════════════════
#  SANITIZATION — keep original data, just make it safe
# ══════════════════════════════════════════════════════════

def _sanitize(df: pd.DataFrame, warnings: list) -> pd.DataFrame:
    """
    Safe sanitization — never drops user data.
    Only: replace inf values, improve dtypes where safe.
    """
    # Replace inf/-inf with NaN in numeric columns
    try:
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            inf_count = np.isinf(df[num_cols].values).sum()
            if inf_count > 0:
                df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
                warnings.append(
                    "{:,} infinite values replaced with blank.".format(int(inf_count)))
    except Exception:
        logger.debug("_sanitize: suppressed exception", exc_info=True)

    # Try to improve dtypes for object columns (safe — won't break values)
    df = _smart_dtype_inference(df, warnings)

    return df


# ── Money, percentages and accounting negatives ───────────
# Real client exports do not hand you clean floats. A sales extract
# carries "$1,200", a discount column carries "15%", a finance export
# writes a loss as "(500)", and a summary sheet writes "1.2k". pandas
# reads every one of those as text.
#
# Measured on a five-column sample of exactly that shape, Analytiq found
# ONE numeric column — `units` — and silently skipped revenue, discount
# and margin. Not a crash: every KPI, chart, correlation, driver and
# forecast simply had nothing to work on, and the client got a report
# about the one column that happened to be typed cleanly. That is the
# kind of defect that loses an account without ever raising an error.
# Single-character currency symbols. "R$" and "US$" are NOT here: put a
# multi-character prefix into a character class and the class quietly
# gains a bare "R", which then matches "R1" — and a Region column of
# R1/R2/R3 was read as the numbers 1, 2, 3, destroying the categories
# the whole analysis groups by. Multi-character prefixes are stripped as
# strings, below.
_CURRENCY_CHARS = "₹$€£¥₩₽฿₺"
_CURRENCY_PREFIXES = ("R$", "US$", "A$", "C$", "NZ$", "HK$", "S$", "NT$")

# A trailing k/m/b multiplier, as written in summary exports.
_SUFFIX_MULTIPLIER = {"k": 1e3, "m": 1e6, "b": 1e9}


# Which character is the decimal point. "€1.850" is 1.85 in a US-style
# file and 1,850 in a German one, and getting it wrong understates a
# revenue figure by a thousand times — the worst class of error this
# product can make, because the number still looks plausible. So the
# question is settled per COLUMN, from the evidence in the column, and
# where the evidence does not settle it the column is left as text and
# the user is told. dataforge-ai's version stripped "," unconditionally
# and turned "€1.850" into 1.85 with no warning.
def _decimal_style(sample) -> str:
    """"us" (1,234.56), "eu" (1.234,56), or "unknown"."""
    import re as _re
    digits = [_re.sub(r"[^\d.,]", "", str(v)) for v in sample]
    digits = [d for d in digits if d]
    if not digits:
        return "unknown"

    # A value carrying both separators settles it outright: whichever
    # comes last is the decimal point.
    for d in digits:
        if "." in d and "," in d:
            return "us" if d.rindex(".") > d.rindex(",") else "eu"

    # Two or more groups of exactly three ("1.234.567") can only be
    # thousands separators.
    for d in digits:
        if len(_re.findall(r"\.\d{3}(?!\d)", d)) >= 2:
            return "eu"
        if len(_re.findall(r",\d{3}(?!\d)", d)) >= 2:
            return "us"

    has_dot = any("." in d for d in digits)
    has_com = any("," in d for d in digits)

    # A separator followed by anything other than exactly three digits is
    # a decimal point — "12.5" and "12,5" are both one and a half dozen.
    if has_dot and any(_re.search(r"\.\d{1,2}(?!\d)|\.\d{4,}", d) for d in digits):
        return "us"
    if has_com and any(_re.search(r",\d{1,2}(?!\d)|,\d{4,}", d) for d in digits):
        return "eu"

    # Only one separator, always followed by exactly three digits, and
    # never twice in one value: "1.850" could be either. Refuse.
    if has_dot and not has_com:
        return "unknown"
    if has_com and not has_dot:
        return "us"          # "1,850" is not a European decimal
    return "us"


def _coerce_numeric_strings(series: pd.Series, style: str = "us") -> pd.Series:
    """Parse the numeric columns that arrive as text.

        "$1,200" -> 1200      "15%"   -> 15        "1.2k"  -> 1200
        "₹3,499" -> 3499      "(500)" -> -500      "3.4M"  -> 3400000

    A percentage keeps the number as written — "15%" becomes 15, not
    0.15 — because that is the figure the column is named for and the
    one the client reads in their own spreadsheet. Anything unparseable
    becomes NaN, and the caller decides whether enough of the column
    converted to be worth trusting.
    """
    import re as _re

    s = series.astype(str).str.strip()

    # Accounting negatives: (500) is a loss of 500, not a footnote.
    negated = s.str.match(r"^\(.*\)$", na=False)
    s = s.str.replace(r"^\((.*)\)$", r"\1", regex=True)

    # A trailing k/m/b, before it gets stripped with everything else.
    mult = pd.Series(1.0, index=s.index)
    for suffix, factor in _SUFFIX_MULTIPLIER.items():
        hit = s.str.match(r"(?i)^-?[\d.,\s]+" + suffix + r"$", na=False)
        mult[hit] = factor

    cleaned = s
    for prefix in _CURRENCY_PREFIXES:
        cleaned = cleaned.str.replace(prefix, "", regex=False)
    cleaned = cleaned.str.replace("[" + _re.escape(_CURRENCY_CHARS) + r"%\s]",
                                  "", regex=True)
    if style == "eu":
        # "1.234,56" -> "1234.56"
        cleaned = (cleaned.str.replace(".", "", regex=False)
                          .str.replace(",", ".", regex=False))
    else:
        cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"(?i)[kmb]$", "", regex=True)

    out = pd.to_numeric(cleaned, errors="coerce") * mult
    # Applied after parsing, so "(1,200)" and "(1.2k)" both come out
    # negative rather than only the ones without formatting.
    out[negated & out.notna()] = -out[negated & out.notna()].abs()
    return out


def _looks_like_money_or_percent(series: pd.Series) -> bool:
    """True when the text in a column is formatted numbers, not words.

    The check is on the DATA, not the column name: an export can call
    the column anything, and a name-based rule would miss "Q3 total" and
    fire on a genuine text column called "price band".
    """
    sample = series.dropna().astype(str).str.strip().head(200)
    if sample.empty:
        return False
    import re as _re

    # Plain pd.to_numeric has already failed by the time this is asked,
    # so a value only reaches here if it carries something extra. That
    # something must be FORMATTING — a currency mark, a group separator,
    # a percent sign, a k/m/b suffix or accounting brackets — and not
    # merely a letter stuck to a digit. Without that requirement "R1"
    # and "Q3" read as numbers and a Region column became 1, 2, 3.
    money = ("(?:" + "|".join(_re.escape(p) for p in _CURRENCY_PREFIXES)
             + "|[" + _re.escape(_CURRENCY_CHARS) + "])")
    number = r"\d[\d.,\s]*"
    alternatives = [
        r"\(\s*{money}?\s*{number}\s*[kKmMbB]?\s*\)",   # (1,200)
        r"-?\s*{money}\s*{number}\s*[kKmMbB]?",            # $1,200
        r"-?\s*{number}\s*{money}",                         # 1200 $
        r"-?\s*{number}\s*%",                               # 15%
        r"-?\s*\d[\d.,\s]*[.,]\d{{3}}(?:[.,]\d+)?",     # 1,234 / 1.234,5
        r"-?\s*\d+(?:[.,]\d+)?\s*[kKmMbB]",               # 1.2k
    ]
    formatted = _re.compile(
        "^(?:" + "|".join(a.format(money=money, number=number)
                          for a in alternatives) + ")$")
    # A plain number is allowed to sit in the column — an accounting
    # column reads "(500), 320, (120), 890", where only the losses carry
    # brackets. Requiring every value to be formatted rejected exactly
    # that column and left a real measure as text.
    plain = _re.compile(r"^-?\s*\d+(?:\.\d+)?$")

    ok = evidence = 0
    for v in sample:
        if formatted.match(v):
            ok += 1
            evidence += 1
        elif plain.match(v):
            ok += 1
    # Nearly all values must parse, and at least one must actually carry
    # formatting — otherwise this is a plain numeric column that
    # pd.to_numeric would already have taken, or something else entirely.
    return ok / len(sample) > 0.90 and evidence > 0


def _smart_dtype_inference(df: pd.DataFrame,
                           warnings: Optional[list] = None) -> pd.DataFrame:
    """
    Carefully improve dtypes.
    Only converts when >80% of values successfully convert — and even
    then, a cell that fails to convert keeps its ORIGINAL value rather
    than being silently destroyed into a blank. A "Revenue" column that's
    90% numbers and one "Pending" stays a column with numbers and the
    word "Pending" in it, not numbers and a hole where "Pending" was.
    NEVER converts columns that look like IDs or product names.
    """
    skip_keywords = ["id", "name", "code", "sku", "url", "link",
                     "image", "description", "address", "email", "phone"]
    # Reading a column differently from how it was stored is a change to
    # the client's data, so it is reported rather than done quietly —
    # they may disagree, and they can only disagree if they are told.
    coerced_cols: list = []
    # Columns whose decimal separator the data does not settle.
    ambiguous_cols: list = []

    protected = set(df.attrs.get("_leading_zero_cols", []))

    for col in text_columns(df):
        col_lower = col.lower()

        # Skip ID-like columns
        if any(kw in col_lower for kw in skip_keywords):
            continue

        # A column read as text specifically to keep its leading zeros
        # must not be inferred back into a number here.
        if col in protected:
            continue

        # Try numeric
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            success_rate = converted.notna().sum() / max(len(df), 1)
            if success_rate > 0.80:
                df[col] = converted.where(converted.notna(), df[col])
                continue
        except Exception:
            logger.debug("_smart_dtype_inference: suppressed exception", exc_info=True)

        # Then the same column with its formatting removed. This runs
        # only after plain parsing has failed and only when the cells
        # look like formatted numbers, so a genuine text column is never
        # dragged through it.
        try:
            if _looks_like_money_or_percent(df[col]):
                style = _decimal_style(df[col].dropna().astype(str).head(200))
                if style == "unknown":
                    ambiguous_cols.append(col)
                    continue
                coerced = _coerce_numeric_strings(df[col], style)
                if coerced.notna().sum() / max(len(df), 1) > 0.80:
                    df[col] = coerced.where(coerced.notna(), df[col])
                    coerced_cols.append(col)
                    logger.info("data_loader: read '%s' as numbers — it was "
                                "stored as formatted text", col)
                    continue
        except Exception:
            logger.debug("currency/percent coercion failed for '%s'", col,
                         exc_info=True)

        # Try datetime (only for date-named columns)
        date_keywords = ["date", "time", "created", "updated", "timestamp"]
        if any(kw in col_lower for kw in date_keywords):
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                success_rate = converted.notna().sum() / max(len(df), 1)
                if success_rate > 0.70:
                    df[col] = converted.where(converted.notna(), df[col])
            except Exception:
                logger.debug("_smart_dtype_inference: suppressed exception", exc_info=True)

    if coerced_cols and warnings is not None:
        warnings.append(
            "Read {} as numbers — {} stored as formatted text "
            "(currency symbols, thousands separators, percent signs or "
            "bracketed negatives). Without this they would have been "
            "treated as words and left out of every chart and "
            "calculation.".format(
                ", ".join("'{}'".format(c) for c in coerced_cols[:6]),
                "they were" if len(coerced_cols) > 1 else "it was"))

    if ambiguous_cols and warnings is not None:
        warnings.append(
            "Left {} as text: the values use a single '.' before three "
            "digits, which is 1.85 in a US-formatted file and 1,850 in a "
            "European one. Nothing in the column settles which, and "
            "guessing wrong would misstate the figure by a thousand "
            "times, so it was not converted. Re-export with a plain "
            "decimal point to include it.".format(
                ", ".join("'{}'".format(c) for c in ambiguous_cols[:6])))

    return df
