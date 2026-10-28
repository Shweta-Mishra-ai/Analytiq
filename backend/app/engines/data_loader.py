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

def _load_csv(f, warnings: list) -> Optional[pd.DataFrame]:
    """Load CSV — tries multiple encodings and separators.

    Encodings are tried strictly. With `encoding_errors="replace"` UTF-8
    never fails, so the later encodings were never reached and a Windows
    export (cp1252) loaded with every accented character replaced by "�" —
    "café" became "caf�" — and no warning. Strict decoding moves on to the
    next encoding instead; latin-1 is last because it accepts any bytes.

    Lines with the wrong number of fields are skipped by the parser, which
    reports them only as a Python warning on the server's stderr. They are
    counted here and reported to the user, because a row that silently
    vanishes changes every total computed from the file.
    """
    import io
    import warnings as _warnings

    encodings  = ["utf-8", "cp1252", "latin-1"]
    separators = [",", ";", "\t", "|"]

    # A fresh buffer per attempt: after a failed read pandas leaves its
    # UTF-8 text wrapper on a reused handle, so every retry decoded as UTF-8
    # whatever encoding it was given.
    f.seek(0)
    raw = f.read()

    def _read(**kw):
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always", pd.errors.ParserWarning)
            df = pd.read_csv(io.BytesIO(raw), low_memory=False,
                             on_bad_lines="warn", **kw)
        skipped = sum(str(w.message).count("Skipping line") for w in caught
                      if issubclass(w.category, pd.errors.ParserWarning))
        return df, skipped

    for enc in encodings:
        for sep in separators:
            try:
                df, skipped = _read(encoding=enc, sep=sep)
            except UnicodeDecodeError:
                break      # this encoding is wrong whatever the separator
            except Exception:
                logger.debug("_load_csv: suppressed exception", exc_info=True)
                continue
            if df.shape[1] >= 2:  # at least 2 columns = real CSV
                if enc != "utf-8":
                    warnings.append(
                        "File is not UTF-8; read as {} — check that accented "
                        "characters look right.".format(enc))
                if sep != ",":
                    warnings.append("Separator detected: '{}'".format(sep))
                if skipped:
                    warnings.append(
                        "{:,} line(s) had the wrong number of fields and were "
                        "skipped — those rows are not in the dataset.".format(skipped))
                return df

    # Last resort — a single-column file
    for enc in encodings:
        try:
            df, skipped = _read(encoding=enc)
        except UnicodeDecodeError:
            continue
        if skipped:
            warnings.append(
                "{:,} line(s) had the wrong number of fields and were "
                "skipped — those rows are not in the dataset.".format(skipped))
        return df
    return None


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
            if data and all(isinstance(r, dict) for r in data):
                # Nested objects become dotted columns (user.name) instead
                # of a column of dicts, which no analysis can use.
                df = pd.json_normalize(data)
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try records format first
            if any(isinstance(v, list) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        else:
            df = pd.read_json(f)

        return _stringify_containers(df, warnings)
    except Exception:
        try:
            f.seek(0)
            return pd.read_json(f)
        except Exception:
            return None


def _stringify_containers(df: pd.DataFrame, warnings: list) -> pd.DataFrame:
    """Turn list/dict cells into JSON text.

    They are unhashable, so value counts, grouping and deduplication all
    raise on a column holding them.
    """
    import json
    converted = []
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s)):
            continue
        is_container = s.map(lambda v: isinstance(v, (list, dict)))
        if is_container.any():
            df[col] = s.map(lambda v: json.dumps(v, default=str)
                            if isinstance(v, (list, dict)) else v)
            converted.append(str(col))
    if converted:
        warnings.append(
            "Nested values in {} were kept as JSON text.".format(
                ", ".join("'{}'".format(c) for c in converted[:5])
                + (" and {} more".format(len(converted) - 5) if len(converted) > 5 else "")))
    return df


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

    # Stripping ("a" and "a ") and the blank-name renames above can both
    # produce two columns with one name; pandas then returns a frame where
    # every caller expects a column, and analysis fails far from the cause.
    names, seen, renamed = list(df.columns), {}, []
    taken = set(names)
    for i, c in enumerate(names):
        if c in seen:
            k = seen[c]
            while "{}_{}".format(c, k) in taken:
                k += 1
            new = "{}_{}".format(c, k)
            seen[c] = k + 1
            taken.add(new)
            renamed.append((c, new))
            names[i] = new
        else:
            seen[c] = 2
    if renamed:
        df.columns = names
        warnings.append(
            "Duplicate column names were made unique: {}.".format(
                ", ".join("'{}' → '{}'".format(a, b) for a, b in renamed[:5])))

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
    df = _smart_dtype_inference(df)

    return df


def _smart_dtype_inference(df: pd.DataFrame) -> pd.DataFrame:
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

    for col in text_columns(df):
        col_lower = col.lower()

        # Skip ID-like columns
        if any(kw in col_lower for kw in skip_keywords):
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

        # Try datetime (only for date-named columns)
        date_keywords = ["date", "time", "created", "updated", "timestamp"]
        if any(kw in col_lower for kw in date_keywords):
            try:
                converted = _parse_dates(df[col])
                success_rate = converted.notna().sum() / max(len(df), 1)
                if success_rate > 0.70:
                    df[col] = converted.where(converted.notna(), df[col])
            except Exception:
                logger.debug("_smart_dtype_inference: suppressed exception", exc_info=True)

    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse dates in whichever day/month order the column actually uses.

    pandas infers a single format from the first value, month-first unless
    told otherwise. A day-first column (31/01/2024 — the usual order in the
    UK, India and most of Europe) whose first value is ambiguous
    (01/02/2024) was read month-first; every value with a day above 12 then
    failed, and the whole column stayed text. Both orders are tried and the
    one that parses more values wins; on a tie the default stands.
    """
    month_first = pd.to_datetime(s, errors="coerce")
    day_first = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return day_first if day_first.notna().sum() > month_first.notna().sum() else month_first
