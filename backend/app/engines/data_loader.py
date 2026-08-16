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
                )
                if df.shape[1] >= 2:  # at least 2 columns = real CSV
                    if enc != "utf-8":
                        warnings.append("Encoding detected: {}".format(enc))
                    if sep != ",":
                        warnings.append("Separator detected: '{}'".format(sep))
                    return df
            except Exception:
                logger.debug("_load_csv: suppressed exception", exc_info=True)
                continue

    # Last resort — no separator detection
    f.seek(0)
    return pd.read_csv(f, encoding_errors="replace", low_memory=False)


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

    # Duplicate names are common in real exports — two "Revenue" columns
    # from a join, or a sheet where a header repeats. `df["revenue"]`
    # then returns a DataFrame instead of a Series, and every engine that
    # calls `.dtype`, `.nunique()` or `pd.to_numeric` on it raises. A fuzz
    # pass over the pipeline found this breaking six of nine entry points
    # — profiling, the story engine, both chart paths, the dashboard spec
    # and the HTML export — from one upload.
    #
    # Suffixing rather than dropping: the second column holds data the
    # user gave us, and silently discarding a column is the one thing a
    # loader must never do.
    counts: dict = {}
    renamed: list = []
    final: list = []
    for name in df.columns:
        if name in counts:
            counts[name] += 1
            new = "{}_{}".format(name, counts[name])
            while new in counts:
                counts[name] += 1
                new = "{}_{}".format(name, counts[name])
            renamed.append(name)
            counts[new] = 0
            final.append(new)
        else:
            counts[name] = 0
            final.append(name)
    if renamed:
        df.columns = final
        warnings.append(
            "{} duplicate column name(s) ({}) were suffixed to keep them "
            "distinct — no data was dropped.".format(
                len(renamed), ", ".join(sorted(set(renamed))[:4])))

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
                converted = pd.to_datetime(df[col], errors="coerce")
                success_rate = converted.notna().sum() / max(len(df), 1)
                if success_rate > 0.70:
                    df[col] = converted.where(converted.notna(), df[col])
            except Exception:
                logger.debug("_smart_dtype_inference: suppressed exception", exc_info=True)

    return df
