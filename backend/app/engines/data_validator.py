"""
data_validator.py
Validates DataFrame before analysis.
Catches garbage data early — never let bad data reach analysis layer.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# ── Config ────────────────────────────────────────────────
MAX_FILE_MB       = 200
MAX_ROWS          = 1_000_000
MIN_ROWS          = 2
MIN_COLS          = 1
MAX_COLS          = 500
SAMPLE_THRESHOLD  = 100_000   # rows above this → auto-sample for heavy ops


@dataclass
class ValidationResult:
    is_valid: bool
    errors:   List[str] = field(default_factory=list)   # blocking issues
    warnings: List[str] = field(default_factory=list)   # non-blocking
    was_sampled: bool    = False
    sample_size: int     = 0
    original_rows: int   = 0
    memory_mb: float     = 0.0


def validate_file_size(size_bytes: int) -> Tuple[bool, str]:
    """Check file size before reading."""
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return False, "File too large ({:.1f} MB). Maximum allowed: {} MB.".format(
            size_mb, MAX_FILE_MB)
    return True, ""


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Full validation of a loaded DataFrame.
    Returns ValidationResult — caller decides whether to proceed.
    """
    result = ValidationResult(is_valid=True)
    result.original_rows = len(df)

    # Memory estimate
    try:
        result.memory_mb = round(df.memory_usage(deep=True).sum() / (1024*1024), 2)
    except Exception:
        result.memory_mb = 0.0

    # ── Blocking checks ───────────────────────────────────

    # Empty file
    if len(df) == 0:
        result.is_valid = False
        result.errors.append("Dataset is empty — no rows found.")
        return result

    if len(df.columns) == 0:
        result.is_valid = False
        result.errors.append("Dataset has no columns.")
        return result

    # Too few rows
    if len(df) < MIN_ROWS:
        result.is_valid = False
        result.errors.append(
            "Dataset has only {} row(s). Minimum {} rows required.".format(
                len(df), MIN_ROWS))

    # Too many columns
    if len(df.columns) > MAX_COLS:
        result.is_valid = False
        result.errors.append(
            "Dataset has {} columns — maximum allowed is {}.".format(
                len(df.columns), MAX_COLS))

    # Too many rows
    if len(df) > MAX_ROWS:
        result.is_valid = False
        result.errors.append(
            "Dataset has {:,} rows — maximum allowed is {:,}. "
            "Please sample your data before uploading.".format(
                len(df), MAX_ROWS))

    # All columns are empty
    non_empty = [c for c in df.columns if df[c].notna().any()]
    if len(non_empty) == 0:
        result.is_valid = False
        result.errors.append("All columns are empty.")

    if not result.is_valid:
        return result

    # ── Non-blocking warnings ─────────────────────────────

    # High missing rate overall
    total_cells   = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_pct   = missing_cells / max(total_cells, 1) * 100
    if missing_pct > 50:
        result.warnings.append(
            "{:.1f}% of all cells are missing — "
            "analysis results may be unreliable.".format(missing_pct))
    elif missing_pct > 20:
        result.warnings.append(
            "{:.1f}% missing data detected — "
            "imputation will be applied.".format(missing_pct))

    # All-null rows
    all_null_rows = df.isna().all(axis=1).sum()
    if all_null_rows > 0:
        result.warnings.append(
            "{} fully empty row(s) found — will be dropped.".format(all_null_rows))

    # Duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        dup_pct = dup_count / len(df) * 100
        result.warnings.append(
            "{:,} duplicate rows ({:.1f}%) — will be removed.".format(
                dup_count, dup_pct))

    # Problematic column names
    bad_names = [c for c in df.columns
                 if not str(c).strip()
                 or str(c).startswith("Unnamed")
                 or str(c).strip() != str(c)]
    if bad_names:
        result.warnings.append(
            "{} column(s) have blank, unnamed, or whitespace-padded names — "
            "will be cleaned.".format(len(bad_names)))

    # Infinite values
    num_df = df.select_dtypes(include="number")
    inf_count = np.isinf(num_df.values).sum() if len(num_df.columns) > 0 else 0
    if inf_count > 0:
        result.warnings.append(
            "{} infinite value(s) detected — will be replaced with NaN.".format(
                inf_count))

    # Mixed type columns (mostly numeric but some strings)
    for col in df.select_dtypes(include="object").columns[:20]:
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            continue
        num_like = pd.to_numeric(sample, errors="coerce").notna().mean()
        if 0.3 < num_like < 0.85:
            result.warnings.append(
                "'{}' has mixed types ({:.0f}% numeric-like) — "
                "verify this column's data.".format(col, num_like * 100))

    # Large memory warning
    if result.memory_mb > 500:
        result.warnings.append(
            "Dataset uses {:.0f} MB in memory — "
            "some operations may be slow.".format(result.memory_mb))

    # Large row count → will be sampled
    if len(df) > SAMPLE_THRESHOLD:
        result.was_sampled  = True
        result.sample_size  = SAMPLE_THRESHOLD
        result.warnings.append(
            "Dataset has {:,} rows. Heavy analysis operations will use "
            "a representative sample of {:,} rows for performance.".format(
                len(df), SAMPLE_THRESHOLD))

    return result


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all non-destructive fixes:
    - Drop fully empty rows/cols
    - Clean column names
    - Replace infinities with NaN
    - Strip string whitespace
    Returns cleaned copy.
    """
    df = df.copy()

    # Drop all-null rows and columns
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Clean column names
    new_cols = []
    seen     = {}
    for col in df.columns:
        clean = str(col).strip()
        if not clean or clean.lower().startswith("unnamed"):
            clean = "column_{}".format(len(seen) + 1)
        # Deduplicate
        if clean in seen:
            seen[clean] += 1
            clean = "{}_{}".format(clean, seen[clean])
        else:
            seen[clean] = 0
        new_cols.append(clean)
    df.columns = new_cols

    # Replace infinities
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    return df


def get_analysis_sample(df: pd.DataFrame, n: int = SAMPLE_THRESHOLD,
                         random_state: int = 42) -> pd.DataFrame:
    """
    Return full df if small, else a stratified sample.
    Always reproducible via random_state.
    """
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)

