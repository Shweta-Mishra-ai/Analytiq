"""
services/column_roles.py — what each column is, as the user confirmed it.

Every analysis has to know which columns are identifiers, dates, measures
and categories, and for a measure whether more is better. Guessed from
names and values alone, that went wrong in ways no single fix could cover:
"paid" read as an identifier because it contains "id", a store number
summed as revenue, delay figures judged as if higher were better. The user
says it once after upload, and every page uses that.

Confirmed roles are stored with the dataset and applied each time it is
loaded, so cleaning, undo and reset keep them. They travel with the data in
``DataFrame.attrs``, which pandas carries through column selection,
filtering, copies, sampling and grouping — so the shared identifier and
direction checks see them without each engine having to be told.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ATTR = "column_roles"
ROLES = ("identifier", "date", "measure", "category", "text")
DIRECTIONS = (1, -1, 0)

# A role that changes a column's type is accepted when at least this share
# of its filled values convert; the rest become blank in the analysed data
# (the uploaded file is untouched) and the user is told how many.
CONVERT_MIN = 0.95


class RoleError(ValueError):
    """A role the column's values cannot support, explained for the user."""


def confirmed(obj) -> Dict[str, dict]:
    """The confirmed roles carried by a DataFrame or Series, if any."""
    return dict((getattr(obj, "attrs", None) or {}).get(ATTR) or {})


def role_of(col, series) -> Optional[dict]:
    """The confirmed role of one column, or None when only guessed."""
    if series is None:
        return None
    return confirmed(series).get(str(col))


# ── detection (the starting point the user confirms) ─────

def _bare(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.attrs = {}
    return out


def detect(df: pd.DataFrame) -> Dict[str, dict]:
    """Best guess for every column, from the same checks the engines use."""
    from app.engines.benchmarking import metric_direction
    from app.engines.domains.base import is_id_column
    from app.services.dtypes import is_text_dtype

    out = {}
    for col in df.columns:
        s = _bare(df[col])
        name = str(col)
        if pd.api.types.is_datetime64_any_dtype(s):
            out[name] = {"role": "date", "direction": 0}
        elif is_id_column(col, s):
            out[name] = {"role": "identifier", "direction": 0}
        elif pd.api.types.is_bool_dtype(s):
            out[name] = {"role": "category", "direction": 0}
        elif pd.api.types.is_numeric_dtype(s):
            out[name] = {"role": "measure", "direction": metric_direction(col)}
        elif is_text_dtype(s):
            filled = s.dropna()
            few = filled.nunique() <= max(50, 0.05 * len(filled))
            out[name] = {"role": "category" if few else "text", "direction": 0}
        else:
            out[name] = {"role": "category", "direction": 0}
    return out


# ── applying roles ───────────────────────────────────────

def _filled(s: pd.Series) -> pd.Series:
    """Values that are present: not missing and not blank text."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return s.notna()
    return s.notna() & s.astype(str).str.strip().ne("")


def _convert(s: pd.Series, col: str, kind: str, strict: bool) -> Tuple[pd.Series, int]:
    """Convert to dates or numbers; returns the series and how many filled
    values could not be read."""
    from app.engines.data_loader import _parse_dates

    if kind == "date":
        converted = _parse_dates(s)
    else:
        text = s.astype(str).str.replace(",", "", regex=False).str.strip()
        converted = pd.to_numeric(text.where(s.notna()), errors="coerce")
    filled = _filled(s)
    n_filled = int(filled.sum())
    lost_mask = filled & converted.isna()
    lost = int(lost_mask.sum())
    if strict and n_filled and (n_filled - lost) / n_filled < CONVERT_MIN:
        examples = ", ".join("'{}'".format(str(v)[:30])
                             for v in s[lost_mask].astype(str).unique()[:3])
        noun = "dates" if kind == "date" else "numbers"
        raise RoleError(
            "Only {:.0%} of the values in '{}' read as {} (for example {}), so it "
            "cannot be treated as a {}. Clean those values first, or choose "
            "another role.".format((n_filled - lost) / n_filled, col, noun, examples,
                                   "date" if kind == "date" else "measure"))
    return converted, lost


def _as_category(s: pd.Series) -> pd.Series:
    """Numbers used as labels (store 7, region 3) become text labels, so no
    analysis sums or averages them. 7.0 stays '7'."""
    if pd.api.types.is_float_dtype(s):
        whole = s.dropna()
        if len(whole) and (whole == whole.round()).all():
            s = s.astype("Int64")
    out = s.astype("string").astype(object).where(s.notna())
    return out.astype("str")


def apply(df: pd.DataFrame, roles: Dict[str, dict],
          strict: bool = True) -> Tuple[pd.DataFrame, list]:
    """Return a copy of ``df`` with the confirmed roles applied, and notes
    for the user. With ``strict`` a role the data cannot support raises
    RoleError; without it (re-applying stored roles after the data changed)
    that column keeps its type and the problem is logged."""
    out = df.copy()
    notes = []
    kept = {}
    for name, spec in (roles or {}).items():
        col = next((c for c in out.columns if str(c) == name), None)
        if col is None:
            continue
        role = spec.get("role")
        direction = int(spec.get("direction") or 0)
        if role not in ROLES or direction not in DIRECTIONS:
            if strict:
                raise RoleError("'{}' is not a valid role for '{}'.".format(role, name))
            continue
        s = out[col]
        try:
            if role == "date" and not pd.api.types.is_datetime64_any_dtype(s):
                out[col], lost = _convert(s, name, "date", strict)
                if lost:
                    notes.append("{:,} value(s) in '{}' could not be read as dates and "
                                 "are left blank in the analysis.".format(lost, name))
            elif role == "measure":
                if pd.api.types.is_datetime64_any_dtype(s):
                    raise RoleError("'{}' holds dates, so it cannot be a measure.".format(name))
                if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                    out[col], lost = _convert(s, name, "number", strict)
                    if lost:
                        notes.append("{:,} value(s) in '{}' are not numbers and are left "
                                     "blank in the analysis.".format(lost, name))
            elif role == "category" and pd.api.types.is_numeric_dtype(s) \
                    and not pd.api.types.is_bool_dtype(s):
                out[col] = _as_category(s)
        except RoleError:
            if strict:
                raise
            logger.warning("column role %s for %r no longer fits the data; left as is",
                           role, name)
            continue
        kept[name] = {"role": role, "direction": direction if role == "measure" else 0}
    out.attrs = {**df.attrs, ATTR: kept}
    return out, notes
