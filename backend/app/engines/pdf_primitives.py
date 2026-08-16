"""
engines/pdf_primitives.py — small shared helpers for the PDF builders.

Ported from dataforge-ai's core/pdf/primitives.py (only the pieces the
health report needs). Both report builders use the same column filter so
they can't disagree about which columns to describe for the same dataset.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


_ID_KW = {"index", "idx", "id", "rowid", "row_id", "empid", "emp_id",
          "order_id", "orderid", "product_id", "customer_id", "user_id"}


def truncate_label(text: str, width: int) -> str:
    """Truncate a label to `width` chars, adding an ellipsis when cut —
    plain slicing ('EnvironmentSatisfaction'[:20] -> 'EnvironmentSatisfact')
    reads as a rendering bug in a client-facing PDF. No-op if it already fits.
    """
    text = str(text)
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[:width - 1].rstrip() + "\u2026"


def is_id_col(col: str, series: pd.Series) -> bool:
    """True if a numeric column looks like an identifier (EmployeeNumber,
    OrderID, ...) or a near-constant column (EmployeeCount, StandardHours)
    rather than a real analytic variable. Shared by the Main Report and
    Health Report builders so BOTH pick the same columns for describe/
    correlation tables — previously only the Main Report filtered these
    out, so the two reports' first-8-columns correlation/stats tables
    disagreed on the same dataset.
    """
    import re as _re
    cl = col.lower().strip()
    if cl in _ID_KW or _re.search(r'\bid\b|\bindex\b', cl):
        return True
    if len(series.dropna()) > 10:
        try:
            diffs = series.dropna().sort_values().diff().dropna()
            if (diffs == 1).mean() > 0.90:
                return True
        except Exception:
            logger.warning("id-col check failed", exc_info=True)
    return False
