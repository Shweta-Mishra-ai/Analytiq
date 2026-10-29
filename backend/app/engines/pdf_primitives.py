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
    """True if a column is an identifier. Delegates to the one shared test,
    so a role the user confirmed decides it here too.

    This used to be a second, looser test of its own: it ignored confirmed
    roles, raised on text columns, and disagreed with the main report about
    which columns to describe.
    """
    from app.engines.domains.base import is_id_column
    return is_id_column(col, series)

