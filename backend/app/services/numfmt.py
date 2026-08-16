"""
services/numfmt.py — writing a number the way a person writes one.

Python's `%g` shortens a float by switching to scientific notation, which
is right for a p-value and wrong for money. The executive summary of a
finance report opened with:

    Median 'revenue' ranges from 7.72e+04 in 'Support' to 8.89e+05 in
    'Retail' — a 11.5× spread

Nobody writes a revenue figure that way, and a reader who meets `8.89e+05`
in the first sentence of a document they have paid for stops reading it as
a document. The same figures written as `77.2k` and `889k` carry exactly
as much information and cost nothing.

One definition, used by the chart axes, the chart headlines and the
report text alike, so the same figure is written the same way wherever it
appears. Two different roundings of one number on one page reads as a
mistake even when both are correct.
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def human_number(value: float) -> str:
    """A number as it would be written in a sentence.

    2,400,000,000 → "2.4bn"; 3,242,612 → "3.2m"; 685,941 → "686k";
    62.0 → "62"; 0.2367 → "0.24".
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return "n/a"
    a = abs(value)
    if a >= 1_000_000_000:
        return "{:,.1f}bn".format(value / 1_000_000_000)
    if a >= 1_000_000:
        return "{:,.1f}m".format(value / 1_000_000)
    if a >= 1_000:
        return "{:,.0f}k".format(value / 1_000)
    if a >= 10:
        return "{:,.0f}".format(value)
    return "{:,.2f}".format(value).rstrip("0").rstrip(".")


def human_money(value: float, currency: str = "") -> str:
    """`human_number` with a currency symbol in front of it, if given."""
    return "{}{}".format(currency, human_number(value))
