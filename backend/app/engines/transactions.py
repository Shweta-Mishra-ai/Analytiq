"""
engines/transactions.py — what an order-line extract is made of.

A shop's export is usually one row per line item: an order (invoice)
number that repeats for every product on the order, a quantity, a unit
price, and often no revenue column at all. Every engine used to guess
these columns on its own, and each guess failed differently on the same
file: frequency counted line items as orders ("138.7 orders per
customer"), a customer's "spend" was the sum of the unit prices of what
they bought, and the scorecard found no revenue and skipped the business
figures altogether.

This module is the single place those columns are found:

  order_column      the order / invoice identifier, if any
  quantity_column   units on the line
  unit_price_column the per-unit price (never itself a revenue figure)
  derive_line_revenue adds Revenue = quantity × unit price when the file
                    has no revenue measure, so totals are real sales
  cancellation_mask the lines that reverse a sale (a negative quantity,
                    or an invoice number in the 'C…' cancellation form)

A derived revenue keeps the sign of the quantity: returns and
cancellations subtract, so a total is net sales, as an accountant would
report it.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DERIVED_REVENUE = "Revenue"

_ORDER_WORDS = ("order", "invoice", "transaction", "receipt", "basket", "bill")
_ID_WORDS = ("id", "no", "num", "number", "code", "key", "ref", "reference")
# Words that make an order-ish name something else: "order date",
# "invoice amount", "order quantity", "order status".
_NOT_ORDER = ("date", "time", "day", "month", "year", "amount", "total", "value",
              "price", "qty", "quantity", "status", "type", "count", "line", "item")
_QTY_WORDS = ("quantity", "qty", "units", "unit_count")
_PRICE_WORDS = ("price", "unitprice", "rate")
_REVENUE_WORDS = ("revenue", "sales", "turnover", "gmv", "amount", "total",
                  "spend", "value", "income", "billing")


def _words(name) -> list:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return [w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w]


def order_column(df: pd.DataFrame, exclude=()) -> Optional[str]:
    """The order / invoice identifier: a name built on an order word
    ("Invoice", "Order ID", "InvoiceNo", "transaction_number") whose
    values repeat — one order spans several lines — but not on every row."""
    n = len(df)
    if n == 0:
        return None
    for c in df.columns:
        if c in exclude:
            continue
        ws = _words(c)
        if not ws or not any(w.startswith(k) for w in ws for k in _ORDER_WORDS):
            continue
        if any(w in _NOT_ORDER for w in ws):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
            continue
        uniq = df[c].nunique(dropna=True)
        if 1 < uniq <= n:
            return c
    return None


def _numeric_named(df: pd.DataFrame, words, exclude_words=()) -> Optional[str]:
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        ws = _words(c)
        joined = "".join(ws)
        if any(w in words or w.startswith(words) for w in ws) or joined in words:
            if not any(w in exclude_words for w in ws):
                return c
    return None


def quantity_column(df: pd.DataFrame) -> Optional[str]:
    return _numeric_named(df, ("quantity", "qty", "units"))


def unit_price_column(df: pd.DataFrame) -> Optional[str]:
    """A per-unit price: "Price", "UnitPrice", "unit_price". Not "Total
    Price" or "Price Paid Total", which are line amounts."""
    return _numeric_named(df, ("price", "unitprice"),
                          exclude_words=("total", "amount", "sum", "line", "paid"))


def revenue_column(df: pd.DataFrame) -> Optional[str]:
    """An existing line or order amount, if the file has one."""
    from app.engines.domains.base import is_id_column
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        ws = _words(c)
        if not any(w.startswith(k) for w in ws for k in _REVENUE_WORDS):
            continue
        if any(w in ("pct", "percent", "rate", "ratio", "share", "count", "growth",
                     "margin", "score", "id", "qty", "quantity", "units") for w in ws):
            continue
        if is_id_column(c, df[c]):
            continue
        return c
    return None


def derive_line_revenue(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """`df` with a Revenue column (quantity × unit price) when the file has
    a quantity and a unit price but no amount of its own, and the note that
    says so; otherwise `df` unchanged and None. The note goes wherever the
    figures are shown, because a reader must know revenue was computed."""
    if DERIVED_REVENUE in df.columns or revenue_column(df) is not None:
        return df, None
    qty, price = quantity_column(df), unit_price_column(df)
    if qty is None or price is None:
        return df, None
    line = (pd.to_numeric(df[qty], errors="coerce")
            * pd.to_numeric(df[price], errors="coerce")).round(2)
    if line.notna().sum() < 10:
        return df, None
    out = df.copy()
    out[DERIVED_REVENUE] = line
    note = (_NOTE_START + "'{r}' was added as '{q}' × '{p}' "
            "on every line. Returns and cancellations have a negative quantity "
            "and subtract, so totals are net sales.").format(
                r=DERIVED_REVENUE, q=qty, p=price)
    return out, note


_NOTE_START = "The file has no revenue column, so "


def is_derivation_note(text) -> bool:
    """True for the note derive_line_revenue writes into a dataset's
    warnings, so a report can repeat it in its basis of preparation."""
    return isinstance(text, str) and text.startswith(_NOTE_START)


def cancellation_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """Lines that reverse a sale: a negative quantity, or an invoice in the
    'C' + number form many shop systems use for a cancellation. None when
    the file has neither a quantity nor an order column to tell."""
    qty, order = quantity_column(df), order_column(df)
    if qty is None and order is None:
        return None
    mask = pd.Series(False, index=df.index)
    if qty is not None:
        mask |= pd.to_numeric(df[qty], errors="coerce").lt(0).fillna(False)
    if order is not None:
        s = df[order].astype("string")
        mask |= s.str.match(r"^\s*[Cc]\d").fillna(False).astype(bool)
    return mask


def count_orders(df: pd.DataFrame, by: Optional[str] = None):
    """Orders, not lines: distinct order numbers when there is an order
    column, otherwise rows. With `by`, a Series per group."""
    order = order_column(df, exclude=(by,) if by else ())
    if by is None:
        return int(df[order].nunique()) if order else len(df)
    g = df.groupby(by, observed=True)
    return g[order].nunique() if order else g.size()
