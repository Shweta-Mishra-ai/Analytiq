"""
engines/present.py — the last step before a number or a name reaches a
reader.

Analysis code thinks in columns and floats; a report is read by someone
who thinks in their own vocabulary. Left to the default formatters, the
gap shows: a median salary printed as ``7.26e+03``, a driver named
``'MonthlyIncome'`` in single quotes, a rating of 3.00 to 4.00. Each one
is individually small and together they are the difference between a
deliverable and a debug dump.

Everything user-facing goes through here.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Acronyms a title-caser would otherwise mangle into "Mrr" or "Id".
_ACRONYMS = {
    "id", "kpi", "mrr", "arr", "arpu", "roi", "roas", "ctr", "cpc", "cpa",
    "cpm", "ltv", "cac", "aov", "sla", "nps", "csat", "ebitda", "hr", "it",
    "fte", "ytd", "mtd", "qtd", "yoy", "mom", "usd", "eur", "gbp", "inr",
    "sku", "url", "api", "ui", "ux", "b2b", "b2c", "saas", "gmv", "aht",
    "otd", "oee", "wip", "bmi", "los", "icu", "er",
}

# Split camelCase and PascalCase, but never inside a run like B2B or
# Q4 — a letter/digit boundary is part of the token, not a word break.
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
                    r"|(?<=[0-9])(?=[A-Z][a-z])")

# Column names the client writes as one word and reads as one word.
# "OverTime" split naively becomes "Over Time", which reads as the phrase
# rather than the thing.
_COMPOUNDS = {
    "overtime": "Overtime", "headcount": "Headcount", "worklife": "Work-Life",
    "workload": "Workload", "timestamp": "Timestamp", "username": "Username",
    "healthcare": "Healthcare", "onboarding": "Onboarding",
    "upsell": "Upsell", "downtime": "Downtime", "throughput": "Throughput",
    "leadtime": "Lead Time", "runrate": "Run Rate", "churn": "Churn",
}


def label(name: Any) -> str:
    """A column name as a person would write it.

    ``MonthlyIncome`` → ``Monthly Income``; ``years_at_company`` → ``Years
    at Company``; ``mrr_usd`` → ``MRR USD``. Never quoted — a report that
    prints 'JobRole' in quotes reads like a stack trace.
    """
    text = str(name).strip().strip("'\"")
    if not text:
        return ""
    flat = re.sub(r"[^a-z0-9]", "", text.lower())
    if flat in _COMPOUNDS:
        return _COMPOUNDS[flat]
    if flat in _ACRONYMS:
        return flat.upper()
    text = _CAMEL.sub(" ", text)
    text = re.sub(r"[_\-.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    small = {"at", "of", "in", "on", "per", "by", "to", "for", "and", "or"}
    words = []
    for i, word in enumerate(text.split(" ")):
        low = word.lower()
        if low in _ACRONYMS:
            words.append(low.upper())
        elif i > 0 and low in small:
            words.append(low)
        elif word.isupper() and len(word) > 1:
            words.append(word)          # already an acronym the user chose
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def value(val: Any) -> str:
    """A data value as it appears to the person who entered it.

    Cleaning turns ``OverTime`` into a 0/1 column, and the report then
    told the client their risk pocket was ``OverTime = True`` — a value
    that appears nowhere in their spreadsheet. Booleans map back to the
    yes/no they came from.
    """
    if val is None:
        return "—"
    # numpy.bool_ is not a Python bool, and pandas hands back numpy
    # scalars — without this the report says "Overtime: True at 39%",
    # naming a value that appears nowhere in the client's own file.
    # Matching on the type name does not work either: in numpy 2 the
    # class is called plain "bool".
    if isinstance(val, (bool, np.bool_)):
        return "Yes" if bool(val) else "No"
    if isinstance(val, str):
        stripped = val.strip().strip("'\"")
        if stripped.lower() in ("true", "false"):
            return "Yes" if stripped.lower() == "true" else "No"
        return stripped
    if isinstance(val, (int, float)):
        return num(val)
    return str(val)


def num(val: Any, unit: str = "", decimals: int | None = None) -> str:
    """A number sized for reading, never in scientific notation.

    Precision follows magnitude rather than a fixed setting: a salary
    wants no decimals (8,024, not 8,024.30), a satisfaction score on a
    1–4 scale wants two.
    """
    try:
        x = float(val)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"

    if decimals is not None:
        body = f"{x:,.{decimals}f}"
    elif abs(x) >= 1_000_000_000:
        body = f"{x / 1_000_000_000:,.2f}bn"
    elif abs(x) >= 1_000_000:
        body = f"{x / 1_000_000:,.2f}m"
    elif abs(x) >= 1000:
        body = f"{x:,.0f}"
    elif abs(x) >= 100:
        body = f"{x:,.0f}"
    elif abs(x) >= 10:
        body = f"{x:,.1f}"
    elif abs(x) >= 1:
        body = f"{x:,.2f}"
    elif x == 0:
        body = "0"
    else:
        # Below 1 the useful precision depends on how small it is; a
        # rate of 0.0004 must not print as "0.00".
        places = min(6, max(2, 1 - int(math.floor(math.log10(abs(x))))))
        # A trailing zero here is noise, not precision: 0.0010 claims a
        # fourth digit the value does not have.
        body = f"{x:,.{places}f}".rstrip("0").rstrip(".")

    if body.startswith("-0") and float(body.replace(",", "")) == 0:
        body = body[1:]                 # no "-0"
    return f"{body}{unit}"


def money(val: Any, symbol: str = "") -> str:
    """Currency: whole units, because nobody budgets to the cent."""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    if abs(x) >= 1_000_000:
        return f"{symbol}{x / 1_000_000:,.2f}m"
    if abs(x) >= 1000:
        return f"{symbol}{x:,.0f}"
    return f"{symbol}{x:,.0f}"


def pct(val: Any, decimals: int = 1) -> str:
    """A percentage already expressed on a 0–100 scale."""
    try:
        x = float(val)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    if abs(x) >= 10:
        decimals = min(decimals, 1)
    return f"{x:,.{decimals}f}%"


def truncate(text: Any, limit: int, ellipsis: str = "…") -> str:
    """Shorten to fit, on a word boundary.

    A hard slice put ``Northwind Manufacturin`` on a cover page and
    ``HR Specialis`` inside a finding. Cutting at a space and marking the
    cut says "there is more here"; cutting mid-word just looks broken.
    """
    s = str(text)
    if len(s) <= limit:
        return s
    if limit <= len(ellipsis):
        return s[:limit]
    cut = s[:limit - len(ellipsis)]
    space = cut.rfind(" ")
    # Only honour the word boundary if it leaves most of the space used;
    # otherwise one long word would shrink the label to nothing.
    # A clean short name reads as deliberately shortened; a name cut
    # mid-word reads as a bug, so the boundary wins unless it would throw
    # away most of the text.
    if space >= int(limit * 0.45):
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + ellipsis


def join_and(items, limit: int = 3) -> str:
    """`a, b and c`, with an honest count when the list runs long."""
    vals = [str(i) for i in items if str(i).strip()]
    if not vals:
        return ""
    if len(vals) > limit:
        return "{} and {} others".format(", ".join(vals[:limit]),
                                         len(vals) - limit)
    if len(vals) == 1:
        return vals[0]
    return "{} and {}".format(", ".join(vals[:-1]), vals[-1])


def sentence(text: Any) -> str:
    """Trim, drop stray quoting, and end on a full stop."""
    s = re.sub(r"\s+", " ", str(text)).strip()
    if not s:
        return ""
    s = s[:1].upper() + s[1:]
    if s[-1] not in ".!?:":
        s += "."
    return s
