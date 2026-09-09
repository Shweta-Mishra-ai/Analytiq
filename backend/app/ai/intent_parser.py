"""
ai/intent_parser.py — answering the ordinary question without a model.

The Chat page offers the user four questions to click, generated from
their own columns:

    Which Department has the highest average Monthly Income?
    Show the 10 rows with the highest Monthly Income
    Compare Years at Company across Department
    Which columns move together?

Every one of those is a group-by, a sort, or a correlation — arithmetic
the app already does in `tool_dispatcher`. The only thing a language
model was ever doing was turning the sentence into `{tool, params}`.

So without an API key the page offered four questions and then answered
none of them: `POST /api/chat/{id}` returned 503 and the suggestions sat
there as buttons that did nothing. A product that shows you a question it
cannot answer is worse than one that never offered.

This module does that mapping with regular expressions. It is not a
natural-language understander and does not pretend to be — it recognises
the shapes people actually type, and returns None for everything else so
a model can take over when one is configured. Being narrow is the point:
a wrong tool call answers confidently about the wrong column, which is
the failure mode this whole codebase exists to avoid.

Everything here runs offline, costs nothing, and cannot hallucinate a
column that is not in the frame.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# How the question refers to an aggregate, and what pandas calls it.
_AGGREGATES = (
    (("average", "avg", "mean", "typical"), "mean"),
    (("total", "sum", "combined", "altogether"), "sum"),
    (("median", "middle"), "median"),
    (("maximum", "max", "largest", "biggest"), "max"),
    (("minimum", "min", "smallest"), "min"),
    (("count", "number of", "how many"), "count"),
)

# Words that mean "the big end" and "the small end" of a sort.
_HIGH = ("highest", "top", "most", "largest", "biggest", "best", "maximum",
         "greatest")
_LOW = ("lowest", "bottom", "least", "smallest", "worst", "minimum", "fewest")


def _words(name: str) -> List[str]:
    """The words in a column name, however it was written.

    `MonthlyIncome`, `monthly_income` and `Monthly Income` are the same
    column asked for three ways, and the UI displays the third while the
    frame stores the first.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return [w for w in re.split(r"[^A-Za-z0-9]+", spaced) if w]


def _pattern_for(column: str):
    """A regex matching this column name in a sentence.

    Anchored on word boundaries, which is the whole point. Matching a
    flattened string instead finds `Age` inside "aver**age**" — and the
    first version of this parser did exactly that, answering "which
    department has the highest average monthly income" with the mean of
    Age. A confidently wrong column is worse than no answer.
    """
    parts = _words(column)
    if not parts:
        return None
    body = r"[\s_\-]*".join(re.escape(p.lower()) for p in parts)
    return re.compile(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])", re.I)


def _columns_in(text: str, columns: List[str]) -> List[str]:
    """Every column named in a phrase, in the order they were mentioned.

    Longer names are matched first and their span blanked out, so
    `Income` cannot also claim the characters inside `MonthlyIncome`.
    """
    haystack = str(text).lower()
    found: List[Tuple[int, str]] = []
    for col in sorted(columns, key=lambda c: -len(str(c))):
        pattern = _pattern_for(col)
        if pattern is None:
            continue
        m = pattern.search(haystack)
        if m:
            haystack = (haystack[:m.start()] + "\x00" * (m.end() - m.start())
                        + haystack[m.end():])
            found.append((m.start(), col))
    return [c for _, c in sorted(found)]


def _numeric(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def _categorical(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]


def _agg_from(text: str) -> Optional[str]:
    for words, func in _AGGREGATES:
        if any(w in text for w in words):
            return func
    return None


def parse(question: str, df: pd.DataFrame) -> Optional[Dict]:
    """`{tool, params, explanation}` for a question this can answer.

    None means "ask a model" — never a guess. A parser that reaches for
    the closest column when it is unsure produces a confident answer
    about the wrong thing, which costs more than declining.
    """
    if not question or df is None or df.empty:
        return None

    q = " " + question.lower().strip() + " "
    columns = [str(c) for c in df.columns]
    named = _columns_in(question, columns)

    for rule in (_rule_correlation_pair, _rule_correlation_general,
                 _rule_top_rows, _rule_group_extreme, _rule_group_compare,
                 _rule_distribution, _rule_value_counts, _rule_describe):
        try:
            hit = rule(q, question, df, columns, named)
        except Exception:
            logger.debug("intent rule %s failed", rule.__name__, exc_info=True)
            continue
        if hit:
            logger.info("chat answered without a model via %s", rule.__name__)
            return hit
    return None


# ── the rules, most specific first ────────────────────────

def _rule_correlation_pair(q, raw, df, columns, named) -> Optional[Dict]:
    """"How does X relate to Y", with both columns named."""
    if not any(w in q for w in ("correlat", "relate", "relationship",
                                "associated", "move together", "linked")):
        return None
    nums = _numeric(df, named)
    if len(nums) < 2:
        return None
    return {
        "tool": "correlation",
        "params": {"col_a": nums[0], "col_b": nums[1]},
        "explanation": "Correlation between {} and {}.".format(
            nums[0], nums[1]),
    }


def _rule_correlation_general(q, raw, df, columns, named) -> Optional[Dict]:
    """"Which columns move together?" — the suggestion the page offers."""
    if not any(w in q for w in ("move together", "correlat", "related to each",
                                "relationships between")):
        return None
    if len(_numeric(df, columns)) < 2:
        return None
    return {
        "tool": "plot_heatmap",
        "params": {},
        "explanation": "Correlation across every numeric column. Read the "
                       "strong colours: those pairs carry the same "
                       "information.",
    }


def _rule_top_rows(q, raw, df, columns, named) -> Optional[Dict]:
    """"Show the 10 rows with the highest X" / "bottom 5 by Y"."""
    if "row" not in q and not re.search(r"\b(top|bottom)\s+\d+", q):
        return None
    direction = None
    if any(w in q for w in _HIGH):
        direction = False        # ascending=False -> highest first
    elif any(w in q for w in _LOW):
        direction = True
    if direction is None:
        return None

    sort_col = None
    for col in named:
        if pd.api.types.is_numeric_dtype(df[col]):
            sort_col = col
            break
    if sort_col is None:
        return None

    found = re.search(r"\b(\d{1,4})\b", q)
    n = int(found.group(1)) if found else 10
    return {
        "tool": "top_n",
        "params": {"sort_col": sort_col, "n": max(1, min(n, 200)),
                   "ascending": direction},
        "explanation": "The {} rows with the {} {}.".format(
            n, "lowest" if direction else "highest", sort_col),
    }


def _rule_group_extreme(q, raw, df, columns, named) -> Optional[Dict]:
    """"Which Department has the highest average Monthly Income?"

    The first suggestion the page offers, and the shape people type most.
    """
    if not any(w in q for w in _HIGH + _LOW):
        return None
    dims = _categorical(df, named)
    metrics = _numeric(df, named)
    if not dims or not metrics:
        return None
    agg = _agg_from(q) or "mean"
    if agg == "count":
        agg = "mean"
    return {
        "tool": "aggregate",
        "params": {"group_col": dims[0], "value_col": metrics[0],
                   "agg_func": agg},
        "explanation": "{} {} by {}, ranked.".format(
            agg.capitalize(), metrics[0], dims[0]),
    }


def _rule_group_compare(q, raw, df, columns, named) -> Optional[Dict]:
    """"Compare Years at Company across Department" / "sales by region"."""
    if not any(w in q for w in ("compare", " by ", "across", "per ",
                                "breakdown", "grouped", "split by")):
        return None
    dims = _categorical(df, named)
    metrics = _numeric(df, named)
    if not dims or not metrics:
        return None
    agg = _agg_from(q) or "mean"
    if agg == "count":
        agg = "mean"
    if any(w in q for w in ("chart", "plot", "graph", "bar", "visual")):
        return {
            "tool": "plot_bar",
            "params": {"x": dims[0], "y": metrics[0],
                       "title": "{} by {}".format(metrics[0], dims[0])},
            "explanation": "{} by {}.".format(metrics[0], dims[0]),
        }
    return {
        "tool": "aggregate",
        "params": {"group_col": dims[0], "value_col": metrics[0],
                   "agg_func": agg},
        "explanation": "{} {} across {}.".format(
            agg.capitalize(), metrics[0], dims[0]),
    }


def _rule_distribution(q, raw, df, columns, named) -> Optional[Dict]:
    """"Distribution of X" / "histogram of X" / "spread of X"."""
    if not any(w in q for w in ("distribut", "histogram", "spread of",
                                "how is", "shape of")):
        return None
    metrics = _numeric(df, named)
    if not metrics:
        return None
    return {
        "tool": "plot_histogram",
        "params": {"column": metrics[0], "nbins": 30,
                   "title": "Distribution of {}".format(metrics[0])},
        "explanation": "How {} is distributed.".format(metrics[0]),
    }


def _rule_value_counts(q, raw, df, columns, named) -> Optional[Dict]:
    """"How many of each Department" / "count by Category"."""
    if not any(w in q for w in ("how many", "count", "number of",
                                "breakdown of", "frequency")):
        return None
    dims = _categorical(df, named)
    if not dims:
        return None
    # A count crossed with a measure is an aggregate, not a tally.
    if _numeric(df, named):
        return None
    return {
        "tool": "count_values",
        "params": {"column": dims[0]},
        "explanation": "How many records fall in each {}.".format(dims[0]),
    }


def _rule_describe(q, raw, df, columns, named) -> Optional[Dict]:
    """"Tell me about X" / "summary of X" / "stats for X"."""
    if not any(w in q for w in ("describe", "summar", "statistics", "stats",
                                "tell me about", "what is")):
        return None
    if not named:
        return None
    return {
        "tool": "describe_column",
        "params": {"column": named[0]},
        "explanation": "A summary of {}.".format(named[0]),
    }


def answerable_examples(df: pd.DataFrame, limit: int = 4) -> List[str]:
    """Questions this parser can answer on THIS frame, for an error
    message that helps rather than apologises."""
    columns = [str(c) for c in df.columns]
    nums = _numeric(df, columns)
    cats = [c for c in _categorical(df, columns)
            if 1 < df[c].nunique(dropna=True) <= 50]
    out: List[str] = []
    if cats and nums:
        out.append("Which {} has the highest average {}?".format(
            cats[0], nums[0]))
    if nums:
        out.append("Show the 10 rows with the highest {}".format(nums[0]))
    if cats:
        out.append("How many records in each {}?".format(cats[0]))
    if len(nums) >= 2:
        out.append("Which columns move together?")
    if nums:
        out.append("Distribution of {}".format(nums[0]))
    return out[:limit]
