"""
engines/chart_message.py — what each chart is actually saying.

A chart titled "revenue by region" tells the reader what is plotted and
nothing about what they should take from it. Every consulting deliverable
does the opposite: the headline states the finding — "North delivers 1.8x
the revenue of any other region" — and the variable names live on the
axes where they belong. It is the cheapest change that separates a
document a client reads from one they flick through.

The messages here are computed from the same data the chart plots, and
obey the rules the rest of the report follows:

  - always a figure, never an adjective on its own;
  - no causal language — a bar chart shows that North is higher, never
    why;
  - when the pattern is unremarkable, say so. "Broadly level across
    regions" is a finding; inventing a leader out of a 2% difference is
    how a chart pack becomes noise.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Below this the difference between the top and the rest is not worth a
# headline; the chart still shows the shape.
MIN_LEAD_RATIO = 1.15
MIN_TREND_PCT = 5.0
CONCENTRATION_PCT = 50.0


# One definition of how a figure is written, shared with the axis
# formatter in `chart_exporter` and with the report text, so a headline
# saying "3.2m" sits above an axis that also reads "3.2m".
from app.services.numfmt import human_number  # noqa: E402

_fmt = human_number


def bar_message(df: pd.DataFrame, x_col: str, y_col: str,
                *, counts: bool = False) -> Optional[str]:
    """Which group leads, and by how much.

    `counts=True` is for a headcount-style chart — "how many rows per
    group" — where x and y are the same column and there is nothing to
    sum. `df.groupby(x)[x].sum()` would add a categorical column to
    itself; a count is the right aggregation and `y_col` is only a
    label here, not a column read from `df`.
    """
    try:
        if counts:
            agg = df.groupby(x_col).size().sort_values(ascending=False)
        else:
            agg = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
        agg = agg[agg > 0]
        if len(agg) < 2:
            return None
        top, second = float(agg.iloc[0]), float(agg.iloc[1])
        share = top / float(agg.sum()) * 100
        # "Dept is broadly level across Dept" repeats the column name
        # for no reason when x and y are the same one — say what the
        # number actually is instead.
        measure = "records" if counts else y_col
        if second > 0 and top / second >= MIN_LEAD_RATIO:
            return ("{} leads {} at {} — {:.1f}x the next group and {:.0f}% "
                    "of the total".format(
                        str(agg.index[0])[:24], measure, _fmt(top),
                        top / second, share))
        return ("{} {} broadly level across {} — the highest group is only "
                "{:.0f}% above the next".format(
                    measure, "are" if counts else "is", x_col,
                    (top / second - 1) * 100 if second else 0))
    except Exception:
        logger.debug("bar message failed", exc_info=True)
        return None


def comparison_message(actual_col: str, plan_col: str, actual_total: float,
                       plan_total: float, worst_group: str = "") -> str:
    """An actual against its plan, written the way it is spoken.

    "revenue came in -7% against budget" makes the reader decode a sign
    before they can read the sentence, and the sign is the whole point.
    "7% below budget" is the same figure in the words a finance reader
    would use out loud.
    """
    if not plan_total:
        return ""
    variance = (actual_total - plan_total) / abs(plan_total) * 100
    if abs(variance) < 0.5:
        head = "{} landed on {} — {} against {}".format(
            actual_col, plan_col, _fmt(actual_total), _fmt(plan_total))
    else:
        head = "{} came in {:.0f}% {} {} — {} against {}".format(
            actual_col, abs(variance),
            "above" if variance > 0 else "below", plan_col,
            _fmt(actual_total), _fmt(plan_total))
    if worst_group:
        head += ", {} furthest behind".format(str(worst_group)[:24])
    return head


def line_message(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[str]:
    """Where the series ended relative to where it started."""
    try:
        work = df[[x_col, y_col]].dropna().sort_values(x_col)
        if len(work) < 6:
            return None
        values = work[y_col].astype(float).to_numpy()
        window = max(len(values) // 10, 1)
        start = float(values[:window].mean())
        end = float(values[-window:].mean())
        if start == 0:
            return None
        change = (end - start) / abs(start) * 100
        if abs(change) < MIN_TREND_PCT:
            return ("{} held broadly flat across the period, within {:.0f}% "
                    "of where it started".format(y_col, abs(change)))
        return ("{} {} {:.0f}% across the period, from {} to {}".format(
            y_col, "rose" if change > 0 else "fell", abs(change),
            _fmt(start), _fmt(end)))
    except Exception:
        logger.debug("line message failed", exc_info=True)
        return None


def histogram_message(df: pd.DataFrame, col: str) -> Optional[str]:
    """Where the mass of the distribution sits."""
    try:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 20:
            return None
        mean, median = float(s.mean()), float(s.median())
        if median == 0:
            return None
        gap = (mean - median) / abs(median) * 100
        if abs(gap) < 10:
            return ("{} is symmetric around {} — the average describes a "
                    "typical record".format(col, _fmt(median)))
        direction = "above" if gap > 0 else "below"
        return ("{} is skewed: the average ({}) sits {:.0f}% {} the median "
                "({}), so the median is the fairer summary".format(
                    col, _fmt(mean), abs(gap), direction, _fmt(median)))
    except Exception:
        logger.debug("histogram message failed", exc_info=True)
        return None


def pie_message(df: pd.DataFrame, x_col: str, y_col: str) -> Optional[str]:
    """How concentrated the total is."""
    try:
        agg = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
        agg = agg[agg > 0]
        total = float(agg.sum())
        if len(agg) < 2 or total <= 0:
            return None
        top_share = float(agg.iloc[0]) / total * 100
        if top_share >= CONCENTRATION_PCT:
            return ("{} of {} comes from {} alone".format(
                "{:.0f}%".format(top_share), y_col, str(agg.index[0])[:24]))
        two = float(agg.iloc[:2].sum()) / total * 100
        return ("{} is spread across {} groups — the largest holds {:.0f}%, "
                "the top two {:.0f}%".format(
                    y_col, len(agg), top_share, two))
    except Exception:
        logger.debug("pie message failed", exc_info=True)
        return None


def scatter_message(df: pd.DataFrame, x_col: str, y_col: str
                    ) -> Optional[str]:
    """Whether two measures move together, stated as association.

    Used when a chart pairs two numeric columns with no time axis
    between them — the honest alternative to calling that pairing a
    "trend". A trend needs a date; a relationship only needs two
    measures, and it is captioned as one.
    """
    try:
        work = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(work) < 20:
            return None
        r = float(work[x_col].corr(work[y_col]))
        if not np.isfinite(r):
            return None
        strength = ("closely" if abs(r) >= 0.6 else
                    "loosely" if abs(r) >= 0.3 else "barely")
        if abs(r) < 0.15:
            return ("{} and {} show no real relationship (r={:.2f}) — "
                    "each varies independently of the other".format(
                        x_col, y_col, r))
        direction = "rises" if r > 0 else "falls"
        return ("{} and {} move {} together (r={:.2f}) — {} {} as {} "
                "increases, an association rather than a cause".format(
                    x_col, y_col, strength, r, y_col, direction, x_col))
    except Exception:
        logger.debug("scatter message failed", exc_info=True)
        return None


def heatmap_message(df: pd.DataFrame, cols: Optional[list] = None
                    ) -> Optional[str]:
    """The strongest relationship on the matrix, stated as association."""
    try:
        num = df[cols] if cols else df.select_dtypes(include="number")
        if num.shape[1] < 2:
            return None
        corr = num.corr(numeric_only=True).abs()
        # `.values` on a frame is read-only under copy-on-write, so
        # fill_diagonal raised and the message was silently dropped — the
        # heatmap kept its "Correlation Matrix" placeholder title and no
        # test noticed, because a missing message is a valid outcome.
        matrix = corr.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(matrix, 0.0)
        corr = pd.DataFrame(matrix, index=corr.index, columns=corr.columns)
        if corr.empty or not np.isfinite(matrix).any():
            return None
        flat = corr.stack().sort_values(ascending=False)
        # A near-perfect correlation is usually one column derived from
        # the other, which is a data note rather than a finding.
        flat = flat[flat < 0.99]
        if flat.empty:
            return None
        (a, b), value = flat.index[0], float(flat.iloc[0])
        if value < 0.3:
            return ("No pair of measures moves together strongly — the "
                    "highest association is {:.2f}".format(value))
        return ("{} and {} move together most closely (r={:.2f}) — an "
                "association, not a cause".format(a, b, value))
    except Exception:
        logger.debug("heatmap message failed", exc_info=True)
        return None
