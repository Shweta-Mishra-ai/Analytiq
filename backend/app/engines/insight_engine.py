"""
engines/insight_engine.py — the quick read of a dataset, for callers
that want a handful of plain statements rather than a full analysis.

Every statement here must be one the data can support. The previous
version could not meet that bar in two ways, and both reached the API:

  * It reported "'revenue' has increased over time" by comparing the
    first half of the rows to the second half. Row order is not time. On
    a file with no date column at all it announced a 60% rise, and the
    same rows shuffled produced no finding — a claim about file order,
    printed as a claim about the business.
  * It grouped by whichever categorical column happened to come first
    and summed whichever numeric column came first, which on an HR
    extract produced "'RM0139' dominates Age — accounts for 0.1% of
    total": an employee ID used as a business dimension.

So: trends need a real date column to order by, dimensions and measures
are chosen for being usable rather than for being first, and identifier
columns take no part in any of it.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from app.engines.domains.base import is_id_column
from app.services.dtypes import text_columns

# A dimension a reader can act on: more than one group, few enough that
# the groups mean something. 400 employee IDs is not a segmentation.
MAX_DIMENSION_LEVELS = 20
MIN_TREND_ROWS = 20


def _measures(df: pd.DataFrame) -> List[str]:
    """Numeric columns that measure something, identifiers excluded."""
    return [c for c in df.select_dtypes(include="number").columns
            if not is_id_column(c, df[c])]


def _dimensions(df: pd.DataFrame) -> List[str]:
    """Categorical columns usable as a grouping, ordered fewest levels
    first — the coarsest split is the one worth leading with."""
    usable = [(df[c].nunique(dropna=True), c) for c in text_columns(df)
              if not is_id_column(c, df[c])
              and 1 < df[c].nunique(dropna=True) <= MAX_DIMENSION_LEVELS]
    return [c for _, c in sorted(usable)]


def _trend(df: pd.DataFrame, date_col: str, col: str) -> Optional[Dict]:
    """Change between the first and second half of a column *in date
    order*. Returns None unless there is a date to order by."""
    pair = df[[date_col, col]].dropna().sort_values(date_col)
    if len(pair) < MIN_TREND_ROWS:
        return None
    s = pair[col]
    mid = len(s) // 2
    first, second = s.iloc[:mid].mean(), s.iloc[mid:].mean()
    if not np.isfinite(first) or not np.isfinite(second):
        return None
    change = (second - first) / max(abs(first), 1e-9) * 100
    if abs(change) <= 15:
        return None
    direction = "risen" if change > 0 else "fallen"
    return {
        "title": f"'{col}' has {direction} over time",
        "body": (f"Average changed by {change:+.1f}% between the earlier "
                 f"and later half of the data, ordered by '{date_col}'."),
        "type": "positive" if change > 0 else "negative",
        "icon": "📈" if change > 0 else "📉",
    }


# A gap smaller than this is noise dressed as a finding.
MATERIAL_GAP_PCT = 10.0


def _widest_gap(df: pd.DataFrame, dimensions: List[str],
                measures: List[str]) -> Optional[Dict]:
    """The clearest difference between groups, or nothing.

    Reporting whichever dimension came first produced findings like "a
    gap of 0% across 3 'segment' groups" — technically true, and not
    worth a reader's attention. This looks across the candidate
    dimensions and measures for a difference big enough to act on, and
    returns None when there isn't one. An empty answer is a result.
    """
    best: Optional[Dict] = None
    best_gap = MATERIAL_GAP_PCT
    for dim in dimensions[:4]:
        for measure in measures[:4]:
            try:
                grp = df.groupby(dim, dropna=True)[measure].mean().dropna()
            except Exception:
                logger.debug("gap search failed on %s/%s", dim, measure,
                             exc_info=True)
                continue
            if len(grp) < 2:
                continue
            high, low = grp.idxmax(), grp.idxmin()
            top, bottom = float(grp.max()), float(grp.min())
            if not bottom:
                continue
            gap = (top - bottom) / abs(bottom) * 100
            if gap <= best_gap:
                continue
            best_gap = gap
            best = {
                "title": f"'{high}' has the highest average {measure}",
                "body": (f"{measure} averages {top:,.1f} in '{high}' against "
                         f"{bottom:,.1f} in '{low}' — a gap of {gap:,.0f}% "
                         f"across {len(grp)} '{dim}' groups. A difference, "
                         f"not yet a cause."),
                "type": "info",
                "icon": "🏆",
            }
    return best


def generate_insights(df: pd.DataFrame) -> List[Dict]:
    insights: List[Dict] = []
    measures = _measures(df)
    dimensions = _dimensions(df)
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    all_num = df.select_dtypes(include="number").columns.tolist()
    all_cat = text_columns(df)

    insights.append({
        "title": f"Dataset has {len(df):,} rows and {len(df.columns)} columns",
        "body": (f"{len(all_num)} numeric, {len(all_cat)} categorical, "
                 f"{len(date_cols)} datetime columns."),
        "type": "info",
        "icon": "📋",
    })

    for col in measures[:3]:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        mean = s.mean()
        cv = s.std() / mean if mean else 0
        if cv > 1:
            insights.append({
                "title": f"High variability in '{col}'",
                "body": f"Values range from {s.min():,.1f} to {s.max():,.1f}.",
                "type": "warning",
                "icon": "📊",
            })
        # A trend is only claimable when something orders the rows in time.
        if date_cols:
            trend = _trend(df, date_cols[0], col)
            if trend:
                insights.append(trend)

    best = _widest_gap(df, dimensions, measures)
    if best:
        insights.append(best)

    if len(measures) >= 2:
        corr = df[measures].corr()
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        vals = corr.where(mask).stack()
        if not vals.empty:
            pair = vals.abs().idxmax()
            r = vals[pair]
            if abs(r) > 0.7:
                direction = "positively" if r > 0 else "negatively"
                insights.append({
                    "title": f"Strong correlation: '{pair[0]}' & '{pair[1]}'",
                    "body": (f"These columns move {direction} together "
                             f"(r = {r:.2f}). Association, not cause."),
                    "type": "info",
                    "icon": "🔗",
                })

    return insights
