import logging
import pandas as pd
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)


from app.services.dtypes import text_columns


def generate_insights(df: pd.DataFrame) -> List[Dict]:
    insights  = []
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = text_columns(df)
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    # ── 1. Dataset size ───────────────────────────────────
    insights.append({
        "title": f"Dataset has {len(df):,} rows and {len(df.columns)} columns",
        "body":  f"{len(num_cols)} numeric, {len(cat_cols)} categorical, {len(date_cols)} datetime columns.",
        "type":  "info",
        "icon":  "📋"
    })

    # ── 2. Numeric variability and trend ──────────────────
    # "Over time" needs a time axis. This used to compare the first and
    # second half of the rows in file order, so any file not sorted by date
    # got a headline like "'salary' has increased over time" about nothing
    # but row order. The trend is now measured along a date column, sorted,
    # and skipped when there is none.
    time_col = date_cols[0] if date_cols else None
    for col in num_cols[:3]:
        s = df[col].dropna()
        if len(s) < 2:
            continue

        mean = s.mean()
        cv = s.std() / abs(mean) if mean != 0 else 0
        if cv > 1:
            insights.append({
                "title": f"High variability in '{col}'",
                "body":  f"Values range from {s.min():,.1f} to {s.max():,.1f}.",
                "type":  "warning",
                "icon":  "📊"
            })

        if time_col is None:
            continue
        series = (df[[time_col, col]].dropna().sort_values(time_col)[col])
        if len(series) < 10:
            continue
        mid    = len(series) // 2
        first  = series.iloc[:mid].mean()
        second = series.iloc[mid:].mean()
        if first == 0:
            continue
        change = (second - first) / abs(first) * 100

        if abs(change) > 15:
            direction = "higher" if change > 0 else "lower"
            insights.append({
                "title": f"'{col}' is {direction} in the later half of the period",
                "body":  (f"Average {change:+.1f}% in the second half of the "
                          f"'{time_col}' range versus the first."),
                "type":  "positive" if change > 0 else "negative",
                "icon":  "📈" if change > 0 else "📉"
            })

    # ── 3. Top category ───────────────────────────────────
    if cat_cols and num_cols:
        try:
            grp     = df.groupby(cat_cols[0])[num_cols[0]].sum()
            total   = df[num_cols[0]].sum()
            # A share only means something when every part is non-negative.
            top_two = grp.nlargest(2)
            # ...and naming a "largest" only means something when it leads:
            # on a tie idxmax just returns whichever group sorts first.
            if (len(grp) >= 2 and total > 0 and (grp >= 0).all()
                    and top_two.iloc[0] > top_two.iloc[1]):
                top     = grp.idxmax()
                pct     = grp.max() / total * 100
                insights.append({
                    "title": f"'{top}' has the largest total {num_cols[0]}",
                    "body":  (f"{pct:.1f}% of the total across {len(grp)} "
                              f"'{cat_cols[0]}' values."),
                    "type":  "info",
                    "icon":  "🏆"
                })
        except Exception:
            logger.debug("generate_insights: suppressed exception", exc_info=True)

    # ── 4. Strong correlation ─────────────────────────────
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        # pandas 3 keeps NaN through stack(); a constant or empty column
        # yields only NaN correlations and idxmax() then raised, turning the
        # whole Insights page into a 500.
        vals = corr.where(mask).stack().dropna()

        if not vals.empty:
            pair = vals.abs().idxmax()
            r    = vals[pair]
            if abs(r) > 0.7:
                direction = "positively" if r > 0 else "negatively"
                insights.append({
                    "title": f"Strong correlation: '{pair[0]}' & '{pair[1]}'",
                    "body":  f"These columns are {direction} correlated (r = {r:.2f}).",
                    "type":  "info",
                    "icon":  "🔗"
                })

    return insights
