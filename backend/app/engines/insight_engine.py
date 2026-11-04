import logging
import pandas as pd
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)


from app.services.dtypes import text_columns


def _totals_differ(df: pd.DataFrame, cat: str, num: str) -> bool:
    """Whether a group total leads for a reason other than chance: a total
    is (rows × value per row), so either the row counts must depart from an
    even split (chi-square) or the values per row must differ between the
    groups (Kruskal-Wallis)."""
    from scipy import stats as _sc
    sub = df[[cat, num]].dropna()
    counts = sub[cat].value_counts()
    try:
        p_counts = float(_sc.chisquare(counts.values).pvalue) if len(counts) >= 2 else 1.0
    except ValueError:
        p_counts = 1.0
    groups = [g[num].values for _, g in sub.groupby(cat) if len(g) >= 3]
    try:
        p_vals = float(_sc.kruskal(*groups).pvalue) if len(groups) >= 2 else 1.0
    except ValueError:
        p_vals = 1.0
    return min(p_counts, p_vals) < 0.025       # two tests, Bonferroni


def generate_insights(df: pd.DataFrame) -> List[Dict]:
    insights  = []
    from app.engines.domains.base import is_id_column
    num_cols  = [c for c in df.select_dtypes(include="number").columns
                 if not pd.api.types.is_bool_dtype(df[c]) and not is_id_column(c, df[c])]
    cat_cols  = [c for c in text_columns(df) if not is_id_column(c, df[c])]
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

        # A 15% swing between halves is ordinary on small or noisy data, so
        # the halves must also differ by more than chance (Mann-Whitney,
        # Bonferroni over the columns checked).
        try:
            from scipy.stats import mannwhitneyu
            p_half = float(mannwhitneyu(series.iloc[:mid], series.iloc[mid:]).pvalue)
        except ValueError:
            p_half = 1.0
        if abs(change) > 15 and p_half < 0.05 / max(len(num_cols[:3]), 1):
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
                if _totals_differ(df, cat_cols[0], num_cols[0]):
                    insights.append({
                        "title": f"'{top}' has the largest total {num_cols[0]}",
                        "body":  (f"{pct:.1f}% of the total across {len(grp)} "
                                  f"'{cat_cols[0]}' values."),
                        "type":  "info",
                        "icon":  "🏆"
                    })
                else:
                    # A leader by a margin chance would produce is not one.
                    insights.append({
                        "title": f"'{num_cols[0]}' is spread about evenly across "
                                 f"'{cat_cols[0]}'",
                        "body":  (f"'{top}' has the most ({pct:.1f}% of the total "
                                  f"across {len(grp)} values), but neither the "
                                  f"number of rows nor the values per row differ "
                                  f"between them by more than chance."),
                        "type":  "info",
                        "icon":  "📊"
                    })
        except Exception:
            logger.warning("generate_insights: suppressed exception", exc_info=True)

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
            # |r| > 0.7 happens by chance on a handful of rows; the strongest
            # of many pairs must also pass a test corrected for all of them.
            both = df[[pair[0], pair[1]]].dropna()
            p_r = 1.0
            if len(both) >= 10:
                from scipy.stats import pearsonr
                p_r = float(pearsonr(both[pair[0]], both[pair[1]]).pvalue)
            if abs(r) > 0.7 and p_r < 0.05 / len(vals):
                direction = "positively" if r > 0 else "negatively"
                insights.append({
                    "title": f"Strong correlation: '{pair[0]}' & '{pair[1]}'",
                    "body":  f"These columns are {direction} correlated (r = {r:.2f}).",
                    "type":  "info",
                    "icon":  "🔗"
                })

    return insights
