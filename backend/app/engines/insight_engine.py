import pandas as pd
import numpy as np
from typing import List, Dict


def generate_insights(df: pd.DataFrame) -> List[Dict]:
    insights  = []
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    # ── 1. Dataset size ───────────────────────────────────
    insights.append({
        "title": f"Dataset has {len(df):,} rows and {len(df.columns)} columns",
        "body":  f"{len(num_cols)} numeric, {len(cat_cols)} categorical, {len(date_cols)} datetime columns.",
        "type":  "info",
        "icon":  "📋"
    })

    # ── 2. Numeric trends ─────────────────────────────────
    for col in num_cols[:3]:
        s = df[col].dropna()
        if len(s) < 2:
            continue

        cv = s.std() / s.mean() if s.mean() != 0 else 0
        if cv > 1:
            insights.append({
                "title": f"High variability in '{col}'",
                "body":  f"Values range from {s.min():,.1f} to {s.max():,.1f}.",
                "type":  "warning",
                "icon":  "📊"
            })

        mid    = len(s) // 2
        first  = s.iloc[:mid].mean()
        second = s.iloc[mid:].mean()
        change = (second - first) / max(abs(first), 1) * 100

        if abs(change) > 15:
            direction = "increased" if change > 0 else "decreased"
            insights.append({
                "title": f"'{col}' has {direction} over time",
                "body":  f"Average changed by {change:+.1f}% from first to second half.",
                "type":  "positive" if change > 0 else "negative",
                "icon":  "📈" if change > 0 else "📉"
            })

    # ── 3. Top category ───────────────────────────────────
    if cat_cols and num_cols:
        try:
            grp     = df.groupby(cat_cols[0])[num_cols[0]].sum()
            top     = grp.idxmax()
            top_val = grp.max()
            total   = df[num_cols[0]].sum()
            pct     = top_val / total * 100 if total else 0
            insights.append({
                "title": f"'{top}' dominates {num_cols[0]}",
                "body":  f"Accounts for {pct:.1f}% of total.",
                "type":  "info",
                "icon":  "🏆"
            })
        except Exception:
            pass

    # ── 4. Strong correlation ─────────────────────────────
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        vals = corr.where(mask).stack()

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
