"""
engines/bi/segments.py — which segments are healthy and which are not.

Health is not "bigger numbers". Scoring every metric as though more is
better ranked the strongest region in a test dataset LAST — highest
revenue, half the churn, 40% lower support cost — listed its low churn
and low cost as weaknesses, and recommended raising churn from 0.04 to
the dataset average of 0.08. Direction is read from the column name, and
a column whose direction the name does not settle is left out of the
score rather than guessed at.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


from typing import List

from app.engines.domains.base import higher_is_better

from app.engines.bi.results import SegmentHealth


#  SEGMENT HEALTH SCORING
# ══════════════════════════════════════════════════════════

def analyze_segment_health(
    df: pd.DataFrame,
    segment_col: str,
    metric_cols: List[str],
) -> List[SegmentHealth]:
    """
    Score each segment across multiple metrics.
    Identifies strengths, weaknesses, opportunities.
    """
    segments = df[segment_col].dropna().unique()
    valid    = [s for s in segments if (df[segment_col] == s).sum() >= 5]

    if len(valid) < 2 or not metric_cols:
        return []

    # Overall means for comparison
    overall = {col: float(df[col].mean()) for col in metric_cols
               if col in df.columns and pd.api.types.is_numeric_dtype(df[col])}
    if not overall:
        return []

    # Which way is good, per metric. None means the column name does not
    # say — 'value', 'amount', 'index'. Those are still reported, with
    # their rank and their distance from average, but they take no part
    # in the health score and are never called a strength or a weakness:
    # calling a number good requires knowing which way good runs.
    direction = {col: higher_is_better(col) for col in overall}

    results = []
    for seg in valid[:10]:
        seg_df  = df[df[segment_col] == seg]
        metrics = {}
        scores  = []

        for col in metric_cols:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            seg_mean = float(seg_df[col].mean())
            avg      = overall.get(col, seg_mean)
            vs_avg   = ((seg_mean - avg) / abs(avg) * 100
                        if avg != 0 else 0)

            # Rank among all segments
            seg_means = df.groupby(segment_col)[col].mean().sort_values(ascending=False)
            rank = int((seg_means.index.tolist().index(seg) + 1)
                       if seg in seg_means.index else len(seg_means))
            n_seg = len(seg_means)

            up = direction.get(col)

            # `rank` counts down from the largest mean, so on a
            # lower-is-better metric the top of that list is the worst
            # segment. Flip it, or the lowest-cost region is reported as
            # ranked last on cost.
            if up is False:
                rank = n_seg - rank + 1

            status = ("top" if rank <= max(1, n_seg // 3)
                      else "bottom" if rank > n_seg - max(1, n_seg // 3)
                      else "mid")
            if up is None:
                status = "n/a"

            # How far above average in the direction that is good.
            favourable_pct = vs_avg if up is not False else -vs_avg

            if up is not None:
                score_val = 50 + favourable_pct * 0.5
                scores.append(max(0, min(100, score_val)))

            metrics[col] = {
                "mean":    round(seg_mean, 4),
                "vs_avg":  round(vs_avg, 2),
                "rank":    rank,
                "n_total": n_seg,
                "status":  status,
                "direction": ("higher" if up is True
                              else "lower" if up is False else "unknown"),
                "favourable_pct": round(favourable_pct, 2) if up is not None else None,
            }

        health_score = round(float(np.mean(scores)), 1) if scores else 50.0

        # Strength and weakness are judged on the favourable direction,
        # never on the raw sign. Metrics of unknown direction sit out.
        judged     = {c: m for c, m in metrics.items()
                      if m["favourable_pct"] is not None}
        strengths  = [c for c, m in judged.items() if m["favourable_pct"] > 10]
        weaknesses = [c for c, m in judged.items() if m["favourable_pct"] < -10]

        # Opportunity
        if weaknesses:
            worst_col = min(judged.items(),
                            key=lambda x: x[1]["favourable_pct"])[0]
            worst = metrics[worst_col]
            verb = "Raise" if worst["direction"] == "higher" else "Bring down"
            opp = ("{} '{}' from {:.2f} to the dataset average of {:.2f} "
                   "— {:.1f}% behind the average in the direction that "
                   "helps.".format(
                       verb, worst_col, worst["mean"],
                       overall.get(worst_col, 0),
                       abs(worst["favourable_pct"])))
        elif strengths:
            best_col = max(judged.items(),
                           key=lambda x: x[1]["favourable_pct"])[0]
            opp = ("Already leading in '{}' — "
                   "leverage this advantage in other segments.".format(best_col))
        elif judged:
            opp = "Performance close to average across all metrics."
        else:
            # Nothing here can be called better or worse without knowing
            # which way each metric runs. Say that, rather than inventing
            # a ranking out of which numbers happen to be larger.
            opp = ("No metric here has a direction its name makes clear, "
                   "so this segment cannot be called healthier or weaker "
                   "than another — only different. The figures above are "
                   "its position, not a verdict.")

        results.append(SegmentHealth(
            segment_name=str(seg),
            segment_col=segment_col,
            n=int(len(seg_df)),
            metrics=metrics,
            health_score=health_score,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunity=opp,
            scored=bool(scores),
        ))

    return sorted(results, key=lambda x: x.health_score, reverse=True)


# ══════════════════════════════════════════════════════════