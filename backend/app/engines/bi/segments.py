"""
engines/bi/segments.py — which segments are healthy and which are not.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


from typing import List

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

            status = ("top" if rank <= max(1, n_seg // 3)
                      else "bottom" if rank > n_seg - max(1, n_seg // 3)
                      else "mid")

            score_val = 50 + vs_avg * 0.5
            score_val = max(0, min(100, score_val))
            scores.append(score_val)

            metrics[col] = {
                "mean":    round(seg_mean, 4),
                "vs_avg":  round(vs_avg, 2),
                "rank":    rank,
                "n_total": n_seg,
                "status":  status,
            }

        health_score = round(float(np.mean(scores)), 1) if scores else 50.0
        strengths    = [col for col, m in metrics.items()
                        if m["vs_avg"] > 10]
        weaknesses   = [col for col, m in metrics.items()
                        if m["vs_avg"] < -10]

        # Opportunity
        if weaknesses:
            worst_col = min(metrics.items(),
                            key=lambda x: x[1]["vs_avg"])[0]
            opp = ("Improve '{}' from {:.2f} to dataset average {:.2f} "
                   "— {:.1f}% improvement opportunity.".format(
                       worst_col,
                       metrics[worst_col]["mean"],
                       overall.get(worst_col, 0),
                       abs(metrics[worst_col]["vs_avg"])))
        elif strengths:
            best_col = max(metrics.items(),
                           key=lambda x: x[1]["vs_avg"])[0]
            opp = ("Already leading in '{}' — "
                   "leverage this advantage in other segments.".format(best_col))
        else:
            opp = "Performance close to average across all metrics."

        results.append(SegmentHealth(
            segment_name=str(seg),
            segment_col=segment_col,
            n=int(len(seg_df)),
            metrics=metrics,
            health_score=health_score,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunity=opp,
        ))

    return sorted(results, key=lambda x: x.health_score, reverse=True)


# ══════════════════════════════════════════════════════════