"""
services/stat_guards.py — statistical honesty helpers.

Used by insight/correlation code so the app never reports a relationship
that doesn't survive basic scrutiny:
  - minimum sample size,
  - p-values with Benjamini-Hochberg correction across the whole family
    of tests (many column pairs tested → many false positives otherwise),
  - effect-size floors,
  - a confidence label the UI/reports can show next to every claim.
"""
from __future__ import annotations
import logging

from typing import List

logger = logging.getLogger(__name__)

MIN_N = 30            # below this, report nothing
EFFECT_FLOOR = 0.30   # |r| below this is noise for business narratives
FDR_Q = 0.05          # accepted false-discovery rate


def bh_adjust(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    prev = 1.0
    for rank_from_end, idx in enumerate(reversed(order)):
        rank = m - rank_from_end
        q = min(prev, pvals[idx] * m / rank)
        adjusted[idx] = q
        prev = q
    return adjusted


def confidence_label(n: int, q: float, effect: float) -> str:
    """high / medium / low confidence for a correlation-style finding."""
    e = abs(effect)
    if n >= 100 and q < 0.01 and e >= 0.5:
        return "high"
    if n >= MIN_N and q < FDR_Q and e >= EFFECT_FLOOR:
        return "medium"
    return "low"


def filter_correlations(results: List[dict]) -> List[dict]:
    """Apply the full honesty pipeline to correlation dicts that carry
    'r', 'p' and 'n'. Returns only findings that survive, each annotated
    with 'q' (adjusted p) and 'confidence'."""
    tested = [r for r in results if r.get("n", 0) >= MIN_N]
    if not tested:
        return []
    qvals = bh_adjust([r["p"] for r in tested])
    out = []
    for r, q in zip(tested, qvals):
        if q < FDR_Q and abs(r["r"]) >= EFFECT_FLOOR:
            r = dict(r)
            r["q"] = round(float(q), 5)
            r["confidence"] = confidence_label(r["n"], q, r["r"])
            out.append(r)
    return out
