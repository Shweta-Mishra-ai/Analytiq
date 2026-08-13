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

from typing import List, Optional

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


def cramers_v(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    """Cramér's V — effect size for a contingency table.

    A chi-square p-value only says "not independent"; on a large table it
    goes significant for a difference too small to act on. V rescales the
    statistic to 0–1 so a finding can be judged on strength, not just on
    whether it cleared 0.05.
    """
    k = min(n_rows, n_cols)
    if n <= 0 or k <= 1:
        return 0.0
    try:
        return float((chi2 / (n * (k - 1))) ** 0.5)
    except (ValueError, ZeroDivisionError):
        logger.debug("cramers_v failed for chi2=%r n=%r", chi2, n, exc_info=True)
        return 0.0


def chi2_validity(expected) -> tuple:
    """Is a chi-square result trustworthy for this table?

    Returns (is_valid, reason). Pearson's chi-square relies on a normal
    approximation that degrades when expected cell counts are small; the
    standard rule is that no expected count may fall below 1 and at most
    20% may fall below 5. Below that, the p-value is not interpretable and
    Fisher's exact test is the correct choice.

    This was previously unchecked — every chi2_contingency call in the app
    took the p-value at face value, so a sparse crosstab could produce a
    confident "significant driver" that would not survive review.
    """
    try:
        import numpy as _np
        exp = _np.asarray(expected, dtype=float)
        if exp.size == 0:
            return False, "empty contingency table"
        n_below_5 = int((exp < 5).sum())
        pct_below_5 = n_below_5 / exp.size * 100
        if float(exp.min()) < 1:
            return False, ("an expected cell count is below 1 — chi-square is "
                           "not valid for this table")
        if pct_below_5 > 20:
            return False, (f"{pct_below_5:.0f}% of expected cell counts are "
                           "below 5 (limit 20%) — chi-square is unreliable here")
        return True, ""
    except Exception:
        logger.warning("chi2 validity check failed", exc_info=True)
        return False, "could not verify chi-square assumptions"


def chi2_association(ct) -> Optional[dict]:
    """Chi-square test of independence with its assumptions actually checked.

    Returns None when the table cannot support the test, so callers skip
    the finding instead of reporting an uninterpretable p-value. On success
    returns chi2, p, dof, n, Cramér's V and an effect label.
    """
    try:
        from scipy import stats as _st
        chi2, p, dof, expected = _st.chi2_contingency(ct)
    except Exception:
        logger.debug("chi2_contingency failed", exc_info=True)
        return None

    valid, reason = chi2_validity(expected)
    if not valid:
        logger.debug("chi-square skipped: %s", reason)
        return None

    n = int(getattr(ct, "values", ct).sum())
    v = cramers_v(float(chi2), n, *getattr(ct, "shape", (2, 2)))
    label = ("negligible" if v < 0.1 else "small" if v < 0.3
             else "moderate" if v < 0.5 else "strong")
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof), "n": n,
            "cramers_v": round(v, 4), "effect_label": label}


def apply_fdr(findings: List[dict], p_key: str = "p",
              q_key: str = "q") -> List[dict]:
    """Annotate a family of tests with Benjamini-Hochberg q-values.

    Scanning every column for "drivers" runs dozens of tests at α=0.05; on
    20 independent columns roughly one false positive is expected by pure
    chance. Reporting that as a driver is how an analysis loses a review.
    Callers should filter on the q-value, not the raw p.
    """
    if not findings:
        return []
    qs = bh_adjust([float(f.get(p_key, 1.0)) for f in findings])
    out = []
    for f, q in zip(findings, qs):
        f = dict(f)
        f[q_key] = round(float(q), 6)
        out.append(f)
    return out


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
