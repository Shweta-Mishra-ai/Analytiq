"""
engines/plain_language.py — the same finding, said in words.

Analytiq is meant to work for a finance director and for the analyst
sitting next to them. Those two need the same result described twice,
not one result pitched at the midpoint: "1 feature(s) have high VIF
(multicollinearity)" is precise and useless to the first reader, while
"some columns overlap" is friendly and useless to the second.

So nothing here softens a computation. Every function takes numbers that
have already been produced by the statistics and returns a sentence that
says what they mean for this dataset and what follows from it. The
technical text keeps its place beside it; a reader who knows what a
variance inflation factor is loses nothing.

Three rules hold throughout:

  * Name the consequence, not the test. "Kruskal-Wallis" tells a reader
    what was run; "the difference is bigger than chance would produce"
    tells them what came of it.
  * Keep the number. A plain sentence without a figure is an opinion,
    and the figure is the reason to believe the sentence.
  * Never promise more than the statistic supports. Association stays
    association here exactly as it does in the technical wording.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

from app.engines.present import label as _L


# ══════════════════════════════════════════════════════════
#  SHAPE OF A SINGLE COLUMN
# ══════════════════════════════════════════════════════════

def skew_plain(column: str, skew: float, mean: Optional[float] = None,
               median: Optional[float] = None) -> str:
    """What a skewed column does to the average a reader would quote."""
    name = _L(column)
    if skew is None:
        return ""
    direction = "high" if skew > 0 else "low"
    gap = ""
    if mean is not None and median is not None and median:
        gap = (" The average is {:,.1f} but the middle value is {:,.1f} — "
               "quote the middle one; it is what a typical record looks "
               "like.".format(mean, median))
    if abs(skew) < 0.5:
        return "{} is spread evenly around its average.".format(name)
    strength = "a few" if abs(skew) < 2 else "a long tail of"
    return ("{} is pulled by {} unusually {} values, so its average sits "
            "above what most records actually show.{}".format(
                name, strength, direction, gap)
            if skew > 0 else
            "{} is pulled by {} unusually {} values, so its average sits "
            "below what most records actually show.{}".format(
                name, strength, direction, gap))


def normality_plain(column: str, normal_enough: Optional[bool]) -> str:
    """Why normality matters to someone who will never run a t-test."""
    name = _L(column)
    if normal_enough is None:
        return "{}: not enough values to judge its shape.".format(name)
    if normal_enough:
        return ("{} follows the usual bell shape, so averages and standard "
                "ranges describe it fairly.".format(name))
    return ("{} is not bell-shaped, so anything quoting its average alone "
            "will mislead. Comparisons on this column use methods that do "
            "not assume that shape.".format(name))


def kurtosis_plain(column: str, excess_kurtosis: Optional[float]) -> str:
    """Tail weight, without the word kurtosis."""
    if excess_kurtosis is None:
        return ""
    name = _L(column)
    if excess_kurtosis > 1:
        return ("{} produces extreme values more often than a normal "
                "spread would — plan for the outliers rather than treating "
                "them as mistakes.".format(name))
    if excess_kurtosis < -1:
        return ("{} has fewer extremes than usual; values cluster in a "
                "narrow band.".format(name))
    return "{} has an ordinary number of extreme values.".format(name)


def entropy_plain(column: str, entropy: Optional[float],
                  n_categories: int, top_value: Optional[str] = None,
                  top_pct: Optional[float] = None) -> str:
    """Entropy is a number about balance. Say balance."""
    if entropy is None or not n_categories:
        return ""
    name = _L(column)
    import math
    even = math.log2(n_categories) if n_categories > 1 else 0
    share = (entropy / even) if even else 1.0
    if share > 0.9:
        return ("{} is spread evenly across its {} groups — no single one "
                "dominates.".format(name, n_categories))
    if top_value is not None and top_pct is not None:
        return ("{} is concentrated: '{}' alone accounts for {:.0f}% of "
                "records, so figures split by this column lean heavily on "
                "that group.".format(name, top_value, top_pct))
    return ("{} is unevenly spread across its {} groups.".format(
        name, n_categories))


def outliers_plain(column: str, pct: Optional[float], n: int = 0) -> str:
    if not pct:
        return ""
    name = _L(column)
    return ("{:.1f}% of {} records ({:,}) sit far outside the normal range. "
            "They are kept, not deleted — an extreme value is as often a "
            "real event as a typing error, and checking which is a "
            "judgement only you can make.".format(pct, name, n))


# ══════════════════════════════════════════════════════════
#  RELATIONSHIPS BETWEEN COLUMNS
# ══════════════════════════════════════════════════════════

def correlation_plain(col_a: str, col_b: str, r: float,
                      significant: bool, n: Optional[int] = None) -> str:
    a, b = _L(col_a), _L(col_b)
    if not significant:
        return ("{} and {} show no dependable relationship in this "
                "data.".format(a, b))
    move = "up and down together" if r > 0 else "in opposite directions"
    shared = r * r * 100
    strength = ("almost perfectly" if abs(r) >= 0.9 else
                "closely" if abs(r) >= 0.7 else
                "somewhat" if abs(r) >= 0.4 else "weakly")
    tail = " across {:,} records.".format(n) if n else "."
    return ("{} and {} move {}, {}: knowing one accounts for about {:.0f}% "
            "of what varies in the other{} This is a pattern, not proof "
            "that either one causes the other.".format(
                a, b, move, strength, shared, tail))


def vif_plain(feature: str, vif: float, verdict: str,
              partner: Optional[str] = None) -> str:
    """The one that most needs translating.

    "1 feature(s) have high VIF (multicollinearity): revenue. Remove or
    combine before regression modeling." is exactly right and tells a
    business reader nothing about why they should care.
    """
    name = _L(feature)
    with_partner = " with {}".format(_L(partner)) if partner else ""
    if verdict in ("High", "Severe"):
        overlap = max(0.0, (1 - 1 / vif) * 100) if vif else 0
        return ("{} repeats information already carried by other "
                "columns{} — about {:.0f}% of it. A model given both "
                "cannot tell which one is doing the work, so it will still "
                "predict well but its account of which one matters "
                "is not trustworthy. Keep one of them.".format(
                    name, with_partner, overlap))
    if verdict == "Moderate":
        return ("{} partly overlaps with other columns{}. Worth watching "
                "if you build a model that has to explain its "
                "reasoning.".format(name, with_partner))
    return ("{} carries information of its own, separate from the other "
            "columns.".format(name))


def group_difference_plain(numeric_col: str, group_col: str,
                           n_groups: int, significant: bool,
                           effect_label: Optional[str] = None,
                           best: Optional[str] = None,
                           worst: Optional[str] = None) -> str:
    metric, dim = _L(numeric_col), _L(group_col)
    if not significant:
        return ("{} is much the same across every {} — the gaps between "
                "them are no bigger than ordinary variation.".format(
                    metric, dim))
    size = {"large": "a wide gap", "medium": "a clear gap",
            "small": "a narrow but real gap"}.get(
                (effect_label or "").lower(), "a real gap")
    named = ""
    if best and worst:
        named = " The highest is '{}' and the lowest '{}'.".format(best, worst)
    return ("{} differs across the {} {} groups by more than chance would "
            "produce — {}.{} Which end is the good one depends on what {} "
            "measures.".format(metric, n_groups, dim, size, named, metric))


def trend_plain(column: str, trend: Optional[str],
                is_stationary: Optional[bool]) -> str:
    name = _L(column)
    if trend == "upward":
        base = "{} is rising over the period covered.".format(name)
    elif trend == "downward":
        base = "{} is falling over the period covered.".format(name)
    else:
        base = ("{} has no consistent direction over the period — it moves "
                "around a steady level.".format(name))
    if is_stationary is False:
        base += (" Its level shifts over time, so a figure from one period "
                 "should not be quoted as if it held throughout.")
    return base


# ══════════════════════════════════════════════════════════
#  THE HEADLINE LIST
# ══════════════════════════════════════════════════════════

def plain_findings(report) -> List[str]:
    """The report's key findings, restated for a reader who does not
    know the vocabulary. Returned alongside the technical list, never
    instead of it."""
    out: List[str] = []

    non_normal = [c for c, r in report.univariate.items()
                  if r.is_normal is False and r.mean is not None]
    if non_normal:
        names = ", ".join(_L(c) for c in non_normal[:4])
        out.append(
            "{} of your number columns ({}) are not evenly spread — a few "
            "large values pull the average up. Where this report compares "
            "groups on those columns it uses methods that do not assume an "
            "even spread, and quotes the middle value rather than the "
            "average.".format(len(non_normal), names))

    heavy = [(c, r) for c, r in report.univariate.items()
             if r.outlier_pct and r.outlier_pct > 5]
    if heavy:
        col, res = max(heavy, key=lambda x: x[1].outlier_pct)
        out.append(outliers_plain(col, res.outlier_pct,
                                  res.outliers_iqr or 0))

    flagged = [m for m in report.multicollinearity
               if m.verdict in ("High", "Severe")]
    if flagged:
        worst = max(flagged, key=lambda m: m.vif)
        out.append(vif_plain(worst.feature, worst.vif, worst.verdict))

    strong = [r for r in report.correlations
              if r.is_significant and r.effect_size
              and abs(r.effect_size) >= 0.5]
    if strong:
        top = strong[0]
        out.append(correlation_plain(top.col_a, top.col_b,
                                     top.effect_size or 0.0, True))

    if report.identifier_cols:
        ids = report.identifier_cols[:4]
        names = ", ".join(_L(c) for c in ids)
        one = len(ids) == 1
        out.append(
            "{} {} a reference number rather than a measurement, so {} "
            "left out of the analysis. Averaging an ID produces a number "
            "with no meaning.".format(
                names, "is" if one else "are",
                "it is" if one else "they are")
            if one else
            "{} are reference numbers rather than measurements, so they "
            "are left out of the analysis. Averaging an ID produces a "
            "number with no meaning.".format(names))

    return out
