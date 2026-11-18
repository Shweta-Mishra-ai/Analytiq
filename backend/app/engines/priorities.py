"""
engines/priorities.py — what to act on first.

Each finding that carries an action is ranked by the money acting on it
could move — stated with how that figure was reached — and by how firm the
finding is. A large figure on thin evidence ranks below a smaller one that
was tested: the score is the money weighted by the evidence (tested or
measured 1.0, indicative 0.5).

Sources: the scorecard's findings (discounting, segment margin, returns,
customer concentration) and the segments that drove the latest change
in the main measure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_WEIGHT = {"tested": 1.0, "measured": 1.0, "beyond chance": 1.0, "indicative": 0.5}


@dataclass
class Priority:
    rank: int
    finding: str
    action: str
    value: float
    value_basis: str
    evidence: str
    level: str          # High / Medium / Low by share of total revenue

    @property
    def strength(self) -> str:
        return self.evidence.split(":")[0].strip().lower()


def build_priorities(df: pd.DataFrame, change=None, limit: int = 6) -> List[Priority]:
    from app.engines.scorecard import build_scorecard
    sc = build_scorecard(df)
    if not sc.revenue_col:
        return []
    # Every value is put on a per-year footing: a discount finding spans the
    # whole file (18 months, say), a change between periods one quarter.
    # Ranked side by side unscaled, they compared different spans of time.
    per_year = _per_year_factor(df, sc)
    items = []
    for m in sc.messages:
        if m.lead or m.value <= 0 or not m.evidence:
            continue
        if m.value_basis.startswith("revenue of the largest customer"):
            # Exposure, not a rate: what would go if that customer left.
            items.append((m.title, m.action, m.value * per_year,
                          m.value_basis.split(" over ")[0].replace(
                              "revenue of", "a year's revenue at risk from losing"),
                          m.evidence))
            continue
        items.append((m.title, m.action, m.value * per_year,
                      m.value_basis + ", a year at the rate of the data", m.evidence))
    if change is None:
        from app.engines.change_drivers import explain_change
        try:
            change = explain_change(df)
        except Exception:
            logger.warning("change explanation for priorities failed", exc_info=True)
            change = None
    if change is not None and change.where:
        months = change.months or (12 if change.period.startswith("12") else 3)
        items.extend([(f, a, v * 12 / months, b + ", a year if it continues", e)
                      for f, a, v, b, e in _from_change(change)])

    total = max(sc.total_revenue, 1e-9)
    scored = sorted(items, key=lambda it: -it[2] * _WEIGHT.get(
        it[4].split(":")[0].strip().lower(), 0.5))
    out = []
    for i, (finding, action, value, basis, evidence) in enumerate(scored[:limit], 1):
        share = value / total
        out.append(Priority(i, finding, action, value, basis, evidence,
                            "High" if share >= 0.05 else "Medium" if share >= 0.01 else "Low"))
    return out


def _per_year_factor(df: pd.DataFrame, sc) -> float:
    """Multiplier from the span of the data to a year (1 without dates)."""
    if not sc.date_col:
        return 1.0
    d = pd.to_datetime(df[sc.date_col], errors="coerce")
    days = (d.max() - d.min()).days + 1 if d.notna().any() else 0
    return 365.0 / days if days >= 60 else 1.0


def _from_change(e) -> List[tuple]:
    """A segment that fell (or, for a fall overall, the one that fell most)
    is a finding to act on; one that grew is an opportunity to find out
    why and repeat it."""
    from app.engines.chart_exporter import fmt
    out = []
    falls = [c for c in e.where.contributions if c.delta < 0][:2]
    for c in falls:
        within = ""
        if e.within and e.within_segment == c.segment and e.within.contributions:
            top = e.within.contributions[0]
            if top.delta < 0:
                within = ", most of it in '{}' ({})".format(top.segment, fmt(top.delta))
        out.append((
            "'{}' fell {} in '{}' ({:+.0f}%) in the {}{}".format(
                e.measure, fmt(abs(c.delta)), c.segment, c.pct or 0, e.period, within),
            "Find out what changed in '{}' — prices, stock, a lost customer or a "
            "competitor — before the next period closes.".format(c.segment),
            abs(c.delta),
            "the fall in '{}' against the {}".format(c.segment, e.prior_period),
            "Beyond chance: {:.1f} times the change chance would produce".format(abs(c.z))))
    rises = [c for c in e.where.contributions if c.delta > 0][:1]
    for c in rises:
        out.append((
            "'{}' grew {} in '{}' ({:+.0f}%) in the {}".format(
                e.measure, fmt(c.delta), c.segment, c.pct or 0, e.period),
            "Find what drove the growth in '{}' and whether it can be repeated in the "
            "segments that fell.".format(c.segment),
            c.delta,
            "the growth in '{}' against the {}".format(c.segment, e.prior_period),
            "Beyond chance: {:.1f} times the change chance would produce".format(abs(c.z))))
    return out
