"""
engines/insights_builder.py
Turns the story engine's top insights into the cards the main report renders.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    severity: str          # critical | high | warning | info
    title: str
    problem: str
    cause: str
    evidence: str
    action: str
    impact: str


def build_top_insights(
    df: pd.DataFrame,
    domain: str = "general",
    story_obj=None,
    attrition=None,
    bi_report=None,
    avg_salary_k: float = 50.0,
) -> list:
    """
    Build structured insights list for PDF top_insights section.
    Works with any domain. Returns list of Insight objects.
    """
    insights = []

    # ── 1. Try story_obj.top_insights first ──────────────────────────────────
    raw = getattr(story_obj, "top_insights", None)
    if raw and isinstance(raw, list) and len(raw) > 0:
        for item in raw:
            if hasattr(item, "severity"):
                insights.append(item)
            elif isinstance(item, dict):
                insights.append(Insight(
                    severity = item.get("severity", "info"),
                    title    = item.get("title", ""),
                    problem  = item.get("problem", ""),
                    cause    = item.get("cause", ""),
                    evidence = item.get("evidence", ""),
                    action   = item.get("action", ""),
                    impact   = item.get("impact", ""),
                    # Without this the report blueprint cannot tell an
                    # attrition finding from a margin one, and every
                    # finding lands under whichever heading comes first.
                    category = item.get("category", "general"),
                ))
        if insights:
            return insights[:6]

    # ── 2. Nothing from the domain engines: no section ─────────────────────
    # This used to build cards from scratch, and every one named a cause the
    # data could not show and a source nobody could trace ("Mercer 2024:
    # career growth = #1 voluntary exit reason", "below industry norm",
    # "top 20% of accounts likely driving 80% of revenue"). A report section
    # left out is better than one a reader can catch inventing.
    return []
