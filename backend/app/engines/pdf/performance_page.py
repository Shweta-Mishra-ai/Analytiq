"""
pdf/performance_page.py — the page every domain gets.

Split out of domain_sections when that module crossed the size the
package guard allows. It is one job: render the section that turns a
domain's registered vocabulary — its KPIs, its outcome, its benchmark
ranges — into a page a practitioner in that field recognises as theirs.

Twelve of the thirteen domains had no deep page at all; finance had a
bespoke one and the rest went straight from the findings to descriptive
statistics, so a healthcare report and a logistics report were the same
report with different words in the narrative.
"""
from __future__ import annotations

import logging

import pandas as pd
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Spacer

from app.engines.present import label as _PL, truncate as _fit
from app.engines.pdf.primitives import _gtable, _sec

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  THE PERFORMANCE PAGE EVERY DOMAIN GETS
# ══════════════════════════════════════════════════════════
# Twelve of the thirteen domains had no deep page at all: finance had a
# bespoke one and the rest went straight from the findings to the
# descriptive statistics. So a healthcare report and a logistics report
# were the same report with different words in the narrative, and
# neither carried a page a practitioner in that field would recognise as
# their own.
#
# This page is generic in code and specific in output, because the
# registry already knows what each domain measures: its KPIs, the
# outcome it cares about, which end of that outcome is the bad one, and
# its published benchmark ranges. Everything below is read from the
# spec, so registering a domain gets it this page.

# How many rows of a breakdown a reader will use before it becomes an
# extract. The same discipline as the period table above.
BREAKDOWN_ROWS = 8

# A dimension with more levels than this is an identifier in disguise;
# below two there is nothing to compare.
MAX_DIMENSION_LEVELS = 12
MIN_DIMENSION_LEVELS = 2

# A group smaller than this cannot carry a rate anybody should act on.
MIN_GROUP_ROWS = 25


def _verdict_against(card, domain: str = "general") -> str:
    """How this KPI reads against its published range, in words.

    The KPI carries a benchmark *key* ("attrition_rate"), which is what
    indexes the published table. Passing it to the column-name lookup —
    which is what an earlier version did — matched nothing, so every row
    of every scorecard read "No published range" even where a sourced
    range existed.
    """
    try:
        from app.engines.industry_benchmarks import DOMAIN_BENCHMARKS
        from app.engines.domains.registry import spec_for
        if card.value is None:
            return "No published range — read against your own history."
        # The card carries the range already rendered as prose. The
        # numeric bounds needed for a verdict live in the published
        # table, indexed by the key on the KPI's own spec.
        key = next((k.benchmark for k in spec_for(domain).kpis
                    if k.label == card.label and getattr(k, "benchmark", "")),
                   "")
        if not key:
            return "No published range — read against your own history."
        bm = (DOMAIN_BENCHMARKS.get(str(domain).lower(), {}) or {}).get(key)
        if bm is None:
            return "No published range — read against your own history."
        low = float(getattr(bm, "low", 0) or 0)
        high = float(getattr(bm, "high", 0) or 0)
        if high <= 0:
            return "No published range — read against your own history."
        value = float(card.value)
        if value < low:
            side = "below"
        elif value > high:
            side = "above"
        else:
            return "Within the typical range of {:g}-{:g}.".format(low, high)
        good = card.higher_is_better
        if good is None:
            reading = "outside"
        elif (side == "above") == bool(good):
            reading = "better than"
        else:
            reading = "worse than"
        return "{} the typical {:g}-{:g} range — {} typical.".format(
            side.capitalize(), low, high, reading)
    except Exception:
        logger.debug("benchmark verdict failed", exc_info=True)
        return "No published range — read against your own history."


def _dimensions(df):
    """Categorical columns worth breaking a rate down by."""
    from app.engines.domains.base import is_id_column
    out = []
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            if is_id_column(col, df[col]):
                continue
            levels = int(df[col].nunique(dropna=True))
            if MIN_DIMENSION_LEVELS <= levels <= MAX_DIMENSION_LEVELS:
                out.append((col, levels))
        except Exception:
            logger.debug("dimension check failed on %s", col, exc_info=True)
    out.sort(key=lambda pair: pair[1])
    return [c for c, _ in out]


def _outcome_column(df, spec):
    """The column this domain's outcome lives in, however it is named."""
    from app.engines.domains._common import binary_mask, find_col
    try:
        if spec.outcome_keywords:
            col = find_col(df, spec.outcome_keywords)
            if col is not None and binary_mask(df[col]) is not None:
                return col, (spec.outcome_noun or "the outcome"), spec.outcome_good
    except Exception:
        logger.debug("declared outcome lookup failed", exc_info=True)
    try:
        from app.engines.domains.outcome_discovery import discover_outcomes
        found = discover_outcomes(df, limit=1)
        if found:
            return found[0].column, found[0].noun, bool(found[0].good)
    except Exception:
        logger.debug("outcome discovery failed", exc_info=True)
    return None, "", False


def _domain_performance_page(story, s, T, df, config, CW, profile=None,
                             domain="general"):
    """KPI scorecard, outcome breakdown, and segment ranking."""
    from app.engines.domains.registry import label_for, spec_for
    from app.engines.kpi_engine import compute_kpis

    spec = spec_for(domain)
    label = label_for(domain) or "Business"
    _sec(story, s, T, "{} Performance Analysis".format(label.title()),
         "Headline measures, who the outcome happens to, and how "
         "segments rank — computed from the submitted data")

    wrote_something = False

    # ── 1. KPI scorecard ──────────────────────────────────
    try:
        cards = compute_kpis(df, domain, limit=6)
    except Exception:
        logger.warning("KPI computation failed", exc_info=True)
        cards = []
    if cards:
        wrote_something = True
        story.append(Paragraph("Headline Measures", s["h3"]))
        rows = []
        for card in cards:
            if card.value is None:
                continue
            if card.unit == "%":
                shown = "{:,.1f}%".format(float(card.value))
            elif abs(float(card.value)) >= 1000:
                shown = "{:,.0f}".format(float(card.value))
            else:
                shown = "{:,.2f}".format(float(card.value))
            rows.append([card.label, shown,
                         _fit(_PL(card.source_column or "—"), 24),
                         _verdict_against(card, domain)])
        if rows:
            _gtable(story, T, ["Measure", "Value", "From column", "Reading"],
                    rows, [CW * 0.22, CW * 0.14, CW * 0.22, CW * 0.42])
            story.append(Spacer(1, 3 * mm))

    # ── 2. Who the outcome happens to ─────────────────────
    outcome_col, noun, good = _outcome_column(df, spec)
    if outcome_col is not None:
        try:
            from app.engines.domains._common import binary_mask
            mask = binary_mask(df[outcome_col])
        except Exception:
            logger.debug("outcome mask failed", exc_info=True)
            mask = None
        if mask is not None:
            overall = float(mask.mean() * 100)
            dims = [c for c in _dimensions(df) if c != outcome_col][:2]
            for dim in dims:
                try:
                    frame = pd.DataFrame({dim: df.loc[mask.index, dim],
                                          "_hit": mask.values})
                    grouped = frame.groupby(dim, observed=True)["_hit"]
                    table = grouped.agg(["size", "mean"])
                    table = table[table["size"] >= MIN_GROUP_ROWS]
                    if len(table) < 2:
                        continue
                    table["rate"] = table["mean"] * 100
                    table = table.sort_values("rate", ascending=bool(good))
                    table = table.head(BREAKDOWN_ROWS)
                except Exception:
                    logger.debug("outcome breakdown failed on %s", dim,
                                 exc_info=True)
                    continue
                wrote_something = True
                story.append(Paragraph(
                    "{} by {}".format(noun.title(), _PL(dim)), s["h3"]))
                story.append(Paragraph(
                    "Across the whole file the rate is {:.1f}%. Groups are "
                    "ordered {}-first, and any holding fewer than {} records "
                    "is left out — a rate over a handful of rows moves on one "
                    "case.".format(overall,
                                   "lowest" if good else "highest",
                                   MIN_GROUP_ROWS),
                    s["note"]))
                rows = []
                for name, row in table.iterrows():
                    delta = float(row["rate"]) - overall
                    rows.append([
                        _fit(str(name), 28),
                        "{:,.0f}".format(float(row["size"])),
                        "{:.1f}%".format(float(row["rate"])),
                        "{:+.1f} pp".format(delta),
                    ])
                _gtable(story, T,
                        [_PL(dim), "Records", noun.title() + " rate",
                         "vs overall"],
                        rows, [CW * 0.34, CW * 0.16, CW * 0.25, CW * 0.25])
                story.append(Spacer(1, 3 * mm))

    # ── 3. Where the value sits ───────────────────────────
    try:
        measure = _primary_measure(df, spec)
    except Exception:
        logger.debug("primary measure lookup failed", exc_info=True)
        measure = None
    if measure is not None:
        for dim in _dimensions(df)[:1]:
            try:
                grouped = df.groupby(dim, observed=True)[measure]
                table = grouped.agg(["size", "sum", "median"])
                table = table[table["size"] >= MIN_GROUP_ROWS]
                if len(table) < 2:
                    continue
                total = float(table["sum"].sum())
                if total <= 0:
                    continue
                table = table.sort_values("sum", ascending=False).head(
                    BREAKDOWN_ROWS)
            except Exception:
                logger.debug("segment ranking failed on %s", dim, exc_info=True)
                continue
            wrote_something = True
            story.append(Paragraph(
                "{} by {}".format(_PL(measure), _PL(dim)), s["h3"]))
            rows = []
            running = 0.0
            for name, row in table.iterrows():
                share = float(row["sum"]) / total * 100
                running += share
                rows.append([
                    _fit(str(name), 26),
                    "{:,.0f}".format(float(row["size"])),
                    "{:,.0f}".format(float(row["sum"])),
                    "{:.1f}%".format(share),
                    "{:.1f}%".format(running),
                ])
            _gtable(story, T,
                    [_PL(dim), "Records", "Total " + _PL(measure),
                     "Share", "Cumulative"],
                    rows, [CW * 0.28, CW * 0.14, CW * 0.22, CW * 0.16,
                           CW * 0.20])
            story.append(Paragraph(
                "Cumulative share is the concentration check: when the top "
                "two lines carry most of the column, an average across all "
                "of them describes none of them.", s["note"]))
            story.append(Spacer(1, 3 * mm))

    if not wrote_something:
        # Never leave the heading standing over nothing.
        story.append(Paragraph(
            "This dataset carries no measure, outcome flag or grouping "
            "column that this section can compute from. The findings and "
            "statistics sections cover what it does support.", s["body"]))


def _primary_measure(df, spec):
    """The number this domain is mostly about."""
    from app.engines.domains._common import binary_mask, find_measure
    from app.engines.domains.base import is_id_column

    def _usable(col):
        """A column that can be summed into a meaningful total."""
        if col is None or col not in df.columns:
            return False
        if is_id_column(col, df[col]):
            return False
        # The domain's chart metrics include its outcome — `on_time` for
        # logistics — and that route reached this function before the
        # numeric filter below could reject it.
        return binary_mask(df[col]) is None

    if spec.chart_metrics:
        col = find_measure(df, list(spec.chart_metrics))
        if _usable(col):
            return col
    # A 0/1 flag summed by group is a count wearing the word "Total":
    # a logistics page reported "Total On Time 1,355 — 54.1% share",
    # which is the number of on-time shipments, not an amount of
    # anything. Flags are covered by the rate tables above.
    numeric = [c for c in df.select_dtypes(include="number").columns
               if not is_id_column(c, df[c])
               and df[c].nunique(dropna=True) > 2]
    if not numeric:
        return None
    # The widest-spread column carries the most information.
    best, best_spread = None, -1.0
    for col in numeric:
        try:
            series = df[col].dropna()
            if series.empty or float(series.sum()) <= 0:
                continue
            spread = float(series.std()) / max(abs(float(series.mean())), 1e-9)
            if spread > best_spread:
                best, best_spread = col, spread
        except Exception:
            logger.debug("spread check failed on %s", col, exc_info=True)
    return best
