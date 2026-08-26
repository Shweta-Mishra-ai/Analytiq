"""
engines/kpi_engine.py — the numbers a reader of this kind of data looks
for first.

WHAT THIS REPLACES
------------------
The KPI panel used to return four data-quality counts (rows, columns,
missing %, duplicates) followed by the *sum of the first four numeric
columns*, whatever they happened to be. On an HR extract that produced
"Σ EmployeeNumber" and "Σ Age" — a total of employee ID numbers, and a
total of ages. Neither is a quantity, and neither is a KPI.

Two things were missing. The aggregation has to match the metric: a total
of a satisfaction rating means nothing, and a total of an identifier means
less. And the panel has to know what business it is looking at: the first
figure on an HR dashboard is attrition, on a finance one it is margin, on
a marketing one it is return on spend.

Specs live on DomainSpec.kpis, so a new domain brings its own KPIs in the
same file as its insight engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from app.engines.domains._common import (
    binary_rate, find_col, find_measure,
)

logger = logging.getLogger(__name__)

# Aggregations a KPI may use. "rate" is the share of a binary column;
# "ratio" divides one resolved column by another.
KINDS = ("count", "sum", "mean", "median", "rate", "ratio", "nunique")


@dataclass(frozen=True)
class KpiSpec:
    """One number, and how to get it from an arbitrary dataset."""
    key: str
    label: str
    kind: str
    # Column-name fragments to look for, most specific first.
    columns: Tuple[str, ...] = ()
    # For "ratio": what to divide by.
    denominator: Tuple[str, ...] = ()
    unit: str = ""
    # Key into industry_benchmarks for this domain, when a published range
    # exists for the metric.
    benchmark: str = ""
    # True when more is better, False when less is, None when it depends.
    # Drives nothing but the wording — a KPI panel that colours "defect
    # rate" green for being high is worse than one with no colour at all.
    higher_is_better: Optional[bool] = None
    exclude: Tuple[str, ...] = ()
    note: str = ""


@dataclass
class KpiCard:
    label: str
    value: Optional[float]
    unit: str = ""
    format: str = "num"
    source_column: str = ""
    benchmark: str = ""
    higher_is_better: Optional[bool] = None
    note: str = ""

    def as_dict(self) -> Dict:
        return {
            "label": self.label, "value": self.value, "unit": self.unit,
            "format": self.format, "source_column": self.source_column,
            "benchmark": self.benchmark,
            "higher_is_better": self.higher_is_better, "note": self.note,
        }


def _fmt_kind(kind: str, unit: str) -> str:
    if unit == "%":
        return "pct"
    if kind in ("count", "nunique"):
        return "int"
    return "num"


def _resolve(df: pd.DataFrame, spec: KpiSpec) -> Optional[str]:
    if spec.kind in ("count",):
        return None
    if spec.kind == "rate":
        # A rate needs a binary column, not merely a numeric one.
        for kw in spec.columns:
            col = find_col(df, [kw], exclude=spec.exclude)
            if col is not None and binary_rate(df[col]) is not None:
                return col
        return None
    return find_measure(df, spec.columns, exclude=spec.exclude)


def compute_kpi(df: pd.DataFrame, spec: KpiSpec,
                domain: str = "general") -> Optional[KpiCard]:
    """One KPI, or None when this dataset cannot support it.

    Returning None matters: a panel that shows "Gross Margin —" for a
    dataset with no cost column is claiming to have looked and found
    nothing, when in fact the question did not apply.
    """
    try:
        if spec.kind == "count":
            value, col = float(len(df)), ""
        else:
            col = _resolve(df, spec)
            if col is None:
                return None
            series = df[col]
            if spec.kind == "rate":
                value = binary_rate(series)
                if value is None:
                    return None
            elif spec.kind == "nunique":
                value = float(series.nunique(dropna=True))
            elif spec.kind == "ratio":
                denom_col = find_measure(df, spec.denominator,
                                         exclude=spec.exclude)
                if denom_col is None:
                    return None
                num = float(pd.to_numeric(series, errors="coerce").sum())
                den = float(pd.to_numeric(df[denom_col],
                                          errors="coerce").sum())
                if den == 0 or not np.isfinite(den):
                    return None
                value = num / den
                if spec.unit == "%":
                    value *= 100
                col = "{} / {}".format(col, denom_col)
            else:
                num = pd.to_numeric(series, errors="coerce").dropna()
                if num.empty:
                    return None
                value = float(getattr(num, spec.kind)())
                # A share stored as 0-1 reads as 0.3% rather than 30%.
                if spec.unit == "%" and abs(value) <= 1.0:
                    value *= 100
        if value is None or not np.isfinite(float(value)):
            return None
    except Exception:
        logger.debug("KPI %r could not be computed", spec.key, exc_info=True)
        return None

    benchmark = ""
    if spec.benchmark:
        try:
            from app.engines.industry_benchmarks import (
                DOMAIN_BENCHMARKS, format_benchmark_context)
            bm = DOMAIN_BENCHMARKS.get(domain, {}).get(spec.benchmark)
            if bm is not None:
                benchmark = format_benchmark_context(bm)
        except Exception:
            logger.debug("benchmark lookup failed for %r", spec.key,
                         exc_info=True)

    return KpiCard(
        label=spec.label, value=round(float(value), 2), unit=spec.unit,
        format=_fmt_kind(spec.kind, spec.unit), source_column=str(col),
        benchmark=benchmark, higher_is_better=spec.higher_is_better,
        note=spec.note)


def _fallback_kpis(df: pd.DataFrame, limit: int) -> List[KpiCard]:
    """When the domain has no specs, or none of them resolved.

    Ranked measures with the aggregation each one deserves, rather than
    the sum of whatever came first. Identifiers and generated missingness
    indicators are excluded.
    """
    cards: List[KpiCard] = []
    try:
        from app.engines.chart_exporter import _agg_for_metric, _rank_measures
        numeric = df.select_dtypes(include="number").columns.tolist()
        for col in _rank_measures(df, numeric)[:limit]:
            agg, is_score = _agg_for_metric(col)
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            value = float(series.sum() if agg == "sum" else series.mean())
            cards.append(KpiCard(
                label="{} {}".format("Total" if agg == "sum" else "Average",
                                     str(col).replace("_", " ").strip()),
                value=round(value, 2),
                format="num", source_column=str(col),
                note=("Averaged rather than totalled — a total of a score "
                      "has no meaning." if is_score else "")))
    except Exception:
        logger.warning("fallback KPIs failed", exc_info=True)
    return cards


def compute_kpis(df: pd.DataFrame, domain: str = "general",
                 limit: int = 6) -> List[KpiCard]:
    """The headline numbers for this dataset, in this domain."""
    if df is None or df.empty:
        return []
    try:
        from app.engines.domains.registry import spec_for
        specs = spec_for(domain).kpis
    except Exception:
        logger.warning("KPI specs unavailable for domain %r", domain,
                       exc_info=True)
        specs = ()

    cards: List[KpiCard] = []
    for spec in specs:
        card = compute_kpi(df, spec, domain)
        if card is not None:
            cards.append(card)
        if len(cards) >= limit:
            break

    if len(cards) < 2:
        # Not enough of this domain's vocabulary is present to fill a
        # panel — fall back to the strongest measures in the frame.
        seen = {c.source_column for c in cards}
        for card in _fallback_kpis(df, limit - len(cards)):
            if card.source_column not in seen:
                cards.append(card)
    return cards[:limit]


def data_quality_cards(df: pd.DataFrame) -> List[KpiCard]:
    """Shape and completeness.

    Kept, but returned separately: these describe the file, not the
    business, and presenting "Duplicates: 3" beside "Revenue: 4.2M" as
    though they answer the same kind of question is what made the old
    panel unreadable.
    """
    if df is None or df.empty:
        return []
    return [
        KpiCard(label="Records", value=float(len(df)), format="int"),
        KpiCard(label="Fields", value=float(df.shape[1]), format="int"),
        KpiCard(label="Missing", value=round(float(df.isna().mean().mean())
                                             * 100, 1),
                unit="%", format="pct", higher_is_better=False),
        KpiCard(label="Repeated rows", value=float(df.duplicated().sum()),
                format="int", higher_is_better=False,
                note="Reported, not removed — see Data Preparation."),
    ]
