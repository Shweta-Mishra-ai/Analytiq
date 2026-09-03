"""
engines/charts/selection.py — which charts a report should contain.

The judgement half. Not every column pair makes a chart worth a page:
an identifier plotted against anything is noise, a metric with no spread
across a dimension says nothing, and a breakdown by a protected
characteristic is a legal problem rather than an insight. This module
decides what is worth drawing; plots.py draws it.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")                      # no display; PNG bytes only
import pandas as pd                        # noqa: E402


logger = logging.getLogger(__name__)

from typing import List, NamedTuple, Optional, Tuple

from app.engines.domains.base import is_id_column
from app.engines.charts.style import (
    _SCORE_KEYWORDS, _agg_for_metric, _is_grouping_column, _pretty,
)
from app.engines.charts.plots import (
    make_bar_chart, make_correlation_heatmap, make_histogram,
    make_line_chart, make_ranked_bar_chart,
)


_ID_METRIC_KW = ("id", "index", "number", "code", "zip", "phone", "guid", "uuid")




# Generated missingness companions (see data_cleaner). They belong to
# modelling, not to a client-facing chart: "Monthly Income Was Missing by
# Department" is not a finding.
def _is_generated_indicator(col) -> bool:
    return str(col).endswith("__was_missing")


def _rank_measures(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Order numeric columns by how much they look like a business measure.

    Charting the first numeric column in the frame is how a finance report
    ended up with "invoice_id by category", a distribution of invoice_id,
    and a pie chart of summed invoice IDs — three of five charts plotting
    an identifier. Identifiers are dropped outright; the rest are ranked so
    recognised measures (revenue, profit, cost, amount…) lead, then any
    column that actually varies.
    """
    MEASURE_WORDS = (
        "revenue", "sales", "profit", "margin", "cost", "expense", "amount",
        "total", "price", "value", "spend", "budget", "income", "salary",
        "score", "rating", "quantity", "qty", "units", "count", "rate",
        "satisfaction", "tenure", "age", "duration", "balance", "opex",
    )
    scored = []
    for c in cols:
        if is_id_column(c, df[c]) or _is_generated_indicator(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty or s.nunique() <= 1:
            continue          # constant column charts as a flat line — useless
        name = str(c).lower()
        score = 0
        if any(w in name for w in MEASURE_WORDS):
            score += 10
        # A column with real spread carries more information than a near-
        # constant one; use CV so it is scale-independent.
        mean = float(s.mean())
        if mean != 0:
            cv = abs(float(s.std()) / mean)
            score += min(cv, 2.0)
        scored.append((score, c))
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored]


def _pick_best_metric(num_cols, df=None, cat_cols=None):
    """
    Headline metric = a numeric column with REAL variation, not an ID and not
    a flat Likert score. The old version always returned the first
    'satisfaction/level' column, which on many datasets is a near-constant
    1-5 scale — producing charts where every bar looks identical.
    """
    if not num_cols:
        return None
    if df is None:
        for c in num_cols:
            if any(kw in c.lower() for kw in _SCORE_KEYWORDS):
                return c
        return num_cols[0]

    best, best_score = None, -1.0
    for c in num_cols:
        cl = c.lower()
        if any(k in cl for k in _ID_METRIC_KW) or _is_generated_indicator(c):
            continue
        # Value-aware check as well as the name check: a monotonic 1-step
        # integer sequence is an identifier whatever it is called.
        if df is not None and c in df.columns and is_id_column(c, df[c]):
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 10 or s.nunique() < 3:
            continue
        m = float(s.mean())
        cv = float(s.std()) / abs(m) if m else 0.0
        score = min(cv, 2.0)                       # variation is the main signal
        if any(k in cl for k in _SCORE_KEYWORDS):  # mild tiebreak toward KPIs
            score += 0.15
        if score > best_score:
            best, best_score = c, score
    # No `or num_cols[0]` fallback: when every numeric column is an
    # identifier, the honest answer is "nothing to chart". The dataforge
    # original fell back to the first column, which put "Record Id by
    # Label" into a client report.
    return best


# Sensitive / demographic dimensions. Charting an arbitrary metric BY these
# ('Stock Option Level by Marital Status') manufactures a spurious — and
# sometimes discriminatory — comparison. Only chart by them when the metric
# is clearly related (e.g. a pay-equity review), never as the auto-headline.
_SENSITIVE_DIMS = ("marital", "gender", "sex", "race", "ethnic", "religion",
                   "nationality", "disab", "age", "orientation")


def _best_metric_by_category(df, num_cols, cat_cols, exclude_pairs=None,
                             allow_sensitive=False):
    """
    Pick the (categorical, numeric) pair where the metric's group means vary
    the MOST (relative spread) — the chart that actually shows a finding.
    Skips sensitive/demographic dimensions so we never auto-generate a
    spurious 'metric by marital status' chart. Returns (cat, metric, spread).
    """
    exclude_pairs = exclude_pairs or set()
    best = None
    for cat in cat_cols:
        if not allow_sensitive and any(k in cat.lower() for k in _SENSITIVE_DIMS):
            continue
        nun = df[cat].nunique(dropna=True)
        if not (2 <= nun <= 25) or not _is_grouping_column(df, cat):
            continue
        for metric in num_cols:
            if (cat, metric) in exclude_pairs or \
                    any(k in metric.lower() for k in _ID_METRIC_KW) or \
                    _is_generated_indicator(metric) or \
                    is_id_column(metric, df[metric]):
                continue
            try:
                # Rank pairs by the SAME aggregation the chart will render
                # (mean for scores, sum for additive metrics) so the category
                # a chart highlights matches what the narrative reports.
                agg_func, _ = _agg_for_metric(metric)
                g = df.groupby(cat)[metric].agg(agg_func).dropna()
                if len(g) < 2:
                    continue
                lo, hi = float(g.min()), float(g.max())
                spread = (hi - lo) / abs(hi) if hi else 0.0
                if best is None or spread > best[2]:
                    best = (cat, metric, spread)
            except Exception:
                logger.debug("pair spread failed %s x %s", cat, metric, exc_info=True)
    return best


class ChartSpec(NamedTuple):
    """What a chart was actually drawn from.

    The narrator used to recover this by parsing the chart's title back
    into column names. Titles are prettified — ``JobRole`` becomes "Job
    Role" — so the lookup missed, fell through to the first categorical
    column, and captioned a chart of seven job roles with a confident
    paragraph about three departments. Carrying the columns forward makes
    that impossible rather than unlikely.
    """
    kind: str                       # bar | hist | trend | correlation
    metric: Optional[str]           # the measured column, if any
    dimension: Optional[str]        # what it is broken down by, if any


def generate_all_charts(
    df: pd.DataFrame,
    theme_name: str = "Corporate Light",
    max_charts: int = 5,
) -> List[Tuple[str, bytes, "ChartSpec"]]:
    """Auto-generate best charts for this dataset.

    FIX: previously always used num_cols[0] for every chart and a numeric-bin
    line chart with unreadable x-axis labels when no datetime column existed.
    Now picks the best score metric and falls back to a second categorical
    breakdown or ranked bar chart instead of meaningless numeric bins.
    """
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "string"]).columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()
    charts    = []

    best_metric = _pick_best_metric(num_cols, df=df, cat_cols=cat_cols)
    if best_metric is None:
        # Every numeric column is an identifier. Returning nothing beats
        # filling a client report with charts of row numbers.
        logger.info("no non-identifier numeric column to chart among %d",
                    len(num_cols))
        return []

    # Chart 1: the (category, metric) pair that VARIES most — the chart that
    # actually shows a finding, instead of flat bars on a near-constant score.
    pair1 = _best_metric_by_category(df, num_cols, cat_cols) if (num_cols and cat_cols) else None
    if pair1:
        best_cat, chart1_metric, _ = pair1
        title = "{} by {}".format(_pretty(chart1_metric), _pretty(best_cat))
        try:
            charts.append((title,
                           make_bar_chart(df, best_cat, chart1_metric,
                                          title, theme_name),
                           ChartSpec("bar", chart1_metric, best_cat)))
        except Exception:
            logger.warning("chart 1 failed", exc_info=True)
    else:
        best_cat = next((c for c in cat_cols
                         if _is_grouping_column(df, c)
                         and not any(k in c.lower() for k in _SENSITIVE_DIMS)),
                        None) if cat_cols else None
        if best_cat is None and cat_cols:
            # Only fall back to a qualifying column — never force cat_cols[0]
            # regardless of cardinality. A column like a review_id/order_id
            # with tens of thousands of unique values (or, on a small
            # dataset, an ID column whose cardinality coincidentally looks
            # 'reasonable') is not a usable chart dimension just because it
            # happened to be first; forcing it produces charts with as many
            # bars as there are rows.
            best_cat = next((c for c in cat_cols if _is_grouping_column(df, c)), None)
        chart1_metric = best_metric
        if best_cat and best_metric:
            title = "{} by {}".format(_pretty(best_metric), _pretty(best_cat))
            try:
                charts.append((title,
                               make_bar_chart(df, best_cat, best_metric,
                                              title, theme_name),
                               ChartSpec("bar", best_metric, best_cat)))
            except Exception:
                logger.warning("chart 1 fallback failed", exc_info=True)
        # If no categorical dimension qualifies, Chart 1 is simply skipped —
        # Chart 3 below already shows Distribution: {best_metric}, so a
        # histogram fallback here would just be a duplicate of that chart.

    # 2. Second view — datetime trend OR the next most-varying (cat, metric)
    #    pair, never repeating chart 1's pair.
    if date_cols and best_metric:
        title = "{} Over Time".format(_pretty(best_metric))
        try:
            charts.append((title,
                           make_line_chart(df, date_cols[0], best_metric,
                                           title, theme_name),
                           ChartSpec("trend", best_metric, date_cols[0])))
        except Exception:
            logger.warning("chart 2 (trend) failed", exc_info=True)
    else:
        pair2 = (_best_metric_by_category(
            df, num_cols, cat_cols,
            exclude_pairs={(pair1[0], pair1[1])} if pair1 else set())
            if (num_cols and cat_cols) else None)
        if pair2 and pair2[2] > 0.03:
            c2, m2, _ = pair2
            title = "{} by {}".format(_pretty(m2), _pretty(c2))
            try:
                charts.append((title,
                               make_bar_chart(df, c2, m2, title,
                                              theme_name),
                               ChartSpec("bar", m2, c2)))
            except Exception:
                logger.warning("chart 2 (pair) failed", exc_info=True)
        elif len(num_cols) >= 2:
            second_metric = next((c for c in num_cols if c != best_metric), num_cols[0])
            title = "Distribution: {}".format(_pretty(second_metric))
            try:
                charts.append((title,
                               make_histogram(df, second_metric, title,
                                              theme_name),
                               ChartSpec("hist", second_metric, None)))
            except Exception:
                logger.warning("chart 2 (hist) failed", exc_info=True)

    # 3. Histogram — distribution of primary metric
    if best_metric:
        title = "Distribution: {}".format(_pretty(best_metric))
        try:
            charts.append((title,
                           make_histogram(df, best_metric, title, theme_name),
                           ChartSpec("hist", best_metric, None)))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # 4. Correlation heatmap
    if len(num_cols) >= 3:
        try:
            charts.append(("Correlation Matrix",
                           make_correlation_heatmap(df, "Correlation Matrix",
                                                    theme_name),
                           ChartSpec("correlation", None, None)))
        except Exception:
            logger.warning("%s unexpected failure", exc_info=True)

    # 5. Ranked bar of a DIFFERENT metric than charts 1-3 used. Repeating the
    #    same variable a 4th time is what makes a report feel padded/AI-made.
    #    Pick the second metric with the most spread (highest CV), ranked by
    #    the category where it varies most.
    alt_metric = _pick_second_metric(df, num_cols, exclude={best_metric})
    if alt_metric and best_cat:
        rank_cat = _pick_high_spread_dimension(df, alt_metric, exclude=set()) or best_cat
        title = "{} Ranking by {}".format(_pretty(alt_metric), _pretty(rank_cat))
        try:
            charts.append((title,
                           make_ranked_bar_chart(df, rank_cat, alt_metric,
                                                 title, theme_name),
                           ChartSpec("bar", alt_metric, rank_cat)))
        except Exception:
            logger.warning("ranked chart 5 failed", exc_info=True)
    elif best_cat and best_metric:
        # Only one numeric metric exists — a ranked view of it is still useful
        title = "{} Ranking by {}".format(_pretty(best_metric), _pretty(best_cat))
        try:
            charts.append((title,
                           make_ranked_bar_chart(df, best_cat, best_metric,
                                                 title, theme_name),
                           ChartSpec("bar", best_metric, best_cat)))
        except Exception:
            logger.warning("ranked chart 5 fallback failed", exc_info=True)

    return charts[:max_charts]


def _pick_second_metric(df, num_cols, exclude=None):
    """Numeric column (not in exclude, not an ID) with the highest coefficient
    of variation — the metric that varies most, hence most worth a second chart."""
    exclude = exclude or set()
    best, best_cv = None, 0.0
    for c in num_cols:
        if c in exclude or any(k in c.lower() for k in ("id", "number", "code")):
            continue
        try:
            s = df[c].dropna()
            if len(s) < 20 or s.nunique() < 5:
                continue
            m = float(s.mean())
            cv = float(s.std()) / abs(m) if m else 0.0
            if cv > best_cv:
                best, best_cv = c, cv
        except Exception:
            logger.debug("cv check failed for %s", c, exc_info=True)
    return best


def _pick_high_spread_dimension(df, metric, exclude=None):
    """Categorical column (2-15 groups) whose group-means of `metric` show
    the LARGEST relative spread — where the metric actually varies."""
    exclude = exclude or set()
    best, best_spread = None, 0.0
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c in exclude or not _is_grouping_column(df, c, max_unique=15):
            continue
        if any(k in c.lower() for k in _SENSITIVE_DIMS):
            continue
        try:
            g = df.groupby(c)[metric].mean().dropna()
            if len(g) < 2:
                continue
            lo, hi = float(g.min()), float(g.max())
            spread = (hi - lo) / abs(hi) if hi else 0.0
            if spread > best_spread:
                best, best_spread = c, spread
        except Exception:
            logger.debug("spread check failed for %s", c, exc_info=True)
    return best if best_spread > 0.05 else None
