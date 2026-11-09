"""
domains/general_depth.py — the analyses that apply whatever the data is
about.

Most files a freelancer receives are not recognisably HR, finance, sales
or e-commerce. They land in the general engine, which could describe
distributions, surface correlations and compare two segments — a
description of the data rather than an account of the business in it.

The three analyses here need no domain vocabulary and answer questions a
client actually asks:

  - Is the main measure trending, and is the trend distinguishable from
    noise? A slope with no significance test is a line drawn through
    scatter.
  - Are the extreme values concentrated somewhere, or spread? Ten
    outliers in one segment is a process problem; ten spread across
    forty is the tail of a distribution.
  - Which segments are too small to conclude anything about? A 90%
    figure on nine records is the finding a client's own analyst will
    challenge first, and it is better challenged here.

Every result is significance-tested and effect-floored before it is
reported, on the same basis as the domain engines.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.engines.domains.base import build_insight, is_id_column
from app.services.dtypes import text_columns
from app.services.stat_guards import MIN_N

logger = logging.getLogger(__name__)

# A trend explaining less than this of the variance is a line through
# scatter, however significant the slope.
MIN_TREND_R2 = 0.10
# Unusual months. Each month is compared with the six before it, scaled by
# the series' own month-to-month noise (robust: MAD of the changes). The
# cut-off is Bonferroni over the months tested, never below 4, with a 30%
# floor on the change. Simulated on pure noise, trend and ±40% seasonality
# this flags at most 4% of series with no real event; a six-month MAD as
# the scale flagged 88%. Monthly totals of fewer than 30 records are mostly
# noise and are not tested.
UNUSUAL_WINDOW = 6
UNUSUAL_MIN_Z = 4.0
UNUSUAL_MIN_CHANGE = 0.30
UNUSUAL_MIN_ROWS = 30
# Below this a segment's rate is an anecdote; a client's analyst will say
# so before anyone acts on it.
MIN_SEGMENT_N = 30


def _measures(df: pd.DataFrame) -> List[str]:
    return [c for c in df.select_dtypes(include="number").columns
            if not is_id_column(c, df[c]) and df[c].nunique(dropna=True) > 2]


def _time_column(df: pd.DataFrame):
    """The date records happened on. A hire, birth or start date is not
    one: a measure "trending" over hire date is tenure and seniority showing
    through, which the report printed as a key finding about salary."""
    from app.engines.chart_exporter import _event_date
    return _event_date(df)


# ══════════════════════════════════════════════════════════
#  TREND
# ══════════════════════════════════════════════════════════

def trend_over_time(df: pd.DataFrame, insights: List, findings: List,
                    risks: List, opps: List) -> None:
    """Is the main measure moving, and is the movement real?

    Reported with R² alongside the direction, because the two answer
    different questions: the slope says which way, R² says whether the
    line describes the data at all. A significant slope through a cloud
    with an R² of 0.02 is a true statement about a trend nobody could
    act on.
    """
    from app.engines.chart_exporter import _measures as _ranked_measures
    time_col = _time_column(df)
    measures = _ranked_measures(df)
    if not time_col or not measures:
        return
    try:
        target = measures[0]
        work = df[[time_col, target]].dropna().sort_values(time_col)
        if len(work) < MIN_N:
            return

        x = (work[time_col] - work[time_col].min()).dt.total_seconds().to_numpy()
        y = work[target].astype(float).to_numpy()
        if x.std() == 0 or y.std() == 0:
            return

        fit = scipy_stats.linregress(x, y)
        r2 = float(fit.rvalue ** 2)
        if fit.pvalue >= 0.05 or r2 < MIN_TREND_R2:
            findings.append(
                "'{}' shows no trend over the period beyond ordinary variation "
                "(R²={:.2f}, p={:.2g}, n={:,}).".format(
                    target, r2, fit.pvalue, len(work)))
            return

        span_days = max((work[time_col].max() - work[time_col].min()).days, 1)
        per_day = float(fit.slope) * 86400
        change = per_day * span_days
        start_level = float(y[:max(len(y) // 10, 1)].mean())
        pct = (change / start_level * 100) if start_level else 0.0
        direction = "rising" if change > 0 else "falling"

        findings.append(
            "'{}' is {} over the {:,}-day period covered: {:+,.1f} in total "
            "({:+.1f}% against the opening level), R²={:.2f}, p={:.2g}, "
            "n={:,}.".format(target, direction, span_days, change, pct, r2,
                             fit.pvalue, len(work)))

        if abs(pct) >= 10:
            severity = ("high" if change < 0 and abs(pct) >= 20
                        else "warning" if change < 0 else "positive")
            insights.append(build_insight(
                title="'{}' {} {:.0f}% across the period".format(
                    target, direction, abs(pct)),
                problem="'{}' moved {:+,.1f} over {:,} days, {:+.1f}% against "
                        "where it started.".format(
                            target, change, span_days, pct),
                cause="What is driving the movement is not identifiable from "
                      "this data — a trend line establishes that the level "
                      "changed, not why. Splitting the same series by segment "
                      "is the next step.",
                evidence="Least-squares fit over {:,} observations: R²={:.2f} "
                         "(the line explains {:.0f}% of the variation), "
                         "p={:.2g}, slope {:+,.4g} per day.".format(
                             len(work), r2, r2 * 100, fit.pvalue, per_day),
                action="1. Split '{}' by your main segment column and refit  "
                       "2. Check whether the change is a step or a drift  "
                       "3. Compare against the same period last year if you "
                       "have it".format(target),
                impact="Continuing at the fitted rate, the same period again "
                       "would move '{}' by about {:+,.1f} — a straight-line "
                       "projection, not a forecast.".format(target, change),
                severity=severity, category="trend",
            ))
        if change > 0 and pct >= 10:
            opps.append(
                "'{}' has risen {:.0f}% over the period; identifying what "
                "changed is worth more than the figure itself.".format(
                    target, pct))
        elif change < 0 and pct <= -10:
            risks.append(
                "'{}' has fallen {:.0f}% over the {:,} days covered "
                "(R²={:.2f}).".format(target, abs(pct), span_days, r2))
    except Exception:
        logger.warning("trend analysis failed", exc_info=True)


# ══════════════════════════════════════════════════════════
#  WHERE THE EXTREMES SIT
# ══════════════════════════════════════════════════════════

def outlier_concentration(df: pd.DataFrame, insights: List, findings: List,
                          risks: List, opps: List) -> None:
    """Are the extreme values concentrated in one segment?

    An outlier count on its own is a data-quality note. Where the
    outliers sit is a business finding: ten in one branch is a process
    problem, ten spread across forty is the tail of a distribution and
    means nothing.
    """
    measures = _measures(df)
    segments = [c for c in text_columns(df)
                if 2 <= df[c].nunique(dropna=True) <= 30
                and not is_id_column(c, df[c])]
    if not measures or not segments:
        return
    try:
        target, seg_col = measures[0], segments[0]
        work = df[[seg_col, target]].dropna()
        if len(work) < MIN_N:
            return

        q1, q3 = work[target].quantile(0.25), work[target].quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            return
        lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
        work = work.assign(_outlier=(work[target] < lo) | (work[target] > hi))
        n_out = int(work["_outlier"].sum())
        if n_out < 5:
            return

        by_seg = work.groupby(seg_col)["_outlier"].agg(["size", "sum"])
        by_seg = by_seg[by_seg["size"] >= 20]
        if len(by_seg) < 2:
            return
        by_seg["rate"] = by_seg["sum"] / by_seg["size"]
        overall = n_out / len(work)

        worst = by_seg["rate"].idxmax()
        worst_rate = float(by_seg.loc[worst, "rate"])
        worst_n = int(by_seg.loc[worst, "size"])
        if worst_rate <= overall * 1.5 or by_seg.loc[worst, "sum"] < 3:
            findings.append(
                "The {:,} extreme values in '{}' are spread across '{}' rather "
                "than concentrated in any one group — consistent with the tail "
                "of a distribution rather than a process fault.".format(
                    n_out, target, seg_col))
            return

        # A rate this much above the pool is worth testing rather than
        # asserting: with small groups it happens by chance regularly.
        table = np.array([
            [int(by_seg.loc[worst, "sum"]),
             worst_n - int(by_seg.loc[worst, "sum"])],
            [n_out - int(by_seg.loc[worst, "sum"]),
             len(work) - worst_n - (n_out - int(by_seg.loc[worst, "sum"]))],
        ])
        try:
            _odds, p = scipy_stats.fisher_exact(table)
        except ValueError:
            return
        if p >= 0.05:
            return

        findings.append(
            "Extreme values in '{}' concentrate in '{}': {:.1f}% of its {:,} "
            "records against {:.1f}% overall (Fisher exact p={:.3g}).".format(
                target, str(worst), worst_rate * 100, worst_n, overall * 100, p))
        insights.append(build_insight(
            title="{:.0f}% of '{}' records are extreme values, against "
                  "{:.0f}% overall".format(worst_rate * 100, str(worst),
                                           overall * 100),
            problem="'{}' holds {:,} of the {:,} extreme values in '{}' from "
                    "{:,} records.".format(str(worst),
                                           int(by_seg.loc[worst, "sum"]),
                                           n_out, target, worst_n),
            cause="A concentration like this is usually a different process, "
                  "a different unit of measure, or a data-entry route that "
                  "differs for that group. Which of the three applies is not "
                  "in this data.",
            evidence="Outliers defined as beyond 3×IQR ({:.4g} to {:.4g}). "
                     "Fisher exact test of '{}' against the rest: "
                     "p={:.3g}.".format(lo, hi, str(worst), p),
            action="1. Pull ten of the extreme '{}' records and read them  "
                   "2. Confirm the unit and the collection route match the "
                   "other groups  3. Exclude and re-run to see what the "
                   "figures become".format(str(worst)),
            impact="Every average that includes this group is pulled by these "
                   "records; excluding them changes the headline figure.",
            severity="warning", category="quality",
        ))
    except Exception:
        logger.warning("outlier concentration failed", exc_info=True)


# ══════════════════════════════════════════════════════════
#  SEGMENTS TOO SMALL TO CONCLUDE FROM
# ══════════════════════════════════════════════════════════

def thin_segments(df: pd.DataFrame, insights: List, findings: List,
                  risks: List, opps: List) -> None:
    """Name the groups whose figures cannot carry a conclusion.

    A 90% rate on nine records will be quoted back at the client, and it
    is better dealt with in the report than in the meeting. This does not
    remove the segment — it says which numbers are indicative.
    """
    segments = [c for c in text_columns(df)
                if 2 <= df[c].nunique(dropna=True) <= 50
                and not is_id_column(c, df[c])]
    if not segments or len(df) < MIN_N:
        return
    try:
        col = segments[0]
        counts = df[col].value_counts()
        thin = counts[counts < MIN_SEGMENT_N]
        if thin.empty or len(thin) == len(counts):
            return

        covered = int(thin.sum())
        findings.append(
            "{} of the {} '{}' groups hold fewer than {} records each ({:,} "
            "rows in total, {:.1f}% of the data). Figures for those groups "
            "are indicative: a rate computed on {} records moves several "
            "points if one record changes.".format(
                len(thin), len(counts), col, MIN_SEGMENT_N, covered,
                covered / len(df) * 100, int(thin.min())))
        if len(thin) >= 3:
            opps.append(
                "Grouping the {} smallest '{}' values into an 'Other' "
                "category would let the remaining segments be compared "
                "without the noise those groups contribute.".format(
                    len(thin), col))
    except Exception:
        logger.warning("thin segment check failed", exc_info=True)


def _complete_months(df: pd.DataFrame, metric: str, time_col: str):
    """Monthly totals (or averages, for a measure that does not add up)
    and record counts over the months the data fully covers, as the trend
    chart draws them."""
    from app.engines.chart_exporter import MONTH_END, is_additive
    data = df[[time_col, metric]].dropna().set_index(time_col).sort_index()
    if data.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int)
    monthly = data[metric].resample(MONTH_END).agg("sum" if is_additive(metric) else "mean")
    rows = data[metric].resample(MONTH_END).count()
    first, last = data.index.min(), data.index.max()
    keep = monthly.index
    if len(keep) and first.day > 3:
        keep = keep[1:]
    if len(keep) and last.day < 25:
        keep = keep[:-1]
    return monthly.loc[keep].astype(float), rows.loc[keep]


def _unusual(values: np.ndarray, seasonal: bool):
    """([(index, log deviation, z) for each month out of line], months
    tested, cut-off). In seasonal
    mode the series is the year-on-year change, so December is compared
    with how Decembers usually move, not with the autumn before it."""
    from scipy.stats import norm
    x = np.log(values)
    s, off = (x[12:] - x[:-12], 12) if seasonal else (x, 0)
    if len(s) <= UNUSUAL_WINDOW + 2:
        return [], 0, 0.0
    d = np.diff(s)
    sig = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2)
    if sig <= 0:
        return [], 0, 0.0
    tested = len(s) - UNUSUAL_WINDOW
    cut = max(UNUSUAL_MIN_Z, float(norm.isf(0.05 / (2 * tested))))
    # The trailing median is itself estimated from six months.
    scale = sig * np.sqrt(1 + 1.57 / UNUSUAL_WINDOW)
    out = []
    for k in range(UNUSUAL_WINDOW, len(s)):
        dev = float(s[k] - np.median(s[k - UNUSUAL_WINDOW:k]))
        z = dev / scale
        if abs(z) >= cut and abs(np.expm1(dev)) >= UNUSUAL_MIN_CHANGE:
            out.append((k + off, dev, z))
    return out, tested, cut


def unusual_months(df: pd.DataFrame, insights: List, findings: List,
                   risks: List, opps: List) -> None:
    """Months that break from the ones before them.

    A trend line answers "is it moving"; it says nothing about a single
    month that doubled or collapsed, which is usually either the most
    important event in the period or a data-load error (a duplicated or
    missing extract).
    """
    from app.engines.chart_exporter import _measures as _ranked_measures
    from app.engines.chart_exporter import fmt
    from app.i18n import month_year
    time_col = _time_column(df)
    measures = _ranked_measures(df)
    if not time_col or not measures:
        return
    try:
        metric = measures[0]
        monthly, rows = _complete_months(df, metric, time_col)
        if (len(monthly) < UNUSUAL_WINDOW + 3 or (monthly <= 0).any()
                or float(rows.median()) < UNUSUAL_MIN_ROWS):
            return
        seasonal = len(monthly) >= 24
        flagged, tested, cut = _unusual(monthly.to_numpy(), seasonal)
        if not flagged:
            return
        flagged.sort(key=lambda t: -abs(t[2]))
        vals = monthly.to_numpy()
        shown = []
        for i, dev, z in flagged[:3]:
            expected = vals[i] / np.exp(dev)
            shown.append((monthly.index[i], vals[i], expected, np.expm1(dev) * 100, z))
        if seasonal:
            items = " · ".join("{}: {} against about {} expected from the year before "
                               "({:+.0f}%)".format(month_year(m), fmt(v), fmt(e), pct)
                               for m, v, e, pct, _z in shown)
            findings.append("Months where '{}' broke from its usual year-on-year "
                            "change: {}.".format(metric, items))
        else:
            items = " · ".join("{}: {} against a median of {} ({:+.0f}%)".format(
                month_year(m), fmt(v), fmt(e), pct) for m, v, e, pct, _z in shown)
            findings.append("Months where '{}' broke from the six before it: {}.".format(
                metric, items))
        m, v, e, pct, z = shown[0]
        if seasonal:
            problem = ("'{}' was {} in {}, {:+.0f}% against about {} expected from the "
                       "same month a year earlier and the recent year-on-year change."
                       .format(metric, fmt(v), month_year(m), pct, fmt(e)))
            how = "year-on-year changes of the six months before it"
        else:
            problem = ("'{}' was {} in {}, {:+.0f}% against the median of the six "
                       "months before it ({}).".format(metric, fmt(v), month_year(m), pct, fmt(e)))
            how = "six months before it"
        cause = ("A single month this far out is either a real event (a promotion, an "
                 "outage, a large one-off order) or a data problem such as a duplicated "
                 "or missing extract; the data alone cannot tell which.")
        if not seasonal:
            cause += (" With under two years of data a seasonal peak cannot be ruled "
                      "out either.")
        insights.append(build_insight(
            title="'{}' in {} broke from its recent pattern".format(metric, month_year(m)),
            problem=problem, cause=cause,
            evidence=("{:.1f} times the usual month-to-month variation away from the {}; "
                      "the cut-off, {:.1f}, allows for the {} months tested. {} of {} "
                      "flagged.".format(abs(z), how, cut, tested, len(flagged), tested)),
            action="1. Check the month's records for duplicates or a missing load  "
                   "2. If the data is right, find the event behind it before using the "
                   "month in a trend or forecast",
            impact="",
            severity="warning" if seasonal else "info", category="trend",
        ))
    except Exception:
        logger.warning("unusual month check failed", exc_info=True)


def run_general_depth(df: pd.DataFrame, insights: List, findings: List,
                      risks: List, opps: List) -> None:
    trend_over_time(df, insights, findings, risks, opps)
    unusual_months(df, insights, findings, risks, opps)
    outlier_concentration(df, insights, findings, risks, opps)
    thin_segments(df, insights, findings, risks, opps)
