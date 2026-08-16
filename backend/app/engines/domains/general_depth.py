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

from app.services.numfmt import human_number

logger = logging.getLogger(__name__)

# A trend explaining less than this of the variance is a line through
# scatter, however significant the slope.
MIN_TREND_R2 = 0.10
# Below this a segment's rate is an anecdote; a client's analyst will say
# so before anyone acts on it.
MIN_SEGMENT_N = 30


def _measures(df: pd.DataFrame) -> List[str]:
    return [c for c in df.select_dtypes(include="number").columns
            if not is_id_column(c, df[c]) and df[c].nunique(dropna=True) > 2]


def _time_column(df: pd.DataFrame):
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


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
    time_col = _time_column(df)
    measures = _measures(df)
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
                "'{}' shows no trend over the period that is distinguishable "
                "from ordinary variation (R²={:.2f}, p={:.2g}, n={:,}). The "
                "level, not the direction, is what this data supports.".format(
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
                # `%g` on the slope printed "+1.2e+02" — a hundred and
                # twenty, written in a notation nobody uses for a rate of
                # change. The p-value keeps exponent form because that is
                # how a p-value is written; a slope is a quantity and gets
                # the same treatment as every other figure in the report.
                evidence="Least-squares fit over {:,} observations: R²={:.2f} "
                         "(the line explains {:.0f}% of the variation), "
                         "p{}, slope {} per day.".format(
                             len(work), r2, r2 * 100,
                             "<0.001" if fit.pvalue < 0.001
                             else "={:.3f}".format(fit.pvalue),
                             ("+" if per_day >= 0 else "-")
                             + human_number(abs(per_day))),
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

        # A group can hold nearly all the extremes simply by sitting at a
        # different level: on a retail file, 99% of Electronics rows were
        # "extreme values" in `unit_price` because Electronics costs
        # twenty times what Grocery does. That is the shape of the
        # catalogue, not a collection fault, and calling it one sent the
        # reader looking for a problem that does not exist.
        from app.engines.domains.base import outliers_explained_by_group
        if outliers_explained_by_group(df, target) == seg_col:
            findings.append(
                "'{}' sits at a different level in '{}' than elsewhere, so "
                "most of its records fall outside a range computed across the "
                "whole file. Within '{}' they are unremarkable — the column "
                "should be read per group.".format(str(worst), seg_col, seg_col))
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
            evidence="Outliers defined as beyond 3×IQR ({} to {}). "
                     "Fisher exact test of '{}' against the rest: "
                     "p={:.3g}.".format(human_number(lo), human_number(hi),
                                        str(worst), p),
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


def run_general_depth(df: pd.DataFrame, insights: List, findings: List,
                      risks: List, opps: List) -> None:
    trend_over_time(df, insights, findings, risks, opps)
    outlier_concentration(df, insights, findings, risks, opps)
    thin_segments(df, insights, findings, risks, opps)
