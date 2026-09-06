"""
engines/bi/cohort.py — comparing groups against each other.

The comparison picks a test the data supports rather than the familiar
one — see engines/statistics.compare_groups for why that distinction
changed the answers this module gives.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from app.engines.statistics import (assess_normality, compare_groups)


from app.services.stat_guards import is_binned_from

from app.engines.bi.results import CohortResult



# ══════════════════════════════════════════════════════════

def analyze_cohort(
    df: pd.DataFrame,
    cohort_col: str,
    metric_col: str,
) -> CohortResult:
    """
    Compare metric across cohorts (segments).
    Statistical test + ranking + gap analysis.
    """
    # Filter to useful groups
    vc      = df[cohort_col].value_counts()
    valid   = vc[vc >= 5].index.tolist()
    df_filt = df[df[cohort_col].isin(valid)]

    if len(valid) < 2:
        return CohortResult(
            cohort_col=cohort_col, metric_col=metric_col,
            cohorts=[], best_cohort="N/A", worst_cohort="N/A",
            gap=0, gap_pct=0, is_significant=False,
            p_value=1.0, test_used="N/A",
            interpretation="Need at least 2 cohorts with 5+ records each.",
            recommendations=[],
        )

    # Group stats
    grp   = df_filt.groupby(cohort_col)[metric_col]
    means = grp.mean()
    medians = grp.median()
    stds  = grp.std()
    ns    = grp.count()
    overall_mean = float(df_filt[metric_col].mean())

    cohorts = []
    for i, name in enumerate(means.sort_values(ascending=False).index):
        m   = float(means[name])
        vs  = ((m - overall_mean) / abs(overall_mean) * 100
               if overall_mean != 0 else 0)
        cohorts.append({
            "name":       str(name),
            "n":          int(ns[name]),
            "mean":       round(m, 4),
            "median":     round(float(medians[name]), 4),
            "std":        round(float(stds.get(name, 0)), 4),
            "rank":       i + 1,
            "vs_avg_pct": round(vs, 2),
            "status":     "above" if vs > 5 else "below" if vs < -5 else "avg",
        })

    best_cohort  = cohorts[0]["name"]
    worst_cohort = cohorts[-1]["name"]
    gap          = cohorts[0]["mean"] - cohorts[-1]["mean"]
    gap_pct      = (gap / abs(cohorts[-1]["mean"]) * 100
                    if cohorts[-1]["mean"] != 0 else 0)

    # Statistical test
    groups    = [df_filt[df_filt[cohort_col] == g][metric_col].dropna().values
                 for g in valid]
    # Two bugs lived here, both of the same kind — an assumption asserted
    # rather than checked:
    #
    #   * `ttest_ind` and `f_oneway` were called with their defaults,
    #     which assume the groups have equal spread. Nobody checked, and
    #     with real business data a small cohort is nearly always more
    #     variable than a large one, so the p-values were optimistic.
    #   * Normality was decided by Shapiro-Wilk on the largest group,
    #     over up to 5,000 rows. At that size the test rejects normality
    #     on essentially any real data, so the branch it guards was
    #     close to dead — and it judged every cohort by one of them.
    #
    # Both now go through the shared helpers: normality by the shape of
    # the distribution rather than by an over-powered test, and the
    # comparison chosen from what the data supports.
    try:
        verdicts = [assess_normality(pd.Series(g)) for g in groups]
        is_normal = all(v is not None and v.normal_enough for v in verdicts)
        stat, p_val, test_used = compare_groups(groups, is_normal)
    except Exception:
        logger.debug("cohort comparison failed", exc_info=True)
        stat, p_val, test_used = 0.0, 1.0, "N/A"

    # bool(), not numpy's np.True_: this value is serialised into the
    # API response and compared with `is True` by callers.
    is_sig = bool(p_val < 0.05)

    # Cohorts that are ranges of the very metric being compared. Group
    # revenue by revenue_band and the highest band has the highest
    # revenue — with a vanishing p-value and a huge gap, because that is
    # the definition of the band, not a fact about the business. The
    # engine used to answer this with "investigate what '700+' does
    # differently to achieve 596.9% higher revenue".
    #
    # The numbers are still returned: the ranking of a binned column is a
    # legitimate distribution summary, and callers chart it. What changes
    # is that nothing is claimed about cause, and no recommendation
    # invites anyone to act on it.
    is_definitional = is_binned_from(df_filt, cohort_col, metric_col)

    interp = (
        "{} cohorts compared on '{}'. "
        "Best: '{}' (mean={:.2f}), Worst: '{}' (mean={:.2f}). "
        "Gap: {:.1f}% ({} difference, {} p={:.4f}).".format(
            len(valid), metric_col,
            best_cohort, cohorts[0]["mean"],
            worst_cohort, cohorts[-1]["mean"],
            gap_pct,
            "statistically significant" if is_sig else "NOT significant",
            test_used, p_val)
    )

    if is_definitional:
        interp += (
            " These cohorts are ranges of {} itself, so the gap between "
            "the top and bottom group restates how the ranges were drawn "
            "rather than revealing anything about them. Read this as the "
            "shape of the distribution, not as a difference to act on; "
            "group by something measured independently — a region, a "
            "channel, a plan — to compare like with like.".format(metric_col)
        )

    recs = []
    if is_definitional:
        recs.append(
            "Compare '{}' across a column that is not derived from it — "
            "'{}' is {} in bands, so the ranking is guaranteed.".format(
                metric_col, cohort_col, metric_col)
        )
    elif is_sig:
        recs.append(
            "Significant difference confirmed — investigate what '{}' "
            "does differently to achieve {:.1f}% higher '{}'.".format(
                best_cohort, gap_pct, metric_col)
        )
        below_avg = [c for c in cohorts if c["status"] == "below"]
        if below_avg:
            recs.append(
                "{} cohort(s) below average: {}. "
                "Prioritize improvement in these segments.".format(
                    len(below_avg),
                    ", ".join([c["name"] for c in below_avg[:3]]))
            )
    else:
        recs.append(
            "No significant difference between cohorts — "
            "'{}' does not meaningfully segment '{}' performance.".format(
                cohort_col, metric_col)
        )

    return CohortResult(
        cohort_col=cohort_col, metric_col=metric_col,
        cohorts=cohorts[:15],
        best_cohort=best_cohort, worst_cohort=worst_cohort,
        gap=round(gap, 4), gap_pct=round(gap_pct, 2),
        is_significant=is_sig, p_value=round(float(p_val), 6),
        test_used=test_used, interpretation=interp,
        recommendations=recs,
        is_definitional=is_definitional,
    )


# ══════════════════════════════════════════════════════════