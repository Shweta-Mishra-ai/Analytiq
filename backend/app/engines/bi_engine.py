"""
bi_engine.py — Business Intelligence engine.
Benchmarking, root cause analysis, cohort analysis, Pareto.
Senior analyst level — not just averages.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")
from scipy import stats as scipy_stats


# ══════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    column:          str
    mean:            float
    median:          float
    p25:             float
    p75:             float
    p90:             float
    top_10_pct:      float    # value threshold for top 10%
    bottom_10_pct:   float    # value threshold for bottom 10%
    above_avg_pct:   float    # % of rows above average
    cv:              float    # coefficient of variation
    benchmark_label: str      # "High variance", "Consistent", etc.
    interpretation:  str


@dataclass
class RootCauseResult:
    target_col:      str
    low_performer_threshold: float
    n_low_performers: int
    low_pct:         float
    drivers:         List[Dict]   # [{factor, impact, direction, detail}]
    top_driver:      str
    interpretation:  str
    recommendations: List[str]


@dataclass
class CohortResult:
    cohort_col:      str
    metric_col:      str
    cohorts:         List[Dict]   # [{name, n, mean, median, rank, vs_avg_pct}]
    best_cohort:     str
    worst_cohort:    str
    gap:             float        # best - worst mean
    gap_pct:         float        # gap as % of worst
    is_significant:  bool
    p_value:         float
    test_used:       str
    interpretation:  str
    recommendations: List[str]


@dataclass
class ParetoResult:
    group_col:       str
    value_col:       str
    agg_fn:          str         # "sum" or "mean"
    groups:          List[Dict]  # [{name, value, cumulative_pct, in_top_20}]
    top_20_pct_groups: int       # how many groups make up 80% of value
    top_groups_share:  float     # % of total value from top 20% groups
    pareto_holds:    bool        # True if top 20% drives >= 60% of value
    interpretation:  str


@dataclass
class SegmentHealth:
    segment_name:    str
    segment_col:     str
    n:               int
    metrics:         Dict[str, Dict]  # {metric: {mean, vs_avg, rank, status}}
    health_score:    float    # 0-100
    strengths:       List[str]
    weaknesses:      List[str]
    opportunity:     str


@dataclass
class BIReport:
    benchmarks:      List[BenchmarkResult]   = field(default_factory=list)
    root_causes:     List[RootCauseResult]   = field(default_factory=list)
    cohorts:         List[CohortResult]      = field(default_factory=list)
    pareto:          List[ParetoResult]      = field(default_factory=list)
    segments:        List[SegmentHealth]     = field(default_factory=list)
    key_insights:    List[str]               = field(default_factory=list)
    executive_brief: str                     = ""


# ══════════════════════════════════════════════════════════
#  BENCHMARKING
# ══════════════════════════════════════════════════════════

def analyze_benchmark(df: pd.DataFrame, col: str) -> BenchmarkResult:
    """Full benchmark analysis for one numeric column."""
    s = df[col].dropna()

    mean   = float(s.mean())
    median = float(s.median())
    std    = float(s.std())
    cv     = std / abs(mean) if mean != 0 else 0

    p25  = float(s.quantile(0.25))
    p75  = float(s.quantile(0.75))
    p90  = float(s.quantile(0.90))
    p10  = float(s.quantile(0.10))

    above_avg_pct = float((s > mean).mean() * 100)

    if cv < 0.1:
        label = "Very Consistent — low variation across records"
    elif cv < 0.3:
        label = "Consistent — moderate variation"
    elif cv < 0.6:
        label = "Variable — significant spread"
    else:
        label = "Highly Variable — large spread, investigate outliers"

    # Mean vs median interpretation
    diff_pct = abs(mean - median) / abs(median) * 100 if median != 0 else 0
    if diff_pct > 20:
        central = ("Mean ({:.2f}) is {:.0f}% away from median ({:.2f}) — "
                   "skewed distribution, use median for central tendency.").format(
                       mean, diff_pct, median)
    else:
        central = ("Mean ({:.2f}) and median ({:.2f}) are close — "
                   "symmetric distribution.").format(mean, median)

    interp = (
        "{} | {:.1f}% of records are above average. "
        "Top 10% threshold: {:.2f}. Bottom 10%: {:.2f}. {}".format(
            label, above_avg_pct, p90, p10, central)
    )

    return BenchmarkResult(
        column=col, mean=round(mean, 4), median=round(median, 4),
        p25=round(p25, 4), p75=round(p75, 4), p90=round(p90, 4),
        top_10_pct=round(p90, 4), bottom_10_pct=round(p10, 4),
        above_avg_pct=round(above_avg_pct, 2),
        cv=round(cv, 4), benchmark_label=label,
        interpretation=interp,
    )


# ══════════════════════════════════════════════════════════
#  ROOT CAUSE ANALYSIS
# ══════════════════════════════════════════════════════════

def analyze_root_cause(
    df: pd.DataFrame,
    target_col: str,
    threshold_pct: float = 25.0,   # bottom X% = low performers
) -> RootCauseResult:
    """
    Find what drives low performance on target_col.
    Compares low performers vs high performers on all other columns.
    """
    s         = df[target_col].dropna()
    threshold = float(s.quantile(threshold_pct / 100))
    low_mask  = df[target_col] <= threshold
    high_mask = df[target_col] > threshold

    n_low     = int(low_mask.sum())
    low_pct   = round(n_low / max(len(df), 1) * 100, 1)

    low_df    = df[low_mask]
    high_df   = df[high_mask]

    drivers = []

    # Numeric features — compare means
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c != target_col]
    for col in num_cols[:15]:
        try:
            low_vals  = low_df[col].dropna()
            high_vals = high_df[col].dropna()
            if len(low_vals) < 5 or len(high_vals) < 5:
                continue

            low_mean  = float(low_vals.mean())
            high_mean = float(high_vals.mean())
            diff      = high_mean - low_mean
            diff_pct  = abs(diff) / abs(low_mean) * 100 if low_mean != 0 else 0

            # Statistical significance
            try:
                _, p = scipy_stats.mannwhitneyu(
                    low_vals, high_vals, alternative="two-sided")
            except Exception:
                p = 1.0

            if p < 0.05 and diff_pct > 5:
                direction = "higher" if diff > 0 else "lower"
                impact    = min(diff_pct / 100, 1.0)   # normalize to 0-1

                detail = (
                    "Low performers have {:.1f}% {} '{}' "
                    "({:.2f} vs {:.2f}, p={:.4f})".format(
                        diff_pct, direction if diff < 0 else
                        "lower" if direction == "higher" else "higher",
                        col, low_mean, high_mean, p)
                )
                drivers.append({
                    "factor":    col,
                    "impact":    round(impact, 4),
                    "direction": "negative" if diff > 0 else "positive",
                    "low_mean":  round(low_mean, 4),
                    "high_mean": round(high_mean, 4),
                    "diff_pct":  round(diff_pct, 1),
                    "p_value":   round(p, 4),
                    "detail":    detail,
                    "dtype":     "numeric",
                })
        except Exception:
            continue

    # Categorical features — compare distributions
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if 2 <= df[c].nunique() <= 20]
    for col in cat_cols[:8]:
        try:
            # Chi-square test
            ct = pd.crosstab(df[col], low_mask)
            if ct.shape[1] < 2:
                continue
            chi2, p, dof, _ = scipy_stats.chi2_contingency(ct)
            if p >= 0.05:
                continue

            # Find which category is most common in low performers
            low_vc  = low_df[col].value_counts(normalize=True)
            high_vc = high_df[col].value_counts(normalize=True)

            if len(low_vc) == 0:
                continue

            worst_cat = low_vc.index[0]
            low_pct_cat  = round(low_vc.iloc[0] * 100, 1)
            high_pct_cat = round(high_vc.get(worst_cat, 0) * 100, 1)
            diff_pct = abs(low_pct_cat - high_pct_cat)

            detail = (
                "In low performers, '{}' = '{}' in {:.0f}% of cases "
                "vs {:.0f}% in high performers (chi2 p={:.4f})".format(
                    col, worst_cat, low_pct_cat, high_pct_cat, p)
            )
            drivers.append({
                "factor":      col,
                "impact":      round(min(diff_pct / 100, 1.0), 4),
                "direction":   "categorical",
                "key_category": str(worst_cat),
                "low_pct":     low_pct_cat,
                "high_pct":    high_pct_cat,
                "p_value":     round(p, 4),
                "detail":      detail,
                "dtype":       "categorical",
            })
        except Exception:
            continue

    # Sort by impact
    drivers.sort(key=lambda x: x["impact"], reverse=True)
    top_driver = drivers[0]["factor"] if drivers else "No significant driver found"

    # Interpretation
    if not drivers:
        interp = (
            "No statistically significant drivers found for low '{}' performance. "
            "Consider collecting additional data.".format(target_col)
        )
        recs = ["Collect more granular data to identify root causes."]
    else:
        top = drivers[0]
        interp = (
            "{:.0f}% of records ({:,}) are in the bottom {:.0f}% of '{}'. "
            "Top driver: '{}' — {}".format(
                low_pct, n_low, threshold_pct, target_col,
                top["factor"], top["detail"])
        )
        recs = []
        for d in drivers[:3]:
            if d["dtype"] == "numeric":
                recs.append(
                    "Focus on '{}' — low performers show {:.1f}% difference. "
                    "Bring to high-performer level ({:.2f}) from current {:.2f}.".format(
                        d["factor"], d["diff_pct"],
                        d["high_mean"], d["low_mean"])
                )
            else:
                recs.append(
                    "Investigate '{}' = '{}' segment — "
                    "over-represented in low performers ({:.0f}% vs {:.0f}%).".format(
                        d["factor"], d.get("key_category", ""),
                        d.get("low_pct", 0), d.get("high_pct", 0))
                )

    return RootCauseResult(
        target_col=target_col,
        low_performer_threshold=round(threshold, 4),
        n_low_performers=n_low,
        low_pct=low_pct,
        drivers=drivers[:10],
        top_driver=top_driver,
        interpretation=interp,
        recommendations=recs,
    )


# ══════════════════════════════════════════════════════════
#  COHORT ANALYSIS
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
    try:
        # Normality check on largest group
        largest = max(groups, key=len)
        _, p_norm = scipy_stats.shapiro(
            largest[:min(len(largest), 5000)])
        is_normal = p_norm > 0.05

        if is_normal and len(groups) == 2:
            stat, p_val  = scipy_stats.ttest_ind(*groups)
            test_used    = "Independent t-test"
        elif is_normal:
            stat, p_val  = scipy_stats.f_oneway(*groups)
            test_used    = "One-Way ANOVA"
        else:
            stat, p_val  = scipy_stats.kruskal(*groups)
            test_used    = "Kruskal-Wallis"
    except Exception:
        p_val, test_used = 1.0, "N/A"

    is_sig = p_val < 0.05

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

    recs = []
    if is_sig:
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
    )


# ══════════════════════════════════════════════════════════
#  PARETO ANALYSIS
# ══════════════════════════════════════════════════════════

def analyze_pareto(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    agg_fn: str = "sum",
) -> ParetoResult:
    """80/20 Pareto analysis."""
    fn  = np.sum if agg_fn == "sum" else np.mean
    agg = (df.groupby(group_col)[value_col]
             .agg(fn)
             .sort_values(ascending=False))

    total     = agg.sum()
    cum_pct   = (agg.cumsum() / total * 100).round(2)
    n         = len(agg)
    top_20_n  = max(1, int(np.ceil(n * 0.20)))

    groups_list = []
    for i, (name, val) in enumerate(agg.items()):
        groups_list.append({
            "rank":           i + 1,
            "name":           str(name)[:30],
            "value":          round(float(val), 4),
            "pct_of_total":   round(float(val / total * 100), 2),
            "cumulative_pct": round(float(cum_pct[name]), 2),
            "in_top_20":      i < top_20_n,
        })

    top_share = round(float(agg.iloc[:top_20_n].sum() / total * 100), 2)
    pareto_holds = top_share >= 60

    interp = (
        "Top {:.0f}% of '{}' groups ({} out of {}) account for {:.1f}% "
        "of total '{}'. {}".format(
            20, group_col, top_20_n, n, top_share, value_col,
            "Pareto principle HOLDS — concentrate on top performers." if pareto_holds
            else "Pareto principle does NOT hold — value is distributed evenly.")
    )

    return ParetoResult(
        group_col=group_col, value_col=value_col, agg_fn=agg_fn,
        groups=groups_list, top_20_pct_groups=top_20_n,
        top_groups_share=top_share, pareto_holds=pareto_holds,
        interpretation=interp,
    )


# ══════════════════════════════════════════════════════════
#  SEGMENT HEALTH SCORING
# ══════════════════════════════════════════════════════════

def analyze_segment_health(
    df: pd.DataFrame,
    segment_col: str,
    metric_cols: List[str],
) -> List[SegmentHealth]:
    """
    Score each segment across multiple metrics.
    Identifies strengths, weaknesses, opportunities.
    """
    segments = df[segment_col].dropna().unique()
    valid    = [s for s in segments if (df[segment_col] == s).sum() >= 5]

    if len(valid) < 2 or not metric_cols:
        return []

    # Overall means for comparison
    overall = {col: float(df[col].mean()) for col in metric_cols
               if col in df.columns and pd.api.types.is_numeric_dtype(df[col])}
    if not overall:
        return []

    results = []
    for seg in valid[:10]:
        seg_df  = df[df[segment_col] == seg]
        metrics = {}
        scores  = []

        for col in metric_cols:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            seg_mean = float(seg_df[col].mean())
            avg      = overall.get(col, seg_mean)
            vs_avg   = ((seg_mean - avg) / abs(avg) * 100
                        if avg != 0 else 0)

            # Rank among all segments
            seg_means = df.groupby(segment_col)[col].mean().sort_values(ascending=False)
            rank = int((seg_means.index.tolist().index(seg) + 1)
                       if seg in seg_means.index else len(seg_means))
            n_seg = len(seg_means)

            status = ("top" if rank <= max(1, n_seg // 3)
                      else "bottom" if rank > n_seg - max(1, n_seg // 3)
                      else "mid")

            score_val = 50 + vs_avg * 0.5
            score_val = max(0, min(100, score_val))
            scores.append(score_val)

            metrics[col] = {
                "mean":    round(seg_mean, 4),
                "vs_avg":  round(vs_avg, 2),
                "rank":    rank,
                "n_total": n_seg,
                "status":  status,
            }

        health_score = round(float(np.mean(scores)), 1) if scores else 50.0
        strengths    = [col for col, m in metrics.items()
                        if m["vs_avg"] > 10]
        weaknesses   = [col for col, m in metrics.items()
                        if m["vs_avg"] < -10]

        # Opportunity
        if weaknesses:
            worst_col = min(metrics.items(),
                            key=lambda x: x[1]["vs_avg"])[0]
            opp = ("Improve '{}' from {:.2f} to dataset average {:.2f} "
                   "— {:.1f}% improvement opportunity.".format(
                       worst_col,
                       metrics[worst_col]["mean"],
                       overall.get(worst_col, 0),
                       abs(metrics[worst_col]["vs_avg"])))
        elif strengths:
            best_col = max(metrics.items(),
                           key=lambda x: x[1]["vs_avg"])[0]
            opp = ("Already leading in '{}' — "
                   "leverage this advantage in other segments.".format(best_col))
        else:
            opp = "Performance close to average across all metrics."

        results.append(SegmentHealth(
            segment_name=str(seg),
            segment_col=segment_col,
            n=int(len(seg_df)),
            metrics=metrics,
            health_score=health_score,
            strengths=strengths,
            weaknesses=weaknesses,
            opportunity=opp,
        ))

    return sorted(results, key=lambda x: x.health_score, reverse=True)


# ══════════════════════════════════════════════════════════
#  KEY INSIGHTS GENERATOR
# ══════════════════════════════════════════════════════════

def _generate_key_insights(
    report: BIReport, df: pd.DataFrame
) -> Tuple[List[str], str]:
    insights = []

    # Pareto insights
    for p in report.pareto:
        if p.pareto_holds:
            insights.append(
                "Pareto holds for '{}' by '{}': top {:.0f}% of groups "
                "drive {:.0f}% of value. Focus resources on top performers.".format(
                    p.value_col, p.group_col,
                    20, p.top_groups_share)
            )
        else:
            insights.append(
                "Value in '{}' is evenly distributed across '{}' segments — "
                "no single group dominates.".format(p.value_col, p.group_col)
            )

    # Root cause insights
    for rc in report.root_causes:
        if rc.drivers:
            insights.append(
                "Root cause of low '{}': '{}' is the top driver "
                "({:.0f}% difference between low and high performers).".format(
                    rc.target_col, rc.top_driver,
                    rc.drivers[0]["diff_pct"] if rc.drivers[0].get("diff_pct") else 0)
            )

    # Cohort insights
    sig_cohorts = [c for c in report.cohorts if c.is_significant]
    for c in sig_cohorts[:2]:
        insights.append(
            "'{}' significantly segments '{}': best cohort '{}' "
            "outperforms worst '{}' by {:.1f}%.".format(
                c.cohort_col, c.metric_col,
                c.best_cohort, c.worst_cohort, c.gap_pct)
        )

    # Segment health insights
    if report.segments:
        best = report.segments[0]
        worst = report.segments[-1]
        insights.append(
            "Healthiest segment: '{}' (score={:.0f}/100). "
            "Needs most attention: '{}' (score={:.0f}/100).".format(
                best.segment_name, best.health_score,
                worst.segment_name, worst.health_score)
        )

    # Executive brief
    brief = "Business Intelligence analysis completed. "
    if insights:
        brief += insights[0] + " "
    if sig_cohorts:
        brief += "{} significant cohort difference(s) identified. ".format(
            len(sig_cohorts))
    if report.root_causes:
        brief += "Root cause analysis run on {} metric(s).".format(
            len(report.root_causes))

    return insights, brief


# ══════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

def run_bi(df: pd.DataFrame, max_rows: int = 50_000) -> BIReport:
    """
    Full BI pipeline.
    Auto-selects best columns for each analysis.
    """
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if 2 <= df[c].nunique() <= 25]

    report = BIReport()

    # 1. Benchmarks — top 4 numeric cols
    for col in num_cols[:4]:
        try:
            report.benchmarks.append(analyze_benchmark(df, col))
        except Exception:
            continue

    # 2. Root cause — top numeric as target
    for col in num_cols[:2]:
        try:
            report.root_causes.append(
                analyze_root_cause(df, col, threshold_pct=25))
        except Exception:
            continue

    # 3. Cohort analysis — top cat × top numeric
    for cat in cat_cols[:2]:
        for num in num_cols[:2]:
            try:
                report.cohorts.append(analyze_cohort(df, cat, num))
            except Exception:
                continue

    # 4. Pareto — top cat × top numeric
    for cat in cat_cols[:1]:
        for num in num_cols[:2]:
            try:
                agg = "mean" if "rate" in num.lower() or "score" in num.lower() \
                      or "rating" in num.lower() else "sum"
                report.pareto.append(analyze_pareto(df, cat, num, agg))
            except Exception:
                continue

    # 5. Segment health
    if cat_cols and num_cols:
        try:
            report.segments = analyze_segment_health(
                df, cat_cols[0], num_cols[:4])
        except Exception:
            pass

    # 6. Key insights + brief
    report.key_insights, report.executive_brief = _generate_key_insights(report, df)

    return report

