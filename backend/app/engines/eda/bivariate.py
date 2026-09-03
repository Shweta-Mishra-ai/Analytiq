"""
engines/eda/bivariate.py — two columns, and whether their relationship
is worth reporting.

Correlation with an interval rather than a bare coefficient, and group
comparison that picks a test the data actually supports rather than the
familiar one.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


from scipy.stats import pearsonr, spearmanr

from app.engines.eda.results import BivariateResult, GroupComparisonResult
from app.engines.statistics import (compare_groups)


#  BIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for two groups."""
    pooled_std = np.sqrt(
        ((len(a) - 1) * a.std()**2 + (len(b) - 1) * b.std()**2)
        / (len(a) + len(b) - 2)
    )
    return float(abs(a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def _effect_label(d: float) -> str:
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def analyze_bivariate_numeric(
    df: pd.DataFrame, col_a: str, col_b: str,
    is_normal_a: bool, is_normal_b: bool,
) -> BivariateResult:
    """
    Numeric vs Numeric bivariate test.
    Picks Pearson (normal) or Spearman (non-normal).
    """
    common = df[[col_a, col_b]].dropna()
    a, b   = common[col_a].values, common[col_b].values

    if len(common) < 10:
        return BivariateResult(
            col_a=col_a, col_b=col_b,
            test_name="N/A", statistic=0, p_value=1,
            is_significant=False,
            interpretation="Insufficient data.")

    if is_normal_a and is_normal_b:
        r, p       = pearsonr(a, b)
        test_name  = "Pearson Correlation"
        effect_s   = abs(r)
        effect_lbl = ("Negligible" if effect_s < 0.1 else
                      "Small" if effect_s < 0.3 else
                      "Medium" if effect_s < 0.5 else "Large")
    else:
        r, p       = spearmanr(a, b)
        test_name  = "Spearman Rank Correlation"
        effect_s   = abs(r)
        effect_lbl = ("Negligible" if effect_s < 0.1 else
                      "Small" if effect_s < 0.3 else
                      "Medium" if effect_s < 0.5 else "Large")

    sig  = p < 0.05
    dirn = "positive" if r > 0 else "negative"

    interp = (
        "{} {} correlation between '{}' and '{}' "
        "(r={:.3f}, p={:.4f}, effect={}).".format(
            effect_lbl, dirn, col_a, col_b,
            round(r, 3), round(p, 4), effect_lbl)
    )
    if sig and effect_lbl in ("Medium", "Large"):
        rec = "Significant relationship — consider including both in models or investigating causation."
    elif sig:
        rec = "Statistically significant but small effect — may not be practically important."
    else:
        rec = "No significant relationship detected at p=0.05."

    return BivariateResult(
        col_a=col_a, col_b=col_b,
        test_name=test_name,
        statistic=round(float(r), 4),
        p_value=round(float(p), 6),
        is_significant=sig,
        effect_size=round(float(effect_s), 4),
        effect_label=effect_lbl,
        interpretation=interp,
        recommendation=rec,
    )


def analyze_group_comparison(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    is_normal: bool,
) -> GroupComparisonResult:
    """
    Compare numeric column across groups.
    Uses ANOVA (normal) or Kruskal-Wallis (non-normal).
    With eta-squared effect size.
    """
    groups     = df.groupby(group_col)[numeric_col].apply(
        lambda x: x.dropna().values
    )
    groups     = {k: v for k, v in groups.items() if len(v) >= 3}
    n_groups   = len(groups)

    if n_groups < 2:
        return GroupComparisonResult(
            numeric_col=numeric_col, group_col=group_col,
            n_groups=n_groups, test_used="N/A",
            statistic=0, p_value=1, is_significant=False,
            interpretation="Need at least 2 groups with 3+ samples.")

    group_arrays = list(groups.values())

    # Choose the test — and then actually use the answer.
    #
    # This block used to run Levene's test for equal variances, store the
    # result, and then call one-way ANOVA regardless. That is worse than
    # not checking at all: ANOVA assumes equal variances, and when they
    # differ it reports significance more often than it should. Testing
    # the assumption and discarding the answer produced exactly the false
    # positives the check exists to prevent.
    #
    # Welch's ANOVA does not make that assumption and costs almost
    # nothing in power when variances happen to be equal, so the only
    # reason to prefer the classic form is convention.
    stat, p, test_name = compare_groups(group_arrays, is_normal)

    # Eta-squared effect size
    all_values = np.concatenate(group_arrays)
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_arrays)
    ss_total   = sum((all_values - grand_mean)**2)
    eta_sq     = ss_between / ss_total if ss_total > 0 else 0
    eta_lbl    = ("Negligible" if eta_sq < 0.01 else
                  "Small" if eta_sq < 0.06 else
                  "Medium" if eta_sq < 0.14 else "Large")

    # Group stats
    group_stats = {}
    for grp, vals in groups.items():
        group_stats[str(grp)] = {
            "n":      len(vals),
            "mean":   round(float(vals.mean()), 4),
            "median": round(float(np.median(vals)), 4),
            "std":    round(float(vals.std()), 4),
        }

    sig = p < 0.05
    interp = (
        "{} test: {} difference in '{}' across {} groups of '{}' "
        "(F/H={:.3f}, p={:.4f}, eta²={:.3f} — {} effect).".format(
            test_name,
            "Significant" if sig else "No significant",
            numeric_col, n_groups, group_col,
            round(float(stat), 3), round(float(p), 4),
            round(eta_sq, 4), eta_lbl)
    )

    # Post-hoc hint
    post_hoc = []
    if sig and n_groups > 2:
        post_hoc.append(
            "Significant difference detected — run post-hoc pairwise comparisons "
            "(Tukey HSD or Dunn test) to identify which groups differ."
        )
    elif sig and n_groups == 2:
        grp_names = list(group_stats.keys())
        m1 = group_stats[grp_names[0]]["mean"]
        m2 = group_stats[grp_names[1]]["mean"]
        higher = grp_names[0] if m1 > m2 else grp_names[1]
        lower  = grp_names[1] if m1 > m2 else grp_names[0]
        post_hoc.append(
            "'{}' has significantly higher '{}' than '{}' "
            "({:.2f} vs {:.2f}).".format(
                higher, numeric_col, lower,
                max(m1, m2), min(m1, m2))
        )

    return GroupComparisonResult(
        numeric_col=numeric_col, group_col=group_col,
        n_groups=n_groups, test_used=test_name,
        statistic=round(float(stat), 4),
        p_value=round(float(p), 6),
        is_significant=sig,
        effect_size=round(float(eta_sq), 4),
        effect_label=eta_lbl,
        group_stats=group_stats,
        interpretation=interp,
        post_hoc=post_hoc,
    )


# ══════════════════════════════════════════════════════════