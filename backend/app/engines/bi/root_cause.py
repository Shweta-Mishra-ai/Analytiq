"""
engines/bi/root_cause.py — which factor best explains a performance gap.

The most easily abused analysis in the product: with enough columns
something always correlates. The gates here — an effect-size floor, a
false-discovery correction, and a refusal to treat an identifier or a
non-performance column as a target — are what stop it from manufacturing
an explanation.
"""
from __future__ import annotations

import logging

import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

from app.engines.present import label as _L, num as _N, value as _V
from app.engines.statistics import (clamp_p, cohens_d,
                                    effect_label, format_p)


from app.services.dtypes import text_columns
from app.services.stat_guards import (apply_fdr, chi2_association,
                                      is_binned_from, is_restatement)

from app.engines.bi.results import RootCauseResult


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

    # Numeric features — compare means.
    #
    # A column that restates the target — the same money in thousands, an
    # exact copy under another name, its log or its rank — separates the
    # two groups perfectly, wins the impact ranking, and turns the
    # headline into a tautology: "low revenue is driven by low revenue_k,
    # bring revenue_k up to 0.62". It is excluded before any test is run,
    # because a driver that is the target restated explains nothing.
    num_cols = []
    restated = []
    for c in df.select_dtypes(include="number").columns:
        if c == target_col:
            continue
        if is_restatement(df[c], df[target_col]):
            restated.append(c)
            continue
        num_cols.append(c)
    if restated:
        logger.info("root cause: excluded %s — restatement(s) of '%s'",
                    ", ".join(restated), target_col)
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

            # A shortfall is expressed against the group that has more,
            # so it stays inside 0-100%. Measured the other way round it
            # produced "129.6% lower", which is not a thing.
            shortfall = (abs(diff) / abs(high_mean) * 100
                         if high_mean else 0.0)
            effect = cohens_d(low_vals, high_vals)
            if (p < 0.05 and shortfall > 5
                    and effect_label(effect or 0) != "negligible"):
                impact = min(shortfall / 100, 1.0)
                detail = (
                    "The low group averages {} against {} for the high "
                    "group on {} — {:.0f}% {}, {} ({} difference)".format(
                        _N(low_mean), _N(high_mean), _L(col), shortfall,
                        "less" if diff > 0 else "more", format_p(p),
                        effect_label(effect or 0))
                )
                drivers.append({
                    "factor":    col,
                    "impact":    round(impact, 4),
                    "direction": "negative" if diff > 0 else "positive",
                    "low_mean":  round(low_mean, 4),
                    "high_mean": round(high_mean, 4),
                    "diff_pct":  round(shortfall, 1),
                    "effect":    round(abs(effect or 0.0), 3),
                    "p_value":   clamp_p(p),
                    "detail":    detail,
                    "dtype":     "numeric",
                })
        except Exception:
            logger.debug("analyze_root_cause: suppressed exception", exc_info=True)
            continue

    # Categorical features — compare distributions.
    #
    # The categorical equivalent of the exclusion above: a band column
    # cut from the target sorts the low and high groups perfectly, so
    # chi-square finds a crushing association and the report announces
    # "what separates the low Revenue group from the high one most is
    # Revenue Band". It separates them because that is how the bands
    # were drawn.
    cat_cols = [c for c in text_columns(df)
                if 2 <= df[c].nunique() <= 20
                and not is_binned_from(df, c, target_col)]
    for col in cat_cols[:8]:
        try:
            # Chi-square with its assumptions verified. The raw
            # chi2_contingency p-value used to be taken at face value, so a
            # sparse crosstab (an expected cell count below 5) could yield a
            # confident "significant driver" that no reviewer would accept.
            # chi2_association returns None when the table can't support the
            # test, and also gives Cramér's V so strength is judged, not just
            # significance.
            ct = pd.crosstab(df[col], low_mask)
            if ct.shape[1] < 2:
                continue
            assoc = chi2_association(ct)
            if assoc is None:
                continue
            p = assoc["p"]
            if p >= 0.05 or assoc["cramers_v"] < 0.1:
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
                "{} = {} in {:.0f}% of the low group against {:.0f}% of "
                "the high group (chi-square {}, Cramér's V {:.2f} — {} "
                "association, n={:,})".format(
                    _L(col), _V(worst_cat), low_pct_cat, high_pct_cat,
                    format_p(p), assoc["cramers_v"], assoc["effect_label"],
                    assoc["n"])
            )
            drivers.append({
                "factor":      col,
                "impact":      round(min(diff_pct / 100, 1.0), 4),
                "direction":   "categorical",
                "key_category": str(worst_cat),
                "low_pct":     low_pct_cat,
                "high_pct":    high_pct_cat,
                "p_value":     round(p, 4),
                "cramers_v":   assoc["cramers_v"],
                "effect_label": assoc["effect_label"],
                "n":           assoc["n"],
                "detail":      detail,
                "dtype":       "categorical",
            })
        except Exception:
            logger.debug("analyze_root_cause: suppressed exception", exc_info=True)
            continue

    # Multiple-comparison correction across the whole driver scan.
    # This function tests up to 15 numeric plus 8 categorical columns, each
    # at α=0.05. Across ~20 independent columns roughly one "significant
    # driver" is expected from chance alone — and it would have been
    # reported as the headline root cause. Benjamini-Hochberg controls the
    # false-discovery rate over the family; drivers are kept only if they
    # survive it.
    if drivers:
        drivers = apply_fdr(drivers, p_key="p_value", q_key="q_value")
        survivors = [d for d in drivers if d["q_value"] < 0.05]
        if survivors:
            drivers = survivors
        else:
            # Nothing survived correction: report none rather than present a
            # finding that is indistinguishable from noise.
            logger.info("root cause: %d candidate driver(s) failed FDR "
                        "correction — reporting none", len(drivers))
            drivers = []

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
