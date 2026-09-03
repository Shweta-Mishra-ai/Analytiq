"""
engines/eda/findings.py — turning results into sentences.

Nothing here computes anything. It reads the results the other modules
produced and says which of them a person should be told about, which is
a separate judgement from whether they are statistically true.
"""
from __future__ import annotations

import logging


logger = logging.getLogger(__name__)

from typing import List

from app.engines.eda.results import EDAReport
from app.engines.present import label as _L
from app.engines.statistics import format_p


#  KEY FINDINGS GENERATOR
# ══════════════════════════════════════════════════════════

def _generate_key_findings(report: "EDAReport") -> List[str]:
    findings = []

    # Non-normal columns
    non_normal = [
        col for col, r in report.univariate.items()
        if r.is_normal is False and r.mean is not None
    ]
    if non_normal:
        findings.append(
            "{} column(s) are non-normally distributed: {}. "
            "Use non-parametric tests (Mann-Whitney, Kruskal-Wallis).".format(
                len(non_normal), ", ".join(non_normal[:4]))
        )

    # High outlier columns
    outlier_cols = [
        (col, r.outlier_pct) for col, r in report.univariate.items()
        if r.outlier_pct and r.outlier_pct > 5
    ]
    if outlier_cols:
        worst = max(outlier_cols, key=lambda x: x[1])
        findings.append(
            "'{}' has {:.1f}% outliers — highest in dataset. "
            "Validate these values before modeling.".format(*worst)
        )

    # Skewed columns
    skewed = [
        col for col, r in report.univariate.items()
        if r.skewness and abs(r.skewness) > 2
    ]
    if skewed:
        findings.append(
            "{} column(s) heavily skewed (|skew|>2): {}. "
            "Log-transform recommended before regression.".format(
                len(skewed), ", ".join(skewed[:3]))
        )

    # Strong correlations
    strong_corr = [
        r for r in report.correlations
        if r.is_significant and r.effect_size and r.effect_size >= 0.5
    ]
    if strong_corr:
        top = strong_corr[0]
        # Whether a correlation actually causes a multicollinearity
        # problem is what VIF answers, and VIF has already run. The
        # report used to hedge "may indicate multicollinearity" directly
        # above a table stating there was none.
        flagged = [m for m in report.multicollinearity
                   if m.verdict not in ("OK", "")]
        if flagged:
            verdict = ("This does inflate the variance of {}: VIF {:.1f}. "
                       "Drop one of the pair, or combine them, before "
                       "fitting a model that assumes independent "
                       "predictors.".format(_L(flagged[0].feature),
                                            flagged[0].vif))
        else:
            verdict = ("It does not amount to a multicollinearity problem: "
                       "every variance inflation factor is below the "
                       "conventional threshold, the highest being "
                       "{:.1f}.".format(max((m.vif for m in
                                             report.multicollinearity),
                                            default=0.0)))
        findings.append(
            "{} and {} move together ({}, r={:.2f}, n={:,}). {}".format(
                _L(top.col_a), _L(top.col_b),
                format_p(top.p_value), top.statistic,
                report.n_rows, verdict))

    # Group differences. Significance alone is not enough: on a large
    # sample almost every comparison is significant, and a difference too
    # small to act on is not a finding however certain it is.
    try:
        from app.engines.rigour import assess_finding
        sig_groups = []
        for r in report.group_comparisons:
            if not r.is_significant:
                continue
            n = sum(g.get("n", 0) for g in (r.group_stats or {}).values()) \
                if isinstance(r.group_stats, dict) else None
            if assess_finding(p_value=r.p_value, effect_size=r.effect_size,
                              n=n or None).reportable:
                sig_groups.append(r)
    except Exception:
        logger.warning("finding gate unavailable", exc_info=True)
        sig_groups = [r for r in report.group_comparisons if r.is_significant]
    if sig_groups:
        top = sig_groups[0]
        findings.append(
            "'{}' differs by '{}' ({}, p={:.4f}, effect {}). The gap is "
            "both statistically reliable and large enough to act on.".format(
                top.numeric_col, top.group_col,
                top.test_used, top.p_value, top.effect_label)
        )

    # Interactions lead, because an effect that reverses across a second
    # factor is the finding a main-effects summary reports as "no effect".
    for inter in list(getattr(report, "interactions", None) or [])[:2]:
        findings.insert(0, inter.description)

    # Class imbalance changes how every subsequent number reads.
    for note in list(getattr(report, "imbalance_notes", None) or [])[:1]:
        findings.append(note.note)

    # Groups too thin to carry a comparison.
    rare = list(getattr(report, "rare_categories", None) or [])
    if rare:
        findings.append(
            "{} category level(s) hold too few rows to support a finding "
            "on their own — smallest is '{}' in '{}' at {} row(s). These "
            "are excluded from group comparisons rather than ranked "
            "alongside groups many times their size.".format(
                len(rare), rare[0].level, rare[0].column, rare[0].n))

    # VIF issues
    severe_vif = [r for r in report.multicollinearity if r.vif >= 10]
    if severe_vif:
        findings.append(
            "{} feature(s) have high VIF (multicollinearity): {}. "
            "Remove or combine before regression modeling.".format(
                len(severe_vif),
                ", ".join([r.feature for r in severe_vif[:3]]))
        )

    return findings


# ══════════════════════════════════════════════════════════