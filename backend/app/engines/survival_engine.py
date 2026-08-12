"""
engines/survival_engine.py — Generalized survival analysis (time-to-event).

Not HR-specific despite "attrition" being the most common use case — this
works for ANY duration+event pair:
  - HR: employee tenure until attrition ("time-to-attrition")
  - Ecommerce/Sales: customer relationship length until churn
  - Finance: loan/subscription duration until default/cancellation

Implements Kaplan-Meier survival estimation and the log-rank test for
comparing survival between groups, both hand-rolled (no lifelines
dependency), consistent with this codebase's existing pattern of
implementing statistical methods directly via numpy/scipy.

All computation is done directly from the submitted dataset — no external
benchmarks or assumed baseline survival curves.

Ported from dataforge-ai's core/survival_engine.py (pure scipy/numpy/pandas,
no framework coupling).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  DATACLASSES
# ══════════════════════════════════════════════════════════

@dataclass
class SurvivalPoint:
    time:             float
    at_risk:          int
    events:           int
    survival_prob:    float
    ci_lower:         float
    ci_upper:         float


@dataclass
class SurvivalCurve:
    label:            str
    n_total:          int
    n_events:         int          # e.g. number who left/churned
    n_censored:       int          # still employed/active — event not yet observed
    median_survival:  Optional[float]   # None if median never reached (>50% survive)
    points:           List[SurvivalPoint] = field(default_factory=list)
    milestone_probs:  Dict[float, float] = field(default_factory=dict)  # {1.0: 0.92, 2.0: 0.81, ...}


@dataclass
class GroupComparisonResult:
    group_a:          str
    group_b:          str
    curve_a:          SurvivalCurve
    curve_b:          SurvivalCurve
    logrank_stat:     float
    p_value:          float
    is_significant:   bool
    verdict:          str


@dataclass
class SurvivalReport:
    duration_col:     str
    event_col:        str
    overall_curve:    SurvivalCurve
    group_col:        Optional[str] = None
    group_curves:     List[SurvivalCurve] = field(default_factory=list)
    pairwise_comparisons: List[GroupComparisonResult] = field(default_factory=list)
    summary:          str = ""
    warnings:         List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
#  KAPLAN-MEIER ESTIMATOR
# ══════════════════════════════════════════════════════════

def _kaplan_meier(durations: np.ndarray, events: np.ndarray, label: str = "") -> SurvivalCurve:
    """
    Compute the Kaplan-Meier survival estimate.

    durations: time-to-event OR time-to-censoring for each subject
    events:    1 if the event was observed (e.g. left/churned), 0 if censored
              (still active — we don't know when/if they'll leave)

    Uses Greenwood's formula for the variance/confidence interval.
    """
    n_total = len(durations)
    n_events = int(events.sum())
    n_censored = n_total - n_events

    # Sort by duration
    order = np.argsort(durations)
    d_sorted = durations[order]
    e_sorted = events[order]

    unique_times = np.unique(d_sorted[e_sorted == 1])  # only compute steps at event times

    points: List[SurvivalPoint] = []
    survival = 1.0
    variance_sum = 0.0  # for Greenwood's formula
    at_risk = n_total

    for t in unique_times:
        # At risk = everyone with duration >= t
        at_risk = int((d_sorted >= t).sum())
        d_i = int(((d_sorted == t) & (e_sorted == 1)).sum())  # events at this exact time

        if at_risk == 0:
            continue

        survival *= (1 - d_i / at_risk)

        if at_risk > d_i:
            variance_sum += d_i / (at_risk * (at_risk - d_i))

        se = survival * np.sqrt(variance_sum) if variance_sum > 0 else 0.0
        z = 1.96  # 95% CI
        ci_lower = max(0.0, survival - z * se)
        ci_upper = min(1.0, survival + z * se)

        points.append(SurvivalPoint(
            time=float(t), at_risk=at_risk, events=d_i,
            survival_prob=round(survival, 4),
            ci_lower=round(ci_lower, 4), ci_upper=round(ci_upper, 4),
        ))

    # Median survival time — first time point where survival <= 0.5
    median_survival = None
    for p in points:
        if p.survival_prob <= 0.5:
            median_survival = p.time
            break

    # Milestone survival probabilities (common tenure checkpoints)
    max_duration = float(durations.max()) if len(durations) else 0.0
    milestones = [m for m in [1.0, 2.0, 3.0, 5.0, 10.0] if m <= max_duration]
    milestone_probs = {}
    for m in milestones:
        # Find the last point at or before this milestone
        applicable = [p for p in points if p.time <= m]
        milestone_probs[m] = applicable[-1].survival_prob if applicable else 1.0

    return SurvivalCurve(
        label=label, n_total=n_total, n_events=n_events, n_censored=n_censored,
        median_survival=median_survival, points=points, milestone_probs=milestone_probs,
    )


# ══════════════════════════════════════════════════════════
#  LOG-RANK TEST  (compare survival between 2 groups)
# ══════════════════════════════════════════════════════════

def _log_rank_test(
    dur_a: np.ndarray, evt_a: np.ndarray,
    dur_b: np.ndarray, evt_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Log-rank test — the standard test for whether two survival curves
    differ significantly. Returns (chi2_statistic, p_value).
    """
    all_times = np.unique(np.concatenate([dur_a[evt_a == 1], dur_b[evt_b == 1]]))
    if len(all_times) == 0:
        return 0.0, 1.0

    O_a_total, E_a_total, V_total = 0.0, 0.0, 0.0

    for t in all_times:
        n_a = int((dur_a >= t).sum())
        n_b = int((dur_b >= t).sum())
        n = n_a + n_b
        if n == 0:
            continue

        d_a = int(((dur_a == t) & (evt_a == 1)).sum())
        d_b = int(((dur_b == t) & (evt_b == 1)).sum())
        d = d_a + d_b
        if d == 0:
            continue

        # Expected events in group A under the null (proportional to at-risk share)
        e_a = d * n_a / n
        O_a_total += d_a
        E_a_total += e_a

        if n > 1:
            v = (d * (n_a / n) * (n_b / n) * (n - d)) / (n - 1)
            V_total += v

    if V_total == 0:
        return 0.0, 1.0

    chi2_stat = (O_a_total - E_a_total) ** 2 / V_total
    p_value = float(1 - scipy_stats.chi2.cdf(chi2_stat, df=1))
    return float(chi2_stat), p_value


# ══════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ══════════════════════════════════════════════════════════

def run_survival_analysis(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: Optional[str] = None,
    max_groups: int = 5,
) -> SurvivalReport:
    """
    Run Kaplan-Meier survival analysis on a dataset.

    duration_col: numeric column — time elapsed (e.g. tenure in years,
                  days since first purchase)
    event_col:    binary column — 1/True/"Yes" if the event occurred
                  (e.g. attrition, churn), 0/False/"No" if censored
                  (still active — we don't know their eventual outcome)
    group_col:    optional categorical column to compare survival across
                  groups (e.g. department, satisfaction band) via
                  log-rank test

    Raises:
        TypeError  — if df is not a DataFrame
        ValueError — if columns are missing, event column isn't binary-
        coercible, or there's insufficient data (<10 valid rows)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"run_survival_analysis expects pd.DataFrame, got {type(df)}")
    for col in (duration_col, event_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    warnings_list: List[str] = []

    work = df[[duration_col, event_col] + ([group_col] if group_col else [])].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")

    # Coerce event column to binary 0/1 — accepts True/False, 1/0, "Yes"/"No"
    def _to_binary(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in ("1", "1.0", "true", "yes", "y", "left", "churned", "attrited"):
            return 1
        if s in ("0", "0.0", "false", "no", "n", "stayed", "active", "retained"):
            return 0
        try:
            return 1 if float(v) != 0 else 0
        except (ValueError, TypeError):
            return np.nan

    work["_event"] = work[event_col].apply(_to_binary)

    n_before = len(work)
    work = work.dropna(subset=[duration_col, "_event"])
    n_dropped = n_before - len(work)
    if n_dropped > 0:
        warnings_list.append(
            f"{n_dropped} rows excluded — invalid duration or unrecognized "
            f"event value in '{event_col}' (expected Yes/No, 1/0, True/False)."
        )

    # Negative duration (e.g. -2 years tenure) is physically impossible for
    # a time-elapsed metric — almost always a data entry error — and would
    # silently distort the Kaplan-Meier curve and median estimate if
    # included unfiltered.
    negative_mask = work[duration_col] < 0
    n_negative = int(negative_mask.sum())
    if n_negative > 0:
        work = work[~negative_mask]
        warnings_list.append(
            f"{n_negative} rows excluded — negative '{duration_col}' values "
            "are not physically valid for a time-elapsed metric (likely data "
            "entry errors)."
        )

    if len(work) < 10:
        raise ValueError(
            f"Only {len(work)} valid rows after cleaning — need at least 10 "
            "for meaningful survival analysis."
        )

    if (work["_event"] == 1).sum() == 0:
        raise ValueError(
            f"No events found in '{event_col}' — every record is censored "
            "(nobody left/churned in this dataset). Survival analysis "
            "requires at least some observed events."
        )

    durations = work[duration_col].values.astype(float)
    events = work["_event"].values.astype(int)

    overall_curve = _kaplan_meier(durations, events, label="Overall")

    group_curves: List[SurvivalCurve] = []
    pairwise: List[GroupComparisonResult] = []

    if group_col:
        groups = work[group_col].dropna().unique().tolist()
        if len(groups) > max_groups:
            # Keep the largest groups only
            top_groups = work[group_col].value_counts().head(max_groups).index.tolist()
            warnings_list.append(
                f"'{group_col}' has {len(groups)} distinct values — showing "
                f"the {max_groups} largest groups only."
            )
            groups = top_groups

        group_data = {}
        for g in groups:
            sub = work[work[group_col] == g]
            if len(sub) < 5:
                continue
            curve = _kaplan_meier(
                sub[duration_col].values.astype(float),
                sub["_event"].values.astype(int),
                label=str(g),
            )
            group_curves.append(curve)
            group_data[g] = sub

        # Pairwise log-rank tests between all group pairs (capped to avoid
        # combinatorial explosion — only compare each group to the largest)
        if len(group_curves) >= 2:
            sorted_groups = sorted(group_data.items(), key=lambda x: -len(x[1]))
            baseline_label, baseline_data = sorted_groups[0]
            for label, sub in sorted_groups[1:]:
                try:
                    chi2, p = _log_rank_test(
                        baseline_data[duration_col].values.astype(float),
                        baseline_data["_event"].values.astype(int),
                        sub[duration_col].values.astype(float),
                        sub["_event"].values.astype(int),
                    )
                    is_sig = p < 0.05
                    curve_a = next(c for c in group_curves if c.label == str(baseline_label))
                    curve_b = next(c for c in group_curves if c.label == str(label))
                    verdict = (
                        f"'{label}' vs '{baseline_label}': survival curves differ "
                        f"significantly (p={p:.4f})." if is_sig else
                        f"'{label}' vs '{baseline_label}': no significant "
                        f"difference in survival (p={p:.4f})."
                    )
                    pairwise.append(GroupComparisonResult(
                        group_a=str(baseline_label), group_b=str(label),
                        curve_a=curve_a, curve_b=curve_b,
                        logrank_stat=round(chi2, 4), p_value=round(p, 6),
                        is_significant=is_sig, verdict=verdict,
                    ))
                except Exception:
                    logger.warning("Log-rank test failed for group '%s'", label, exc_info=True)

    # ── Narrative summary ────────────────────────────────────────────────────
    event_rate = overall_curve.n_events / overall_curve.n_total * 100
    parts = [
        f"{event_rate:.1f}% of {overall_curve.n_total:,} records experienced the "
        f"event (e.g. attrition/churn) within the observed period."
    ]
    if overall_curve.median_survival is not None:
        parts.append(
            f"Median survival time: {overall_curve.median_survival:.1f} "
            f"({duration_col} units) — half of all subjects experience the "
            "event by this point."
        )
    else:
        parts.append(
            "Median survival was not reached — over 50% of subjects had not "
            "experienced the event by the end of the observed period."
        )
    if pairwise:
        n_sig = sum(1 for c in pairwise if c.is_significant)
        if n_sig > 0:
            parts.append(
                f"{n_sig} of {len(pairwise)} group comparison(s) show a "
                f"statistically significant difference in survival — some "
                f"groups are retained meaningfully longer than others."
            )
        else:
            parts.append(
                "No significant difference in survival was found between groups."
            )

    return SurvivalReport(
        duration_col=duration_col, event_col=event_col,
        overall_curve=overall_curve, group_col=group_col,
        group_curves=group_curves, pairwise_comparisons=pairwise,
        summary=" ".join(parts), warnings=warnings_list,
    )
