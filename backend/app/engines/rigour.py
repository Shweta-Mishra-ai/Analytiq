"""
engines/rigour.py — the gate between "a number was computed" and "this is
a finding".

WHY THIS EXISTS
---------------
Given an imbalanced target (91% one class) and features that were pure
random noise, the pipeline reported:

    accuracy 0.9125   f1 0.0   roc_auc 0.4457
    "Excellent classifier: 91.2% accuracy on held-out data."
    "Most important predictor: 'f0' (31% contribution)."

An AUC of 0.4457 is worse than a coin flip, and an F1 of 0.0 means the
model never once predicted the minority class — it learned to say "no"
every time. Accuracy was high because 91% of the answers are "no". The
report called it excellent and then attributed a third of its reasoning
to a column of noise.

Nothing was broken in the arithmetic. What was missing was the comparison
every analyst makes first: against what? A model that cannot beat
"predict the majority class every time" has found nothing, and its
feature importances describe the shape of the noise it was fitted to.

This module makes that comparison, and the verdicts it returns are used
to decide what may be presented as a finding at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# A model must beat its baseline by at least this much of the headroom
# that was available to it. Beating a 91% baseline by 0.3pp is noise.
MIN_BASELINE_LIFT = 0.10
# Below this, ranking is no better than chance in any useful sense.
MIN_USABLE_AUC = 0.60
# A single feature that separates the target this well on its own is
# almost always leakage — something recorded at or after the outcome.
LEAKAGE_AUC = 0.97
# Minimum rows before any of this is worth saying out loud.
MIN_ROWS_FOR_A_CLAIM = 50
# Conventional significance, and a small-but-real effect.
MAX_P_VALUE = 0.05
MIN_EFFECT_SIZE = 0.20


# ══════════════════════════════════════════════════════════
#  MODEL VERDICTS
# ══════════════════════════════════════════════════════════

@dataclass
class ModelVerdict:
    """Whether a fitted model has actually found anything."""
    usable: bool
    task: str
    baseline_score: float
    baseline_strategy: str
    model_score: float
    metric: str
    lift: float                 # share of available headroom captured
    auc: Optional[float] = None
    minority_recall: Optional[float] = None
    reason: str = ""
    verdict: str = ""

    @property
    def headline(self) -> str:
        """One sentence, safe to print in a client report."""
        return self.verdict


def _majority_baseline(y: pd.Series) -> Tuple[float, str]:
    counts = pd.Series(y).value_counts(normalize=True, dropna=True)
    if counts.empty:
        return 0.0, "no data"
    top = float(counts.iloc[0])
    return top, "always predict '{}'".format(counts.index[0])


def assess_classifier(y_true, y_pred, y_proba=None,
                      auc: Optional[float] = None) -> ModelVerdict:
    """Is this classifier better than the obvious guess?

    The obvious guess is the majority class. `lift` is the share of the
    headroom above that baseline which the model actually captured, so a
    model scoring 91.5% against a 91.0% baseline reports a lift of 0.06
    rather than a flattering "91.5% accurate".
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    n = len(y_true)

    base, strategy = _majority_baseline(y_true)
    acc = float((y_true == y_pred).mean()) if n else 0.0
    headroom = max(1.0 - base, 1e-9)
    lift = (acc - base) / headroom

    # Did it ever predict the minority class at all?
    minority_recall = None
    try:
        counts = y_true.value_counts()
        if len(counts) == 2:
            minority = counts.index[-1]
            mask = y_true == minority
            if mask.sum():
                minority_recall = float((y_pred[mask] == minority).mean())
    except Exception:
        logger.debug("minority recall failed", exc_info=True)

    if auc is None and y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            logger.debug("auc unavailable", exc_info=True)
            auc = None

    # Which yardstick applies depends on what the model produces.
    #
    # Accuracy against the majority-class baseline is the right test only
    # when there is no ranking to judge. On an imbalanced problem a useful
    # risk model deliberately scores *below* that baseline: an attrition
    # model that catches 78% of leavers at AUC 0.74 has accuracy 0.72
    # against a 0.81 baseline, because at a 0.5 threshold it accepts false
    # positives in order to find the leavers at all. Judging that on
    # accuracy would reject exactly the models worth having — the mirror
    # image of calling a coin flip "excellent".
    #
    # So: when a ranking exists, judge the ranking. Threshold choice is a
    # separate decision, made against the cost of each error.
    reasons = []
    if n < MIN_ROWS_FOR_A_CLAIM:
        reasons.append("only {:,} rows — too few to support a claim".format(n))

    if auc is not None and auc == auc:
        if auc < MIN_USABLE_AUC:
            reasons.append("AUC {:.2f} — ranking is {}".format(
                auc, "worse than chance" if auc < 0.5
                else "barely above chance"))
    else:
        # No probabilities to rank by, so accuracy is all there is.
        if lift < MIN_BASELINE_LIFT:
            reasons.append(
                "captures {:.0f}% of the headroom above the baseline"
                .format(max(lift, 0) * 100))
        if minority_recall is not None and minority_recall == 0.0:
            reasons.append("never predicts the minority class")

    usable = not reasons

    if usable:
        parts = ["The model reaches {:.1f}% accuracy against a {:.1f}% "
                 "baseline ({})".format(acc * 100, base * 100, strategy)]
        if auc is not None and auc == auc:
            parts.append("and ranks with AUC {:.2f}".format(auc))
        verdict = ", ".join(parts) + "."
        if lift < 0 and auc is not None and auc == auc:
            # Explain the apparent contradiction before a reader spots it
            # and stops trusting the rest of the page.
            verdict += (
                " Accuracy sits below the baseline because the model is "
                "tuned to find the minority class rather than to be right "
                "most often: at the default threshold it catches {:.0f}% of "
                "them, where always guessing the majority catches none. "
                "Where to set that threshold is a business decision about "
                "the cost of a miss against the cost of a false alarm."
            ).format((minority_recall or 0) * 100)
    else:
        verdict = (
            "No reliable predictive signal was found. The model reaches "
            "{:.1f}% accuracy, but always predicting the majority class "
            "reaches {:.1f}% — so the model adds {}. {}. Feature "
            "importances are not reported: with no signal to explain, they "
            "describe the noise the model was fitted to."
        ).format(
            acc * 100, base * 100,
            "nothing" if lift <= 0 else "{:.1f} percentage points".format(
                (acc - base) * 100),
            "; ".join(r.capitalize() if i == 0 else r
                      for i, r in enumerate(reasons)))

    return ModelVerdict(
        usable=usable, task="classification", baseline_score=round(base, 4),
        baseline_strategy=strategy, model_score=round(acc, 4),
        metric="accuracy", lift=round(lift, 4),
        auc=round(auc, 4) if auc is not None else None,
        minority_recall=(round(minority_recall, 4)
                         if minority_recall is not None else None),
        reason="; ".join(reasons), verdict=verdict)


def assess_regressor(y_true, y_pred) -> ModelVerdict:
    """Is this regressor better than predicting the mean?

    R² is exactly this comparison, so a negative R² means the model is
    worse than a horizontal line through the data.
    """
    y_true = np.asarray(pd.Series(y_true).astype(float))
    y_pred = np.asarray(pd.Series(y_pred).astype(float))
    n = len(y_true)

    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    ss_res = float(((y_true - y_pred) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    reasons = []
    if n < MIN_ROWS_FOR_A_CLAIM:
        reasons.append("only {:,} rows — too few to support a claim".format(n))
    if r2 < MIN_BASELINE_LIFT:
        reasons.append("R² {:.2f} — {}".format(
            r2, "worse than predicting the mean" if r2 < 0
            else "barely better than predicting the mean"))
    usable = not reasons

    if usable:
        verdict = ("The model explains {:.0f}% of the variation in the "
                   "target, against a baseline of predicting the mean."
                   ).format(r2 * 100)
    else:
        verdict = ("No reliable predictive signal was found. {}. Predicting "
                   "the mean for every row would do about as well, so the "
                   "drivers below are not reported."
                   ).format("; ".join(r.capitalize() for r in reasons))

    return ModelVerdict(
        usable=usable, task="regression", baseline_score=0.0,
        baseline_strategy="always predict the mean", model_score=round(r2, 4),
        metric="r2", lift=round(r2, 4), reason="; ".join(reasons),
        verdict=verdict)


# ══════════════════════════════════════════════════════════
#  TARGET LEAKAGE
# ══════════════════════════════════════════════════════════

@dataclass
class LeakageFinding:
    column: str
    separation: float
    reason: str


def detect_leakage(df: pd.DataFrame, target: str,
                   max_cols: int = 40) -> List[LeakageFinding]:
    """Columns that predict the target far too well on their own.

    A single column separating the outcome almost perfectly is rarely a
    discovery. It is usually something recorded at the same moment as the
    outcome or afterwards — a `churn_date` on a churn model, a
    `settlement_amount` on a claim model. Reported as leakage rather than
    celebrated as the top driver, because a model built on it looks
    excellent in validation and is useless in production, where the column
    does not exist yet.
    """
    findings: List[LeakageFinding] = []
    if target not in df.columns:
        return findings
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        logger.info("scikit-learn unavailable — leakage check skipped")
        return findings

    y_raw = df[target]
    codes = pd.Series(pd.factorize(y_raw)[0], index=df.index)
    if codes.nunique() != 2:
        return findings          # binary targets only
    y = (codes == codes.value_counts().index[-1]).astype(int)

    for col in list(df.columns)[:max_cols]:
        if col == target:
            continue
        s = df[col]
        try:
            # Leakage through absence. A `churn_date` is NULL for exactly
            # the customers who did not churn, so the column's *presence*
            # gives the answer away even though its values, taken alone,
            # look unremarkable. This is missed entirely by scoring the
            # values, because dropping the nulls leaves only one class.
            miss = s.isna()
            if 0 < miss.sum() < len(s):
                sep_miss = float(roc_auc_score(y, (~miss).astype(int)))
                sep_miss = max(sep_miss, 1 - sep_miss)
                if sep_miss >= LEAKAGE_AUC:
                    findings.append(LeakageFinding(
                        column=str(col), separation=round(sep_miss, 4),
                        reason=(
                            "'{}' is populated for almost exactly the rows "
                            "where {} takes one value ({:.2f} separation from "
                            "presence alone). Whether the field is filled in "
                            "gives the outcome away, so a model using it will "
                            "not work on new data where the outcome has not "
                            "happened yet."
                        ).format(col, target, sep_miss)))
                    continue
        except Exception:
            logger.debug("missingness leakage check failed for %r", col,
                         exc_info=True)
        try:
            if pd.api.types.is_numeric_dtype(s):
                x = pd.to_numeric(s, errors="coerce")
                ok = x.notna() & y.notna()
                if ok.sum() < MIN_ROWS_FOR_A_CLAIM or x[ok].nunique() < 2:
                    continue
                auc = float(roc_auc_score(y[ok], x[ok]))
                sep = max(auc, 1 - auc)
            else:
                # For a categorical column, how well its best split
                # separates the outcome.
                grouped = y.groupby(s.astype(str)).agg(["mean", "count"])
                grouped = grouped[grouped["count"] >= 5]
                if len(grouped) < 2:
                    continue
                sep = float(grouped["mean"].max() - grouped["mean"].min())
                sep = 0.5 + sep / 2      # onto the same 0.5-1.0 scale
            if sep >= LEAKAGE_AUC:
                findings.append(LeakageFinding(
                    column=str(col), separation=round(sep, 4),
                    reason=(
                        "'{}' separates {} almost perfectly on its own "
                        "({:.2f}). A single column this predictive is usually "
                        "recorded at the same time as the outcome or after "
                        "it, in which case a model using it will not work on "
                        "new data, where the column is not yet known. "
                        "Confirm when this field is populated before "
                        "including it."
                    ).format(col, target, sep)))
        except Exception:
            logger.debug("leakage check failed for %r", col, exc_info=True)
    return findings


# ══════════════════════════════════════════════════════════
#  STATISTICAL FINDINGS
# ══════════════════════════════════════════════════════════

@dataclass
class FindingVerdict:
    reportable: bool
    reason: str = ""


def assess_finding(p_value: Optional[float] = None,
                   effect_size: Optional[float] = None,
                   n: Optional[int] = None) -> FindingVerdict:
    """May a statistical comparison be printed as a finding?

    Significance alone is not enough: on a large enough sample almost
    everything is significant, and a difference too small to act on is not
    a finding however certain it is. Both tests have to pass.
    """
    reasons = []
    if n is not None and n < MIN_ROWS_FOR_A_CLAIM:
        reasons.append("only {:,} rows".format(n))
    if p_value is not None and not (p_value == p_value):
        reasons.append("the test did not converge")
    elif p_value is not None and p_value > MAX_P_VALUE:
        reasons.append("p = {:.3f}, above the 0.05 threshold".format(p_value))
    if effect_size is not None and effect_size == effect_size \
            and abs(effect_size) < MIN_EFFECT_SIZE:
        reasons.append(
            "effect size {:.2f} — statistically detectable but too small to "
            "act on".format(effect_size))
    return FindingVerdict(reportable=not reasons, reason="; ".join(reasons))
