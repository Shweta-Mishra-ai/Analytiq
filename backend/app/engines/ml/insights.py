"""
engines/ml/insights.py — saying what the model means.

Reads a finished report and writes sentences. Computes nothing, which is
why it is separate: the wording changes far more often than the maths.
"""
from __future__ import annotations

import logging


logger = logging.getLogger(__name__)

from typing import List

from app.engines.ml.results import MLReport


#  INSIGHTS GENERATOR
# ══════════════════════════════════════════════════════════

def _generate_insights(report: MLReport) -> List[str]:
    insights = []
    best = report.best_model

    if best is None:
        return ["No model trained successfully."]

    # Performance interpretation
    if report.task == "regression":
        r2 = best.test_score
        if r2 >= 0.85:
            insights.append(
                "Excellent model: R2={:.2f} — model explains {:.0f}% of variance in '{}'.".format(
                    r2, r2*100, report.target_col))
        elif r2 >= 0.70:
            insights.append(
                "Good model: R2={:.2f} — explains {:.0f}% of variance. "
                "Acceptable for business use.".format(r2, r2*100))
        elif r2 >= 0.50:
            insights.append(
                "Moderate model: R2={:.2f} — explains {:.0f}% of variance. "
                "Consider adding more features.".format(r2, r2*100))
        else:
            verdict = getattr(report, "verdict", None)
            insights.append(
                verdict.verdict if verdict is not None and not verdict.usable
                else "Weak model: R2={:.2f} — '{}' is difficult to predict "
                     "from current features.".format(r2, report.target_col))
    else:
        acc = best.test_score
        verdict = getattr(report, "verdict", None)
        if verdict is not None and not verdict.usable:
            # Say what was actually found. Quoting the accuracy alone here
            # described a model with AUC 0.45 and F1 0.0 as "Excellent".
            insights.append(verdict.verdict)
        elif verdict is not None:
            insights.append(verdict.verdict)
        elif acc >= 0.90:
            insights.append(
                "Excellent classifier: {:.1f}% accuracy on held-out data.".format(acc*100))
        elif acc >= 0.75:
            insights.append(
                "Good classifier: {:.1f}% accuracy.".format(acc*100))
        else:
            insights.append(
                "Weak classifier: {:.1f}% accuracy. "
                "Class imbalance or insufficient features may be the cause.".format(acc*100))

    # Overfitting
    if best.overfit_label == "Severe":
        insights.append(
            "WARNING: Severe overfitting detected (train={:.2f} vs test={:.2f}). "
            "Model memorized training data — will not generalize.".format(
                best.train_score, best.test_score))
    elif best.overfit_label == "Mild":
        insights.append(
            "Mild overfitting (gap={:.2f}). "
            "Consider regularization or more training data.".format(best.overfit_gap))

    # Top feature
    if report.feature_importance:
        top = report.feature_importance[0]
        insights.append(
            "Most important predictor: '{}' ({:.0f}% contribution). {}".format(
                top.feature, top.importance*100, top.explanation))

    # Model comparison
    if len(report.models) >= 2:
        best_cv  = report.models[0].cv_score
        worst_cv = report.models[-1].cv_score
        if best_cv - worst_cv > 0.1:
            insights.append(
                "Significant model performance gap: best ({}) CV={:.2f} "
                "vs worst ({}) CV={:.2f}. Model choice matters for this dataset.".format(
                    report.models[0].name, best_cv,
                    report.models[-1].name, worst_cv))

    # Class imbalance warning
    if report.class_balance:
        max_pct = max(report.class_balance.values())
        if max_pct > 0.80:
            insights.append(
                "Class imbalance detected ({:.0f}% dominant class). "
                "Accuracy metric may be misleading — check F1 score.".format(max_pct*100))

    return insights


# ══════════════════════════════════════════════════════════