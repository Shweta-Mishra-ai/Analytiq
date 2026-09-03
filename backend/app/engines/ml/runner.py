"""
engines/ml/runner.py — the pipeline end to end.

Target selection, feature preparation, training, importance, leakage
review and insights, in that order. The only module that knows about all
the others.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from typing import List, Optional

from app.engines.ml.results import MLReport
from app.engines.ml.targets import detect_task
from app.engines.ml.features import prepare_features
from app.engines.ml.training import train_models
from app.engines.ml.importance import get_feature_importance
from app.engines.ml.insights import _generate_insights
from app.engines.present import label as _L
from app.engines import present as _P

from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from app.engines.ml.training import _get_models, _make_pipeline


#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

def run_ml_pipeline(
    df: pd.DataFrame,
    target_col: str,
    selected_features: Optional[List[str]] = None,
    max_rows: int = 50_000,
) -> MLReport:
    """
    Full ML pipeline:
    1. Detect task (regression/classification)
    2. Prepare features
    3. Train + cross-validate multiple models
    4. Feature importance (SHAP if available)
    5. Generate insights
    Returns MLReport.
    """
    # Sample if large
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    task, task_reason = detect_task(df[target_col])
    if task == "unsupported":
        # Refused here rather than allowed to fail somewhere deeper, so
        # the message names the column and the reason instead of
        # surfacing as a type error from inside a scaler.
        raise ValueError(
            f"'{target_col}' cannot be used as a prediction target: "
            f"{task_reason.lower()}.")

    # Class balance for classification
    class_balance = None
    if task == "classification":
        vc = df[target_col].value_counts(normalize=True)
        class_balance = {str(k): round(float(v), 4) for k, v in vc.items()}

    # Prepare features
    X, y, label_encoders, encoding_map = prepare_features(
        df, target_col, selected_features)

    # ── Target-leakage guard ──────────────────────────────
    # A feature that is (nearly) a copy of the target produces a
    # spectacular but meaningless model. Detect and drop, with a warning.
    leakage_warnings = []
    # Kept in the model, but the reader is shown what the model
    # scores without them.
    suspect_cols: list = []
    y_num = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce")
    if y_num.notna().sum() >= 20:
        for col in list(X.columns):
            try:
                x_num = pd.to_numeric(X[col].reset_index(drop=True),
                                      errors="coerce")
                mask = x_num.notna() & y_num.notna()
                if mask.sum() < 20 or x_num[mask].nunique() < 2:
                    continue
                r = float(np.corrcoef(x_num[mask], y_num[mask])[0, 1])
                if abs(r) > 0.98:
                    X = X.drop(columns=[col])
                    leakage_warnings.append(
                        "Dropped '{}' — it is almost identical to the target "
                        "(|r|={:.3f}). Using it would fake a perfect model "
                        "(target leakage).".format(col, abs(r)))
            except Exception:
                logger.debug("run_ml_pipeline: suppressed exception", exc_info=True)
                continue

    # The correlation guard above catches a near-copy of the target. It
    # cannot see leakage carried by a column's *presence* — a churn_date
    # populated for exactly the churners — or by a categorical field, so
    # run the fuller check too and report what it finds.
    detected_leakage = []
    try:
        from app.engines.rigour import detect_leakage
        detected_leakage = detect_leakage(df, target_col)
        for finding in detected_leakage:
            leakage_warnings.append(finding.reason)
            # Only a confirmed leak is removed. A column that merely
            # separates the outcome very well is kept and flagged: on a
            # dataset whose target was defined as x > 0, dropping every
            # strong column dropped x, and the pipeline then reported
            # "no usable feature columns found" — the model destroyed
            # rather than protected.
            if finding.drop and finding.column in X.columns:
                X = X.drop(columns=[finding.column])
            elif finding.column in X.columns:
                suspect_cols.append(finding.column)
    except Exception:
        logger.warning("leakage detection failed", exc_info=True)

    if len(X.columns) == 0:
        report = MLReport(
            task=task, target_col=target_col,
            feature_cols=[], n_rows_used=len(df), n_features=0,
        )
        report.warnings.append("No usable feature columns found after preprocessing.")
        return report

    if len(X) < 20:
        report = MLReport(
            task=task, target_col=target_col,
            feature_cols=list(X.columns), n_rows_used=len(X), n_features=len(X.columns),
        )
        report.warnings.append("Too few rows ({}) for reliable ML. Need at least 20.".format(len(X)))
        return report

    # Train models
    model_results, X_test, y_test, target_encoder = train_models(X, y, task)
    best = next((m for m in model_results if m.is_best), None)

    # Feature importance
    feat_importance = []
    shap_values     = None
    if best and best.model is not None:
        feat_importance, shap_values = get_feature_importance(
            best, list(X.columns), X_test, task
        )

    # Predictions on test set
    y_pred = None
    if best and best.model is not None:
        try:
            y_pred = best.model.predict(X_test)
        except Exception:
            logger.debug("run_ml_pipeline: suppressed exception", exc_info=True)

    report = MLReport(
        task=task,
        target_col=target_col,
        feature_cols=list(X.columns),
        n_rows_used=len(X),
        n_features=len(X.columns),
        class_balance=class_balance,
        models=model_results,
        best_model=best,
        feature_importance=feat_importance,
        shap_values=shap_values,
        shap_feature_names=list(X.columns) if shap_values is not None else None,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        preprocessor=None,
        label_encoders=label_encoders,
        encoding_map=encoding_map,
        target_encoder=target_encoder,
        warnings=[task_reason] + leakage_warnings,
        leakage=detected_leakage,
    )

    # ── Did the model beat the obvious guess? ─────────────
    # Without this comparison a 91% accuracy on a 91%-imbalanced target
    # reads as a triumph. It is a model that learned to say "no".
    try:
        from app.engines.rigour import assess_classifier, assess_regressor
        if y_pred is not None and y_test is not None and len(y_test):
            if task == "classification":
                proba = None
                try:
                    if best.model is not None and hasattr(best.model,
                                                          "predict_proba"):
                        p = best.model.predict_proba(X_test)
                        if p.shape[1] == 2:
                            proba = p[:, 1]
                except Exception:
                    logger.debug("predict_proba unavailable", exc_info=True)
                report.verdict = assess_classifier(
                    y_test, y_pred, y_proba=proba,
                    auc=best.roc_auc if best else None)
            else:
                report.verdict = assess_regressor(y_test, y_pred)
    except Exception:
        logger.warning("model verdict could not be computed", exc_info=True)

    # A model that found nothing has no drivers worth naming: its
    # importances describe the noise it was fitted to.
    if report.verdict is not None and not report.verdict.usable:
        report.feature_importance = []
        report.shap_values = None
        report.shap_feature_names = None

    # What the model is worth without the columns we could not clear.
    # Keeping a suspiciously strong feature is only honest if the reader
    # can see how much of the result rests on it.
    if suspect_cols and best is not None:
        report.suspect_features = list(suspect_cols)
        try:
            reduced = X.drop(columns=[c for c in suspect_cols
                                      if c in X.columns])
            if reduced.shape[1] >= 1:
                strat = (y if task == "classification"
                         and pd.Series(y).nunique() <= 20 else None)
                r_tr, r_te, ry_tr, ry_te = train_test_split(
                    reduced, y, test_size=0.2, random_state=42,
                    stratify=strat)
                pipe = _make_pipeline(_get_models(task)[0][1], task)
                pipe.fit(r_tr, ry_tr)
                pred = pipe.predict(r_te)
                score = (r2_score(ry_te, pred) if task == "regression"
                         else accuracy_score(ry_te, pred))
                report.score_without_suspects = round(float(score), 4)
                report.warnings.append(
                    "Held out {}: the model scores {:.1%} without {} "
                    "against {:.1%} with {}. If that field turns out to be "
                    "recorded at the same time as the outcome, the lower "
                    "figure is the one that will hold on new data.".format(
                        _P.join_and([_L(c) for c in suspect_cols]),
                        score,
                        "them" if len(suspect_cols) > 1 else "it",
                        best.test_score,
                        "them" if len(suspect_cols) > 1 else "it"))
            else:
                report.warnings.append(
                    "Every feature in this model is one that separates the "
                    "outcome almost perfectly on its own. There is nothing "
                    "left to compare against, so treat the score as "
                    "unverified until you confirm when these fields are "
                    "populated.")
        except Exception:
            logger.warning("could not score the model without its suspect "
                           "features", exc_info=True)

    # Suspiciously perfect scores usually mean leakage we couldn't detect
    if best and best.test_score >= 0.995:
        report.warnings.append(
            "Test score is near-perfect ({:.1%}). Real-world data almost "
            "never behaves like this — check whether a feature contains "
            "information that would not be available before the outcome "
            "(target leakage).".format(best.test_score))

    report.insights = _generate_insights(report)
    return report
