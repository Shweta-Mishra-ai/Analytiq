"""
engines/ml/training.py — fitting the candidates and scoring them
honestly.

Cross-validation before the test score, and the gap between them
reported rather than hidden, because a model that memorised the training
set looks excellent right up until it meets new data.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Optional, Tuple

from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.engines.ml.results import ModelResult

from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    # Optional: the pipeline works without it, with one fewer candidate.
    XGBOOST_AVAILABLE = False



#  MODEL TRAINING
# ══════════════════════════════════════════════════════════

def _get_models(task: str) -> List[Tuple[str, Any]]:
    """Return list of (name, model) tuples for given task."""
    if task == "regression":
        models = [
            ("Ridge Regression",    Ridge(alpha=1.0)),
            ("Random Forest",       RandomForestRegressor(
                                        n_estimators=100, random_state=42,
                                        n_jobs=-1)),
            ("Gradient Boosting",   GradientBoostingRegressor(
                                        n_estimators=100, random_state=42)),
        ]
        if XGBOOST_AVAILABLE:
            models.append(("XGBoost", XGBRegressor(
                n_estimators=100, random_state=42,
                verbosity=0, eval_metric="rmse")))
    else:
        models = [
            ("Logistic Regression", LogisticRegression(
                                        max_iter=1000, random_state=42)),
            ("Random Forest",       RandomForestClassifier(
                                        n_estimators=100, random_state=42,
                                        n_jobs=-1)),
            ("Gradient Boosting",   GradientBoostingClassifier(
                                        n_estimators=100, random_state=42)),
        ]
        if XGBOOST_AVAILABLE:
            models.append(("XGBoost", XGBClassifier(
                n_estimators=100, random_state=42,
                verbosity=0, eval_metric="logloss",
                use_label_encoder=False)))
    return models


def _make_pipeline(model, task: str) -> Pipeline:
    """Wrap model in imputer + scaler pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def _evaluate_regression(y_true, y_pred) -> Dict:
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}


def _at_operating_threshold(y_true, y_prob) -> dict:
    """What the model does at a cut chosen rather than assumed.

    0.5 is right only when the classes are balanced and a miss costs
    what a false alarm costs. On an 80/20 attrition file it gave F1 0.17
    and caught 11% of the leavers, which is a model nobody can act on —
    and the very same model, cut where it performs best, catches most of
    them. Reporting only the default was discarding recall the model had
    already earned.

    The threshold itself is a business decision; what this does is show
    the reader there is one to make, and what it is worth.
    """
    if y_prob is None:
        return {}
    try:
        import numpy as _np
        from sklearn.metrics import f1_score, precision_score, recall_score

        proba = _np.asarray(y_prob)
        if proba.ndim == 2:
            if proba.shape[1] != 2:
                return {}         # multi-class: one cut does not describe it
            proba = proba[:, 1]

        classes = sorted(set(_np.asarray(y_true).ravel().tolist()))
        if len(classes) != 2:
            return {}
        positive = classes[-1]
        truth = (_np.asarray(y_true) == positive).astype(int)

        best_t, best_f1 = 0.5, -1.0
        for t in _np.arange(0.05, 0.95, 0.01):
            f1 = float(f1_score(truth, (proba >= t).astype(int),
                                zero_division=0))
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1

        pred = (proba >= best_t).astype(int)
        return {
            "threshold": round(best_t, 2),
            "threshold_basis": ("maximises F1 — balances missed cases "
                                "against false alarms"),
            "precision_at_threshold": round(float(
                precision_score(truth, pred, zero_division=0)), 4),
            "recall_at_threshold": round(float(
                recall_score(truth, pred, zero_division=0)), 4),
            "f1_at_threshold": round(best_f1, 4),
        }
    except Exception:
        logger.debug("operating threshold search failed", exc_info=True)
        return {}


def _evaluate_classification(y_true, y_pred, y_prob=None) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    f1  = f1_score(y_true, y_pred, average=avg, zero_division=0)
    auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                    average="weighted")
            auc = round(auc, 4)
        except Exception:
            logger.debug("_evaluate_classification: suppressed exception", exc_info=True)
    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "roc_auc": auc}


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    target_encoder: Optional[LabelEncoder] = None,
) -> List[ModelResult]:
    """
    Train all models, cross-validate, evaluate on holdout.
    Returns list of ModelResult sorted by cv_score descending.
    """
    # Encode classification target
    if task == "classification":
        if y.dtype == object or str(y.dtype) == "str":
            if target_encoder is None:
                target_encoder = LabelEncoder()
            y = pd.Series(
                target_encoder.fit_transform(y.astype(str)),
                index=y.index
            )
        else:
            y = y.astype(int)

    # Train/test split — stratified for classification
    stratify = y if task == "classification" and y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scoring = "r2" if task == "regression" else "f1_weighted"
    results = []

    for name, model in _get_models(task):
        try:
            pipe = _make_pipeline(model, task)

            # Cross-validation on training set
            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                cv=5, scoring=scoring, n_jobs=-1
            )

            # Fit on full training set
            pipe.fit(X_train, y_train)
            y_pred_train = pipe.predict(X_train)
            y_pred_test  = pipe.predict(X_test)

            # Scores
            if task == "regression":
                train_s = r2_score(y_train, y_pred_train)
                test_s  = r2_score(y_test,  y_pred_test)
                metrics = _evaluate_regression(y_test, y_pred_test)
                metric_name = "R2"
            else:
                train_s = accuracy_score(y_train, y_pred_train)
                test_s  = accuracy_score(y_test,  y_pred_test)
                try:
                    y_prob = pipe.predict_proba(X_test)
                except Exception:
                    y_prob = None
                metrics = _evaluate_classification(y_test, y_pred_test, y_prob)
                metric_name = "Accuracy"
                metrics.update(_at_operating_threshold(y_test, y_prob))

            gap   = train_s - test_s
            o_lbl = ("None" if gap < 0.05
                     else "Mild" if gap < 0.15
                     else "Severe")

            results.append(ModelResult(
                name=name, task=task,
                cv_score=round(float(np.mean(cv_scores)), 4),
                cv_std=round(float(np.std(cv_scores)), 4),
                train_score=round(train_s, 4),
                test_score=round(test_s, 4),
                overfit_gap=round(gap, 4),
                overfit_label=o_lbl,
                metric_name=metric_name,
                mae=metrics.get("mae"),
                rmse=metrics.get("rmse"),
                f1=metrics.get("f1"),
                roc_auc=metrics.get("roc_auc"),
                threshold=metrics.get("threshold"),
                threshold_basis=metrics.get("threshold_basis", ""),
                precision_at_threshold=metrics.get("precision_at_threshold"),
                recall_at_threshold=metrics.get("recall_at_threshold"),
                f1_at_threshold=metrics.get("f1_at_threshold"),
                model=pipe,
            ))

        except Exception as e:
            logger.warning(f"Model '{name}' failed to train and was skipped: {e}")
            results.append(ModelResult(
                name=name, task=task,
                cv_score=-999, cv_std=0,
                train_score=0, test_score=0,
                overfit_gap=0, overfit_label="N/A",
                metric_name="N/A",
                model=None,
            ))

    # "Best" was the top cross-validation score alone, which picked a
    # model scoring 78.9% with AUC 0.621 and mild overfitting over one
    # scoring 80.6% with AUC 0.632 and none. Cross-validation is the
    # right primary signal — it is the one measured on data the model
    # did not choose its parameters on — but when two models are within
    # noise of each other on it, the tie is broken on the things that
    # decide which one to deploy: ranking quality, then held-out score,
    # then how far the model fell from train to test.
    def _rank_key(m):
        return (round(m.cv_score, 2),                 # noise floor
                m.roc_auc if m.roc_auc is not None else 0.0,
                m.test_score,
                -abs(m.overfit_gap))
    results.sort(key=_rank_key, reverse=True)
    if results:
        results[0].is_best = True

    return results, X_test, y_test, target_encoder


# ══════════════════════════════════════════════════════════