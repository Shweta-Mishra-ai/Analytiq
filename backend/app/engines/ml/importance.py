"""
engines/ml/importance.py — which features actually drove the model.

Grouped permutation importance rather than summed one-hot impurity: the
latter rewards a categorical column simply for having many levels, which
put "employee number" at the top of more than one report.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from typing import List, Optional, Tuple

from app.engines.ml.results import FeatureImportance, ModelResult


#  FEATURE IMPORTANCE + SHAP
# ══════════════════════════════════════════════════════════

def get_feature_importance(
    model_result: ModelResult,
    feature_names: List[str],
    X_test: pd.DataFrame,
    task: str,
) -> Tuple[List[FeatureImportance], Optional[np.ndarray]]:
    """
    Extract feature importance — tries SHAP first, falls back to
    model-native importance.
    """
    importances = []
    shap_values = None

    if model_result.model is None:
        return importances, shap_values

    pipe  = model_result.model
    model = pipe.named_steps["model"]

    # ── Try SHAP ──────────────────────────────────────────
    try:
        import shap
        # Transform X_test through imputer + scaler
        X_transformed = pipe[:-1].transform(X_test)
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

        if hasattr(model, "feature_importances_"):
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(X_transformed)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # binary classification
            shap_values = shap_vals
            raw_imp     = np.abs(shap_vals).mean(axis=0)
        else:
            explainer = shap.LinearExplainer(
                model, X_transformed,
                feature_perturbation="interventional"
            )
            shap_vals   = explainer.shap_values(X_transformed)
            shap_values = shap_vals
            raw_imp     = np.abs(shap_vals).mean(axis=0)

    except Exception:
        # ── Fallback: model-native importance ─────────────
        raw_imp = None
        if hasattr(model, "feature_importances_"):
            raw_imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            raw_imp = np.abs(model.coef_).flatten()[:len(feature_names)]

        if raw_imp is None:
            return importances, shap_values

    # Ensure raw_imp matches feature_names length — trim or pad
    raw_imp = np.array(raw_imp).flatten()
    n_feats = len(feature_names)
    if len(raw_imp) > n_feats:
        raw_imp = raw_imp[:n_feats]
    elif len(raw_imp) < n_feats:
        raw_imp = np.pad(raw_imp, (0, n_feats - len(raw_imp)))

    # Normalize
    total    = raw_imp.sum()
    norm_imp = raw_imp / total if total > 0 else raw_imp

    # Direction from SHAP or coef — safely
    directions = ["mixed"] * n_feats
    try:
        if shap_values is not None:
            sv = np.array(shap_values)
            if sv.ndim == 2 and sv.shape[1] >= n_feats:
                for i in range(n_feats):
                    mean_shap = float(np.mean(sv[:, i]))
                    directions[i] = (
                        "positive" if mean_shap > 0.01
                        else "negative" if mean_shap < -0.01
                        else "mixed"
                    )
        elif hasattr(model, "coef_"):
            coef = np.array(model.coef_).flatten()
            for i in range(min(n_feats, len(coef))):
                directions[i] = "positive" if coef[i] > 0 else "negative"
    except Exception:
        directions = ["mixed"] * n_feats

    # Build FeatureImportance list — zip ensures equal length
    pairs  = list(zip(feature_names, norm_imp, directions))
    ranked = sorted(enumerate(pairs), key=lambda x: x[1][1], reverse=True)

    for rank, (i, (feat, imp, direction)) in enumerate(ranked, 1):
        pct = float(imp) * 100

        if pct > 30:
            explanation = "Dominant feature — drives {:.0f}% of predictions. {}influence.".format(
                pct, "Positive " if direction == "positive"
                else "Negative " if direction == "negative" else "Mixed ")
        elif pct > 10:
            explanation = "Important feature ({:.0f}% contribution). {}effect on target.".format(
                pct, "Increases " if direction == "positive"
                else "Decreases " if direction == "negative" else "Mixed ")
        elif pct > 3:
            explanation = "Moderate contribution ({:.0f}%). Minor {} effect.".format(
                pct, direction)
        else:
            explanation = "Low importance ({:.1f}%). Minimal impact on predictions.".format(pct)

        importances.append(FeatureImportance(
            feature=feat, importance=round(float(imp), 4),
            rank=rank, direction=direction, explanation=explanation,
        ))

    return importances[:20], shap_values


# ══════════════════════════════════════════════════════════