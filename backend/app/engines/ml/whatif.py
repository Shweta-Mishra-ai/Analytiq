"""
engines/ml/whatif.py — scoring one hypothetical row.

The interactive half: given a trained model and a set of inputs a user
typed, what does it predict. Kept separate from training because it runs
on a different schedule and must never retrain anything.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

from typing import Dict

from app.engines.ml.results import MLReport


#  WHAT-IF PREDICTION
# ══════════════════════════════════════════════════════════

def predict_what_if(
    ml_report: MLReport,
    input_values: Dict[str, float],
) -> Dict:
    """
    Make a single prediction from user-supplied input values.
    Returns prediction + confidence info.
    """
    if ml_report.best_model is None or ml_report.best_model.model is None:
        return {"error": "No trained model available."}

    try:
        # A caller supplies categories the way a person says them —
        # {"Department": "Sales"} — not the one-hot columns the model was
        # fitted on. Expand them here so the UI never has to know the
        # encoding.
        values = dict(input_values)
        mapping = getattr(ml_report, "encoding_map", {}) or {}
        if mapping:
            sources = {src for src, _level in mapping.values()}
            for dummy, (src, level) in mapping.items():
                if src in values:
                    values[dummy] = 1.0 if str(values[src]) == level else 0.0
            for src in sources:
                values.pop(src, None)
        missing = [c for c in ml_report.feature_cols if c not in values]
        for c in missing:
            values[c] = 0.0
        X_input = pd.DataFrame([values])[ml_report.feature_cols]
        pipe    = ml_report.best_model.model
        pred    = pipe.predict(X_input)[0]

        result = {"prediction": float(pred), "task": ml_report.task}

        if ml_report.task == "classification":
            # Decode label
            if ml_report.target_encoder is not None:
                try:
                    pred_label = ml_report.target_encoder.inverse_transform([int(pred)])[0]
                    result["prediction_label"] = str(pred_label)
                except Exception:
                    result["prediction_label"] = str(pred)

            # Probability
            try:
                proba = pipe.predict_proba(X_input)[0]
                result["probabilities"] = {
                    str(c): round(float(p), 4)
                    for c, p in zip(pipe.classes_, proba)
                }
                result["confidence"] = round(float(max(proba)) * 100, 1)
            except Exception:
                result["confidence"] = None
        else:
            # Regression confidence interval (naive ± 1 RMSE)
            rmse = ml_report.best_model.rmse or 0
            result["lower"] = round(float(pred) - rmse, 4)
            result["upper"] = round(float(pred) + rmse, 4)
            result["confidence_note"] = "±{:.2f} (1x RMSE)".format(rmse)

        return result

    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════