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

# How far past the observed edge an input may sit before the answer
# stops being a prediction. A tenth of the observed span is a step
# beyond what was seen; several times it is invention. The scenario
# engine uses the same tolerance, deliberately — a user should not get
# two different answers to "is this too far?" from two screens.
OUTSIDE_RANGE_TOLERANCE = 0.10


def _out_of_range(ml_report: MLReport, input_values: Dict) -> list:
    """Inputs that sit outside the span the model was fitted on.

    Reported in the caller's own words — the column as they named it and
    the range as the data holds it — so the note can be shown verbatim.
    """
    ranges = getattr(ml_report, "feature_ranges", None) or {}
    if not ranges:
        return []

    from app.engines.present import label as _label, num as _num

    notes = []
    for feature, value in input_values.items():
        bounds = ranges.get(feature)
        if not bounds or value is None:
            continue
        try:
            val = float(value)
            lo, hi = float(bounds["min"]), float(bounds["max"])
        except (TypeError, ValueError):
            continue
        margin = (hi - lo) * OUTSIDE_RANGE_TOLERANCE
        if lo - margin <= val <= hi + margin:
            continue
        notes.append("{} of {} is outside the {} to {} the data covers"
                     .format(_label(feature), _num(val), _num(lo), _num(hi)))
    return notes

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

        # Whether the question is one the model has any standing to
        # answer. A fitted model returns a confident number for ANY
        # input: ask it about a salary of 5,000,000 when nothing above
        # 120,000 was ever seen and it answers, with a confidence
        # interval, and nothing on the screen says the figure is an
        # extrapolation. This is the same defect the scenario engine had
        # — a projection past the edge of the evidence, presented as a
        # finding — and the interactive predictor is where a user is
        # most likely to walk into it, because typing a big number is
        # the obvious thing to try.
        #
        # The prediction is still returned: the caller shows it with the
        # caveat attached rather than refusing to answer.
        out_of_range = _out_of_range(ml_report, input_values)
        result["out_of_range"] = out_of_range
        result["within_training_range"] = not out_of_range
        if out_of_range:
            result["range_note"] = (
                "This prediction is outside what the model has seen: "
                + "; ".join(out_of_range[:3])
                + ". The figure above is the fitted model extended past "
                  "its evidence, not something the data supports."
            )

        return result

    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════