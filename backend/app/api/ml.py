"""
api/ml.py — ML pipeline endpoints (mirrors page 5, ML Predictions).
Every dataset is scoped to the authenticated client (`owner`).
"""
from __future__ import annotations
import logging

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.services.auth import current_owner
from app.services.dataset_store import store
from app.services.serialize import to_jsonable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ml"])


class TrainRequest(BaseModel):
    target: str
    max_rows: int = 50_000


class WhatIfRequest(BaseModel):
    target: str
    inputs: Dict[str, object] = Field(default_factory=dict)


def _df_or_404(owner: str, ds_id: str):
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return df


def _serialize_report(report) -> dict:
    """MLReport → JSON, excluding fitted models/arrays/encoders."""
    def _model(m):
        if m is None:
            return None
        d = {k: to_jsonable(v) for k, v in vars(m).items() if k != "model"}
        return d

    return {
        "task": report.task,
        "target_col": report.target_col,
        "feature_cols": report.feature_cols,
        "n_rows_used": report.n_rows_used,
        "n_features": report.n_features,
        "class_balance": to_jsonable(report.class_balance),
        "models": [_model(m) for m in report.models],
        "best_model": _model(report.best_model),
        "feature_importance": to_jsonable(report.feature_importance),
        "warnings": to_jsonable(report.warnings),
        "insights": to_jsonable(report.insights),
        # Whether the model beat the obvious guess, and any field that
        # knows the answer in advance. Both were computed and then
        # dropped here, so the UI could not show either.
        "verdict": to_jsonable(report.verdict),
        "leakage": to_jsonable(report.leakage),
    }


@router.get("/{ds_id}/targets")
def targets(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    from app.engines.ml_engine import suggest_targets
    return {"targets": to_jsonable(suggest_targets(df))}


@router.post("/{ds_id}/train")
def train(ds_id: str, req: TrainRequest, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    if req.target not in df.columns:
        raise HTTPException(422, f"Target '{req.target}' not in dataset")
    from app.engines.ml_engine import run_ml_pipeline
    try:
        report = run_ml_pipeline(df, req.target)
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")
    store.cache_set(owner, ds_id, f"ml_{req.target}", report)
    store.cache_set(owner, ds_id, "ml_last", report)
    return _serialize_report(report)


@router.get("/{ds_id}/report")
def last_report(ds_id: str, target: Optional[str] = None,
                 owner: str = Depends(current_owner)):
    _df_or_404(owner, ds_id)
    key = f"ml_{target}" if target else "ml_last"
    report = store.cache_get(owner, ds_id, key)
    if report is None:
        raise HTTPException(404, "No trained model yet — call /train first")
    return _serialize_report(report)


@router.post("/{ds_id}/what-if")
def what_if(ds_id: str, req: WhatIfRequest, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    report = store.cache_get(owner, ds_id, f"ml_{req.target}") \
        or store.cache_get(owner, ds_id, "ml_last")
    if report is None:
        raise HTTPException(404, "No trained model yet — call /train first")
    from app.engines.ml_engine import predict_what_if
    try:
        result = predict_what_if(report, req.inputs)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    return to_jsonable(result)
