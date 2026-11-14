"""
api/analytics.py — stats, EDA, BI, story/insights.
Mirrors pages 4 (Business Insights), 6 (Deep EDA), 7 (Business Intel).
All heavy results are cached per dataset content hash. Every dataset is
scoped to the authenticated client (`owner`).
"""
from __future__ import annotations
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.services.auth import current_owner
from app.services.dataset_store import store
from app.services.serialize import to_jsonable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _df_or_404(owner: str, ds_id: str):
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return df


def _cached(owner: str, ds_id: str, key: str, compute):
    obj = store.cache_get(owner, ds_id, key)
    if obj is None:
        obj = compute()
        store.cache_set(owner, ds_id, key, obj)
    return obj


@router.get("/{ds_id}/stats")
def stats(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    from app.engines.stats_engine import analyze
    return to_jsonable(_cached(owner, ds_id, "stats", lambda: analyze(df)))


@router.get("/{ds_id}/eda")
def eda(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    from app.engines.eda_engine import run_eda
    return to_jsonable(_cached(owner, ds_id, "eda", lambda: run_eda(df)))


@router.get("/{ds_id}/bi")
def bi(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    from app.engines.bi_engine import run_bi
    return to_jsonable(_cached(owner, ds_id, "bi", lambda: run_bi(df)))


@router.get("/{ds_id}/story")
def story(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    from app.engines.story_engine import generate_story
    return to_jsonable(_cached(owner, ds_id, "story", lambda: generate_story(df)))


@router.get("/{ds_id}/insights")
def insights(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    from app.engines.insight_engine import generate_insights
    return {"insights": to_jsonable(_cached(
        owner, ds_id, "insights", lambda: generate_insights(df)))}


def _outlook(df) -> dict:
    """What changed and why, the next three months, and what to act on
    first. Each part is None (or empty) when the data cannot support it,
    with the reason, rather than failing the whole page."""
    from app.engines.change_drivers import explain_change_or_reason
    from app.engines.forecast import forecast
    from app.engines.priorities import build_priorities
    out = {"why": None, "forecast": None, "priorities": [], "notes": []}
    reason = "No change to explain: the calculation failed; see the server log."
    try:
        change, reason = explain_change_or_reason(df)
    except Exception:
        logger.warning("explain_change failed", exc_info=True)
        change = None
    if change:
        out["why"] = {"measure": change.measure, "period": change.period,
                      "prior_period": change.prior_period, "prior": change.prior,
                      "current": change.current, "pct": change.pct,
                      "sentences": change.sentences,
                      "segments": [vars(c) for c in (change.where.contributions[:6]
                                                     if change.where else [])],
                      "column": change.where.column if change.where else None}
    else:
        out["notes"].append(reason)
    try:
        fc = forecast(df)
    except Exception:
        logger.warning("forecast failed", exc_info=True)
        fc = None
    if fc:
        out["forecast"] = {"measure": fc.measure, "method": fc.method,
                           "aggregate": fc.aggregate, "error_pct": fc.error_pct,
                           "naive_error_pct": fc.naive_error_pct, "n_months": fc.n_months,
                           "points": [vars(p) for p in fc.points],
                           "history": [{"month": m, "value": v} for m, v in fc.history[-24:]],
                           "sentences": fc.sentences}
    else:
        out["notes"].append("No forecast: it needs at least 12 complete, unbroken months "
                            "of a measure to test a method on before trusting it.")
    try:
        out["priorities"] = [vars(p) for p in build_priorities(df, change=change)]
    except Exception:
        logger.warning("priorities failed", exc_info=True)
    return out


@router.get("/{ds_id}/outlook")
def outlook(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    return to_jsonable(_cached(owner, ds_id, "outlook", lambda: _outlook(df)))


@router.get("/{ds_id}/domain")
def domain(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    from app.engines.story_engine import detect_domain
    name, confidence = detect_domain(df)
    return {"domain": name, "confidence": to_jsonable(confidence)}
