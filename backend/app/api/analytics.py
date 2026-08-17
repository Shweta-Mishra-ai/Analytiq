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
    """Cached result, or one computed under the concurrency limit.

    The guard sits inside the cache miss on purpose: serving a result
    that already exists costs nothing and must never be refused because
    other people are computing theirs.
    """
    obj = store.cache_get(owner, ds_id, key)
    if obj is not None:
        return obj
    from app.services.load_guard import ANALYSIS, http_slot

    with http_slot(ANALYSIS):
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


@router.get("/{ds_id}/domain")
def domain(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    from app.engines.story_engine import detect_domain
    name, confidence = detect_domain(df)
    return {"domain": name, "confidence": to_jsonable(confidence)}
