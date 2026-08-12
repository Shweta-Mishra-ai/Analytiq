"""
api/advanced_analytics.py — RFM segmentation, A/B testing, and survival
(time-to-event) analysis. Ported from dataforge-ai's stronger analytics
engines to close the gap with Analytiq's existing BI/EDA/stats suite.
Every dataset is scoped to the authenticated client (`owner`).
"""
from __future__ import annotations
import logging

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.auth import current_owner
from app.services.dataset_store import store
from app.services.serialize import to_jsonable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["advanced-analytics"])


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


# ══════════════════════════════════════════════════════════
#  RFM segmentation
# ══════════════════════════════════════════════════════════

@router.get("/{ds_id}/rfm/columns")
def rfm_columns(ds_id: str, owner: str = Depends(current_owner)):
    """Suggests which columns look like customer_id / date / monetary,
    for a frontend column picker. Returns detected=null if this dataset
    doesn't look like transaction-level data at all."""
    df = _df_or_404(owner, ds_id)
    from app.engines.rfm_engine import detect_rfm_columns
    cols = detect_rfm_columns(df)
    return {"detected": to_jsonable(cols) if cols else None}


@router.get("/{ds_id}/rfm")
def rfm(
    ds_id: str,
    customer_col: Optional[str] = None,
    date_col: Optional[str] = None,
    monetary_col: Optional[str] = None,
    quantity_col: Optional[str] = None,
    price_col: Optional[str] = None,
    owner: str = Depends(current_owner),
):
    df = _df_or_404(owner, ds_id)
    from app.engines.rfm_engine import RFMColumns, run_rfm

    columns = None
    if customer_col or date_col:
        if not (customer_col and date_col):
            raise HTTPException(422, "customer_col and date_col must both be given together")
        for c in (customer_col, date_col, monetary_col, quantity_col, price_col):
            if c and c not in df.columns:
                raise HTTPException(422, f"Column '{c}' not in dataset")
        columns = RFMColumns(customer_id=customer_col, date_col=date_col,
                              monetary_col=monetary_col, quantity_col=quantity_col,
                              price_col=price_col)

    cache_key = f"rfm_{customer_col}_{date_col}_{monetary_col}" if columns else "rfm_auto"
    try:
        report = _cached(owner, ds_id, cache_key, lambda: run_rfm(df, columns))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return to_jsonable(report)


# ══════════════════════════════════════════════════════════
#  A/B testing
# ══════════════════════════════════════════════════════════

class ABTestRequest(BaseModel):
    group_col: str
    metric_col: str
    group_a_value: Optional[str] = None
    group_b_value: Optional[str] = None
    # For a binary/conversion metric_col: which value counts as a
    # "conversion" (e.g. "Yes", "1", "Converted"). Auto-detected from a
    # common-token list when omitted.
    success_value: Optional[str] = None
    confidence_level: float = 0.95


_POSITIVE_TOKENS = {"1", "1.0", "true", "yes", "y", "converted", "success", "won", "purchased"}


def _is_binary_series(s: pd.Series) -> bool:
    return 0 < s.dropna().nunique() <= 2


def _measure_columns(df: pd.DataFrame, numeric_only: bool = False) -> list[str]:
    """Columns offerable as a metric/duration/driver — identifiers excluded.

    Without this an ID column is a valid-looking pick and, being first in
    most frames, becomes the default: the survival page opened on
    "median survival time: 108.0 (customer_id units)" and the what-if
    simulator defaulted to projecting a change in customer_id. Both are
    meaningless, and neither surfaces as an error.
    """
    from app.engines.domains.base import is_id_column
    out = []
    for c in df.columns:
        if numeric_only and not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if is_id_column(c, df[c]):
            continue
        out.append(c)
    return out


@router.get("/{ds_id}/ab-test/fields")
def ab_test_fields(ds_id: str, owner: str = Depends(current_owner)):
    """Candidate group (2+ distinct values) and metric columns, for a
    frontend column picker."""
    df = _df_or_404(owner, ds_id)
    measures = set(_measure_columns(df))
    group_candidates = [c for c in df.columns
                         if c in measures and 2 <= df[c].nunique() <= 20]
    metric_candidates = [c for c in df.columns
                          if c in measures
                          and (pd.api.types.is_numeric_dtype(df[c])
                               or _is_binary_series(df[c]))]
    return {"group_columns": group_candidates, "metric_columns": metric_candidates}


@router.post("/{ds_id}/ab-test")
def ab_test(ds_id: str, req: ABTestRequest, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    from app.engines.ab_test_engine import run_conversion_test, run_continuous_test

    if req.group_col not in df.columns:
        raise HTTPException(422, f"Column '{req.group_col}' not in dataset")
    if req.metric_col not in df.columns:
        raise HTTPException(422, f"Column '{req.metric_col}' not in dataset")

    counts = df[req.group_col].dropna().astype(str).value_counts()
    if req.group_a_value and req.group_b_value:
        group_a_value, group_b_value = req.group_a_value, req.group_b_value
    else:
        top2 = counts.head(2).index.tolist()
        if len(top2) < 2:
            raise HTTPException(422, f"'{req.group_col}' needs at least 2 distinct groups")
        group_a_value, group_b_value = top2[0], top2[1]

    sub_a = df[df[req.group_col].astype(str) == str(group_a_value)]
    sub_b = df[df[req.group_col].astype(str) == str(group_b_value)]
    if sub_a.empty or sub_b.empty:
        raise HTTPException(422, "One of the selected groups has no rows")

    try:
        if _is_binary_series(df[req.metric_col]):
            def _positive_count(sub: pd.DataFrame) -> int:
                s = sub[req.metric_col].dropna().astype(str).str.strip().str.lower()
                if req.success_value:
                    return int((s == str(req.success_value).strip().lower()).sum())
                return int(s.isin(_POSITIVE_TOKENS).sum())

            result = run_conversion_test(
                conversions_a=_positive_count(sub_a), n_a=len(sub_a),
                conversions_b=_positive_count(sub_b), n_b=len(sub_b),
                variant_a_name=str(group_a_value), variant_b_name=str(group_b_value),
                metric_name=req.metric_col, confidence_level=req.confidence_level,
            )
        else:
            result = run_continuous_test(
                values_a=sub_a[req.metric_col], values_b=sub_b[req.metric_col],
                variant_a_name=str(group_a_value), variant_b_name=str(group_b_value),
                metric_name=req.metric_col, confidence_level=req.confidence_level,
            )
    except ValueError as e:
        raise HTTPException(422, str(e))

    return to_jsonable(result)


# ══════════════════════════════════════════════════════════
#  Survival analysis
# ══════════════════════════════════════════════════════════

class SurvivalRequest(BaseModel):
    duration_col: str
    event_col: str
    group_col: Optional[str] = None


@router.get("/{ds_id}/survival/fields")
def survival_fields(ds_id: str, owner: str = Depends(current_owner)):
    """Candidate duration (numeric), event (binary-ish), and group
    (low-cardinality categorical) columns, for a frontend column picker."""
    df = _df_or_404(owner, ds_id)
    measures = set(_measure_columns(df))
    duration_candidates = [c for c in df.columns
                            if c in measures and pd.api.types.is_numeric_dtype(df[c])
                            and not _is_binary_series(df[c])]
    event_candidates = [c for c in df.columns if _is_binary_series(df[c])]
    group_candidates = [c for c in df.columns
                         if c in measures and 2 <= df[c].nunique() <= 20]
    return {"duration_columns": duration_candidates, "event_columns": event_candidates,
            "group_columns": group_candidates}


@router.post("/{ds_id}/survival")
def survival(ds_id: str, req: SurvivalRequest, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    from app.engines.survival_engine import run_survival_analysis

    for c in (req.duration_col, req.event_col, req.group_col):
        if c and c not in df.columns:
            raise HTTPException(422, f"Column '{c}' not in dataset")

    cache_key = f"survival_{req.duration_col}_{req.event_col}_{req.group_col}"
    try:
        report = _cached(
            owner, ds_id, cache_key,
            lambda: run_survival_analysis(df, req.duration_col, req.event_col, req.group_col),
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    return to_jsonable(report)


# ══════════════════════════════════════════════════════════
#  Scenario / what-if projection
# ══════════════════════════════════════════════════════════

class ScenarioRequest(BaseModel):
    driver_col: str
    target_col: str
    change_pct: float = 10.0


@router.get("/{ds_id}/scenario/fields")
def scenario_fields(ds_id: str, owner: str = Depends(current_owner)):
    """Numeric columns that are real measures — identifiers excluded, since
    projecting "what if customer_id rose 10%" is meaningless."""
    df = _df_or_404(owner, ds_id)
    return {"numeric_columns": _measure_columns(df, numeric_only=True)}


@router.post("/{ds_id}/scenario")
def scenario(ds_id: str, req: ScenarioRequest, owner: str = Depends(current_owner)):
    """"What if <driver> moved by X%?" — projects the effect on a target
    metric from the historical linear relationship. The response carries
    an explicit `reliable` flag and a causal caveat; a weak relationship
    still returns a result so the caller can say "not reliable enough"
    rather than showing nothing."""
    df = _df_or_404(owner, ds_id)
    from app.engines.bi_engine import analyze_scenario

    for c in (req.driver_col, req.target_col):
        if c not in df.columns:
            raise HTTPException(422, f"Column '{c}' not in dataset")

    result = analyze_scenario(df, req.driver_col, req.target_col, req.change_pct)
    if result is None:
        raise HTTPException(
            422,
            "Cannot project this scenario — both columns must be numeric, "
            "distinct, and share at least 10 rows with values in both.")
    return to_jsonable(result)


# ══════════════════════════════════════════════════════════
#  Internal benchmarking (metric vs its own top quartile)
# ══════════════════════════════════════════════════════════

@router.get("/{ds_id}/benchmarks")
def benchmarks(ds_id: str, max_metrics: int = 5,
                owner: str = Depends(current_owner)):
    """Benchmarks each directional numeric metric against its own
    top-quartile performers — no external/industry assumptions."""
    df = _df_or_404(owner, ds_id)
    from app.engines.benchmarking import compute_benchmarks
    results = _cached(owner, ds_id, f"benchmarks_{max_metrics}",
                       lambda: compute_benchmarks(df, max_metrics=max_metrics))
    return {"benchmarks": to_jsonable(results)}


@router.get("/{ds_id}/industry-benchmarks")
def industry_benchmarks(ds_id: str, owner: str = Depends(current_owner)):
    """Looks up published industry reference ranges for recognised metric
    names in the detected domain. Only returns columns with a known
    reference range — never invents one."""
    df = _df_or_404(owner, ds_id)
    from app.engines.industry_benchmarks import lookup_benchmark, format_benchmark_context
    from app.engines.story_engine import detect_domain

    domain, _confidence = detect_domain(df)
    out = []
    for col in df.columns:
        bm = lookup_benchmark(domain, col)
        if bm is not None:
            out.append({"column": col, "benchmark": to_jsonable(bm),
                        "context": format_benchmark_context(bm)})
    return {"domain": domain, "benchmarks": out}


# ══════════════════════════════════════════════════════════
#  Predictive drivers (what predicts churn / attrition)
# ══════════════════════════════════════════════════════════

@router.get("/{ds_id}/drivers")
def drivers(ds_id: str, target: Optional[str] = None,
            owner: str = Depends(current_owner)):
    """Ranks which factors most predict a binary outcome (attrition,
    churn, default), with model quality and the highest-risk profile.
    Auto-detects a suitable target column when `target` is omitted."""
    df = _df_or_404(owner, ds_id)
    from app.engines.predictive import compute_drivers, find_binary_target

    target_col = target or find_binary_target(df)
    if not target_col:
        raise HTTPException(
            422,
            "No binary outcome column detected. Pass ?target=<column> "
            "naming a two-value column such as churn/attrition/left.")
    if target_col not in df.columns:
        raise HTTPException(422, f"Column '{target_col}' not in dataset")

    result = _cached(owner, ds_id, f"drivers_{target_col}",
                      lambda: compute_drivers(df, target_col))
    if result is None:
        raise HTTPException(
            422,
            f"Could not model '{target_col}' — it must be effectively binary "
            "with both classes present and enough rows to train on.")
    return to_jsonable(result)


# ══════════════════════════════════════════════════════════
#  Dataset comparison (period over period)
# ══════════════════════════════════════════════════════════

class ComparisonRequest(BaseModel):
    other_dataset_id: str
    label_a: str = "Period A"
    label_b: str = "Period B"


@router.post("/{ds_id}/compare")
def compare(ds_id: str, req: ComparisonRequest,
            owner: str = Depends(current_owner)):
    """Diff-style comparison of two of the caller's own datasets — schema
    drift, row-count change, and per-column distribution shifts with
    significance testing. Both datasets are resolved under `owner`, so a
    client can never compare against another client's data."""
    df_a = _df_or_404(owner, ds_id)
    df_b = store.get_df(owner, req.other_dataset_id)
    if df_b is None:
        raise HTTPException(404, "Comparison dataset not found")

    from app.engines.comparison_engine import run_comparison
    try:
        report = run_comparison(df_a, df_b, label_a=req.label_a, label_b=req.label_b)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e))
    return to_jsonable(report)
