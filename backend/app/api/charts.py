"""
api/charts.py — Power BI-style dashboard endpoints.
Every endpoint accepts a `filters` list so all tiles cross-filter together.
Charts are returned as Plotly JSON (rendered by react-plotly on the frontend).
"""
from __future__ import annotations
import logging

import json
from typing import List, Optional

import pandas as pd
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.engines import chart_engine
from app.services.auth import current_owner
from app.services.dataset_store import store
from app.services.filters import apply_filters, field_catalog
from app.services.serialize import to_jsonable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/charts", tags=["charts"])


class FilterSpec(BaseModel):
    column: str
    op: str = "eq"
    value: object = None


class ChartRequest(BaseModel):
    type: str                       # bar|line|area|scatter|histogram|pie|heatmap|table
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    # "auto" asks the server which aggregation the metric deserves —
    # summing an age gives 25,000 years and answers nothing.
    agg: str = "sum"                # auto|sum|mean|count|median|min|max
    nbins: int = 30
    top_n: int = 20
    title: str = ""
    filters: List[FilterSpec] = Field(default_factory=list)


class FiltersBody(BaseModel):
    filters: List[FilterSpec] = Field(default_factory=list)


def _df(owner: str, ds_id: str, filters: List[FilterSpec] | None = None) -> pd.DataFrame:
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return apply_filters(df, [f.model_dump() for f in (filters or [])])


def _fig_json(fig) -> dict:
    return json.loads(pio.to_json(fig))


@router.get("/{ds_id}/fields")
def fields(ds_id: str, owner: str = Depends(current_owner)):
    df = _df(owner, ds_id)
    return {"fields": field_catalog(df)}


@router.post("/{ds_id}/kpis")
def kpis(ds_id: str, body: FiltersBody, owner: str = Depends(current_owner)):
    """The headline numbers for this dataset, in its domain.

    This used to return four data-quality counts followed by the sum of
    the first four numeric columns — on an HR extract, a total of employee
    ID numbers and a total of ages. KPIs are now resolved from the
    domain's own specs with the aggregation each metric deserves, and the
    file-shape figures are returned separately because they answer a
    different kind of question.
    """
    df = _df(owner, ds_id, body.filters)
    try:
        from app.engines.story_engine import detect_domain
        domain, confidence = detect_domain(df)
    except Exception:
        logger.warning("domain detection failed for KPIs", exc_info=True)
        domain, confidence = "general", 0.0

    from app.engines.kpi_engine import compute_kpis, data_quality_cards
    cards = compute_kpis(df, domain)
    quality = data_quality_cards(df)

    return {
        "kpis": to_jsonable([c.as_dict() for c in cards]),
        "data_quality": to_jsonable([c.as_dict() for c in quality]),
        "domain": domain,
        "domain_confidence": confidence,
    }


@router.post("/{ds_id}/recommend")
def recommend(ds_id: str, body: FiltersBody, owner: str = Depends(current_owner)):
    df = _df(owner, ds_id, body.filters)
    # Pass the detected domain so chart selection can prefer the metrics
    # this kind of business actually leads with, rather than whichever
    # numeric column happens to come first.
    try:
        from app.engines.story_engine import detect_domain
        domain, _ = detect_domain(df)
    except Exception:
        logger.warning("domain detection failed for chart recommendation",
                       exc_info=True)
        domain = "general"
    charts = chart_engine.recommend_charts(df, domain)
    return {"charts": [{"title": t, "figure": _fig_json(f)} for t, f in charts]}


@router.post("/{ds_id}/recommend-tiles")
def recommend_tiles(ds_id: str, body: FiltersBody,
                    owner: str = Depends(current_owner)):
    """The dashboard's opening tiles, as specs it can re-render.

    The page used to choose these in the browser from column order, and
    the result was a finance dashboard opening on "budget by account" —
    a budget is assigned per account, so the bars restate the file's own
    construction — and a healthcare one on average age rather than
    length of stay. Column roles are known here; column order is all the
    browser has.
    """
    df = _df(owner, ds_id, body.filters)
    try:
        from app.engines.story_engine import detect_domain
        domain, _ = detect_domain(df)
    except Exception:
        logger.warning("domain detection failed for tile recommendation",
                       exc_info=True)
        domain = "general"
    return {"tiles": chart_engine.recommend_tiles(df, domain),
            "domain": domain}


@router.post("/{ds_id}/build")
def build(ds_id: str, req: ChartRequest, owner: str = Depends(current_owner)):
    df = _df(owner, ds_id, req.filters)
    if df.empty:
        raise HTTPException(422, "No rows match the current filters")
    t = req.type

    # A column the frame does not have is a bad request, not a server
    # fault. Without this check plotly raised deep inside make_figure and
    # the client got a 500 with no usable message — which is what a
    # dashboard still holding the previous dataset's tiles produced when
    # the user switched files.
    for role, name in (("x", req.x), ("y", req.y), ("color", req.color)):
        if name and name not in df.columns:
            raise HTTPException(
                422, "Column '{}' (used as {}) is not in this dataset."
                     .format(name, role))

    try:
        if t == "histogram":
            fig = chart_engine.make_histogram(df, req.x or req.y, req.nbins, req.title)
        elif t == "heatmap":
            if df.select_dtypes(include="number").shape[1] < 2:
                raise HTTPException(422, "Need 2+ numeric columns for a heatmap")
            fig = chart_engine.make_heatmap(df)
        elif t == "scatter":
            fig = chart_engine.make_scatter(df, req.x, req.y, req.color, req.title)
        elif t in ("bar", "line", "area", "pie"):
            if not req.x or not req.y:
                raise HTTPException(422, "x and y are required")
            grouped, y_col = _aggregate(df, req)
            if t == "bar":
                fig = chart_engine.make_bar(grouped, req.x, y_col, req.title)
            elif t == "pie":
                fig = chart_engine.make_pie(grouped, req.x, y_col, req.title)
            else:
                fig = chart_engine.make_line(grouped, req.x, y_col, req.title)
                if t == "area":
                    fig.update_traces(fill="tozeroy")
        elif t == "table":
            grouped = (_aggregate(df, req)[0] if (req.x and req.y)
                       else df.head(req.top_n))
            from app.services.serialize import df_records
            return {"table": df_records(grouped, req.top_n)}
        else:
            raise HTTPException(422, f"Unknown chart type '{t}'")
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(422, f"Column not found: {e}")
    except Exception as e:
        # A 500 that leaves nothing in the log cannot be diagnosed. This
        # one reached a user's dashboard and the server had recorded only
        # the status line, so the cause had to be guessed at from the
        # outside. The traceback goes to the log; the client still gets
        # the short message.
        logger.exception("chart build failed: type=%s x=%r y=%r agg=%r",
                         t, req.x, req.y, req.agg)
        raise HTTPException(500, f"Chart build failed: {e}")

    return {"figure": _fig_json(fig)}


def _aggregate(df: pd.DataFrame, req: ChartRequest):
    """(grouped frame, the y column to plot).

    The y column is returned because a yes/no measure is converted to a
    percentage under a new name, and the caller has to plot the name
    that actually exists in the frame it gets back.
    """
    x, y, agg = req.x, req.y, req.agg
    if agg == "auto" and y:
        # One rule for how a metric is aggregated, shared with the report
        # generator, so a dashboard tile and the same chart in the PDF
        # never disagree.
        from app.engines.chart_exporter import _agg_for_metric
        agg, _is_score = _agg_for_metric(str(y))
    if x not in df.columns or y not in df.columns:
        raise KeyError(x if x not in df.columns else y)
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        # resample by sensible period for time axes
        tmp = df[[x, y]].dropna().set_index(x).sort_index()
        span_days = (tmp.index.max() - tmp.index.min()).days or 1
        rule = "D" if span_days <= 92 else ("W" if span_days <= 730 else "ME")
        agg_fn = "count" if agg == "count" else agg
        out = getattr(tmp[y].resample(rule), agg_fn)().reset_index()
        return out, y
    if agg == "count":
        out = df.groupby(x, dropna=True)[y].count().reset_index()
        return out.sort_values(y, ascending=False).head(req.top_n), y
    else:
        frame = df
        if agg in ("mean", "sum") and not pd.api.types.is_numeric_dtype(df[y]):
            # A rate chart over a Yes/No column is the most useful tile
            # a dashboard can carry — "attrition rate by department" —
            # and pandas cannot mean a string, so the whole tile failed
            # with "dtype 'str' does not support operation 'mean'".
            # Mapping the affirmative side to 1 makes the average the
            # rate, which is what the title claims it is.
            from app.engines.domains._common import binary_mask
            mask = binary_mask(df[y])
            if mask is None:
                raise HTTPException(
                    422,
                    "'{}' is text, so it cannot be averaged. Pick a numeric "
                    "column, or a yes/no column to chart as a rate."
                    .format(y))
            # Two things a first draft of this got wrong.
            #
            # Scale follows the aggregation. Averaging 0/100 gives the
            # rate; SUMMING percentages does not — three affirmative
            # rows would read as 300%. A sum of a flag is a count of it,
            # so summing keeps the 0/1 form.
            #
            # And a missing value is not a "no". Filling the gaps with
            # False counted every blank row against the rate; left as
            # NaN, groupby skips them, which is what "rate among the
            # records that recorded it" means.
            as_rate = agg == "mean"
            values = mask.reindex(df.index).astype("float64")
            if as_rate:
                values = values * 100.0
            rate_name = ("{} rate (%)".format(y) if as_rate
                         else "{} count".format(y))
            frame = df.assign(**{rate_name: values})
            y = rate_name
        out = frame.groupby(x, dropna=True)[y].agg(agg).reset_index()
    return out.sort_values(y, ascending=False).head(req.top_n), y
