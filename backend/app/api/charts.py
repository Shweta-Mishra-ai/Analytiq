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
    agg: str = "sum"                # sum|mean|count|median|min|max
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
    df = _df(owner, ds_id, body.filters)
    # The measures lead; the file's shape follows. This row used to open
    # with Rows / Columns / Missing % / Duplicates — a profiling strip,
    # not a KPI row — and then summed the first four numeric columns
    # whatever they were, so the headline figure on a sales dashboard was
    # "Σ order_id 405,450". Ranking is shared with the chart builders so
    # the cards and the tiles agree about what the business measure is.
    from app.engines.chart_engine import rank_measures

    cards: List[dict] = []
    for col in rank_measures(df)[:4]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s):
            cards.append({
                "label": f"Σ {col}", "value": float(s.sum()), "format": "num",
                "mean": float(s.mean()),
            })
    cards.append({"label": "Rows", "value": len(df), "format": "int"})
    cards.append({"label": "Columns", "value": df.shape[1], "format": "int"})
    cards.append({
        "label": "Missing %",
        "value": round(float(df.isna().mean().mean()) * 100, 1),
        "format": "pct"})
    cards.append({"label": "Duplicates", "value": int(df.duplicated().sum()),
                  "format": "int"})
    return {"kpis": to_jsonable(cards)}


@router.get("/{ds_id}/layout")
def layout(ds_id: str, owner: str = Depends(current_owner)):
    """The starting dashboard: which tiles, of what, and how big.

    The browser used to decide this from the raw field list, taking the
    first numeric column and the first categorical one. On a sales export
    that put `order_id` on the hero tile and never charted revenue at
    all. The same ranking that drives the PDF and the recommended charts
    now drives the default dashboard, so the three cannot disagree.

    Sizes are part of the answer: a dashboard where every tile is the
    same 6x5 rectangle has no reading order. The lead measure gets the
    wide tile.
    """
    from app.engines.chart_engine import _cat_columns, rank_measures

    df = _df(owner, ds_id)
    measures = rank_measures(df)
    cats = _cat_columns(df)
    dates = df.select_dtypes(include="datetime").columns.tolist()
    if not measures:
        return {"tiles": []}

    lead = measures[0]
    second = measures[1] if len(measures) > 1 else lead
    tiles: List[dict] = []

    def _add(type_, w, h, **spec):
        tiles.append({"id": "t{}".format(len(tiles) + 1), "type": type_,
                      "w": w, "h": h, "agg": "sum", **spec})

    if dates:
        _add("line", 12, 5, x=dates[0], y=lead,
             title="{} over time".format(lead))
    if cats:
        _add("bar", 7, 5, x=cats[0], y=lead,
             title="{} by {}".format(lead, cats[0]))
    if len(cats) > 1:
        _add("pie", 5, 5, x=cats[1], y=lead,
             title="{} share by {}".format(lead, cats[1]))
    if second != lead:
        _add("histogram", 5, 4, x=second,
             title="Distribution of {}".format(second))
    if len(measures) >= 3:
        _add("heatmap", 7, 4, title="Correlation matrix")

    # Lay the tiles out left to right, wrapping at twelve columns.
    # Grid position is `gx`/`gy`, not `x`/`y`: those already carry the
    # column names the tile plots, and one of the two would have won.
    col = row = row_h = 0
    for t in tiles:
        if col + t["w"] > 12:
            col, row = 0, row + row_h
            row_h = 0
        t["gx"], t["gy"] = col, row
        col += t["w"]
        row_h = max(row_h, t["h"])
    return {"tiles": tiles}


@router.post("/{ds_id}/recommend")
def recommend(ds_id: str, body: FiltersBody, owner: str = Depends(current_owner)):
    df = _df(owner, ds_id, body.filters)
    charts = chart_engine.recommend_charts(df)
    return {"charts": [{"title": t, "figure": _fig_json(f)} for t, f in charts]}


@router.post("/{ds_id}/build")
def build(ds_id: str, req: ChartRequest, owner: str = Depends(current_owner)):
    df = _df(owner, ds_id, req.filters)
    if df.empty:
        raise HTTPException(422, "No rows match the current filters")
    t = req.type

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
            grouped = _aggregate(df, req)
            if t == "bar":
                fig = chart_engine.make_bar(grouped, req.x, req.y, req.title)
            elif t == "pie":
                fig = chart_engine.make_pie(grouped, req.x, req.y, req.title)
            else:
                fig = chart_engine.make_line(grouped, req.x, req.y, req.title)
                if t == "area":
                    fig.update_traces(fill="tozeroy")
        elif t == "table":
            grouped = _aggregate(df, req) if (req.x and req.y) else df.head(req.top_n)
            from app.services.serialize import df_records
            return {"table": df_records(grouped, req.top_n)}
        else:
            raise HTTPException(422, f"Unknown chart type '{t}'")
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(422, f"Column not found: {e}")
    except Exception as e:
        raise HTTPException(500, f"Chart build failed: {e}")

    return {"figure": _fig_json(fig)}


def _aggregate(df: pd.DataFrame, req: ChartRequest) -> pd.DataFrame:
    x, y, agg = req.x, req.y, req.agg
    if x not in df.columns or y not in df.columns:
        raise KeyError(x if x not in df.columns else y)
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        # resample by sensible period for time axes
        tmp = df[[x, y]].dropna().set_index(x).sort_index()
        span_days = (tmp.index.max() - tmp.index.min()).days or 1
        rule = "D" if span_days <= 92 else ("W" if span_days <= 730 else "ME")
        agg_fn = "count" if agg == "count" else agg
        out = getattr(tmp[y].resample(rule), agg_fn)().reset_index()
        return out
    if agg == "count":
        out = df.groupby(x, dropna=True)[y].count().reset_index()
    else:
        out = df.groupby(x, dropna=True)[y].agg(agg).reset_index()
    return out.sort_values(y, ascending=False).head(req.top_n)
