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
    y2: Optional[str] = None
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

    The tiles themselves come from the domain. A fixed line-bar-pie-
    histogram-heatmap set went out for every file, which is a chart grid
    rather than a dashboard: a finance director opens a P&L expecting
    margin, cost structure and budget variance, and an HR director
    expects headcount and attrition by department. See
    engines/dashboard_spec, which also picks the mark from the question
    — composition as a donut, a trend as a line, a relationship as a
    scatter — instead of from whichever column type came to hand.
    """
    from app.engines.dashboard_spec import build_spec, layout_tiles
    from app.engines.domain_detect import detect

    df = _df(owner, ds_id)
    verdict = detect(df)
    tiles = layout_tiles(build_spec(df, verdict.domain))
    return {"tiles": tiles, "domain": verdict.domain,
            "domain_reason": verdict.reason}


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
        elif t == "comparison":
            if not (req.x and req.y and req.y2):
                raise HTTPException(422, "A comparison needs x and two "
                                         "measures")
            fig = chart_engine.make_comparison(df, req.x, req.y, req.y2,
                                               req.title)
        elif t in ("bar", "line", "area", "pie"):
            if not req.x or not req.y:
                raise HTTPException(422, "x and y are required")
            grouped = _aggregate(df, req)
            # A row count comes back in a column called `count`, because
            # a group cannot also be the measure. Passing `req.y` here
            # would look for `department` in a frame whose measure column
            # is `count`.
            y = "count" if req.x == req.y else req.y
            if t == "bar":
                fig = chart_engine.make_bar(grouped, req.x, y, req.title)
            elif t == "pie":
                fig = chart_engine.make_pie(grouped, req.x, y, req.title)
            else:
                fig = chart_engine.make_line(grouped, req.x, y, req.title)
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


class ExportBody(FiltersBody):
    title: str = "Dashboard"
    subtitle: str = ""
    # Whose name goes in the footer. The app never names the tool or the
    # model that produced a deliverable — the person delivering it signs
    # their own work.
    prepared_by: str = ""
    tiles: List[ChartRequest] = Field(default_factory=list)


@router.post("/{ds_id}/export/html")
def export_html(ds_id: str, body: ExportBody,
                owner: str = Depends(current_owner)):
    """The dashboard as one self-contained HTML file.

    A PDF is finished: the reader sees the cuts you chose, and asking for
    the same view on one region means coming back to you. Half of what a
    client commissions analysis for is the ability to poke at it. This
    opens from a file:// path with no server and no network, so it can be
    attached to an email.

    The tiles are built through the same code path the screen uses — if
    the caller sends none, the server's own default layout is used — so
    the exported file cannot show different figures from the app.
    """
    from fastapi.responses import HTMLResponse

    from app.engines.dashboard_export import build_dashboard_html

    df = _df(owner, ds_id, body.filters)
    if df.empty:
        raise HTTPException(422, "No rows match the current filters")

    specs = body.tiles
    widths: List[int] = [6] * len(specs)
    questions: List[str] = [""] * len(specs)
    if not specs:
        served = layout(ds_id, owner=owner)["tiles"]
        specs = [ChartRequest(type=t["type"], x=t.get("x"), y=t.get("y"),
                              y2=t.get("y2"), agg=t.get("agg", "sum"),
                              title=t.get("title", ""))
                 for t in served]
        widths = [t.get("w", 6) for t in served]
        questions = [t.get("question", "") for t in served]

    tiles: List[dict] = []
    # Light, because this is the artefact that gets read in a meeting,
    # printed, and pasted into a deck. The interface stays dark.
    # The domain's own palette, so a workforce review and a P&L are not
    # indistinguishable across a desk.
    from app.engines.domain_detect import detect as _detect_domain
    domain = _detect_domain(_df(owner, ds_id)).domain
    with chart_engine.use_theme(chart_engine.theme_for(domain)):
        for spec, width, question in zip(specs, widths, questions):
            try:
                built = build(ds_id, spec, owner=owner)
            except HTTPException:
                # One tile that cannot be built is not a reason to refuse
                # the document; the others still say something.
                logger.info("export: skipping %s tile", spec.type,
                            exc_info=True)
                continue
            if "figure" not in built:
                continue
            tiles.append({"title": spec.title or spec.type, "w": width,
                          "question": question, "figure": built["figure"]})

    if not tiles:
        raise HTTPException(422, "Nothing in this dataset charts as a "
                                 "business measure, so there is no "
                                 "dashboard to export")

    page = build_dashboard_html(
        df, tiles, kpis(ds_id, FiltersBody(filters=body.filters),
                        owner=owner)["kpis"],
        title=body.title, subtitle=body.subtitle,
        prepared_by=body.prepared_by, domain=domain)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-"
                   for ch in (body.title or "dashboard")).strip("-") or "dashboard"
    return HTMLResponse(page, headers={
        "Content-Disposition": 'attachment; filename="{}.html"'.format(safe)})


def _measure_series(df: pd.DataFrame, col: str) -> pd.Series:
    """The column as something that can be averaged.

    A rate tile — "attrition rate by department" — asks for the mean of a
    column an HRIS exports as "Yes"/"No", and pandas raises on that.
    Normalising a two-valued flag to 0/1 is what makes the mean the rate
    the tile is named after; anything else is left alone and will fail
    loudly rather than produce a number nobody can trace.
    """
    if pd.api.types.is_numeric_dtype(df[col]):
        return df[col]
    if pd.api.types.is_bool_dtype(df[col]):
        return df[col].astype(float)
    from app.engines.column_roles import left_mask

    found = left_mask(df[[col]])
    if found:
        return found[1].astype(float)
    coerced = pd.to_numeric(df[col], errors="coerce")
    if coerced.notna().mean() >= 0.8:
        return coerced
    return df[col]


def _aggregate(df: pd.DataFrame, req: ChartRequest) -> pd.DataFrame:
    x, y, agg = req.x, req.y, req.agg
    if x not in df.columns or y not in df.columns:
        raise KeyError(x if x not in df.columns else y)

    # Counting rows per group is `x` grouped by itself. `groupby(x)[x]`
    # then tries to reset an index column that already exists, so a
    # headcount tile raised "cannot insert department, already exists".
    if x == y:
        if agg != "count":
            raise HTTPException(422, "A column cannot be aggregated against "
                                     "itself except as a count")
        out = df.groupby(x, dropna=True).size().reset_index(name="count")
        return out.sort_values("count", ascending=False).head(req.top_n)

    if agg in ("mean", "sum") and not pd.api.types.is_numeric_dtype(df[y]):
        df = df.assign(**{y: _measure_series(df, y)})
        if not pd.api.types.is_numeric_dtype(df[y]):
            raise HTTPException(
                422, "'{}' cannot be averaged — it is not a measure or a "
                     "two-valued flag".format(y))

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
