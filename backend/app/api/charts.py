"""
api/charts.py — Power BI-style dashboard endpoints.
Every endpoint accepts a `filters` list so all tiles cross-filter together.
Charts are returned as Plotly JSON (rendered by react-plotly on the frontend).
"""
from __future__ import annotations
import logging

import json
from typing import List, Literal, Optional

import pandas as pd
import plotly.io as pio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.engines import chart_engine
from app.services.auth import current_owner
from app.services.dataset_store import store
from app.services.filters import FilterError, apply_filters, field_catalog
from app.services.serialize import to_jsonable

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/charts", tags=["charts"])


class FilterSpec(BaseModel):
    column: str
    op: str = "eq"
    value: object = None


class ChartRequest(BaseModel):
    type: str                       # bar|line|area|scatter|histogram|pie|heatmap|rate|table
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    # A closed set: the value is looked up as a pandas method by name, so an
    # open string let a request call arbitrary resampler/groupby attributes.
    agg: Literal["sum", "mean", "count", "median", "min", "max"] = "sum"
    nbins: int = Field(30, ge=1, le=500)
    top_n: int = Field(20, ge=1, le=1000)
    title: str = ""
    filters: List[FilterSpec] = Field(default_factory=list)


class FiltersBody(BaseModel):
    filters: List[FilterSpec] = Field(default_factory=list)


def _df(owner: str, ds_id: str, filters: List[FilterSpec] | None = None) -> pd.DataFrame:
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    try:
        return apply_filters(df, [f.model_dump() for f in (filters or [])])
    except FilterError as e:
        raise HTTPException(422, str(e))


def _fig_json(fig) -> dict:
    return json.loads(pio.to_json(fig))


@router.get("/{ds_id}/fields")
def fields(ds_id: str, owner: str = Depends(current_owner)):
    df = _df(owner, ds_id)
    return {"fields": field_catalog(df)}


@router.post("/{ds_id}/kpis")
def kpis(ds_id: str, body: FiltersBody, owner: str = Depends(current_owner)):
    """Headline figures for the current filters.

    A measure is totalled only when adding it up means something (revenue,
    units, hours); salary, age or a score is averaged. The cards used to
    read "Σ age" and "Σ tenure_years" — sums nobody can use.
    """
    from app.engines.chart_exporter import _label, _measures, _outcome, fmt, is_additive
    df = _df(owner, ds_id, body.filters)
    cards = [{"label": "Rows", "value": len(df), "format": "int",
              "sub": "after filters" if body.filters else "in the dataset"}]
    out = _outcome(df) if len(df) else None
    if out:
        col, y, event = out
        cards.append({"label": "{} rate".format(_label(col)),
                      "value": round(float(y.mean()) * 100, 1), "format": "pct",
                      "sub": "{:,} of {:,} are '{}'".format(int(y.sum()), int(y.notna().sum()), event)})
    # Ranked measures first; then any other number that is not an identifier
    # or a flag (a constant revenue column still has a total).
    from app.engines.domains.base import is_id_column
    ranked = _measures(df)
    rest = [c for c in df.select_dtypes(include="number").columns
            if c not in ranked and not pd.api.types.is_bool_dtype(df[c])
            and not is_id_column(c, df[c]) and (not out or c != out[0])]
    for col in (ranked + rest)[:4]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not len(s):
            continue
        if is_additive(col):
            cards.append({"label": "Total {}".format(_label(col).lower()),
                          "value": float(s.sum()), "format": "num",
                          "sub": "average {}".format(fmt(float(s.mean())))})
        else:
            cards.append({"label": "Average {}".format(_label(col).lower()),
                          "value": float(s.mean()), "format": "num",
                          "sub": "median {}".format(fmt(float(s.median())))})
    missing = round(float(df.isna().mean().mean()) * 100, 1) if df.size else 0.0
    cards.append({"label": "Missing values", "value": missing, "format": "pct",
                  "sub": "of all cells"})
    return {"kpis": to_jsonable(cards)}


def auto_tiles(df: pd.DataFrame) -> List[dict]:
    """The dashboard's opening tiles, chosen the way the report's charts are:
    an outcome as a rate, a measure totalled only when that means something
    and averaged otherwise, a trend only over a date things happened on (not
    a hire or birth date), and a correlation grid only when something
    correlates. The tiles used to sum the first numeric column — salary,
    tenure — by the first text column and draw a pie of the sums.
    """
    from app.engines.chart_exporter import (_event_date, _group_column, _label,
                                            _measures, _outcome, is_additive)
    measures = _measures(df)
    group = _group_column(df)
    second = _group_column(df, exclude=(group,)) if group else None
    out = _outcome(df)
    date = _event_date(df)
    if out:
        measures = [m for m in measures if m != out[0]]

    def agg(m):
        return "sum" if is_additive(m) else "mean"

    def word(m):
        return "Total" if is_additive(m) else "Average"

    tiles = []
    if out and group:
        tiles.append({"title": "{} rate by {}".format(_label(out[0]), _label(group).lower()),
                      "type": "rate", "x": group, "y": out[0]})
    if measures and group:
        m = measures[0]
        tiles.append({"title": "{} {} by {}".format(word(m), _label(m).lower(), _label(group).lower()),
                      "type": "bar", "x": group, "y": m, "agg": agg(m)})
    if measures and date:
        m = measures[0]
        tiles.append({"title": "{} {} over time".format(word(m), _label(m).lower()),
                      "type": "line", "x": date, "y": m, "agg": agg(m)})
    if out and second:
        tiles.append({"title": "{} rate by {}".format(_label(out[0]), _label(second).lower()),
                      "type": "rate", "x": second, "y": out[0]})
    if measures:
        tiles.append({"title": "Distribution of {}".format(_label(measures[0]).lower()),
                      "type": "histogram", "x": measures[0]})
    if len(measures) > 1 and (second or group):
        m, g = measures[1], second or group
        tiles.append({"title": "{} {} by {}".format(word(m), _label(m).lower(), _label(g).lower()),
                      "type": "bar", "x": g, "y": m, "agg": agg(m)})
    if len(measures) >= 3:
        corr = df[measures[:8]].corr(method="spearman").abs()
        np_corr = corr.where(~pd.DataFrame(
            [[i == j for j in range(len(corr))] for i in range(len(corr))],
            index=corr.index, columns=corr.columns))
        if float(np_corr.max().max()) >= 0.3:
            tiles.append({"title": "How the measures relate", "type": "heatmap"})
    for i, t in enumerate(tiles, 1):
        t["id"] = "t{}".format(i)
    return tiles[:6]


@router.get("/{ds_id}/auto-tiles")
def auto_tiles_route(ds_id: str, owner: str = Depends(current_owner)):
    return {"tiles": auto_tiles(_df(owner, ds_id))}


@router.post("/{ds_id}/recommend")
def recommend(ds_id: str, body: FiltersBody, owner: str = Depends(current_owner)):
    """The opening tiles, rendered."""
    df = _df(owner, ds_id, body.filters)
    charts = []
    for t in auto_tiles(df):
        req = ChartRequest(type=t["type"], x=t.get("x"), y=t.get("y"),
                           agg=t.get("agg", "sum"), title=t["title"])
        try:
            charts.append({"title": t["title"], "figure": _figure(df, req)})
        except HTTPException:
            continue
    return {"charts": charts}


@router.post("/{ds_id}/build")
def build(ds_id: str, req: ChartRequest, owner: str = Depends(current_owner)):
    df = _df(owner, ds_id, req.filters)
    if df.empty:
        raise HTTPException(422, "No rows match the current filters")
    if req.type == "table":
        grouped = _aggregate(df, req) if (req.x and req.y) else df.head(req.top_n)
        from app.services.serialize import df_records
        return {"table": df_records(grouped, req.top_n)}
    return {"figure": _figure(df, req)}


def _rate_figure(df: pd.DataFrame, req: ChartRequest):
    """Share of rows with the outcome (left, churned, returned) per group,
    with the overall rate drawn across for reference."""
    from app.engines.predictive import _to_binary
    if not req.x or not req.y:
        raise HTTPException(422, "x and y are required")
    if req.x not in df.columns or req.y not in df.columns:
        raise KeyError(req.x if req.x not in df.columns else req.y)
    coded = _to_binary(df[req.y])
    if coded is None:
        raise HTTPException(422, f"'{req.y}' is not a two-value outcome")
    y, event = coded
    y = y.where(df[req.y].notna())
    tmp = pd.DataFrame({req.x: df[req.x].astype(str).where(df[req.x].notna()), "y": y}).dropna()
    if tmp.empty:
        raise HTTPException(422, f"No rows have both '{req.x}' and '{req.y}'")
    g = tmp.groupby(req.x)["y"].agg(["mean", "size"])
    g = g[g["size"] >= 5].sort_values("mean", ascending=False).head(req.top_n)
    label = "% '{}'".format(event)
    grouped = pd.DataFrame({req.x: g.index, label: (g["mean"] * 100).round(1).values})
    fig = chart_engine.make_bar(grouped, req.x, label, req.title)
    overall = float(tmp["y"].mean() * 100)
    fig.add_hline(y=overall, line_dash="dash", line_color="#94A3B8",
                  annotation_text="overall {:.0f}%".format(overall),
                  annotation_position="top left")
    return fig


def _figure(df: pd.DataFrame, req: ChartRequest) -> dict:
    t = req.type
    try:
        if t == "rate":
            fig = _rate_figure(df, req)
        elif t == "histogram":
            fig = chart_engine.make_histogram(df, req.x or req.y, req.nbins, req.title)
        elif t == "heatmap":
            from app.engines.chart_exporter import _measures
            cols = _measures(df)[:8]
            if len(cols) < 2:
                raise HTTPException(422, "Need 2+ measure columns for a heatmap")
            fig = chart_engine.make_heatmap(df[cols])
        elif t == "scatter":
            fig = chart_engine.make_scatter(df, req.x, req.y, req.color, req.title)
        elif t in ("bar", "line", "area", "pie"):
            if not req.x or not req.y:
                raise HTTPException(422, "x and y are required")
            if t == "pie" and req.agg not in ("sum", "count"):
                raise HTTPException(
                    422, f"A pie shows shares of a total, and a {req.agg} is not a "
                         f"share of anything — use sum or count, or a bar chart")
            # A pie needs every group: its slices are shares of the whole, so
            # trimming to the top N first would misstate every percentage.
            grouped = _aggregate(df, req, limit=(t != "pie"))
            if t == "bar":
                fig = chart_engine.make_bar(grouped, req.x, req.y, req.title)
            elif t == "pie":
                fig = chart_engine.make_pie(grouped, req.x, req.y, req.title)
            else:
                fig = chart_engine.make_line(grouped, req.x, req.y, req.title)
                if t == "area":
                    fig.update_traces(fill="tozeroy")
                omitted = grouped.attrs.get("omitted") or []
                if omitted:
                    fig.update_xaxes(title_text="{} ({} partial {} excluded)".format(
                        req.x, " and ".join(omitted), grouped.attrs["period"]))
        else:
            raise HTTPException(422, f"Unknown chart type '{t}'")
    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(422, f"Column not found: {e}")
    except Exception as e:
        logger.warning("chart build failed", exc_info=True)
        raise HTTPException(500, f"Chart build failed: {e}")
    return _fig_json(fig)


def _drop_partial_periods(out: pd.DataFrame, x: str, rule: str, agg: str,
                          first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Remove a first or last week/month the data only partly covers.

    A sum or count over half a month is about half a month's worth, so a
    file ending on the 10th drew a final point that looked like revenue
    collapsing. Means and medians are not biased that way and are left
    alone, as are daily buckets. What was removed is recorded in
    `out.attrs["omitted"]` for the caller to state on the chart.
    """
    if agg not in ("sum", "count") or rule not in ("W", "ME") or len(out) < 3:
        return out
    labels = pd.to_datetime(out[x])
    if rule == "W":
        starts = labels - pd.Timedelta(days=6)
    else:
        starts = labels - pd.offsets.MonthBegin(1)
    keep = pd.Series(True, index=out.index)
    omitted = []
    first_d, last_d = first.normalize(), last.normalize()
    if first_d > starts.iloc[0].normalize():
        keep.iloc[0] = False
        omitted.append("first")
    if last_d < labels.iloc[-1].normalize():
        keep.iloc[-1] = False
        omitted.append("last")
    trimmed = out[keep].reset_index(drop=True)
    trimmed.attrs["omitted"] = omitted
    trimmed.attrs["period"] = "week" if rule == "W" else "month"
    return trimmed


def _aggregate(df: pd.DataFrame, req: ChartRequest, limit: bool = True) -> pd.DataFrame:
    x, y, agg = req.x, req.y, req.agg
    if x not in df.columns or y not in df.columns:
        raise KeyError(x if x not in df.columns else y)
    # Both checks used to surface as a 500 from pandas ("cannot insert x,
    # already exists"; "datetime64 type does not support operation 'sum'").
    if x == y:
        raise HTTPException(422, "Pick different columns for x and y")
    if agg != "count" and not (pd.api.types.is_numeric_dtype(df[y])
                               and not pd.api.types.is_bool_dtype(df[y])):
        raise HTTPException(
            422, f"'{y}' is not numeric, so it cannot be summarised with "
                 f"{agg} — use count, or pick a numeric column for y")
    if pd.api.types.is_object_dtype(df[x]):
        # A mostly-numeric column keeps its non-numeric cells as text
        # ("Pending"), and grouping on mixed floats and strings fails while
        # sorting the keys. Group on the values as written.
        df = df.assign(**{x: df[x].where(df[x].isna(), df[x].astype(str))})
    if pd.api.types.is_datetime64_any_dtype(df[x]):
        # resample by sensible period for time axes
        tmp = df[[x, y]].dropna().set_index(x).sort_index()
        if tmp.empty:
            raise HTTPException(422, f"No rows have both '{x}' and '{y}'")
        span_days = (tmp.index.max() - tmp.index.min()).days or 1
        rule = "D" if span_days <= 92 else ("W" if span_days <= 730 else "ME")
        agg_fn = "count" if agg == "count" else agg
        out = getattr(tmp[y].resample(rule), agg_fn)().reset_index()
        return _drop_partial_periods(out, x, rule, agg, tmp.index.min(), tmp.index.max())
    if agg == "count":
        out = df.groupby(x, dropna=True)[y].count().reset_index()
    else:
        out = df.groupby(x, dropna=True)[y].agg(agg).reset_index()
    out = out.sort_values(y, ascending=False)
    return out.head(req.top_n) if limit else out
