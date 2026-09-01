"""
api/reports.py — report generation & data export (mirrors page 8, Reports).
Reproduces the full senior-analyst PDF pipeline:
profile → domain → story → insights → stats → BI → charts+narratives → PDF.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import config
from app.engines.data_cleaner import table_name_from_filename
from app.services.auth import current_owner
from app.services.dataset_store import store
from app.services.metrics import metrics
from app.services.serialize import to_jsonable

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reports", tags=["reports"])


class PdfRequest(BaseModel):
    title: str = "Data Analysis Report"
    subtitle: str = ""
    client_name: str = "Client"
    confidential: bool = False
    theme_name: str = ""            # empty → auto by domain
    include_stats: bool = True
    include_bi: bool = True
    include_ml: bool = False
    # 0 means "not supplied". It used to default to 8.0, and every report
    # then stated a cost "at the 8K unit cost entered at report setup" —
    # naming a figure the client had never entered and could not have
    # recognised.
    avg_salary_k: float = 0.0
    max_charts: int = 5
    # Who prepared the report — the freelancer, consultancy or in-house
    # analyst delivering it. It appears in the basis of preparation and
    # under the methodology, because a review deliverable is signed by a
    # person or a firm. Nothing about the tooling is named anywhere.
    prepared_by: str = ""
    # "pdf" (default) or "pptx". The deck is the same analysis rendered for
    # a room rather than for a desk — in consulting it is usually the
    # thing that actually gets presented.
    format: str = "pdf"
    # Periods to project. Capped in the engine; a longer horizon widens
    # the interval rather than adding confidence.
    forecast_horizon: int = 3


def _df_or_404(owner: str, ds_id: str):
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return df


@router.get("/{ds_id}/csv")
def export_csv(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8-sig"))
    return StreamingResponse(buf, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=analytiq_cleaned_data.csv"})


@router.get("/{ds_id}/excel")
def export_excel(ds_id: str, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)
    buf = io.BytesIO()
    import pandas as pd
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        num = df.select_dtypes(include="number")
        if not num.empty:
            num.describe().T.to_excel(writer, sheet_name="Summary")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=analytiq_export.xlsx"})


def _count_skipped(skipped: list) -> None:
    """A section that dropped out is an engine that failed. Counting them
    by name turns "the report looks thin" into "forecast has failed 40
    times since the last restart"."""
    for section in skipped:
        metrics.record_failure(f"engine.{section}", "section omitted")


@router.post("/{ds_id}/pdf")
def generate_pdf(ds_id: str, req: PdfRequest, owner: str = Depends(current_owner)):
    # "Reports are slow" is unanswerable without a number, and the answer
    # is rarely build_pdf itself — it is usually one engine upstream of it.
    # The whole request is timed, and every section that drops out is
    # counted, so /api/metrics can say which one.
    with metrics.timed(f"report.{req.format}"):
        return _generate_pdf(ds_id, req, owner)


def _generate_pdf(ds_id: str, req: PdfRequest, owner: str):
    df = _df_or_404(owner, ds_id)

    from app.engines.data_profiler import profile_dataset
    from app.engines.story_engine import detect_domain, generate_story
    from app.engines.pdf_builder import build_pdf
    from app.engines.chart_exporter import generate_all_charts

    # Sections that failed to build. A report that quietly drops a section
    # looks the same as one that never had it, so the names come back on a
    # response header instead of vanishing into a debug log.
    skipped: list = []

    # 1. profile
    try:
        profile = profile_dataset(df)
    except Exception:
        logger.warning("profiling failed — quality figures omitted",
                       exc_info=True)
        profile = None
        skipped.append("profile")

    # 2. domain
    try:
        domain_name, _ = detect_domain(df)
    except Exception:
        domain_name = "general"

    # 3. story
    story_obj = None
    exec_summary, findings, risks, opportunities, actions = (
        "Analysis completed by Analytiq.", [], [], [], [])
    try:
        story_obj = generate_story(df)
        exec_summary = story_obj.executive_summary
        findings = story_obj.key_findings
        risks = story_obj.business_risks
        opportunities = story_obj.opportunities
        actions = story_obj.recommended_actions
    except Exception as e:
        logger.warning("story engine failed — narrative sections fall back "
                       "to defaults: %s", e, exc_info=True)
        skipped.append("story")

    # 4. structured insights
    top_insights = []
    try:
        from app.engines.insights_builder import build_top_insights
        top_insights = build_top_insights(
            df=df, domain=domain_name, story_obj=story_obj,
            attrition=getattr(story_obj, "attrition", None),
            avg_salary_k=float(req.avg_salary_k),
        )
    except Exception as e:
        logger.warning("insight builder failed — findings section will be "
                       "thin: %s", e, exc_info=True)
        skipped.append("insights")

    # 5. stats
    stats_report = None
    if req.include_stats:
        stats_report = store.cache_get(owner, ds_id, "stats")
        if stats_report is None:
            try:
                from app.engines.stats_engine import analyze
                stats_report = analyze(df)
                store.cache_set(owner, ds_id, "stats", stats_report)
            except Exception:
                logger.warning("statistics failed — section omitted",
                               exc_info=True)
                skipped.append("stats")

    # 6. BI
    bi_report = None
    if req.include_bi:
        bi_report = store.cache_get(owner, ds_id, "bi")
        if bi_report is None:
            try:
                from app.engines.bi_engine import run_bi
                bi_report = run_bi(df)
                store.cache_set(owner, ds_id, "bi", bi_report)
            except Exception:
                logger.warning("business intelligence failed — section "
                               "omitted", exc_info=True)
                skipped.append("bi")

    # 7. ML (only if previously trained)
    ml_report = store.cache_get(owner, ds_id, "ml_last") if req.include_ml else None

    # 8. predictive drivers + risk concentration
    #
    # predictive.py has been in the codebase throughout, but build_pdf had
    # no parameter to receive its output, so this ran (when anything called
    # it at all) and was discarded. The report now carries it.
    predictive = top_cluster = None
    driver_chart = risk_heatmap = None
    try:
        from app.engines.predictive import (
            find_binary_target, compute_drivers, find_top_cluster,
            pick_heatmap_dims,
        )
        target = find_binary_target(df)
        if target:
            predictive = compute_drivers(df, target)
            top_cluster = find_top_cluster(df, target)
            if predictive is not None and predictive.top_drivers:
                try:
                    from app.engines.chart_exporter import (
                        make_driver_importance_chart, make_risk_heatmap)
                    driver_chart = make_driver_importance_chart(
                        predictive.top_drivers, theme_name=req.theme_name,
                        target_label=str(target).replace("_", " "))
                    dims = pick_heatmap_dims(df, target)
                    if dims:
                        risk_heatmap = make_risk_heatmap(
                            df, target, dims[0], dims[1],
                            theme_name=req.theme_name,
                            event_label=str(target).replace("_", " "))
                except Exception:
                    logger.warning("predictive charts failed — the section "
                                   "falls back to its table form",
                                   exc_info=True)
        else:
            logger.info("no binary target column — predictive section skipped")
    except Exception:
        logger.warning("predictive analysis failed — section skipped",
                       exc_info=True)
        skipped.append("predictive")

    # 9. outlook
    forecast = None
    try:
        from app.engines.forecast_engine import run_forecast
        forecast = run_forecast(df, horizon=int(req.forecast_horizon))
    except Exception:
        logger.warning("forecast failed — section omitted", exc_info=True)
        skipped.append("forecast")

    # 10. charts + AI narratives
    chart_data = []
    theme_name = req.theme_name
    try:
        charts = generate_all_charts(df, theme_name, max_charts=req.max_charts)
        for title, img_bytes, spec in charts:
            if not img_bytes:
                continue
            try:
                from app.ai.report_narrator import generate_chart_narrative
                narrative = generate_chart_narrative(
                    df, title, config.groq_api_key, domain_name, spec=spec)
            except Exception:
                narrative = "Chart generated from dataset analysis."
            chart_data.append((title, img_bytes, narrative))
    except Exception as e:
        logger.warning("chart export failed — report has no figures: %s", e,
                       exc_info=True)
        skipped.append("charts")

    # 9. build PDF
    pdf_config = {
        "title": req.title,
        "subtitle": req.subtitle,
        "client_name": req.client_name,
        "confidential": req.confidential,
        "theme_name": theme_name,
        "logo_path": None,
        # Names the table the Data Preparation SQL is written against, so a
        # reader recognises their own warehouse object rather than a
        # placeholder.
        "prepared_by": req.prepared_by.strip(),
        "source_table": table_name_from_filename(
            getattr(store.get_meta(owner, ds_id), "filename", "")),
    }
    try:
        pdf_bytes = build_pdf(
            df=df, config=pdf_config, profile=profile,
            cleaning_summary=store.cache_get(owner, ds_id, "clean_report"),
            stats_report=stats_report, bi_report=bi_report,
            ml_report=ml_report, chart_data=chart_data,
            executive_summary=exec_summary, findings=findings,
            risks=risks, opportunities=opportunities,
            recommendations=actions, top_insights=top_insights,
            attrition=getattr(story_obj, "attrition", None),
            domain=domain_name,
            predictive=predictive, top_cluster=top_cluster,
            driver_chart=driver_chart, risk_heatmap=risk_heatmap,
            avg_salary_k=float(req.avg_salary_k), forecast=forecast,
        )
    except Exception as e:
        logger.exception("PDF build failed")
        raise HTTPException(500, f"PDF build failed: {e}")

    if req.format == "pptx":
        from app.engines.deck_builder import build_deck
        from app.engines.kpi_engine import compute_kpis
        try:
            deck = build_deck(
                df=df, config=pdf_config, domain=domain_name,
                kpis=[c.as_dict() for c in compute_kpis(df, domain_name)],
                executive_summary=exec_summary, findings=findings,
                top_insights=top_insights, recommendations=actions,
                chart_data=chart_data, predictive=predictive,
                forecast=forecast)
        except Exception as e:
            logger.exception("deck build failed")
            raise HTTPException(500, f"Deck build failed: {e}")
        deck_headers = {"Content-Disposition":
                        "attachment; filename=analytiq_report.pptx"}
        if skipped:
            # The deck is built from the same analysis, so it drops the
            # same sections — and used to say nothing about it.
            deck_headers["X-Analytiq-Skipped-Sections"] = ",".join(skipped)
            _count_skipped(skipped)
        return StreamingResponse(
            io.BytesIO(deck),
            media_type=("application/vnd.openxmlformats-officedocument"
                        ".presentationml.presentation"),
            headers=deck_headers)

    headers = {"Content-Disposition":
               "attachment; filename=analytiq_report.pdf"}
    if skipped:
        # A report that quietly drops a section is indistinguishable from
        # one that never had it. Name them so the caller can tell.
        headers["X-Analytiq-Skipped-Sections"] = ",".join(skipped)
        logger.warning("report built with sections omitted: %s",
                       ", ".join(skipped))
        _count_skipped(skipped)
    return StreamingResponse(
        io.BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)


class HealthPdfRequest(BaseModel):
    # Shown in the header/footer of every page — set it to your own or your
    # client's agency name when delivering the report.
    agency_name: str = "Analytiq"


@router.get("/{ds_id}/health")
def health_summary(ds_id: str, owner: str = Depends(current_owner)):
    """Health score, grade and the niche insight cards as JSON — the same
    content the Health Report PDF renders, for previewing before download."""
    df = _df_or_404(owner, ds_id)
    from app.engines.health_engine import build_report_payload, compute_health
    from app.engines.story_engine import detect_domain

    health = compute_health(df)
    try:
        domain_name, _ = detect_domain(df)
    except Exception:
        logger.warning("detect_domain failed for health summary", exc_info=True)
        domain_name = "general"
    try:
        payload = build_report_payload(df, domain_name)
    except Exception:
        logger.exception("health report payload failed")
        payload = {"executive_summary": "", "insights": [], "key_findings": [],
                   "risks": [], "opportunities": [], "actions": []}
    return {"domain": domain_name, "health": to_jsonable(health),
            **{k: to_jsonable(v) for k, v in payload.items()}}


@router.post("/{ds_id}/health-pdf")
def generate_health_pdf(ds_id: str, req: HealthPdfRequest,
                         owner: str = Depends(current_owner)):
    """Client-facing Data Health & Business Insights report (PDF)."""
    df = _df_or_404(owner, ds_id)
    meta = store.get_meta(owner, ds_id)
    filename = meta.filename if meta else f"{ds_id}.csv"

    from app.engines.health_engine import build_report_payload, compute_health
    from app.engines.health_pdf_builder import build_health_pdf
    from app.engines.story_engine import detect_domain

    try:
        domain_name, _ = detect_domain(df)
    except Exception:
        logger.warning("detect_domain failed for health PDF", exc_info=True)
        domain_name = "general"

    try:
        health = compute_health(df)
        payload = build_report_payload(df, domain_name)
        pdf_bytes = build_health_pdf(
            df, domain_name, health, payload["insights"], filename,
            agency_name=req.agency_name,
            executive_summary=payload["executive_summary"],
            key_findings=payload["key_findings"],
            risks=payload["risks"],
            opportunities=payload["opportunities"],
            actions=payload["actions"])
    except Exception as e:
        logger.exception("Health PDF build failed")
        raise HTTPException(500, f"Health report build failed: {e}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition":
                 "attachment; filename=analytiq_health_report.pdf"})
