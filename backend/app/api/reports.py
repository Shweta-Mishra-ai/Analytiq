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
from app.services.auth import current_owner
from app.services.dataset_store import store
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
    avg_salary_k: float = 8.0       # for HR impact estimates
    max_charts: int = 5


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


@router.post("/{ds_id}/pdf")
def generate_pdf(ds_id: str, req: PdfRequest, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)

    from app.engines.data_profiler import profile_dataset
    from app.engines.story_engine import detect_domain, generate_story
    from app.engines.pdf_builder import build_pdf
    from app.engines.chart_exporter import generate_all_charts

    # 1. profile
    try:
        profile = profile_dataset(df)
    except Exception:
        profile = None

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
        logger.warning(f"story_engine failed: {e}")

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
        logger.warning(f"insights_builder failed: {e}")

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
                logger.debug("generate_pdf: suppressed exception", exc_info=True)

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
                logger.debug("generate_pdf: suppressed exception", exc_info=True)

    # 7. ML (only if previously trained)
    ml_report = store.cache_get(owner, ds_id, "ml_last") if req.include_ml else None

    # 8. charts + AI narratives
    chart_data = []
    theme_name = req.theme_name
    try:
        charts = generate_all_charts(df, theme_name, max_charts=req.max_charts)
        for title, img_bytes in charts:
            if not img_bytes:
                continue
            try:
                from app.ai.report_narrator import generate_chart_narrative
                narrative = generate_chart_narrative(
                    df, title, config.groq_api_key, domain_name)
            except Exception:
                narrative = "Chart generated from dataset analysis."
            chart_data.append((title, img_bytes, narrative))
    except Exception as e:
        logger.warning(f"chart export failed: {e}")

    # 9. build PDF
    pdf_config = {
        "title": req.title,
        "subtitle": req.subtitle,
        "client_name": req.client_name,
        "confidential": req.confidential,
        "theme_name": theme_name,
        "logo_path": None,
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
        )
    except Exception as e:
        logger.exception("PDF build failed")
        raise HTTPException(500, f"PDF build failed: {e}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=analytiq_report.pdf"})


class HealthPdfRequest(BaseModel):
    # Shown in the header/footer of every page — set it to your own or your
    # client's agency name when delivering the report.
    agency_name: str = "Analytiq"


@router.get("/{ds_id}/health")
def health_summary(ds_id: str, owner: str = Depends(current_owner)):
    """Health score, grade and the niche insight cards as JSON — the same
    content the Health Report PDF renders, for previewing before download."""
    df = _df_or_404(owner, ds_id)
    from app.engines.health_engine import compute_health, build_full_insights
    from app.engines.story_engine import detect_domain

    health = compute_health(df)
    try:
        domain_name, _ = detect_domain(df)
    except Exception:
        logger.warning("detect_domain failed for health summary", exc_info=True)
        domain_name = "general"
    try:
        insights = build_full_insights(df, domain_name)
    except Exception:
        logger.exception("build_insights failed for health summary")
        insights = []
    return {"domain": domain_name, "health": to_jsonable(health),
            "insights": to_jsonable(insights)}


@router.post("/{ds_id}/health-pdf")
def generate_health_pdf(ds_id: str, req: HealthPdfRequest,
                         owner: str = Depends(current_owner)):
    """Client-facing Data Health & Business Insights report (PDF)."""
    df = _df_or_404(owner, ds_id)
    meta = store.get_meta(owner, ds_id)
    filename = meta.filename if meta else f"{ds_id}.csv"

    from app.engines.health_engine import compute_health, build_full_insights
    from app.engines.health_pdf_builder import build_health_pdf
    from app.engines.story_engine import detect_domain

    try:
        domain_name, _ = detect_domain(df)
    except Exception:
        logger.warning("detect_domain failed for health PDF", exc_info=True)
        domain_name = "general"

    try:
        health = compute_health(df)
        insights = build_full_insights(df, domain_name)
        pdf_bytes = build_health_pdf(df, domain_name, health, insights,
                                      filename, agency_name=req.agency_name)
    except Exception as e:
        logger.exception("Health PDF build failed")
        raise HTTPException(500, f"Health report build failed: {e}")

    return StreamingResponse(
        io.BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition":
                 "attachment; filename=analytiq_health_report.pdf"})
