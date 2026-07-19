"""
api/reports.py — report generation & data export (mirrors page 8, Reports).
Reproduces the full senior-analyst PDF pipeline:
profile → domain → story → insights → stats → BI → charts+narratives → PDF.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import config
from app.services.dataset_store import store

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


def _df_or_404(ds_id: str):
    df = store.get_df(ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return df


@router.get("/{ds_id}/csv")
def export_csv(ds_id: str):
    df = _df_or_404(ds_id)
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8-sig"))
    return StreamingResponse(buf, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=analytiq_cleaned_data.csv"})


@router.get("/{ds_id}/excel")
def export_excel(ds_id: str):
    df = _df_or_404(ds_id)
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
def generate_pdf(ds_id: str, req: PdfRequest):
    df = _df_or_404(ds_id)

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
        stats_report = store.cache_get(ds_id, "stats")
        if stats_report is None:
            try:
                from app.engines.stats_engine import analyze
                stats_report = analyze(df)
                store.cache_set(ds_id, "stats", stats_report)
            except Exception:
                pass

    # 6. BI
    bi_report = None
    if req.include_bi:
        bi_report = store.cache_get(ds_id, "bi")
        if bi_report is None:
            try:
                from app.engines.bi_engine import run_bi
                bi_report = run_bi(df)
                store.cache_set(ds_id, "bi", bi_report)
            except Exception:
                pass

    # 7. ML (only if previously trained)
    ml_report = store.cache_get(ds_id, "ml_last") if req.include_ml else None

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
            cleaning_summary=store.cache_get(ds_id, "clean_report"),
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
