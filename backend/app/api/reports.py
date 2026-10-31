"""
api/reports.py — report generation & data export (mirrors page 8, Reports).
Reproduces the full senior-analyst PDF pipeline:
profile → domain → story → insights → stats → BI → charts+narratives → PDF.
"""
from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.engines.data_cleaner import table_name_from_filename
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
    # Who prepared the report — the freelancer, consultancy or in-house
    # analyst delivering it. It appears in the basis of preparation and
    # under the methodology, because a review deliverable is signed by a
    # person or a firm. Nothing about the tooling is named anywhere.
    prepared_by: str = ""


def _df_or_404(owner: str, ds_id: str):
    df = store.get_df(owner, ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    return df


_FORMULA_LEADS = ("=", "+", "-", "@", "\t", "\r")


def _neutralise_formulas(df):
    """A copy of `df` whose text cells cannot run as spreadsheet formulas.

    The exported file is opened in Excel or Sheets, which evaluate any cell
    beginning with = + - or @ — so an uploaded value like
    =HYPERLINK("http://…") becomes a live link or command in the recipient's
    spreadsheet. A leading apostrophe is the standard defence: the
    spreadsheet shows the text as typed and runs nothing. Only text cells
    are touched, and a value that is just a number ("-5") is left alone.
    """
    import pandas as pd
    from app.services.dtypes import is_text_dtype

    def fix(v):
        if isinstance(v, str) and v.startswith(_FORMULA_LEADS):
            try:
                float(v)
                return v
            except ValueError:
                return "'" + v
        return v

    out = df.copy()
    for col in out.columns:
        if is_text_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].map(fix)
    return out


@router.get("/{ds_id}/csv")
def export_csv(ds_id: str, owner: str = Depends(current_owner)):
    df = _neutralise_formulas(_df_or_404(owner, ds_id))
    # Sent whole. A StreamingResponse over a BytesIO iterates it line by
    # line: 60,000 rows went out as 60,000 chunks and took 12 s instead of 2.
    return Response(df.to_csv(index=False).encode("utf-8-sig"), media_type="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=analytiq_cleaned_data.csv"})


def _excel_safe_times(df):
    """Excel cannot store a timezone, and xlsxwriter refuses the whole
    export when a column has one. Write the wall-clock time and name the
    zone in the header, so nothing about the value is lost."""
    import pandas as pd
    out, renames = df, {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) and getattr(s.dt, "tz", None) is not None:
            if out is df:
                out = df.copy()
            out[col] = s.dt.tz_localize(None)
            renames[col] = f"{col} ({s.dt.tz})"
    return out.rename(columns=renames) if renames else out


@router.get("/{ds_id}/excel")
def export_excel(ds_id: str, owner: str = Depends(current_owner)):
    df = _excel_safe_times(_df_or_404(owner, ds_id))
    buf = io.BytesIO()
    import pandas as pd
    # Strings are written as text. xlsxwriter's default turns any value
    # beginning with "=" into a live formula in the delivered workbook.
    # Not constant_memory: it is faster, but pandas does not write row by
    # row, and in that mode every text and date cell came back empty.
    with pd.ExcelWriter(buf, engine="xlsxwriter", engine_kwargs={
            "options": {"strings_to_formulas": False}}) as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        num = df.select_dtypes(include="number")
        if not num.empty:
            num.describe().T.to_excel(writer, sheet_name="Summary")
    return Response(
        buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=analytiq_export.xlsx"})


@router.post("/{ds_id}/pdf")
def generate_pdf(ds_id: str, req: PdfRequest, owner: str = Depends(current_owner)):
    df = _df_or_404(owner, ds_id)

    from app.engines.data_profiler import profile_dataset
    from app.engines.story_engine import detect_domain, generate_story
    from app.engines.pdf_builder import build_pdf
    from app.engines.chart_exporter import build_report_charts

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
    exec_summary, findings, risks, opportunities, actions = ("", [], [], [], [])
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

    # 8. charts, each captioned from the numbers it plots. The text used to
    # be written separately, by guessing the chart's columns back from its
    # title, and described identifiers and the wrong measure.
    from app.engines import report_theme
    theme_name = report_theme.resolve(req.theme_name, domain_name)
    chart_data = []
    try:
        chart_data = [(c.title, c.png, c.caption) for c in build_report_charts(
            df, theme_name, domain_name, max_charts=req.max_charts)]
    except Exception:
        logger.warning("chart export failed", exc_info=True)

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
        )
    except Exception as e:
        logger.exception("PDF build failed")
        raise HTTPException(500, f"PDF build failed: {e}")

    return Response(
        pdf_bytes, media_type="application/pdf",
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

    return Response(
        pdf_bytes, media_type="application/pdf",
        headers={"Content-Disposition":
                 "attachment; filename=analytiq_health_report.pdf"})
