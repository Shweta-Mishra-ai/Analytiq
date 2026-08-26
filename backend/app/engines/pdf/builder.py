"""
engines/pdf/builder.py — assembles the report.

build_pdf keeps the signature it has always had; callers are unaffected
by the package split.
"""
import logging
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether,
)
from reportlab.pdfgen import canvas as CV

logger = logging.getLogger(__name__)

from app.engines.pdf.theme import (
    THEMES, HR_BENCHMARKS, _domain_theme, _c, _styles, _ReportCanvas,
    _build_cover, W, H, CW_DEFAULT,
)
from app.engines.pdf.primitives import (
    _sec, _kpi_row, _narrative_box, _gtable, _insight_card, _toc, _clean,
)
from app.engines.pdf.narrative_sections import (
    _exec_summary, _top_insights, _dq_note, _readiness_block,
    _benchmark_section, _attrition_page, _domain_label,
    _has_reference_ranges, _exec_dashboard, _predictive_section,
)
from app.engines.pdf.data_sections import (
    _data_prep_section, _dataset_overview, _estimates_block, _stats_section,
    _bi_section, _chart_page, _recommendations,
)
from app.engines.pdf.domain_sections import (
    _appendix, _prepared_by_line, _domain_deep_page, has_deep_page,
)
from app.engines.report_blueprints import blueprint_for
from app.services.dtypes import is_text_dtype, text_columns


# ══════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTION  (identical signature to original)
# ══════════════════════════════════════════════════════════

def _has_predictive_section(predictive) -> bool:
    """Whether the report carries a Predictive Risk section.

    True both when the model found drivers and when it found nothing:
    "we looked and found no signal" is a result the reader needs, and
    gating on `top_drivers` alone made a failed model indistinguishable
    from one that was never attempted.
    """
    if predictive is None:
        return False
    if getattr(predictive, "top_drivers", None):
        return True
    verdict = getattr(predictive, "verdict", None)
    return verdict is not None and not verdict.usable



def build_pdf(
    df: pd.DataFrame,
    config: dict,
    profile=None,
    cleaning_summary: dict = None,
    stats_report=None,
    bi_report=None,
    ml_report=None,
    chart_data: list = None,
    executive_summary: str = "",
    findings: list = None,
    risks: list = None,
    opportunities: list = None,
    recommendations: list = None,
    top_insights: list = None,
    attrition=None,
    domain: str = "general",
    predictive=None,
    top_cluster=None,
    driver_chart: bytes = None,
    risk_heatmap: bytes = None,
    avg_salary_k: float = 0.0,
) -> bytes:
    """Build the report.

    The first sixteen parameters are unchanged and still positional-safe;
    everything after `domain` is optional and additive.

    `predictive` is a predictive.DriverResult and `top_cluster` a
    predictive.TopCluster. Both engines have been in the codebase all
    along, but build_pdf had no parameter to receive their output, so the
    Predictive Risk section simply did not exist and the work was thrown
    away. `driver_chart` and `risk_heatmap` are optional pre-rendered PNGs
    for that section; it falls back to a table when they are absent.
    """
    from pypdf import PdfWriter, PdfReader

    findings        = findings       or []
    risks           = risks          or []
    opportunities   = opportunities  or []
    recommendations = recommendations or []
    chart_data      = chart_data     or []
    top_insights    = top_insights   or []

    # ── Theme selection ───────────────────────────────────
    theme_name = config.get("theme_name", "")
    if theme_name not in THEMES:
        auto_key  = _domain_theme(domain)
        theme_name = auto_key
    T = THEMES[theme_name]
    config["domain"]     = domain
    config["theme_name"] = theme_name

    report_title = config.get("title", "Data Analysis Report")
    client_name  = config.get("client_name", "Client")
    report_date  = datetime.now().strftime("%B %d, %Y")

    # ── KPI preview for cover ─────────────────────────────
    n_rows    = len(df)
    miss_pct  = df.isna().mean().mean() * 100
    n_charts  = len(chart_data)
    qual      = getattr(profile, "overall_quality_score", "—")

    kpis_cover = [
        {"label": "RECORDS",      "value": "{:,}".format(n_rows),
         "sub":   "Clean dataset", "color": T["accent"]},
        {"label": "QUALITY",      "value": str(qual),
         "sub":   "/ 100",         "color": T["positive"]},
        {"label": "CHARTS",       "value": str(n_charts),
         "sub":   "Incl. in report","color": T["accent"]},
        {"label": "MISSING DATA", "value": "{:.1f}%".format(miss_pct),
         "sub":   "0% = perfect",  "color": T["positive"] if miss_pct < 1 else T["warning"]},
    ]

    # ── Cover page ────────────────────────────────────────
    cover_bytes = _build_cover(T, config, kpis_cover)

    # ── Content pages ─────────────────────────────────────
    content_buf = io.BytesIO()
    M   = 18 * mm
    CW  = W - 2 * M

    def canvas_maker(fn, **kw):
        return _ReportCanvas(fn, T=T,
                             report_title=report_title,
                             client_name=client_name,
                             report_date=report_date, **kw)

    doc = BaseDocTemplate(
        content_buf, pagesize=A4,
        leftMargin=M, rightMargin=M,
        topMargin=30*mm, bottomMargin=17*mm,
    )
    frame = Frame(M, 17*mm, CW, H - 47*mm,
                  leftPadding=0, rightPadding=0,
                  topPadding=0,  bottomPadding=0)
    tpl   = PageTemplate(id="main", frames=[frame], onPage=lambda c,d: None)
    doc.addPageTemplates([tpl])

    s     = _styles(T)
    story = []

    # ── Build TOC entries ─────────────────────────────────
    sec_num = 1
    toc     = []
    def _add_toc(title):
        nonlocal sec_num
        toc.append((sec_num, title))
        sec_num += 1

    _add_toc("Report at a Glance")
    _add_toc("Executive Summary")
    _add_toc("Data Quality & Transparency Note")
    if cleaning_summary:
        _add_toc("Data Preparation")
    if _has_reference_ranges(domain, df):
        _add_toc("Performance Against Published Ranges")
    _add_toc("{} — Findings".format(
        blueprint_for(domain).label))
    if attrition:
        _add_toc("Attrition Deep Dive")
    if _has_predictive_section(predictive):
        _add_toc("Predictive Risk Analysis")
    if has_deep_page(domain):
        _add_toc("{} Analysis".format(_domain_label(domain).title()))
    _add_toc("Dataset Overview & Descriptive Statistics")
    if stats_report:
        _add_toc("Statistical Analysis")
    if bi_report:
        _add_toc("Business Intelligence")
    for i, (t, _, _) in enumerate(chart_data, 1):
        _add_toc("Chart {}: {}".format(i, t[:28]))
    _add_toc("Recommendations & Action Plan")
    _add_toc("Appendix — Methodology & Sources")

    # ── Assemble story ────────────────────────────────────
    _toc(story, s, T, toc, CW)
    story.append(PageBreak())

    _exec_dashboard(story, s, T, df, profile, top_insights,
                    executive_summary, CW, top_cluster=top_cluster,
                    driver_result=predictive, avg_salary_k=avg_salary_k)
    story.append(PageBreak())

    _exec_summary(story, s, T, executive_summary,
                  findings, risks, opportunities, CW)
    story.append(PageBreak())

    _dq_note(story, s, T, df, profile, CW)
    story.append(PageBreak())

    if cleaning_summary:
        _data_prep_section(story, s, T, cleaning_summary, CW,
                           table=config.get("source_table") or "source_table")
        story.append(PageBreak())

    if _has_reference_ranges(domain, df):
        _benchmark_section(story, s, T, domain, CW, df=df)
        story.append(PageBreak())

    _top_insights(story, s, T, top_insights, CW, domain=domain)
    story.append(PageBreak())

    if attrition:
        _attrition_page(story, s, T, attrition, CW)
        story.append(PageBreak())

    if _has_predictive_section(predictive):
        _predictive_section(story, s, T, predictive, CW,
                            avg_salary_k=avg_salary_k,
                            top_cluster=top_cluster,
                            driver_chart=driver_chart,
                            risk_heatmap=risk_heatmap)
        story.append(PageBreak())

    if has_deep_page(domain):
        if _domain_deep_page(story, s, T, df, config, CW, domain,
                             profile=profile):
            story.append(PageBreak())

    _dataset_overview(story, s, T, df, profile, CW)
    _estimates_block(story, s, T, df, CW)
    story.append(PageBreak())

    if stats_report:
        _stats_section(story, s, T, stats_report, CW)
        story.append(PageBreak())

    if bi_report:
        _bi_section(story, s, T, bi_report, CW)
        story.append(PageBreak())

    for i, (title, img_bytes, narrative) in enumerate(chart_data, 1):
        _chart_page(story, s, T, img_bytes, title, narrative, i, CW)
        story.append(PageBreak())

    _recommendations(story, s, T, recommendations, CW)
    story.append(PageBreak())

    _appendix(story, s, T, config, CW, domain=domain)

    # ── Build PDF ─────────────────────────────────────────
    doc.build(story, canvasmaker=canvas_maker)
    content_buf.seek(0)

    # ── Merge cover + content ─────────────────────────────
    writer = PdfWriter()
    for pg in PdfReader(io.BytesIO(cover_bytes)).pages:
        writer.add_page(pg)
    for pg in PdfReader(content_buf).pages:
        writer.add_page(pg)

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()
