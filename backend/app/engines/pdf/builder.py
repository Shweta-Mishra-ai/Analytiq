"""
engines/pdf/builder.py — assembles the report.

build_pdf keeps the signature it has always had; callers are unaffected
by the package split.
"""
import logging
import io
from datetime import datetime

import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    PageBreak,
)

logger = logging.getLogger(__name__)

from app.engines.present import truncate as _fit

from app.engines.pdf.theme import (
    THEMES, _domain_theme, _styles, _ReportCanvas,
    _build_cover, W, H,
)
from app.engines.pdf.primitives import (
    _sec, _toc,
)
from app.engines.pdf.narrative_sections import (
    _exec_summary, _top_insights, _dq_note, _benchmark_section, _attrition_page, _domain_label,
    _has_reference_ranges, _exec_dashboard, _forecast_section,
)
from app.engines.pdf.predictive_sections import _predictive_section
from app.engines.pdf.lineage import _data_prep_section
from app.engines.pdf.data_sections import (
    _dataset_overview, _estimates_block, _stats_section,
    _bi_section, _chart_page, _governance_section,
    _recommendations,
)
from app.engines.pdf.domain_sections import (
    APPENDIX_TITLE, _appendix, _domain_deep_page,
    has_deep_page,
)
from app.engines.report_blueprints import blueprint_for


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
    forecast=None,
    top_cluster=None,
    driver_chart: bytes = None,
    risk_heatmap: bytes = None,
    avg_salary_k: float = 0.0,
    governance=None,
    integrity=None,
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

    # A cover reports the business, not the file. Three of the four tiles
    # used to be metadata — a quality score, a missing-data percentage,
    # and a count of the charts inside — which told a client nothing they
    # had asked to know. The headline finding leads instead, and the
    # file-level figures keep one tile between them.
    n_action = sum(1 for i in (top_insights or [])
                   if str(getattr(i, "severity", "")).lower()
                   in ("critical", "high"))
    headline = None
    if attrition is not None and getattr(attrition, "rate", None) is not None:
        headline = {"label": "ATTRITION RATE",
                    "value": "{:.1f}%".format(attrition.rate),
                    "sub": "{:,} of {:,} left".format(
                        attrition.n_left, attrition.n_total),
                    "color": T["negative"] if attrition.severity in
                             ("critical", "high") else T["positive"]}
    elif predictive is not None and getattr(predictive, "base_rate", None):
        headline = {"label": "EVENT RATE", "sub": "across the period",
                    "value": "{:.1f}%".format(predictive.base_rate),
                    "color": T["accent"]}

    try:
        qual_txt = "{:.0f}".format(float(qual))
    except (TypeError, ValueError):
        qual_txt = str(qual)

    kpis_cover = [
        {"label": "RECORDS",      "value": "{:,}".format(n_rows),
         "sub":   "{} fields".format(df.shape[1]), "color": T["accent"]},
    ]
    if headline:
        kpis_cover.append(headline)
    kpis_cover.append(
        {"label": "NEED ACTION", "value": str(n_action),
         "sub": "of {} findings".format(len(top_insights or [])),
         "color": T["negative"] if n_action else T["positive"]})
    kpis_cover.append(
        {"label": "DATA QUALITY", "value": "{} / 100".format(qual_txt),
         "sub": "{:.1f}% missing".format(miss_pct),
         "color": T["positive"] if miss_pct < 1 else T["warning"]})

    # ── Cover page ────────────────────────────────────────
    # Off unless a model is assigned to the cover_art task, in which
    # case this returns the image plus the caption that must accompany
    # it. Never touches an exhibit — see ai/imagery.py.
    cover_art = None
    try:
        from app.ai import imagery
        cover_art = imagery.generate_cover(
            title=config.get("title", "Data Analysis Report"),
            domain=str(domain or ""))
    except Exception:
        logger.warning("cover artwork step failed — using the flat cover",
                       exc_info=True)

    cover_bytes = _build_cover(T, config, kpis_cover, cover_art=cover_art)

    # ── Content pages ─────────────────────────────────────
    content_buf = io.BytesIO()
    M   = 18 * mm
    CW  = W - 2 * M

    def canvas_maker(fn, **kw):
        return _ReportCanvas(fn, T=T,
                             report_title=report_title,
                             client_name=client_name,
                             report_date=report_date,
                             confidential=bool(config.get("confidential")),
                             agency_name=config.get("agency_name")
                             or "Analytiq", **kw)

    class _PageIndexDoc(BaseDocTemplate):
        """Records the page each section heading came to rest on.

        ReportLab only knows where a flowable landed once it has laid it
        out, which is why the contents needs two passes: one to find out,
        one to print it. The pass is cheap next to the analysis itself,
        and the alternative is a contents page that numbers the sections
        rather than locating them.
        """

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.section_pages: dict = {}

        def afterFlowable(self, flowable):
            title = getattr(flowable, "_toc_title", None)
            if title and title not in self.section_pages:
                self.section_pages[str(title)] = self.page

    doc = _PageIndexDoc(
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

    # ── Build TOC entries ─────────────────────────────────
    sec_num = 1
    toc     = []
    def _add_toc(title):
        nonlocal sec_num
        toc.append((sec_num, title, 0))
        sec_num += 1

    _add_toc("Report at a Glance")
    _add_toc("Executive Summary")
    _add_toc("Data Quality & Transparency Note")
    if governance is not None:
        _add_toc("Data Governance")
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
    if forecast is not None:
        _add_toc("Outlook")
    if has_deep_page(domain):
        _add_toc("{} Analysis".format(_domain_label(domain).title()))
    _add_toc("Dataset Overview & Descriptive Statistics")
    if stats_report:
        _add_toc("Statistical Analysis")
    if bi_report:
        _add_toc("Business Intelligence")
    if chart_data:
        # One numbered entry for the exhibits, with the individual charts
        # indented beneath it. Listed flat, each chart sat in the contents
        # as a peer of the executive summary.
        _add_toc("Exhibits")
        for entry in chart_data:
            t = entry[0]
            # Matches the section heading, which is the chart's own title
            # — a contents line that reads differently from the page it
            # points at makes the reader check they are in the right place.
            toc.append((None, _fit(str(t), 46), 1))
    _add_toc("Recommendations & Action Plan")
    _add_toc(APPENDIX_TITLE)

    # ── Assemble story ────────────────────────────────────
    def _assemble(page_index: dict) -> list:
        # Exhibit numbers are handed out during assembly and the story is
        # assembled twice — once to find the page numbers, once for real.
        # Without this reset the second pass carried on from where the
        # first left off and the report opened at "Exhibit 7".
        s["_exhibit"] = {"n": 0}
        story = []
        _toc(story, s, T, toc, CW, pages=page_index)
        story.append(PageBreak())
        _rest(story)
        return story

    def _rest(story):

        _exec_dashboard(story, s, T, df, profile, top_insights,
                        executive_summary, CW, top_cluster=top_cluster,
                        driver_result=predictive, avg_salary_k=avg_salary_k)
        story.append(PageBreak())

        _exec_summary(story, s, T, executive_summary,
                      findings, risks, opportunities, CW)
        story.append(PageBreak())

        _dq_note(story, s, T, df, profile, CW)
        story.append(PageBreak())

        if governance is not None:
            _governance_section(story, s, T, governance, CW,
                                integrity=integrity)
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

        if forecast is not None:
            _forecast_section(story, s, T, forecast, CW)
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

        if chart_data:
            _sec(story, s, T, "Exhibits",
                 "Each chart with the reading it supports")
            for i, entry in enumerate(chart_data, 1):
                title, img_bytes, narrative = entry[0], entry[1], entry[2]
                # The spec rides along when the caller has it, so the page
                # can print the figures the chart was drawn from.
                spec = entry[3] if len(entry) > 3 else None
                _chart_page(story, s, T, img_bytes, title, narrative, i, CW,
                            df=df, spec=spec)
                story.append(PageBreak())

        _recommendations(story, s, T, recommendations, CW,
                         insights=top_insights)
        story.append(PageBreak())

        _appendix(story, s, T, config, CW, domain=domain,
                  cleaning_summary=cleaning_summary,
                  source_table=config.get("source_table")
                  or "source_table")
    # ── Build PDF ─────────────────────────────────────────
    # First pass discovers where each heading landed; it is thrown away.
    # Contents rows are a fixed height whether or not they carry a page
    # number, so the second pass paginates identically to the first and
    # the numbers it prints are the ones the reader will find.
    probe = _PageIndexDoc(io.BytesIO(), pagesize=A4,
                          leftMargin=M, rightMargin=M,
                          topMargin=30*mm, bottomMargin=17*mm)
    probe.addPageTemplates([PageTemplate(
        id="main", frames=[Frame(M, 17*mm, CW, H - 47*mm,
                                 leftPadding=0, rightPadding=0,
                                 topPadding=0, bottomPadding=0)],
        onPage=lambda c, d: None)])
    try:
        probe.build(_assemble({}))
        page_index = probe.section_pages
    except Exception:
        # A contents without page numbers still lists the sections; a
        # failed report does not.
        logger.warning("could not resolve contents page numbers",
                       exc_info=True)
        page_index = {}

    doc.build(_assemble(page_index), canvasmaker=canvas_maker)
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
