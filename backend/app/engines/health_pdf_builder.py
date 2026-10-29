"""
core/health_pdf_builder.py
Health Report PDF builder — extracted from pages/11_Health_Report.py.
Single responsibility: given health dict + df, produce PDF bytes.
Call: build_health_pdf(df, niche, health, config) -> bytes
"""
from __future__ import annotations
import io
import os
import datetime
import logging
import tempfile
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, KeepTogether, HRFlowable,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfgen import canvas as rl_canvas
from pypdf import PdfReader, PdfWriter
from app.engines.pdf_primitives import truncate_label, is_id_col

logger = logging.getLogger(__name__)


def _clean_text(text: object) -> str:
    """Make engine-authored text safe and clean for a reportlab Paragraph.

    Engine strings carry markdown emphasis (**bold**) that reportlab does
    not understand and would render literally, and may contain &, < or >
    from column names — which reportlab parses as markup and which would
    otherwise raise or silently swallow the rest of the paragraph.
    """
    s = str(text or "")
    s = s.replace("**", "").replace(" ", " ")
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.strip()

def build_health_pdf(df: pd.DataFrame, niche: str, health: dict,
                     insights: list, fname: str,
                     agency_name: str = "Analytiq",
                     executive_summary: str = "",
                     key_findings: Optional[List[str]] = None,
                     risks: Optional[List[str]] = None,
                     opportunities: Optional[List[str]] = None,
                     actions: Optional[List[str]] = None) -> bytes:
    """Client-facing data health + business insights PDF report.

    The narrative arguments are optional and default to empty, so existing
    two-positional-argument callers keep working. When supplied they add
    the Executive Summary / Key Findings / Risks / Opportunities /
    Recommendations sections that make this read as a consulting
    deliverable rather than a list of alerts — and they carry analysis the
    insight cards alone drop (several domain analyses emit findings and
    risks but no card).
    """
    agency_name = (agency_name or "Analytiq")[:40]
    key_findings  = list(key_findings or [])
    risks         = list(risks or [])
    opportunities = list(opportunities or [])
    actions       = list(actions or [])
    import io as _io
    from reportlab.lib.colors import white, black
    from reportlab.platypus import (
        PageBreak, Image, CondPageBreak,
    )
    from reportlab.platypus.tableofcontents import TableOfContents
    from reportlab.pdfgen import canvas as CV
    from reportlab.lib import colors as rl_colors

    W, H = A4
    CW   = W - 36 * mm
    now  = datetime.datetime.now().strftime("%B %d, %Y")

    # The same palette as the main report and its charts (report_theme).
    from app.engines import report_theme
    from app.engines.chart_exporter import fmt
    T = report_theme.get("", niche)
    accent_hex  = T["accent"]
    accent      = HexColor(accent_hex)
    dark        = HexColor(T["primary"])
    ink         = HexColor(T["text"])
    gray        = HexColor(T["text_muted"])
    border      = HexColor(T["border"])
    light       = HexColor(T["bg_light"])
    light2      = HexColor(T["bg_card"])
    score_color = HexColor(health["color"])

    # Embedded Carlito, Helvetica only if the bundled files are missing
    # (see engines/pdf_fonts.py). The fonts were never committed, so every
    # report fell back to Helvetica, which has no rupee sign.
    from app.engines.pdf_fonts import FONT_BOLD as _BB, FONT_ITALIC as _BI, FONT_REG as _BF

    def ps(name, **kw): return ParagraphStyle(name, **kw)
    ST = {
        "h1":   ps("h1",   fontName=_BB, fontSize=15, textColor=dark,
                   spaceAfter=3, spaceBefore=2, leading=19, keepWithNext=1),
        "h2":   ps("h2",   fontName=_BB, fontSize=12, textColor=dark,
                   spaceBefore=10, spaceAfter=4, leading=15, keepWithNext=1),
        "h3":   ps("h3",   fontName=_BB, fontSize=10.5, textColor=dark,
                   spaceBefore=8, spaceAfter=3, leading=14, keepWithNext=1),
        "body": ps("body", fontName=_BF, fontSize=9.5, textColor=ink,
                   leading=14.5, spaceAfter=3, alignment=TA_LEFT),
        "sm":   ps("sm",   fontName=_BF, fontSize=8, textColor=gray,
                   leading=11, spaceAfter=2),
        "act":  ps("act",  fontName=_BB, fontSize=9, textColor=accent,
                   leading=13, spaceAfter=3),
        "ctr":  ps("ctr",  fontName=_BF, fontSize=9, textColor=dark,
                   alignment=TA_CENTER),
        "note": ps("note", fontName=_BI, fontSize=8, textColor=gray,
                   leading=11, spaceAfter=2),
    }

    buf = _io.BytesIO()

    # ── Canvas with header/footer ─────────────────────────
    class _Canvas(CV.Canvas):
        def __init__(self, fn, **kw):
            super().__init__(fn, **kw)
            self._sp = []
        def showPage(self):
            self._sp.append(dict(self.__dict__))
            self._startPage()
        def save(self):
            tot = len(self._sp)
            for state in self._sp:
                self.__dict__.update(state)
                self._draw_hf(tot)
                super().showPage()
            super().save()
        def _draw_cover(self, tot):
            """White cover with one band of the report colour, the grade,
            and who it is for — drawn on the canvas, outside the frame."""
            self.setFillColor(dark)
            self.rect(0, 0, 14 * mm, H, fill=1, stroke=0)
            self.setFillColor(accent)
            self.rect(14 * mm, 0, 1.2 * mm, H, fill=1, stroke=0)
            x0 = 30 * mm
            self.setFillColor(dark)
            self.setFont(_BB, 12)
            self.drawString(x0, H - 30 * mm, agency_name)
            self.setFillColor(accent)
            self.setFont(_BB, 8.5)
            self.drawString(x0, H - 92 * mm, T.get("domain_label", "ANALYTICS"))
            self.setFillColor(dark)
            self.setFont(_BB, 30)
            self.drawString(x0, H - 106 * mm, "Data Health &")
            self.drawString(x0, H - 120 * mm, "Business Insights")
            self.setFillColor(gray)
            self.setFont(_BF, 11)
            self.drawString(x0, H - 131 * mm, fname[:70])
            self.setStrokeColor(accent)
            self.setLineWidth(1.2)
            self.line(x0, H - 138 * mm, x0 + 40 * mm, H - 138 * mm)

            badge_y = H - 175 * mm
            self.setFillColor(score_color)
            self.circle(x0 + 14 * mm, badge_y, 14 * mm, fill=1, stroke=0)
            self.setFillColor(white)
            self.setFont(_BB, 19)
            self.drawCentredString(x0 + 14 * mm, badge_y - 2 * mm, str(health["score"]))
            self.setFont(_BF, 7)
            self.drawCentredString(x0 + 14 * mm, badge_y - 8 * mm, "/ 100")
            self.setFillColor(dark)
            self.setFont(_BB, 13)
            self.drawString(x0 + 34 * mm, badge_y + 3 * mm,
                            "Grade {} — {}".format(health["grade"], health["label"]))
            self.setFillColor(gray)
            self.setFont(_BF, 9)
            self.drawString(x0 + 34 * mm, badge_y - 4 * mm,
                            "{:,} rows · {} columns".format(health["rows"], health["cols"]))

            mx = x0
            for label, value in (("Dataset", fname[:34]), ("Date", now),
                                 ("Prepared by", agency_name)):
                self.setFillColor(gray)
                self.setFont(_BF, 8)
                self.drawString(mx, 30 * mm, label)
                self.setFillColor(ink)
                self.setFont(_BB, 10)
                self.drawString(mx, 24 * mm, value)
                mx += (W - x0 - 22 * mm) / 3
            self.setFillColor(gray)
            self.setFont(_BF, 7.5)
            self.drawString(x0, 14 * mm, "Confidential — prepared for the named client only.")

        def _draw_hf(self, tot):
            # The cover carries no running header/footer or page number,
            # and is not counted.
            if self._pageNumber == 1:
                self._draw_cover(tot)
                return
            self.setStrokeColor(border)
            self.setLineWidth(0.6)
            self.line(18*mm, H - 16*mm, W - 18*mm, H - 16*mm)
            self.setFont(_BB, 8)
            self.setFillColor(dark)
            self.drawString(18*mm, H - 13*mm, "Data Health & Business Insights")
            self.setFont(_BF, 7.5)
            self.setFillColor(gray)
            self.drawRightString(W - 18*mm, H - 13*mm, "{} · {}".format(fname[:40], now))
            self.line(18*mm, 13*mm, W - 18*mm, 13*mm)
            self.drawString(18*mm, 9*mm, "{} · Confidential".format(agency_name))
            self.drawRightString(W - 18*mm, 9*mm,
                                 "Page {} of {}".format(self._pageNumber - 1, tot - 1))

    class _Doc(BaseDocTemplate):
        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph) and flowable.style.name == "sec":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page - 1))

    doc = _Doc(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=22*mm, bottomMargin=18*mm,
        title="Data Health & Business Insights", author=agency_name,
    )
    frame = Frame(18*mm, 18*mm, CW, H - 40*mm, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=lambda c, d: None)])
    story = []

    # ══════════════════════════════════════════════════════
    # PAGE 1: COVER  (drawn on the canvas — see _draw_cover)
    # ══════════════════════════════════════════════════════
    # A single spacer reserves the page; every flowable below lands from
    # page 2 onward.
    story.append(Spacer(1, 1 * mm))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE 2: CONTENTS
    # ══════════════════════════════════════════════════════
    # Built from the same conditions that gate each section below, so it
    # can never advertise a section the report doesn't contain.
    # Only top-level sections are numbered. Key Findings / Risks /
    # Opportunities are subsections of the Executive Summary, so listing
    # them as peers would make the numbering disagree with the headings.
    # Business charts, generated up-front so the contents page knows whether
    # the section will exist. The chart engine excludes identifier and
    # constant columns, so this returns nothing rather than plotting an ID
    # when a dataset has no real measures.
    _charts = []
    try:
        from app.engines.chart_exporter import build_report_charts
        _charts = build_report_charts(df, domain=niche, max_charts=4)
    except Exception:
        logger.warning("chart generation failed for the health report",
                       exc_info=True)

    # The findings section carries the name the domain's reader uses, so
    # a workforce report does not open a section called "Meaningful
    # Business Insights" — a heading that tells the reader nothing and
    # reads as a template's default.
    from app.engines.report_blueprints import blueprint_for
    _insights_heading = "{} — Findings".format(blueprint_for(niche).label)

    _toc_entries = ["Data Health Overview"]
    _sub = [n for n, present in (("Key Findings", key_findings),
                                  ("Risks Identified", risks),
                                  ("Opportunities", opportunities)) if present]
    if executive_summary or _sub:
        _toc_entries.append("Executive Summary")
    if insights:
        _toc_entries.append(_insights_heading)
    if _charts:
        _toc_entries.append("Visual Analysis")
    _toc_entries += ["Descriptive Statistics", "Column Quality Analysis"]
    # The same columns decide whether the section exists and whether the
    # contents page lists it; they used to be two different tests, so a
    # dataset could be numbered for a section it never printed.
    _corr_cols = [c for c in df.select_dtypes(include="number").columns
                  if df[c].nunique() > 2 and not is_id_col(c, df[c])
                  and not pd.api.types.is_bool_dtype(df[c])][:8]
    if len(_corr_cols) >= 2:
        _toc_entries.append("Correlation Analysis")
    if actions:
        _toc_entries.append("Recommended Actions")

    _sec_style = ParagraphStyle("sec", parent=ST["h1"])

    def _section(name: str):
        """A section heading: starts a new page only when little room is
        left, and is listed with its page number on the contents page."""
        try:
            num = "{:02d}&nbsp;&nbsp;".format(_toc_entries.index(name) + 1)
        except ValueError:
            num = ""
        story.append(CondPageBreak(50 * mm))
        return Paragraph('<font color="{}">{}</font>{}'.format(accent_hex, num, name),
                         _sec_style)

    story.append(Paragraph("Contents", ParagraphStyle("toc_t", parent=ST["h1"],
                                                      keepWithNext=0)))
    story.append(HRFlowable(width="100%", thickness=0.8, color=accent, spaceAfter=8))
    _toc = TableOfContents()
    _toc.dotsMinLevel = 0
    _toc.levelStyles = [ParagraphStyle("toc0", fontName=_BF, fontSize=10.5, leading=20,
                                       textColor=ink)]
    story.append(_toc)
    # The basis of preparation sits under the contents. At the end it was
    # pushed onto a page of its own: three lines on an otherwise blank page.
    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=gray, spaceAfter=4))
    story.append(Paragraph(
        "<b>Basis of preparation.</b> Prepared by {} on {}. Source: {}. "
        "All figures derive solely from the supplied dataset and describe the "
        "period it covers. Statistical association is reported where present; "
        "it does not establish causation. Recommendations assume the data is "
        "complete and accurate as supplied.".format(
            _clean_text(agency_name), now, _clean_text(fname[:40])),
        ST["sm"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE 3: HEALTH SCORE
    # ══════════════════════════════════════════════════════
    story.append(Spacer(1, 6*mm))
    story.append(_section("Data Health Overview"))
    story.append(Paragraph(_clean_text(fname[:70]), ST["sm"]))
    story.append(HRFlowable(width="100%", thickness=2.5, color=accent, spaceAfter=6))
    story.append(Paragraph(
        "Generated: {}  ·  {:,} rows  ·  {} columns  ·  Domain: <b>{}</b>".format(
            now, health["rows"], health["cols"], niche.upper()),
        ST["sm"]))
    story.append(Spacer(1, 6*mm))

    # Health score KPI box
    # FIX: fontSize 36 → 28 + explicit row height to prevent score bleeding into footer
    story.append(Paragraph("Overall Data Health Score", ST["h2"]))
    score_para = Paragraph(
        "<b>{}<font size=13>/100</font></b>".format(health["score"]),
        ParagraphStyle("sc", fontName=_BB, fontSize=26, textColor=score_color,
                       alignment=TA_CENTER, leading=30))
    grade_para = Paragraph(
        "<b>Grade: {}  —  {}</b>".format(health["grade"], health["label"]),
        ParagraphStyle("gr", fontName=_BB, fontSize=10,
                       textColor=score_color, alignment=TA_CENTER, leading=13))

    kpi_row = [
        [score_para, grade_para,
         Paragraph("Missing: <b>{}%</b>".format(health["missing_pct"]),
                   ParagraphStyle("kv", fontName=_BF, fontSize=9, textColor=dark, alignment=TA_CENTER)),
         Paragraph("Duplicates: <b>{}%</b>".format(health["dup_pct"]),
                   ParagraphStyle("kv", fontName=_BF, fontSize=9, textColor=dark, alignment=TA_CENTER)),
         Paragraph("Outlier cols: <b>{}%</b>".format(health["outlier_pct"]),
                   ParagraphStyle("kv", fontName=_BF, fontSize=9, textColor=dark, alignment=TA_CENTER)),
        ]
    ]
    # FIX: explicit rowHeights prevents text from escaping table bounds and
    # overlapping with the page-number circle drawn at 5.5 mm in the footer
    kpi_tbl = Table(kpi_row, colWidths=[CW*x for x in [0.24,0.24,0.173,0.173,0.174]],
                    rowHeights=[58])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,0), light),
        ("BACKGROUND",    (1,0),(1,0), HexColor("#EFF6FF")),
        ("BACKGROUND",    (2,0),(-1,0), light2),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("BOX",           (0,0),(-1,-1), 1.5, accent),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, HexColor("#E5E7EB")),
        ("LINEBELOW",     (0,0),(-1,0),  2, accent),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 5*mm))

    # Dataset summary table
    story.append(Paragraph("Dataset Summary", ST["h3"]))
    num_cols_list = df.select_dtypes(include="number").columns.tolist()
    cat_cols_list = df.select_dtypes(include=["object","string"]).columns.tolist()
    date_cols_list= df.select_dtypes(include="datetime").columns.tolist()
    missing_total = df.isna().sum().sum()
    dup_count     = df.duplicated().sum()

    # Header text in white: it was the dark body style on the dark band, so
    # the header row printed as an empty bar.
    _hdr = ParagraphStyle("sumh", fontName=_BB, fontSize=8, textColor=white)
    summary_data = [
        [Paragraph("Measure", _hdr),
         Paragraph("Value", ParagraphStyle("sumhr", parent=_hdr, alignment=TA_RIGHT))],
        ["Rows",             "{:,}".format(health["rows"])],
        ["Columns",          str(health["cols"])],
        ["Numeric columns",  str(len(num_cols_list))],
        ["Text columns",     str(len(cat_cols_list))],
        ["Date columns",     str(len(date_cols_list))],
        ["Missing values",   "{:,} ({:.1f}%)".format(int(missing_total), health["missing_pct"])],
        ["Duplicate rows",   "{:,} ({:.1f}%)".format(int(dup_count), health["dup_pct"])],
    ]
    for i in range(1, len(summary_data)):
        summary_data[i] = [
            Paragraph(str(summary_data[i][0]), ST["sm"]),
            Paragraph(str(summary_data[i][1]),
                      ParagraphStyle("sv", fontName=_BB, fontSize=8.5,
                                     textColor=dark, alignment=TA_RIGHT))
        ]

    sum_tbl = Table(summary_data, colWidths=[CW*0.6, CW*0.4])
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  dark),
        ("TEXTCOLOR",     (0,0),(-1,0),  white),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
        ("ALIGN",         (1,0),(-1,-1), "RIGHT"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("BOX",           (0,0),(-1,-1), 0.5, HexColor("#E5E7EB")),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, HexColor("#E5E7EB")),
    ]))
    story.append(sum_tbl)

    # ══════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY + NARRATIVE SECTIONS
    # ══════════════════════════════════════════════════════
    # Skipped entirely when no narrative was supplied, so the report never
    # shows an empty heading.
    if executive_summary or key_findings or risks or opportunities:
        story.append(_section("Executive Summary"))
        story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=6))

        if executive_summary:
            story.append(Paragraph(_clean_text(executive_summary), ST["body"]))
            story.append(Spacer(1, 5 * mm))

        def _bullet_section(heading: str, items: list, bullet_hex: str,
                            max_items: int = 8) -> None:
            if not items:
                return
            story.append(Paragraph(heading, ST["h2"]))
            rows = []
            for item in items[:max_items]:
                marker = Paragraph(
                    '<font color="{}">•</font>'.format(bullet_hex),
                    ParagraphStyle("bm", fontName=_BF, fontSize=7.5, leading=13))
                text = Paragraph(_clean_text(item),
                                 ParagraphStyle("bt", fontName=_BF, fontSize=9.5,
                                                textColor=dark, leading=14))
                rows.append([marker, text])
            tbl = Table(rows, colWidths=[6 * mm, CW - 6 * mm])
            tbl.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING",    (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 4 * mm))

        _bullet_section("Key Findings", key_findings, accent_hex)
        _bullet_section("Risks Identified", risks, T["negative"])
        _bullet_section("Opportunities", opportunities, T["positive"])

    # ══════════════════════════════════════════════════════
    # PAGE 2: BUSINESS INSIGHTS
    # ══════════════════════════════════════════════════════
    if insights:
        story.append(_section(_insights_heading))
        story.append(Paragraph(
            "What the data shows, why it matters, and what to check next. Every "
            "figure is computed from the uploaded dataset.", ST["body"]))
        story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=6))

    # Severity shows in the label and a narrow rule only. Body and action
    # text were set in the severity colour on a tinted fill — red on pink
    # for a whole card — which read as alarm rather than analysis.
    SEV = {"critical": (T["negative"], "High priority"),
           "warning":  (T["warning"],  "Watch"),
           "positive": (T["positive"], "Strength"),
           "info":     (T["info"],     "Context")}
    lbl_st  = ParagraphStyle("it", fontName=_BB, fontSize=7.5, leading=10, spaceAfter=2)
    ttl_st  = ParagraphStyle("itl", fontName=_BB, fontSize=10.5, textColor=dark,
                             spaceAfter=3, leading=14)
    body_st = ParagraphStyle("ib", fontName=_BF, fontSize=9.3, textColor=ink,
                             leading=13.8, spaceAfter=4)
    act_st  = ParagraphStyle("ia", parent=body_st, spaceAfter=0)
    from app.engines.pdf_builder import _steps
    for i, ins in enumerate(insights, 1):
        sev_hex, sev_word = SEV.get(ins["severity"], SEV["info"])
        tag_p = Paragraph('<font color="{}">{}</font>&nbsp;&nbsp;<font color="{}">{}</font>'.format(
            sev_hex, sev_word.upper(), T["text_muted"], _clean_text(ins["tag"].title())), lbl_st)
        title_p = Paragraph("{}. {}".format(i, _clean_text(ins["title"])), ttl_st)
        body_p = Paragraph(_clean_text(ins["body"].replace("**", "")), body_st)
        action_p = Paragraph("<b>Next step:</b> {}".format(
            _steps(str(ins["action"]).replace("**", ""))), act_st)

        card = Table([[tag_p], [title_p], [body_p], [action_p]], colWidths=[CW])
        card.setStyle(TableStyle([
            ("LINEBEFORE",    (0, 0), (0, -1), 2.5, HexColor(sev_hex)),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING",    (0, 0), (-1, 0), 7),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("BACKGROUND",    (0, 0), (-1, -1), light2),
        ]))
        story.append(KeepTogether([card, Spacer(1, 4*mm)]))

    # ══════════════════════════════════════════════════════
    # VISUAL ANALYSIS
    # ══════════════════════════════════════════════════════
    # The report previously carried only thumbnail histograms and a
    # correlation heatmap — diagnostics, not the business picture a client
    # reads. These are the same charts as the main report, each captioned
    # with what it shows rather than left to speak for itself.
    if _charts:
        story.append(_section("Visual Analysis"))
        story.append(Paragraph("Each caption is computed from the figures plotted.",
                               ST["sm"]))
        story.append(HRFlowable(width="100%", thickness=0.8, color=accent,
                                 spaceAfter=6))

        from PIL import Image as _PIL
        for idx, chart in enumerate(_charts, 1):
            try:
                w_px, h_px = _PIL.open(_io.BytesIO(chart.png)).size
                width, height = CW, CW * h_px / w_px
                if height > 105 * mm:
                    height, width = 105 * mm, 105 * mm * w_px / h_px
                img = Image(_io.BytesIO(chart.png), width=width, height=height)
                img.hAlign = "LEFT"
                story.append(KeepTogether([
                    Paragraph("{}. {}".format(idx, _clean_text(chart.title)), ST["h3"]),
                    img,
                    Paragraph(_clean_text(chart.caption), ST["body"]),
                    Spacer(1, 5 * mm),
                ]))
            except Exception:
                logger.warning("could not place chart %r in the health report",
                               chart.title, exc_info=True)

    # ══════════════════════════════════════════════════════
    # PAGE 3: DESCRIPTIVE STATISTICS
    # ══════════════════════════════════════════════════════
        story.append(_section("Descriptive Statistics"))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=5))

    # One row per measure, figures formatted as a reader writes them. The
    # old table printed 59333.386, and the histograms under it included
    # employee_id.
    _measures = [c for c in num_cols_list if not is_id_col(c, df[c])
                 and not pd.api.types.is_bool_dtype(df[c])]
    if _measures:
        th = ParagraphStyle("dth", fontName=_BB, fontSize=8, textColor=white,
                            alignment=TA_RIGHT)
        hdr = [Paragraph("Measure", ParagraphStyle("dth0", parent=th, alignment=TA_LEFT))] + \
              [Paragraph(h, th) for h in ("Count", "Mean", "Median", "Min", "25th pct",
                                          "75th pct", "Max")]
        rows = [hdr]
        num_r = ParagraphStyle("dtr", fontName=_BF, fontSize=8.2, textColor=ink,
                               alignment=TA_RIGHT)
        for c in _measures[:10]:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if v.empty:
                continue
            q = v.quantile([0.25, 0.5, 0.75])
            rows.append([Paragraph(_clean_text(c), ParagraphStyle(
                             "dtl", fontName=_BF, fontSize=8.2, textColor=ink))] +
                        [Paragraph(x, num_r) for x in (
                            fmt(len(v)), fmt(v.mean()), fmt(q[0.5]), fmt(v.min()),
                            fmt(q[0.25]), fmt(q[0.75]), fmt(v.max()))])
        stat_tbl = Table(rows, colWidths=[CW * x for x in
                                          (0.24, 0.10, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11)],
                         repeatRows=1)
        stat_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  dark),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
            ("LINEBELOW",     (0,0),(-1,-1), 0.3, border),
        ]))
        story.append(stat_tbl)
        story.append(Spacer(1, 6*mm))

    # ══════════════════════════════════════════════════════
    # PAGE 4: COLUMN QUALITY TABLE
    # ══════════════════════════════════════════════════════
        story.append(_section("Column Quality Analysis"))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=5))
    story.append(Paragraph(
        "Completeness, distinct values and range for each column. Columns more "
        "than 5% empty, or holding a single value, need attention before "
        "analysis.",
        ST["body"]))
    story.append(Spacer(1, 4*mm))

    th_st2 = ParagraphStyle("th2", fontName=_BB, fontSize=8,
                             textColor=white, alignment=TA_CENTER)
    td_st2 = ParagraphStyle("td2", fontName=_BF, fontSize=8, textColor=dark)
    td_c2  = ParagraphStyle("tc2", fontName=_BF, fontSize=8, textColor=dark,
                             alignment=TA_CENTER)

    hdr2  = [Paragraph(h, th_st2) for h in
             ["Column", "Type", "Missing%", "Unique", "Min", "Max", "Sample Value", "Status"]]
    rows2 = []
    for col in df.columns[:25]:
        sc = df[col]
        miss = "{:.1f}%".format(sc.isna().mean()*100)
        uniq = "{:,}".format(sc.nunique())
        sample_val = str(sc.dropna().iloc[0])[:20] if len(sc.dropna()) > 0 else "—"

        if pd.api.types.is_bool_dtype(sc):
            mn = str(bool(sc.dropna().min())) if len(sc.dropna()) > 0 else "—"
            mx = str(bool(sc.dropna().max())) if len(sc.dropna()) > 0 else "—"
        elif pd.api.types.is_numeric_dtype(sc):
            mn = fmt(float(sc.dropna().min())) if len(sc.dropna()) > 0 else "—"
            mx = fmt(float(sc.dropna().max())) if len(sc.dropna()) > 0 else "—"
            if len(sc.dropna()) > 0 and not is_id_col(col, sc):
                sample_val = fmt(float(sc.dropna().iloc[0]))
        elif pd.api.types.is_datetime64_any_dtype(sc):
            mn = sc.dropna().min().strftime("%Y-%m-%d") if len(sc.dropna()) > 0 else "—"
            mx = sc.dropna().max().strftime("%Y-%m-%d") if len(sc.dropna()) > 0 else "—"
            sample_val = str(sc.dropna().iloc[0])[:10] if len(sc.dropna()) > 0 else "—"
        else:
            mn = "—"
            mx = "—"

        # Status
        miss_f = sc.isna().mean()*100
        if miss_f > 20:
            status = "HIGH MISSING"
            st_c   = HexColor("#DC2626")
        elif miss_f > 5:
            status = "REVIEW"
            st_c   = HexColor("#D97706")
        elif sc.nunique() == 1:
            status = "CONSTANT"
            st_c   = HexColor("#DC2626")
        elif is_id_col(col, sc):
            # The shared identifier test. "Every value different" also
            # described hire_date and every continuous measure.
            status = "IDENTIFIER"
            st_c   = HexColor("#2563EB")
        else:
            status = "OK"
            st_c   = HexColor("#059669")

        rows2.append([
            # Every cell is data, so every cell is escaped: a value such as
            # "<b>&amp;" crashed the whole report as unbalanced markup.
            Paragraph(_clean_text(truncate_label(col, 20)), td_st2),
            Paragraph(_clean_text(str(sc.dtype)[:8]), td_c2),
            Paragraph(miss, ParagraphStyle("mv", fontName=_BB, fontSize=8,
                           textColor=HexColor("#DC2626") if miss_f > 5 else dark,
                           alignment=TA_CENTER)),
            Paragraph(uniq, td_c2),
            Paragraph(_clean_text(mn), td_c2),
            Paragraph(_clean_text(mx), td_c2),
            Paragraph(_clean_text(sample_val), td_st2),
            Paragraph(status, ParagraphStyle("sv3", fontName=_BB, fontSize=7.5,
                                             textColor=st_c, alignment=TA_CENTER)),
        ])

    # "Unique" needs 0.09 width minimum — at 0.07 the header wrapped
    # mid-word ("Uniqu / e") in the rendered PDF.
    col_tbl2 = Table([hdr2] + rows2,
                     colWidths=[CW*x for x in [0.19,0.08,0.08,0.08,0.12,0.12,0.18,0.15]])
    col_tbl2.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  dark),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("BOX",           (0,0),(-1,-1), 0.5, HexColor("#E5E7EB")),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, HexColor("#E5E7EB")),
    ]))
    story.append(col_tbl2)
    if len(df.columns) > 25:
        story.append(Paragraph(
            "The first 25 of {} columns are listed.".format(len(df.columns)), ST["sm"]))

    # ══════════════════════════════════════════════════════
    # PAGE 5: CORRELATION + DISCLAIMER
    # ══════════════════════════════════════════════════════
    if len(_corr_cols) >= 2:
        story.append(_section("Correlation Analysis"))
        story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=5))
        story.append(Paragraph(
            "Pairs of measures that move together (Spearman rank correlation). "
            "Moving together is association; it does not show that one drives "
            "the other.", ST["body"]))
        story.append(Spacer(1, 3*mm))
        try:
            corr_cols = _corr_cols
            if len(corr_cols) >= 2:
                corr = df[corr_cols].corr(method="spearman")
                pairs = []
                for ri in range(len(corr_cols)):
                    for ci in range(ri + 1, len(corr_cols)):
                        r_val = float(corr.iloc[ri, ci])
                        if abs(r_val) > 0.2:
                            pairs.append((corr_cols[ri], corr_cols[ci], r_val))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                if pairs:
                    thl = ParagraphStyle("cth", fontName=_BB, fontSize=8, textColor=white)
                    thr = ParagraphStyle("cthr", parent=thl, alignment=TA_RIGHT)
                    tdl = ParagraphStyle("ctd", fontName=_BF, fontSize=8.2, textColor=ink)
                    tdr = ParagraphStyle("ctdr", parent=tdl, alignment=TA_RIGHT)
                    corr_rows = [[Paragraph("Measure", thl), Paragraph("With", thl),
                                  Paragraph("r", thr), Paragraph("Variance shared", thr),
                                  Paragraph("Strength", thl)]]
                    for ca, cb, rv in pairs[:8]:
                        strength = ("Strong" if abs(rv) > 0.6 else "Moderate"
                                    if abs(rv) > 0.4 else "Weak")
                        corr_rows.append([
                            Paragraph(_clean_text(ca), tdl), Paragraph(_clean_text(cb), tdl),
                            Paragraph("{:+.2f}".format(rv), tdr),
                            Paragraph("{:.0f}%".format(rv * rv * 100), tdr),
                            Paragraph("{} {}".format(strength, "positive" if rv > 0
                                                     else "negative"), tdl)])
                    corr_tbl = Table(corr_rows, colWidths=[CW * x for x in
                                                          (0.26, 0.26, 0.12, 0.16, 0.20)])
                    corr_tbl.setStyle(TableStyle([
                        ("BACKGROUND",    (0,0),(-1,0),  dark),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
                        ("TOPPADDING",    (0,0),(-1,-1), 4),
                        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                        ("LEFTPADDING",   (0,0),(-1,-1), 6),
                        ("RIGHTPADDING",  (0,0),(-1,-1), 6),
                        ("LINEBELOW",     (0,0),(-1,-1), 0.3, border),
                    ]))
                    story.append(corr_tbl)
                else:
                    story.append(Paragraph(
                        "No pair of measures is correlated beyond |r| = 0.2: each "
                        "varies independently of the others.", ST["body"]))
        except Exception:
            logger.warning("Health Report section failure", exc_info=True)

    # ══════════════════════════════════════════════════════
    # RECOMMENDED ACTIONS
    # ══════════════════════════════════════════════════════
    # Closing a client report on what to DO — not on a correlation table —
    # is what separates a deliverable from a data dump.
    if actions:
        story.append(_section("Recommended Actions"))
        story.append(Paragraph(
            "Follow-ups the findings support. Each rests on this dataset alone; "
            "check it against what the data cannot show before acting.", ST["body"]))
        story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=6))

        rows = []
        for idx, act in enumerate(actions[:10], 1):
            num = Paragraph(
                '<font color="{}"><b>{}</b></font>'.format(accent_hex, idx),
                ParagraphStyle("an", fontName=_BB, fontSize=11,
                               alignment=TA_CENTER, leading=15))
            body = Paragraph(_clean_text(act),
                             ParagraphStyle("ab", fontName=_BF, fontSize=9.5,
                                            textColor=dark, leading=14.5))
            rows.append([num, body])

        act_tbl = Table(rows, colWidths=[10 * mm, CW - 10 * mm])
        act_tbl.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 2),
            ("LINEBELOW",     (0, 0), (-1, -2), 0.4, HexColor("#E5E7EB")),
            ("BACKGROUND",    (0, 0), (-1, -1), light2),
            ("BOX",           (0, 0), (-1, -1), 0.5, HexColor("#E5E7EB")),
        ]))
        story.append(act_tbl)

    # A section heading's subtitle and rule belong with what follows them;
    # without this "Visual Analysis" and its subtitle sat at the foot of a
    # page with every chart overleaf.
    for i, f in enumerate(story):
        if isinstance(f, Paragraph) and f.style.name == "sec":
            for g in story[i + 1:i + 3]:
                if isinstance(g, (Paragraph, HRFlowable)):
                    g.keepWithNext = 1

    from app.engines.pdf_builder import _bind_headings
    doc.multiBuild(_bind_headings(story), canvasmaker=_Canvas)
    buf.seek(0)
    return buf.read()



# ══════════════════════════════════════════════════════════

