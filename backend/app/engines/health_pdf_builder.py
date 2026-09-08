"""
core/health_pdf_builder.py
Health Report PDF builder — extracted from pages/11_Health_Report.py.
Single responsibility: given health dict + df, produce PDF bytes.
Call: build_health_pdf(df, niche, health, config) -> bytes
"""
from __future__ import annotations
import datetime
import logging
from typing import List, Optional

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, KeepTogether, HRFlowable,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from app.engines.pdf_primitives import truncate_label, is_id_col
from app.engines.present import label as _present_label, num as _present_num

logger = logging.getLogger(__name__)


# ── Presentation helpers ──────────────────────────────────
# The health report used to print whatever pandas handed it: raw column
# names (unit_price), raw dtypes (float64, int64, str) and raw floats
# (count 1500.0, mean 2557.499, max 9916.48). Each is correct and none
# belongs in a document a client reads. These route through the same
# present.* layer the rest of the product already uses, so a number
# looks the same in the app, the PDF and the deck.

# Pandas dtype -> what the column actually holds. A finance lead reading
# "float64" learns nothing; "Number" is the fact they wanted.
_DTYPE_WORDS = (
    ("datetime", "Date"), ("timedelta", "Duration"), ("period", "Date"),
    ("bool", "Yes / No"), ("category", "Category"),
    ("int", "Whole number"), ("float", "Number"),
    ("object", "Text"), ("string", "Text"), ("str", "Text"),
)


def _dtype_word(dtype: object) -> str:
    """Plain-language name for a pandas dtype."""
    d = str(dtype).lower()
    for needle, word in _DTYPE_WORDS:
        if needle in d:
            return word
    return "Text"


def _pretty_col(name: object) -> str:
    """Column name as a human label — 'unit_price' -> 'Unit Price'."""
    return _present_label(name)


# Which describe() rows are counts, not measurements. Printing a count
# with decimals ("count 1500.0") is the tell of a report nobody read.
_COUNT_STATS = {"count", "unique", "freq"}


def _stat_text(stat: object, val: object) -> str:
    """One cell of the describe() table, formatted for a reader."""
    name = str(stat).strip().lower()
    if name in _COUNT_STATS:
        try:
            return "{:,}".format(int(float(val)))
        except (TypeError, ValueError):
            return _present_num(val)
    return _present_num(val)


# "1. Decide which end is desirable 2. Check the gap persists 3. Pilot
# the practices" is three instructions, and printing it as one paragraph
# buries two of them mid-line where no reader will find them.
_STEP_SPLIT = __import__("re").compile(r"(?:(?<=^)|(?<=[.\s]))(?=\d{1,2}\.\s+[A-Z])")


def _action_steps(action: object) -> list:
    """One numbered instruction per line, or the whole string if it is
    not a numbered list."""
    text = str(action or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _STEP_SPLIT.split(text) if p.strip()]
    return parts if len(parts) > 1 else [text]


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
    from reportlab.lib.colors import white
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.platypus import (
        PageBreak, Image, CondPageBreak,
    )

    # Sections used to start on a fresh page unconditionally, all nine of
    # them, so a table half a page tall left the other half blank and an
    # eleven-page report carried perhaps six pages of content. A partner
    # deck breaks at PARTS, not at every heading. CondPageBreak asks for
    # enough room to start the section properly and otherwise lets it
    # follow the one above — the reader gets a document that flows
    # instead of a slide deck printed onto A4.
    #
    # The heights are the space a section needs before its heading stops
    # being an orphan: a heading, its standfirst, and enough of the
    # content below to be worth turning to.
    ROOM_TEXT  = 70 * mm    # a heading and several paragraphs
    ROOM_TABLE = 95 * mm    # a heading and a readable slice of a table
    ROOM_CHART = 150 * mm   # a chart is not worth splitting
    from reportlab.pdfgen import canvas as CV
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W, H = A4
    CW   = W - 36 * mm
    now  = datetime.datetime.now().strftime("%B %d, %Y")

    # One accent, for every domain. This used to be a lookup that
    # repainted the whole report green for a sales file and orange for an
    # e-commerce one, so two reports from the same product did not look
    # like the same product. The domain is named in words on the cover;
    # it does not need its own livery.
    from app.engines.palette import PRINT, STATUS_LIGHT, grade_color

    accent_hex  = PRINT["accent"]
    accent      = HexColor(accent_hex)
    dark        = HexColor(PRINT["ground"])
    gray        = HexColor(PRINT["muted"])
    light       = HexColor(PRINT["surface_tint"])
    light2      = HexColor(PRINT["surface_alt"])
    rule_c      = HexColor(PRINT["rule"])
    # The score badge used to take health["color"], a mint #22d3a5 chosen
    # for the dark dashboard. On white paper, beside a blue rule, it read
    # as a colour from a different document — because it was.
    score_color = HexColor(grade_color(health["grade"]))

    # ── Premium fonts — fall back to Helvetica if not found ──────────────
    import os as _os
    from reportlab.pdfbase import pdfmetrics as _pm
    from reportlab.pdfbase.ttfonts import TTFont as _TTF

    # health_pdf_builder sits at backend/app/engines/, so the backend root
    # — where assets/ lives — is three levels up. The old two-level walk
    # resolved to backend/app/assets, which does not exist, so this
    # silently fell back to Helvetica on every run.
    from app.engines.pdf.theme import _FONT_DIR

    _BF, _BB, _BI = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
    _FONTS = [
        ("HDF-Reg",    "Carlito-Regular.ttf",    "_BF"),
        ("HDF-Bold",   "Carlito-Bold.ttf",        "_BB"),
        ("HDF-Italic", "Carlito-Italic.ttf",      "_BI"),
    ]
    for alias, fname_f, var in _FONTS:
        font_path = _os.path.join(_FONT_DIR, fname_f)
        if not _os.path.exists(font_path):
            logger.warning("Font not found at %s — using Helvetica fallback", font_path)
            continue
        try:
            _pm.registerFont(_TTF(alias, font_path))
            if alias == "HDF-Reg":    _BF = alias
            if alias == "HDF-Bold":   _BB = alias
            if alias == "HDF-Italic": _BI = alias
        except Exception:
            logger.warning("Font registration failed for %s", alias, exc_info=True)

    def ps(name, **kw): return ParagraphStyle(name, **kw)
    ST = {
        "h1":   ps("h1",   fontName=_BB, fontSize=17, textColor=accent,
                   spaceAfter=4, spaceBefore=2, leading=21),
        "h2":   ps("h2",   fontName=_BB, fontSize=13, textColor=dark,
                   spaceBefore=10, spaceAfter=4, leading=16),
        "h3":   ps("h3",   fontName=_BB, fontSize=10.5, textColor=accent,
                   spaceBefore=8, spaceAfter=3, leading=14),
        "body": ps("body", fontName=_BF, fontSize=9.5, textColor=dark,
                   leading=15, spaceAfter=3, alignment=TA_JUSTIFY),
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
            """Full-bleed cover. Drawn on the canvas rather than as
            flowables so it can ignore the frame margins and the running
            header/footer, the way a real report cover does."""
            self.setFillColor(dark)
            self.rect(0, 0, W, H, fill=1, stroke=0)
            # accent bar down the left edge
            self.setFillColor(accent)
            self.rect(0, 0, 6 * mm, H, fill=1, stroke=0)

            self.setFillColor(white)
            self.setFont(_BB, 11)
            self.drawString(22 * mm, H - 28 * mm, agency_name.upper())
            self.setFillColor(accent)
            self.rect(22 * mm, H - 32 * mm, 24 * mm, 0.8 * mm, fill=1, stroke=0)

            # Title
            self.setFillColor(white)
            self.setFont(_BB, 30)
            self.drawString(22 * mm, H - 92 * mm, "Data Health &")
            self.drawString(22 * mm, H - 106 * mm, "Business Insights")
            self.setFillColor(HexColor(PRINT["accent_soft"]))
            self.setFont(_BF, 13)
            self.drawString(22 * mm, H - 120 * mm, "Analysis Report")

            # Grade badge
            badge_y = H - 165 * mm
            self.setFillColor(score_color)
            self.circle(34 * mm, badge_y, 17 * mm, fill=1, stroke=0)
            self.setFillColor(white)
            self.setFont(_BB, 21)
            self.drawCentredString(34 * mm, badge_y - 3 * mm, str(health["score"]))
            self.setFont(_BF, 7)
            self.drawCentredString(34 * mm, badge_y - 11 * mm, "/ 100")

            self.setFillColor(white)
            self.setFont(_BB, 14)
            self.drawString(58 * mm, badge_y + 4 * mm,
                            "Grade {} — {}".format(health["grade"], health["label"]))
            self.setFillColor(HexColor("#9FA9B8"))
            self.setFont(_BF, 9)
            self.drawString(58 * mm, badge_y - 4 * mm,
                            "{:,} rows  ·  {} columns  ·  {} domain".format(
                                health["rows"], health["cols"], niche))

            # Meta block
            self.setFillColor(HexColor("#9FA9B8"))
            self.setFont(_BF, 8.5)
            meta_y = 52 * mm
            for label, value in (("DATASET", fname[:52]),
                                 ("PREPARED", now),
                                 ("PREPARED BY", agency_name)):
                self.setFillColor(HexColor("#7A8798"))
                self.setFont(_BF, 6.5)
                self.drawString(22 * mm, meta_y, label)
                self.setFillColor(white)
                self.setFont(_BB, 9)
                self.drawString(22 * mm, meta_y - 5 * mm, value)
                meta_y -= 13 * mm

            self.setFillColor(HexColor("#5C6979"))
            self.setFont(_BF, 6.5)
            self.drawString(22 * mm, 14 * mm,
                            "CONFIDENTIAL  ·  Findings derive solely from the supplied dataset "
                            "and the period it covers")

        def _draw_hf(self, tot):
            # The cover carries no running header/footer or page number.
            if self._pageNumber == 1:
                self._draw_cover(tot)
                return
            # Header
            self.setFillColor(dark)
            self.rect(0, H - 20*mm, W, 20*mm, fill=1, stroke=0)
            self.setFillColor(accent)
            self.rect(0, H - 21*mm, W, 1*mm, fill=1, stroke=0)
            self.setFillColor(accent)
            self.rect(0, H - 20*mm, 3*mm, 20*mm, fill=1, stroke=0)
            self.setFillColor(white)
            self.setFont(_BB, 9.5)
            self.drawString(8*mm, H - 11*mm, f"{agency_name}  ·  Data Health & Business Insights")
            self.setFont(_BF, 7.5)
            self.setFillColor(HexColor(PRINT["accent_soft"]))
            self.drawString(8*mm, H - 17.5*mm, fname[:60])
            self.setFillColor(white)
            self.drawRightString(W - 8*mm, H - 11*mm, now)
            self.setFont(_BF, 7)
            self.drawRightString(W - 8*mm, H - 17.5*mm, "CONFIDENTIAL")
            # Footer
            self.setFillColor(dark)
            self.rect(0, 0, W, 11*mm, fill=1, stroke=0)
            self.setFillColor(accent)
            self.rect(0, 11*mm, W, 0.8*mm, fill=1, stroke=0)
            self.setFillColor(white)
            self.setFont(_BF, 6.5)
            self.drawString(8*mm, 4*mm, f"{agency_name}  ·  Confidential  ·  Prepared for the named recipient")
            # Page circle
            self.setFillColor(accent)
            self.circle(W - 13*mm, 5.5*mm, 4.5*mm, fill=1, stroke=0)
            self.setFillColor(white)
            self.setFont(_BB, 6.5)
            self.drawCentredString(W - 13*mm, 3.8*mm, "{}/{}".format(self._pageNumber, tot))

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=26*mm, bottomMargin=17*mm,
    )
    frame = Frame(18*mm, 17*mm, CW, H - 43*mm, id="main")
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
        from app.engines.chart_exporter import generate_all_charts
        _charts = generate_all_charts(df, theme_name="Corporate Light",
                                       max_charts=4)
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
    if len(df.select_dtypes(include="number").columns) >= 2:
        _toc_entries.append("Correlation Analysis")
    if actions:
        _toc_entries.append("Recommended Actions")

    def _section(name: str):
        """Numbered section heading that stays in step with the contents
        page, because both read from the same list."""
        try:
            num = _toc_entries.index(name) + 1
        except ValueError:
            return Paragraph(name, ST["h1"])
        return Paragraph(
            '<font color="{}">{:02d}</font>&nbsp;&nbsp;{}'.format(accent_hex, num, name),
            ST["h1"])

    story.append(Paragraph("Contents", ST["h1"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=8))
    _toc_rows = []
    for _i, _name in enumerate(_toc_entries, 1):
        _toc_rows.append([
            Paragraph('<font color="{}"><b>{:02d}</b></font>'.format(accent_hex, _i),
                      ParagraphStyle("tn", fontName=_BB, fontSize=9.5, leading=15)),
            Paragraph(_name, ParagraphStyle("tt", fontName=_BF, fontSize=10.5,
                                            textColor=dark, leading=15)),
        ])
    _toc_tbl = Table(_toc_rows, colWidths=[12 * mm, CW - 12 * mm])
    _toc_tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("LINEBELOW",     (0, 0), (-1, -2), 0.4, rule_c),
    ]))
    story.append(_toc_tbl)
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════
    # PAGE 3: HEALTH SCORE
    # ══════════════════════════════════════════════════════
    story.append(Spacer(1, 6*mm))
    story.append(_section("Data Health Overview"))
    story.append(Paragraph(fname[:70], ST["sm"]))
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
        ("BACKGROUND",    (1,0),(1,0), HexColor(PRINT["surface_tint"])),
        ("BACKGROUND",    (2,0),(-1,0), light2),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("BOX",           (0,0),(-1,-1), 1.5, accent),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, rule_c),
        ("LINEBELOW",     (0,0),(-1,0),  2, accent),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 5*mm))

    # When one fault caps the grade, the score and the grade disagree by
    # design — 71/100 shown beside a D. Print the reason immediately below
    # the box, or the reader is left to guess which of the two to believe.
    blocking = health.get("blocking_defect") or ""
    if blocking:
        story.append(Paragraph(
            "<b>Why this grade is capped:</b> " + blocking.replace(
                "&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
            ParagraphStyle("blk", parent=ST["sm"], textColor=HexColor(STATUS_LIGHT["critical"]),
                           backColor=HexColor("#FBF0EF"),
                           borderPadding=6, leading=13)))
        story.append(Spacer(1, 5*mm))

    # Dataset summary table
    story.append(Paragraph("Dataset Summary", ST["h3"]))
    num_cols_list = df.select_dtypes(include="number").columns.tolist()
    cat_cols_list = df.select_dtypes(include=["object","string"]).columns.tolist()
    date_cols_list= df.select_dtypes(include="datetime").columns.tolist()
    missing_total = df.isna().sum().sum()
    dup_count     = df.duplicated().sum()

    _sum_hdr = ParagraphStyle("sumh", fontName=_BB, fontSize=8.5,
                              textColor=white, alignment=TA_CENTER)
    summary_data = [
        [Paragraph("<b>Metric</b>", _sum_hdr), Paragraph("<b>Value</b>", _sum_hdr)],
        ["Total Rows",       "{:,}".format(health["rows"])],
        ["Total Columns",    str(health["cols"])],
        ["Numeric Columns",  str(len(num_cols_list))],
        ["Categorical Cols", str(len(cat_cols_list))],
        ["DateTime Columns", str(len(date_cols_list))],
        ["Missing Values",   "{:,} ({:.1f}%)".format(int(missing_total), health["missing_pct"])],
        ["Duplicate Rows",   "{:,} ({:.1f}%)".format(int(dup_count), health["dup_pct"])],
        ["Memory Usage",     "{:.1f} MB".format(df.memory_usage(deep=True).sum() / 1e6)],
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
        ("BOX",           (0,0),(-1,-1), 0.5, rule_c),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, rule_c),
    ]))
    story.append(sum_tbl)

    # ══════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY + NARRATIVE SECTIONS
    # ══════════════════════════════════════════════════════
    # Skipped entirely when no narrative was supplied, so the report never
    # shows an empty heading.
    if executive_summary or key_findings or risks or opportunities:
        story.append(CondPageBreak(ROOM_TEXT))
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
                    '<font color="{}">■</font>'.format(bullet_hex),
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

        _bullet_section("Key Findings", key_findings, "#2563EB")
        _bullet_section("Risks Identified", risks, "#DC2626")
        _bullet_section("Opportunities", opportunities, STATUS_LIGHT["good"])

    # ══════════════════════════════════════════════════════
    # PAGE 2: BUSINESS INSIGHTS
    # ══════════════════════════════════════════════════════
    story.append(CondPageBreak(ROOM_TEXT))
    story.append(_section(_insights_heading))
    story.append(Paragraph(
        "Each insight follows the format: <b>What → Why it matters → What to do.</b> "
        "All figures are computed directly from the uploaded dataset.",
        ST["body"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=6))

    SEV_COLORS = {
        "critical": ("#DC2626", "#FEF2F2"),
        "warning":  ("#D97706", "#FFFBEB"),
        "positive": (STATUS_LIGHT["good"], "#ECF6F2"),
        "info":     ("#2563EB", "#EFF6FF"),
    }

    for i, ins in enumerate(insights, 1):
        border_c, bg_c = SEV_COLORS.get(ins["severity"], ("#2563EB", "#EFF6FF"))
        bg_hex  = HexColor(bg_c)
        bdr_hex = HexColor(border_c)
        tag_c   = HexColor(border_c)

        tag_p    = Paragraph(ins["tag"],
            ParagraphStyle("it", fontName=_BB, fontSize=7.5,
                           textColor=tag_c, spaceAfter=2))
        title_p  = Paragraph("<b>{}. {}</b>".format(i, ins["title"]),
            ParagraphStyle("itl", fontName=_BB, fontSize=10.5,
                           textColor=dark, spaceAfter=3, leading=14))
        body_p   = Paragraph(ins["body"].replace("**","").replace("*",""),
            ParagraphStyle("ib", fontName=_BF, fontSize=9.5,
                           textColor=dark, leading=14.5, spaceAfter=4,
                           alignment=TA_JUSTIFY))
        # The action is usually a numbered sequence — "1. Decide … 2.
        # Check … 3. Pilot …" — and it arrived as one string, so the
        # three steps ran together into a single wrapped paragraph that
        # nobody could act from. Split on the numbering and give each
        # step its own line.
        action_rows = [[Paragraph(_a, ParagraphStyle(
                            "ia", fontName=_BB, fontSize=9,
                            textColor=HexColor(border_c), leading=13,
                            spaceAfter=2))]
                       for _a in _action_steps(ins["action"])]

        card = Table([[tag_p],[title_p],[body_p]] + action_rows, colWidths=[CW])
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), bg_hex),
            ("LINEBEFORE",    (0,0),(0,-1),  6, bdr_hex),
            ("TOPPADDING",    (0,0),(-1,-1), 9),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
            ("LEFTPADDING",   (0,0),(-1,-1), 16),
            ("RIGHTPADDING",  (0,0),(-1,-1), 12),
            ("BOX",           (0,0),(-1,-1), 0.5, rule_c),
        ]))
        story.append(KeepTogether([card, Spacer(1, 5*mm)]))

    # ══════════════════════════════════════════════════════
    # VISUAL ANALYSIS
    # ══════════════════════════════════════════════════════
    # The report previously carried only thumbnail histograms and a
    # correlation heatmap — diagnostics, not the business picture a client
    # reads. These are the same charts as the main report, each captioned
    # with what it shows rather than left to speak for itself.
    if _charts:
        story.append(CondPageBreak(ROOM_TEXT))
        story.append(_section("Visual Analysis"))
        story.append(Paragraph(
            "Charts are built from measure columns only; identifiers and "
            "columns with no variation are excluded, since neither carries "
            "business meaning.", ST["body"]))
        story.append(HRFlowable(width="100%", thickness=1.5, color=accent,
                                 spaceAfter=6))

        for idx, (chart_title, img_bytes, _spec) in enumerate(_charts, 1):
            if not img_bytes:
                continue
            try:
                # The chart draws its own title (the finding) and its
                # own subtitle (what is plotted), so printing the name a
                # third time above it was pure repetition. A numbered
                # exhibit label is what a report needs here — it gives
                # the reader something to cite without restating the
                # chart.
                block = [
                    Paragraph("Exhibit {}".format(idx), ST["h3"]),
                    Image(_io.BytesIO(img_bytes), width=CW, height=CW * 0.46),
                    Spacer(1, 6 * mm),
                ]
                story.append(KeepTogether(block))
            except Exception:
                logger.warning("could not place chart %r in the health report",
                               chart_title, exc_info=True)

    # ══════════════════════════════════════════════════════
    # PAGE 3: DESCRIPTIVE STATISTICS
    # ══════════════════════════════════════════════════════
    story.append(CondPageBreak(ROOM_CHART))
    story.append(_section("Descriptive Statistics"))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=5))

    if len(num_cols_list) > 0:
        story.append(Paragraph("Numeric Columns Summary", ST["h2"]))
        _desc_cols = [c for c in num_cols_list if not is_id_col(c, df[c])] or num_cols_list
        desc = df[_desc_cols[:8]].describe()
        # describe() returns raw floats, and they were printed as-is:
        # "count 1500.0", "mean 2557.499", "max 9916.48". A count is not
        # a decimal, and no finance team reads 9916.48 as ten thousand
        # without a separator. present.num() applies the same rules the
        # rest of the product uses.
        hdr_vals = ["Stat"] + [_pretty_col(c) for c in desc.columns]
        stat_hdr = [Paragraph("<b>{}</b>".format(h),
                    ParagraphStyle("sh", fontName=_BB, fontSize=7.5,
                                   textColor=white, alignment=TA_CENTER))
                    for h in hdr_vals]
        stat_rows = [stat_hdr]
        for idx in desc.index:
            row = [Paragraph("<b>{}</b>".format(idx),
                             ParagraphStyle("si", fontName=_BB, fontSize=7.5, textColor=dark))]
            for val in desc.loc[idx]:
                row.append(Paragraph(_stat_text(idx, val),
                    ParagraphStyle("sv2", fontName=_BF, fontSize=7.5,
                                   textColor=dark, alignment=TA_CENTER)))
            stat_rows.append(row)

        n_stat_cols = len(hdr_vals)
        stat_tbl = Table(stat_rows,
                         colWidths=[CW*0.1] + [CW*0.9/max(n_stat_cols-1,1)]*(n_stat_cols-1))
        stat_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  dark),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
            ("ALIGN",         (1,0),(-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("BOX",           (0,0),(-1,-1), 0.5, rule_c),
            ("INNERGRID",     (0,0),(-1,-1), 0.3, rule_c),
        ]))
        story.append(stat_tbl)
        story.append(Spacer(1, 6*mm))

    # ── Distribution mini-charts ──────────────────────────
    if len(num_cols_list) >= 2:
        story.append(Paragraph("Distribution Overview", ST["h3"]))
        story.append(Paragraph(
            "Histograms below show the data distribution for the top numeric columns. "
            "Dashed line = mean, dotted = median. Skewed distributions require median for reporting.",
            ST["note"]))
        story.append(Spacer(1, 3*mm))

        try:
            n_charts = min(4, len(num_cols_list))
            fig, axes = plt.subplots(1, n_charts, figsize=(10, 2.8))
            if n_charts == 1:
                axes = [axes]
            fig.patch.set_facecolor("#ffffff")
            # Four columns are not four categories. Giving them four
            # hues invited the reader to look for what blue-versus-green
            # meant, and there was nothing to find.
            from app.engines.palette import SINGLE_LIGHT, STATUS_LIGHT as _SL
            bar_c = SINGLE_LIGHT
            for idx2, (ax2, col) in enumerate(zip(axes, num_cols_list[:n_charts])):
                s2 = df[col].dropna()
                s2 = pd.to_numeric(s2, errors="coerce").dropna()
                if len(s2) == 0:
                    continue
                ax2.hist(s2, bins=20, color=bar_c,
                         alpha=0.9, edgecolor="#FFFFFF", linewidth=0.4)
                ax2.axvline(s2.mean(), color=_SL["critical"], linestyle="--",
                            linewidth=1.3)
                ax2.axvline(s2.median(), color=_SL["good"], linestyle=":",
                            linewidth=1.3)
                ax2.set_title(_pretty_col(col), fontsize=8,
                              fontweight="bold", color=PRINT["ink"], pad=6)
                ax2.set_facecolor(PRINT["surface_alt"])
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                ax2.tick_params(labelsize=6, colors="#0F172A")
            fig.tight_layout(pad=1.2)
            buf2 = _io.BytesIO()
            fig.savefig(buf2, format="png", dpi=160, bbox_inches="tight")
            buf2.seek(0)
            plt.close(fig)
            story.append(Image(buf2, width=CW, height=CW*0.3))
        except Exception:
            logger.warning("Health Report section failure", exc_info=True)

    # ══════════════════════════════════════════════════════
    # PAGE 4: COLUMN QUALITY TABLE
    # ══════════════════════════════════════════════════════
    story.append(CondPageBreak(ROOM_TABLE))
    story.append(_section("Column Quality Analysis"))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=5))
    story.append(Paragraph(
        "Each column is assessed for completeness, uniqueness, data type, "
        "and potential issues. Columns with <b>Missing > 5%</b> or "
        "<b>Unique = 1</b> (constant) need attention before analysis.",
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
            mn = _present_num(sc.dropna().min()) if len(sc.dropna()) > 0 else "—"
            mx = _present_num(sc.dropna().max()) if len(sc.dropna()) > 0 else "—"
        else:
            mn = "—"
            mx = "—"

        # Status
        miss_f = sc.isna().mean()*100
        if miss_f > 20:
            status = "⚠ HIGH MISSING"
            st_c   = HexColor(STATUS_LIGHT["critical"])
        elif miss_f > 5:
            status = "△ REVIEW"
            st_c   = HexColor(STATUS_LIGHT["warning"])
        elif sc.nunique() == 1:
            status = "⚠ CONSTANT"
            st_c   = HexColor(STATUS_LIGHT["critical"])
        elif sc.nunique() == len(df):
            status = "ℹ IDENTIFIER"
            st_c   = HexColor(PRINT["accent"])
        else:
            status = "✓ OK"
            st_c   = HexColor(STATUS_LIGHT["good"])

        rows2.append([
            Paragraph(truncate_label(_pretty_col(col), 22), td_st2),
            Paragraph(_dtype_word(sc.dtype), td_c2),
            Paragraph(miss, ParagraphStyle("mv", fontName=_BB, fontSize=8,
                           textColor=HexColor(STATUS_LIGHT["critical"]) if miss_f > 5 else dark,
                           alignment=TA_CENTER)),
            Paragraph(uniq, td_c2),
            Paragraph(mn, td_c2),
            Paragraph(mx, td_c2),
            Paragraph(sample_val, td_st2),
            Paragraph(status, ParagraphStyle("sv3", fontName=_BB, fontSize=7.5,
                                             textColor=st_c, alignment=TA_CENTER)),
        ])

    # "Unique" needs 0.09 width minimum — at 0.07 the header wrapped
    # mid-word ("Uniqu / e") in the rendered PDF.
    col_tbl2 = Table([hdr2] + rows2,
                     colWidths=[CW*x for x in [0.20,0.09,0.09,0.09,0.08,0.08,0.22,0.15]])
    col_tbl2.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  dark),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
        ("BOX",           (0,0),(-1,-1), 0.5, rule_c),
        ("INNERGRID",     (0,0),(-1,-1), 0.3, rule_c),
    ]))
    story.append(col_tbl2)

    # ══════════════════════════════════════════════════════
    # PAGE 5: CORRELATION + DISCLAIMER
    # ══════════════════════════════════════════════════════
    if len(num_cols_list) >= 3:
        story.append(CondPageBreak(ROOM_TABLE))
        story.append(_section("Correlation Analysis"))
        story.append(HRFlowable(width="100%", thickness=1.5, color=accent, spaceAfter=5))
        story.append(Paragraph(
            "<b>Important:</b> Correlation measures association, NOT causation. "
            "r² tells you what % of variance is shared between two variables. "
            "Strong correlation alone is never sufficient reason to act.",
            ST["body"]))
        story.append(Spacer(1, 4*mm))

        try:
            corr_cols = [c for c in num_cols_list
                         if df[c].nunique() > 2 and not is_id_col(c, df[c])][:8]
            if len(corr_cols) >= 2:
                corr = df[corr_cols].corr().round(2)
                n3   = len(corr)
                sz   = max(5, n3)
                fig3, ax3 = plt.subplots(figsize=(sz, sz * 0.8))
                fig3.patch.set_facecolor("#ffffff")
                ax3.set_facecolor("#f8faff")
                im3  = ax3.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                for ri in range(n3):
                    for ci in range(n3):
                        val3 = corr.values[ri, ci]
                        ax3.text(ci, ri, "{:.2f}".format(val3),
                                 ha="center", va="center", fontsize=8.5,
                                 color="white" if abs(val3) > 0.5 else "#0A1628",
                                 fontweight="bold" if abs(val3) > 0.3 else "normal")
                ax3.set_xticks(range(n3))
                ax3.set_yticks(range(n3))
                # "discount_..." told the reader nothing. Human labels
                # first, and a wider budget, so the truncation that
                # remains still names the column.
                labels3 = [truncate_label(_pretty_col(c), 16) for c in corr.columns]
                ax3.set_xticklabels(labels3, rotation=30, ha="right",
                                    fontsize=8.5, color=PRINT["ink"])
                ax3.set_yticklabels(labels3, fontsize=8.5, color=PRINT["ink"])
                ax3.set_title("Correlation Matrix", fontsize=12, fontweight="bold",
                              color="#0A1628", pad=10)
                ax3.spines[:].set_edgecolor("#d0d8f0")
                plt.colorbar(im3, ax=ax3, shrink=0.8)
                fig3.tight_layout()
                buf3 = _io.BytesIO()
                fig3.savefig(buf3, format="png", dpi=160, bbox_inches="tight")
                buf3.seek(0)
                plt.close(fig3)
                story.append(Image(buf3, width=CW, height=CW * 0.75))
                story.append(Spacer(1, 4*mm))

                # Top correlations table
                story.append(Paragraph("Top Correlations (|r| > 0.2)", ST["h3"]))
                pairs = []
                for ri in range(n3):
                    for ci in range(ri+1, n3):
                        r_val = corr.values[ri, ci]
                        if abs(r_val) > 0.2:
                            pairs.append((corr.index[ri], corr.columns[ci], r_val))
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)

                if pairs:
                    corr_hdr = [Paragraph("<b>{}</b>".format(h), th_st2)
                                for h in ["Column A", "Column B", "r", "r²", "Strength", "Interpretation"]]
                    corr_rows = [corr_hdr]
                    for ca, cb, rv in pairs[:8]:
                        r2 = rv ** 2
                        strength = "Strong" if abs(rv) > 0.6 else "Moderate" if abs(rv) > 0.4 else "Weak"
                        direction = "positive" if rv > 0 else "negative"
                        interp = "{}% variance shared — {} {}.".format(
                            round(r2*100, 1), strength.lower(), direction)
                        corr_rows.append([
                            Paragraph(truncate_label(ca, 16).replace("_"," "), td_st2),
                            Paragraph(truncate_label(cb, 16).replace("_"," "), td_st2),
                            Paragraph("<b>{:.3f}</b>".format(rv),
                                      ParagraphStyle("rv", fontName=_BB, fontSize=8,
                                      textColor=HexColor(STATUS_LIGHT["good"] if rv > 0
                                                 else STATUS_LIGHT["critical"]),
                                      alignment=TA_CENTER)),
                            Paragraph("{:.3f}".format(r2), td_c2),
                            Paragraph(strength, td_c2),
                            Paragraph(interp, td_st2),
                        ])
                    corr_tbl = Table(corr_rows,
                                     colWidths=[CW*x for x in [0.18,0.18,0.08,0.08,0.12,0.36]])
                    corr_tbl.setStyle(TableStyle([
                        ("BACKGROUND",    (0,0),(-1,0),  dark),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, light2]),
                        ("TOPPADDING",    (0,0),(-1,-1), 5),
                        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                        ("LEFTPADDING",   (0,0),(-1,-1), 6),
                        ("BOX",           (0,0),(-1,-1), 0.5, rule_c),
                        ("INNERGRID",     (0,0),(-1,-1), 0.3, rule_c),
                    ]))
                    story.append(corr_tbl)
                else:
                    # Never leave a promised section blank in a client
                    # deliverable — an absent finding is itself a finding.
                    story.append(Paragraph(
                        "No column pairs exceed |r| &gt; 0.2 — the numeric variables "
                        "in this dataset move independently of each other. This is a "
                        "meaningful result: no single metric can be used as a proxy "
                        "for another, and multivariate analysis (segmentation, ML) "
                        "is required to find drivers.", ST["body"]))
        except Exception:
            logger.warning("Health Report section failure", exc_info=True)

    # ══════════════════════════════════════════════════════
    # RECOMMENDED ACTIONS
    # ══════════════════════════════════════════════════════
    # Closing a client report on what to DO — not on a correlation table —
    # is what separates a deliverable from a data dump.
    if actions:
        story.append(CondPageBreak(ROOM_TABLE))
        story.append(_section("Recommended Actions"))
        story.append(Paragraph(
            "Prioritised from the findings above. Sequence reflects urgency "
            "and expected impact, not effort.", ST["body"]))
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
            ("LINEBELOW",     (0, 0), (-1, -2), 0.4, rule_c),
            ("BACKGROUND",    (0, 0), (-1, -1), light2),
            ("BOX",           (0, 0), (-1, -1), 0.5, rule_c),
        ]))
        story.append(act_tbl)

    # ── How this was measured ─────────────────────────────
    # The analysis report carries a full Analytical Method appendix; this
    # one carried a single BASIS OF PREPARATION paragraph, and this is
    # the document that most often reaches a stakeholder on its own. The
    # question it has to answer is not "is the number right" — it is
    # "what would make this number wrong", and a reader cannot judge
    # that from a score and a grade.
    story.append(CondPageBreak(ROOM_TEXT))
    story.append(_section("How This Was Measured"))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent,
                            spaceAfter=6))

    for heading, text in (
        ("What the score is",
         "The health score is the same figure the analysis report prints "
         "for this dataset — completeness at 60%, freedom from duplicate "
         "records at 30%, and per-column quality at 10%. It is not a "
         "second opinion computed here; two different scores for one file "
         "would be a contradiction in the pack you received."),
        ("What the grade is",
         "The grade is a judgement rather than an average, so it has a "
         "floor. One ruinous fault caps it regardless of the score — a "
         "file that is mostly duplicate rows, or one where no column "
         "varies, cannot grade well no matter how complete it is. Where "
         "the grade has been capped, the reason is stated beside it in "
         "words."),
        ("What the findings rest on",
         "Group differences are tested with Kruskal-Wallis or Mann-Whitney "
         "where the distribution is not normal and with ANOVA or Welch's "
         "t-test where it is; the choice follows the data rather than "
         "convenience. Correlations carry their p-value and sample size. "
         "Findings that did not survive correction for multiple testing "
         "were dropped rather than reported with a caveat."),
        ("What it does not support",
         "Everything here describes association within this dataset and "
         "the period it covers. Nothing establishes cause. A column "
         "computed from another — revenue from units and price, a band "
         "cut from the measure it groups — is excluded from driver and "
         "cohort findings rather than reported as an explanation of "
         "itself. Projections are not extended past the range the data "
         "actually covers."),
        ("What would change the answer",
         "Figures describe the file as supplied. If the export was "
         "filtered, covers a partial period, or was taken before a "
         "correction, every number here inherits that. The period this "
         "file covers is stated in the column analysis above; check it "
         "against the period you meant to review."),
    ):
        story.append(Paragraph(
            "<b>{}.</b> {}".format(heading, text), ST["body"]))
        story.append(Spacer(1, 2*mm))

    # ── Disclaimer ────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=gray, spaceAfter=4))
    story.append(Paragraph(
        "<b>BASIS OF PREPARATION</b>  —  Prepared by {} on {}. Source: {}. "
        "All figures derive solely from the supplied dataset and describe the "
        "period it covers. Statistical association is reported where present; "
        "it does not establish causation. Recommendations assume the data is "
        "complete and accurate as supplied.".format(
            agency_name, now, fname[:40]),
        ST["sm"]))

    doc.build(story, canvasmaker=_Canvas)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════
