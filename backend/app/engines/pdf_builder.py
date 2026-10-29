"""
engines/pdf_builder.py — the main client report.

    build_pdf(df, config, profile, cleaning_summary, stats_report,
              bi_report, ml_report, chart_data, executive_summary,
              findings, risks, opportunities, recommendations,
              top_insights, attrition, domain) -> bytes

Layout rules, because each was broken before:
- Sections flow on from one another and start a new page only when too
  little room is left; a heading always travels with what follows it. A
  page break after every section left most pages two-thirds empty, and a
  findings heading alone on its own page.
- The contents page lists real page numbers.
- One palette per report (engines/report_theme), shared with its charts;
  status colours are used for status only.
- Charts keep their own proportions and carry the caption computed from
  their data.
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
    Image, HRFlowable, PageBreak, KeepTogether, CondPageBreak,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfgen import canvas as CV
from app.engines.report_blueprints import blueprint_for
from app.services.dtypes import is_text_dtype, text_columns

from app.engines.pdf_fonts import FONT_BOLD, FONT_ITALIC, FONT_REG
logger = logging.getLogger(__name__)

W, H = A4
CW_DEFAULT = W - 36 * mm   # content width (18mm each side)


# Colours are shared with the charts so a report reads as one document.
from app.engines.report_theme import DOMAIN_THEMES, THEMES  # noqa: E402,F401

# SHRM/Gallup/Mercer benchmarks for HR domain
HR_BENCHMARKS = [
    ["Attrition Rate",         "—",    "10–15%",      "<10%",          "SHRM 2024"],
    ["Employee Satisfaction",  "—",    "0.70 (70%+)", "0.80+",         "Gallup/Mercer"],
    ["Replacement Cost/EE",    "—",    "50–200% sal", "6–9 mo salary", "SHRM/Gallup"],
    ["Mgr-Driven Satisfaction","—",    "70%",         "Manager train", "Gallup 2024"],
    ["Preventable Exits",      "—",    "52%",         "Proactive 1:1", "Gallup 2024"],
]


# ══════════════════════════════════════════════════════════
#  COLOUR HELPERS
# ══════════════════════════════════════════════════════════

def _c(hex_str: str) -> HexColor:
    return HexColor(hex_str)


# ══════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════

def _styles(T: dict) -> dict:
    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "h1":    ps("h1",   fontName=FONT_BOLD, fontSize=17,
                    textColor=_c(T["primary"]),    spaceAfter=4),
        # Headings travel with what follows them, so none is left at the
        # foot of a page with its content overleaf.
        "h2":    ps("h2",   fontName=FONT_BOLD, fontSize=15, leading=19,
                    textColor=_c(T["primary"]),    spaceBefore=4, spaceAfter=2,
                    keepWithNext=1),
        "h3":    ps("h3",   fontName=FONT_BOLD, fontSize=10.5, leading=14,
                    textColor=_c(T["primary"]),    spaceBefore=8, spaceAfter=3,
                    keepWithNext=1),
        "sub":   ps("sub",  fontName=FONT_REG,      fontSize=8.5, leading=12,
                    textColor=_c(T["text_muted"]), spaceAfter=2, keepWithNext=1),
        "body":  ps("body", fontName=FONT_REG,      fontSize=9.5,
                    textColor=_c(T["text"]),       leading=14,  spaceAfter=4,
                    alignment=TA_LEFT),
        "caption": ps("caption", fontName=FONT_REG, fontSize=8.8,
                    textColor=_c(T["text"]),       leading=13, spaceBefore=2,
                    spaceAfter=2),
        "sm":    ps("sm",   fontName=FONT_REG,      fontSize=7.5,
                    textColor=_c(T["text_muted"]), leading=11,  spaceAfter=2),
        "bl":    ps("bl",   fontName=FONT_REG,      fontSize=9,
                    textColor=_c(T["text"]),       leading=13,  spaceAfter=3,
                    leftIndent=10, firstLineIndent=-10),
        "toc":   ps("toc",  fontName=FONT_REG,      fontSize=10,
                    textColor=_c(T["text"]),       leading=16,  spaceAfter=3),
        "wh":    ps("wh",   fontName=FONT_REG,      fontSize=9,
                    textColor=HexColor("#FFFFFF"),  leading=13),
        "wbh":   ps("wbh",  fontName=FONT_BOLD, fontSize=10,
                    textColor=HexColor("#FFFFFF")),
        "note":  ps("note", fontName=FONT_ITALIC, fontSize=7.5,
                    textColor=_c(T["text_muted"]), spaceAfter=3),
        "warn":  ps("warn", fontName=FONT_REG,      fontSize=8.5,
                    textColor=_c(T["text"]),       leading=13,
                    backColor=_c(T["warning_bg"])),
        # Insight card row styles
        "rl":    ps("rl",   fontName=FONT_BOLD, fontSize=8,
                    textColor=HexColor("#FFFFFF"),  alignment=TA_CENTER),
        "rv":    ps("rv",   fontName=FONT_REG,      fontSize=8.5,
                    textColor=_c(T["text"]),       leading=12.5),
    }


# ══════════════════════════════════════════════════════════
#  RUNNING HEADER / FOOTER  (PageCanvas)
# ══════════════════════════════════════════════════════════

class _ReportCanvas(CV.Canvas):
    """Draws premium header + footer on every content page."""

    def __init__(self, fn, T, report_title="", client_name="", report_date="", **kw):
        super().__init__(fn, **kw)
        self._sp          = []
        self.T            = T
        self.report_title = report_title[:60]
        self.client_name  = client_name
        self.report_date  = report_date

    def showPage(self):
        self._sp.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        tot = len(self._sp)
        for st in self._sp:
            self.__dict__.update(st)
            self._draw(tot)
            super().showPage()
        super().save()

    def _draw(self, tot):
        # A thin rule and small type: the old full-width coloured bars on
        # every page outweighed the content between them.
        T = self.T
        self.setStrokeColor(_c(T["border"]))
        self.setLineWidth(0.6)
        self.line(18*mm, H - 16*mm, W - 18*mm, H - 16*mm)
        self.setFont(FONT_BOLD, 8)
        self.setFillColor(_c(T["primary"]))
        self.drawString(18*mm, H - 13*mm, self.report_title)
        self.setFont(FONT_REG, 7.5)
        self.setFillColor(_c(T["text_muted"]))
        self.drawRightString(W - 18*mm, H - 13*mm,
                             "{} · {}".format(self.client_name, self.report_date))
        self.line(18*mm, 13*mm, W - 18*mm, 13*mm)
        self.drawString(18*mm, 9*mm, "Confidential · Prepared for {}".format(
            self.client_name))
        self.drawRightString(W - 18*mm, 9*mm,
                             "Page {} of {}".format(self._pageNumber, tot))


# ══════════════════════════════════════════════════════════
#  COVER PAGE  (drawn on separate canvas, merged via pypdf)
# ══════════════════════════════════════════════════════════

def _build_cover(T: dict, config: dict, kpis_preview: list) -> bytes:
    """A white cover with one band of the report's colour: the title, who
    it is for, and a few facts about the data. The decorative circles,
    stripes and multi-coloured tiles it replaced competed with the title."""
    buf = io.BytesIO()
    cv  = CV.Canvas(buf, pagesize=A4)
    title       = config.get("title", "Data Analysis Report")
    client_name = config.get("client_name", "Client")
    report_date = datetime.now().strftime("%B %d, %Y")
    domain_lbl  = T.get("domain_label", "BUSINESS ANALYTICS")
    x0 = 30*mm

    cv.setFillColor(_c(T["primary"]))
    cv.rect(0, 0, 14*mm, H, fill=1, stroke=0)
    cv.setFillColor(_c(T["accent"]))
    cv.rect(14*mm, 0, 1.2*mm, H, fill=1, stroke=0)

    cv.setFillColor(_c(T["primary"]))
    cv.setFont(FONT_BOLD, 12)
    cv.drawString(x0, H - 30*mm, "Analytiq")

    logo_path = config.get("logo_path", "")
    if logo_path and os.path.exists(logo_path):
        try:
            cv.drawImage(logo_path, W - 68*mm, H - 40*mm, width=48*mm, height=20*mm,
                         preserveAspectRatio=True, mask="auto")
        except Exception:
            logger.debug("cover logo could not be drawn", exc_info=True)

    cv.setFillColor(_c(T["accent"]))
    cv.setFont(FONT_BOLD, 8.5)
    cv.drawString(x0, H * 0.62 + 24*mm, domain_lbl)

    # Title, wrapped to the cover's width by measured text width.
    from reportlab.pdfbase.pdfmetrics import stringWidth
    size = 30 if len(title) <= 40 else 24
    max_w = W - x0 - 22*mm
    words, lines, line = title.split(), [], ""
    for w in words:
        test = (line + " " + w).strip()
        if stringWidth(test, FONT_BOLD, size) <= max_w:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    cv.setFillColor(_c(T["primary"]))
    y = H * 0.62 + 12*mm
    for ln in lines[:3]:
        cv.setFont(FONT_BOLD, size)
        cv.drawString(x0, y, ln)
        y -= size * 1.2
    subtitle = config.get("subtitle", "")
    if subtitle:
        cv.setFillColor(_c(T["text_muted"]))
        cv.setFont(FONT_REG, 11)
        cv.drawString(x0, y - 2*mm, subtitle[:90])
        y -= 8*mm
    cv.setStrokeColor(_c(T["accent"]))
    cv.setLineWidth(1.2)
    cv.line(x0, y - 4*mm, x0 + 40*mm, y - 4*mm)

    # A few facts about the data, in one colour.
    kpis = kpis_preview[:4]
    if kpis:
        bw = (W - x0 - 22*mm) / len(kpis)
        ky = H * 0.30
        for i, kpi in enumerate(kpis):
            kx = x0 + i * bw
            value = str(kpi.get("value", ""))
            vsize = 18
            while vsize > 10 and stringWidth(value, FONT_BOLD, vsize) > bw - 4*mm:
                vsize -= 1
            cv.setFillColor(_c(T["primary"]))
            cv.setFont(FONT_BOLD, vsize)
            cv.drawString(kx, ky, value)
            cv.setFillColor(_c(T["text_muted"]))
            cv.setFont(FONT_REG, 8)
            cv.drawString(kx, ky - 6*mm, str(kpi.get("label", ""))[:24])
        cv.setStrokeColor(_c(T["border"]))
        cv.setLineWidth(0.6)
        cv.line(x0, ky + 10*mm, W - 22*mm, ky + 10*mm)

    meta = [("Prepared for", client_name), ("Date", report_date),
            ("Classification", "Confidential")]
    mw = (W - x0 - 22*mm) / 3
    for i, (k, v) in enumerate(meta):
        mx = x0 + i * mw
        cv.setFillColor(_c(T["text_muted"]))
        cv.setFont(FONT_REG, 8)
        cv.drawString(mx, 30*mm, k)
        cv.setFillColor(_c(T["text"]))
        cv.setFont(FONT_BOLD, 10)
        cv.drawString(mx, 24*mm, v[:30])
    cv.save()
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════
#  REUSABLE COMPONENT HELPERS
# ══════════════════════════════════════════════════════════

def _sec(story: list, s: dict, T: dict, title: str, sub: str = ""):
    """A section heading; it is listed on the contents page and kept on
    the same page as the start of its section."""
    story.append(CondPageBreak(50*mm))
    story.append(Paragraph(title, s["h2"]))
    if sub:
        story.append(Paragraph(sub, s["sub"]))
    rule = HRFlowable(width="100%", thickness=0.8, color=_c(T["accent"]),
                      spaceBefore=2, spaceAfter=6)
    rule.keepWithNext = 1
    story.append(rule)


def _kpi_row(story: list, s: dict, T: dict, kpis: list, CW: float):
    cols = min(4, len(kpis))
    cw   = CW / cols
    vals = [Paragraph(
                "<b>{}</b>".format(k.get("value", "")),
                ParagraphStyle("kv", fontName=FONT_BOLD, fontSize=17,
                               textColor=HexColor(k.get("color", T["primary"])),
                               alignment=TA_CENTER))
            for k in kpis[:cols]]
    lbls = [Paragraph(
                k.get("label", ""),
                ParagraphStyle("kl", fontName=FONT_BOLD, fontSize=7.5,
                               textColor=_c(T["text_muted"]), alignment=TA_CENTER))
            for k in kpis[:cols]]
    subs = [Paragraph(
                k.get("sub", ""),
                ParagraphStyle("ks", fontName=FONT_REG, fontSize=7,
                               textColor=_c(T["text_muted"]), alignment=TA_CENTER))
            for k in kpis[:cols]]
    t = Table([vals, lbls, subs], colWidths=[cw]*cols)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["bg_card"])),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("BOX",           (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, _c(T["border"])),
    ]))
    story.append(t)
    story.append(Spacer(1, 3*mm))


def _narrative_box(story: list, s: dict, T: dict, text: str):
    if not text: return
    t = Table([[Paragraph(text, s["body"])]], colWidths=["100%"])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["bg_light"])),
        ("LINEBEFORE",    (0,0), (0,-1),  4, _c(T["accent"])),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 2*mm))


import re as _re  # noqa: E402
_FIGURE_RE = _re.compile(r"^[-+−]?[\d.,]+\s?(%|[KMB]|x|pp)?$|^—$")


def _gtable(story: list, T: dict, headers: list,
            rows_data: list, col_widths: list,
            severity_col: int = -1):
    """Generic styled table."""
    # Figures right-aligned, text left: centring both made columns of
    # numbers impossible to compare.
    def _is_figure(v) -> bool:
        return bool(_FIGURE_RE.match(str(v).strip()))

    def _cell(v, header=False, right=False):
        return Paragraph(str(v), ParagraphStyle(
            "th" if header else "td",
            fontName=FONT_BOLD if header else FONT_REG, fontSize=8.2,
            leading=11, textColor=HexColor("#FFFFFF") if header else _c(T["text"]),
            alignment=TA_RIGHT if right else TA_LEFT))

    numeric_cols = [i for i in range(len(headers))
                    if i > 0 and rows_data and all(_is_figure(r[i]) for r in rows_data)]
    hrow = [_cell(h, header=True, right=i in numeric_cols) for i, h in enumerate(headers)]
    body = [[_cell(c, right=i in numeric_cols) for i, c in enumerate(row)]
            for row in rows_data]
    tbl = Table([hrow] + body, colWidths=col_widths, repeatRows=1)
    sty = [
        ("BACKGROUND",    (0,0), (-1,0),  _c(T["primary"])),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [HexColor("#FFFFFF"), _c(T["bg_card"])]),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("LINEBELOW",     (0,0), (-1,-1), 0.3, _c(T["border"])),
    ]
    if severity_col >= 0:
        # Status shown as a tinted cell with coloured text, not a solid
        # block of orange in the middle of a table.
        _sev = {"CRITICAL": ("negative", "critical_bg"), "HIGH": ("warning", "warning_bg"),
                "ABOVE": ("warning", "warning_bg"), "OK": ("positive", "positive_bg"),
                "WITHIN": ("positive", "positive_bg"), "BELOW": ("info", "info_bg")}
        for i, row in enumerate(rows_data, 1):
            val = str(row[severity_col]).upper()
            for k, (fg, bg) in _sev.items():
                if k in val:
                    body[i - 1][severity_col] = Paragraph(
                        "<b>{}</b>".format(row[severity_col]), ParagraphStyle(
                            "sev", fontName=FONT_BOLD, fontSize=8, leading=11,
                            textColor=_c(T[fg])))
                    sty.append(("BACKGROUND", (severity_col, i), (severity_col, i),
                                _c(T[bg])))
                    break
        tbl = Table([hrow] + body, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(sty))
    story.append(tbl)
    story.append(Spacer(1, 2*mm))


def _insight_card(story: list, s: dict, T: dict, ins, CW: float, num=None):
    """
    Works with both:
      - Dataclass objects (has .severity, .title, .problem …)
      - Plain dicts (keys: severity, title, problem …)
    """
    def _get(obj, key, default=""):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    sev    = _get(ins, "severity", "info").lower()
    sev_c  = {"critical": T["negative"], "high": T["warning"],
              "warning":  T["info"],     "info": T["text_muted"]}
    sev_bg = {"critical": T["critical_bg"], "high": T["warning_bg"],
              "warning":  T["info_bg"],     "info": T["bg_card"]}
    col = sev_c.get(sev, T["accent"])
    bg  = sev_bg.get(sev, T["bg_card"])

    bs = ParagraphStyle("bi_badge", fontName=FONT_BOLD, fontSize=7.5,
                        textColor=HexColor("#FFFFFF"), alignment=TA_CENTER)
    ts = ParagraphStyle("bi_title", fontName=FONT_BOLD, fontSize=9.5,
                        textColor=_c(T["text"]))
    rl = ParagraphStyle("bi_lbl",   fontName=FONT_BOLD, fontSize=7.5,
                        textColor=_c(T["primary"]), alignment=TA_LEFT)
    rv = ParagraphStyle("bi_val",   fontName=FONT_REG,      fontSize=8.5,
                        textColor=_c(T["text"]),  leading=12.5)

    num_str = "{}. ".format(num) if num else ""
    bs = ParagraphStyle("bi_badge", fontName=FONT_BOLD, fontSize=7.5,
                        textColor=HexColor(col), alignment=TA_CENTER)
    hdr = Table([[
        Paragraph(_get(ins, "severity", "INFO").upper(), bs),
        Paragraph("{}{}".format(num_str, _clean(_get(ins, "title", ""))), ts),
    ]], colWidths=[20*mm, CW - 20*mm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), HexColor(bg)),
        ("BACKGROUND", (1,0), (1,0), _c(T["bg_card"])),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",      (0,0), (0,0),  "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING",(0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0),(-1,-1), 8),
        ("BOX",        (0,0), (-1,-1), 0.5, _c(T["border"])),
    ]))

    lw, vw = 22*mm, CW - 22*mm
    rows = [
        [Paragraph(k, rl), Paragraph(_steps(_get(ins, fk, "")), rv)]
        for k, fk in [("Finding", "problem"), ("Why",  "cause"),
                      ("Evidence","evidence"),("Next step", "action"),
                      ("Impact",  "impact")]
        if _get(ins, fk, "").strip()
    ]
    if not rows:
        rows = [[Paragraph("DETAIL", rl),
                 Paragraph(str(ins)[:200], rv)]]

    body = Table(rows, colWidths=[lw, vw])
    body.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), _c(T["bg_light"])),
        ("VALIGN",  (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("BOX",       (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("INNERGRID", (0,0), (-1,-1), 0.3, _c(T["border"])),
    ]))
    story.append(KeepTogether([hdr, body, Spacer(1, 4*mm)]))


# ══════════════════════════════════════════════════════════
#  TOC PAGE
# ══════════════════════════════════════════════════════════

def _toc(story, s, T, CW):
    """Contents with the page each section starts on. The old list had
    no page numbers at all."""
    story.append(Paragraph("Contents", ParagraphStyle(
        "toc_title", parent=s["h2"], keepWithNext=0)))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_c(T["accent"]),
                            spaceBefore=2, spaceAfter=8))
    toc = TableOfContents()
    toc.dotsMinLevel = 0
    toc.levelStyles = [ParagraphStyle(
        "toc0", fontName=FONT_REG, fontSize=10.5, leading=20,
        textColor=_c(T["text"]), leftIndent=0, firstLineIndent=0)]
    story.append(toc)


def _basis_box(story, s, T, config):
    """The basis of preparation, under the contents. As the last item of the
    appendix it was pushed onto a page of its own whenever the appendix ran
    a few lines long."""
    story.append(Spacer(1, 10*mm))
    disc = Table([[Paragraph(
        "<b>Basis of preparation</b><br/>"
        "Prepared for {} on {}. All figures derive solely from the dataset "
        "supplied for this engagement and describe the period it covers. "
        "Statistical association is reported where present; it does not "
        "establish causation. Any external reference range cited is "
        "indicative and should be validated against the organisation's own "
        "sector and prior periods before it informs a decision. "
        "Recommendations assume the data is complete and accurate as "
        "supplied.{}".format(
            config.get("client_name", "Client"),
            datetime.now().strftime("%B %d, %Y"),
            _prepared_by_line(config)),
        s["body"])]],
        colWidths=["100%"])
    disc.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["bg_light"])),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("LINEBEFORE",    (0,0), (0,-1), 2.5, _c(T["accent"])),
    ]))
    story.append(disc)


def _bind_headings(story: list) -> list:
    """Join each run of headings to the block that follows it.

    ReportLab keeps a heading (keepWithNext) with the next flowable only if
    that flowable is not itself a KeepTogether — and every finding card and
    chart is one. The heading was kept with nothing, so "Findings" could sit
    alone at the foot or top of a page with all its cards overleaf.
    """
    out, run = [], []
    for f in story:
        if getattr(f, "keepWithNext", 0) or (
                isinstance(f, Paragraph) and getattr(f.style, "keepWithNext", 0)):
            run.append(f)
            continue
        if run and isinstance(f, KeepTogether):
            out.append(KeepTogether(run + list(f._content)))
        elif run:
            out.append(KeepTogether(run + [f]))
        else:
            out.append(f)
        run = []
    out.extend(run)
    return out


class _ReportDoc(BaseDocTemplate):
    """Registers every section heading on the contents page."""

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "h2":
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))


# ══════════════════════════════════════════════════════════
#  EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════

def _exec_summary(story, s, T, summary, findings, risks, opps, CW):
    _sec(story, s, T, "Executive Summary")
    if summary:
        _narrative_box(story, s, T, summary)
    if findings:
        story.append(Paragraph("Key Findings", s["h3"]))
        for f in findings[:5]:
            story.append(Paragraph("• " + str(f), s["bl"]))
        story.append(Spacer(1, 2*mm))
    if risks:
        story.append(Paragraph("Risks", s["h3"]))
        for r in risks[:4]:
            story.append(Paragraph(
                '<font color="{}">•</font> {}'.format(T["negative"], _clean(str(r))),
                s["bl"]))
        story.append(Spacer(1, 2*mm))
    if opps:
        story.append(Paragraph("Opportunities", s["h3"]))
        for o in opps[:3]:
            story.append(Paragraph(
                '<font color="{}">•</font> {}'.format(T["positive"], _clean(str(o))),
                s["bl"]))


# ══════════════════════════════════════════════════════════
#  TOP INSIGHTS
# ══════════════════════════════════════════════════════════

def _top_insights(story, s, T, insights, CW, domain: str = "general"):
    """Findings under the headings the domain's reader expects.

    A flat "Top Insights" list is a tool's default output. A finance
    director reads position, then cost structure, then variance; an HR
    director reads workforce profile, then attrition. Grouping the same
    findings under those headings is the difference between a report that
    was written for them and one that was generated.
    """
    from app.engines.insight_guard import guard_insights, withheld_note
    from app.engines.report_blueprints import blueprint_for, group_insights

    bp = blueprint_for(domain)
    # Last check before anything is printed: a finding with no figure, a
    # future asserted as fact, or a cause the data cannot support does
    # more damage than the finding is worth.
    guarded = guard_insights(list(insights or []))
    insights = guarded.kept
    if not insights:
        _sec(story, s, T, "Findings",
             "What the data shows, why it matters, and what to check next")
        story.append(Paragraph(
            "No finding in this dataset met the evidence threshold for "
            "inclusion. That is a result, not an omission: the analysis "
            "ran and found nothing it could support.", s["body"]))
        note = withheld_note(guarded)
        if note:
            story.append(Paragraph(note, s["note"]))
        return

    grouped = group_insights(bp, list(insights))
    _sec(story, s, T, "{} — Findings".format(bp.label),
         "What the data shows, why it matters, and what to check next")

    num = 0
    for section, items in grouped:
        # The heading travels with its first finding. Left to flow, a
        # section title lands at the foot of a page with its content
        # overleaf, which is the kind of thing a reader registers as
        # sloppy without being able to say why.
        head = [Spacer(1, 2 * mm),
                Paragraph(section.title, s["h3"]),
                Paragraph(section.purpose, s["sm"])]
        first: list = []
        _insight_card(first, s, T, items[0], CW, num=num + 1)
        story.append(KeepTogether(head + first))
        num += 1

        for ins in items[1:6]:
            num += 1
            _insight_card(story, s, T, ins, CW, num=num)
        if num >= 10:
            break

    note = withheld_note(guarded)
    if note:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(note, s["note"]))


# ══════════════════════════════════════════════════════════
#  DATA QUALITY NOTE
# ══════════════════════════════════════════════════════════

def _dq_note(story, s, T, df: pd.DataFrame, profile, CW):
    _sec(story, s, T, "Data Quality & Transparency Note",
         "Must read before interpreting any finding")

    raw_rows  = getattr(profile, "total_rows",     len(df))
    dupes     = getattr(profile, "duplicate_rows", 0)
    miss_pct  = getattr(profile, "missing_pct",
                        df.isna().mean().mean() * 100)
    qual      = getattr(profile, "overall_quality_score", "—")
    grade     = getattr(profile, "data_quality_grade", "")

    if dupes > 0:
        dup_pct = dupes / max(raw_rows, 1) * 100
        warn_t = Table([[Paragraph(
            "<b>Deduplication Alert:</b> "
            "{:,} exact duplicate rows ({:.1f}%) were detected in the raw data. "
            "All analysis in this report uses the deduplicated dataset "
            "({:,} clean rows). "
            "If comparing to any prior report on the raw file, numbers will differ — "
            "this is expected and correct.".format(dupes, dup_pct, len(df)),
            s["warn"])]],
            colWidths=["100%"])
        warn_t.setStyle(TableStyle([
            ("LEFTPADDING",  (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING",   (0,0), (-1,-1), 8),
            ("BOTTOMPADDING",(0,0), (-1,-1), 8),
            ("BOX",          (0,0), (-1,-1), 1, _c(T["warning"])),
        ]))
        story.append(warn_t)
        story.append(Spacer(1, 3*mm))

    # KPI strip
    _kpi_row(story, s, T, [
        {"label": "TOTAL ROWS",    "value": "{:,}".format(len(df)),
         "sub": "After deduplication", "color": T["accent"]},
        {"label": "COLUMNS",       "value": str(df.shape[1]),
         "sub": "{} num · {} cat".format(
             len(df.select_dtypes(include="number").columns),
             len(text_columns(df))),
         "color": T["accent"]},
        {"label": "MISSING DATA",  "value": "{:.1f}%".format(miss_pct),
         "sub": "0% = perfect",
         "color": T["positive"] if miss_pct == 0 else T["warning"]},
        {"label": "QUALITY SCORE", "value": str(qual),
         "sub": "Grade {}".format(grade) if grade else "/ 100",
         "color": T["positive"]},
    ], CW)

    _readiness_block(story, s, T, df, CW)

    # DQ table from profile
    recs = getattr(profile, "recommendations", [])
    if recs:
        story.append(Paragraph("Data Quality Recommendations", s["h3"]))
        for rec in recs[:6]:
            sty = "bl"
            story.append(Paragraph("• " + str(rec), s[sty]))


def _readiness_block(story, s, T, df: pd.DataFrame, CW):
    """State whether the data was fit to analyse before showing findings.

    A reader is entitled to know that the revenue column was text and was
    therefore in none of the numbers above it. Putting this on the same
    page as the quality note, rather than in an appendix, is the point:
    it is a precondition for the report, not a footnote to it.
    """
    try:
        from app.engines.readiness import assess_readiness
        rep = assess_readiness(df)
    except Exception:
        logger.warning("readiness assessment failed", exc_info=True)
        return

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Fitness for Analysis", s["h3"]))
    story.append(Paragraph(_clean(rep.summary), s["body"]))

    if rep.blockers:
        rows = [[b.column, b.issue, b.consequence] for b in rep.blockers[:6]]
        _gtable(story, T, ["Column", "Issue", "Effect on the analysis"],
                rows, [CW * x for x in [0.20, 0.25, 0.55]])
        story.append(Paragraph(
            "Findings in this report were produced despite the above. Treat "
            "any figure that depends on an affected column as provisional "
            "until it is resolved.", s["note"]))
    if rep.personal_data_columns:
        story.append(Paragraph(
            "Personal data present: {}. This report and the underlying "
            "extract should be handled and retained accordingly.".format(
                ", ".join(rep.personal_data_columns[:8])),
            s["note"]))


def _clean(text: str) -> str:
    """Escape for ReportLab's mini-HTML parser."""
    return (str(text).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _steps(text: str) -> str:
    """Escaped text with "1. …  2. …" steps on lines of their own; written
    inline they ran together into one sentence."""
    out = _clean(str(text).strip())
    return _re.sub(r"\s{2,}(?=\d+\.\s)", "<br/>", out)


# ══════════════════════════════════════════════════════════
#  DATA PREPARATION  (what was changed, and the SQL for it)
# ══════════════════════════════════════════════════════════

def _data_prep_section(story, s, T, cleaning_summary, CW, table="source_table"):
    """Every transformation applied before analysis, with its SQL.

    A reader who cannot see what was changed between the file they sent
    and the numbers they are reading has to take the whole report on
    trust. Listing each step — and the statement that reproduces it in
    their own warehouse — is what makes the rest checkable.
    """
    if not cleaning_summary:
        return
    actions = cleaning_summary.get("actions")
    if not actions:
        # Older cached summaries carry only the display groups. Flattening
        # them loses execution order, so the SQL block is suppressed rather
        # than printed in an order that would not reproduce the table.
        groups = cleaning_summary.get("groups") or {}
        actions = [a for g in groups.values() for a in g]
        ordered = False
    else:
        ordered = True
    if not actions:
        return

    _sec(story, s, T, "Data Preparation",
         "Every change made to the source data before any figure was computed")

    story.append(Paragraph(
        "The source file contained {:,} rows across {} columns. After the "
        "steps below, {:,} rows and {} columns were carried into the "
        "analysis. Nothing else was altered.".format(
            cleaning_summary.get("original_rows", 0),
            cleaning_summary.get("original_cols", 0),
            cleaning_summary.get("cleaned_rows", 0),
            cleaning_summary.get("cleaned_cols", 0)),
        s["body"]))
    story.append(Spacer(1, 2*mm))

    rows = []
    for a in actions[:18]:
        rows.append([
            getattr(a, "column", ""),
            getattr(a, "issue", ""),
            getattr(a, "action", ""),
            "{:,}".format(getattr(a, "rows_affected", 0) or 0),
        ])
    _gtable(story, T, ["Column", "Observed", "Treatment", "Rows"],
            rows, [CW*x for x in [0.18, 0.30, 0.42, 0.10]])
    if len(actions) > 18:
        story.append(Paragraph(
            "{} further steps of the same kinds are listed in the "
            "accompanying SQL script.".format(len(actions) - 18), s["note"]))

    # ── The same steps as SQL ──────────────────────────────
    sql_actions = [a for a in actions if getattr(a, "sql", "")] if ordered else []
    if not sql_actions:
        return
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("Equivalent SQL", s["h3"]))
    story.append(Paragraph(
        "The analysis itself was performed in pandas. These statements "
        "express the same treatment against <b>{}</b> so your data team can "
        "verify each step, or apply it upstream and stop the issue "
        "recurring. Order matters: deduplicating after imputing does not "
        "give the same table. None of it has been executed.".format(table),
        s["body"]))
    story.append(Spacer(1, 1.5*mm))

    mono = ParagraphStyle(
        "sqlmono", fontName="Courier", fontSize=7.2, leading=9.6,
        textColor=_c(T["text"]), leftIndent=4, spaceAfter=0)
    lines = []
    for a in sql_actions[:14]:
        stmt = a.sql.replace("{table}", '"{}"'.format(table))
        for raw in stmt.splitlines():
            lines.extend(_wrap_sql_line(raw))
        lines.append("")
    block = [[Paragraph(_sql_escape(ln) or "&nbsp;", mono)] for ln in lines[:70]]
    tbl = Table(block, colWidths=[CW])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), _c(T["bg_light"])),
        ("BOX",          (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 0),
        ("BOTTOMPADDING",(0,0), (-1,-1), 0),
    ]))
    story.append(tbl)
    if len(lines) > 70:
        story.append(Paragraph(
            "Truncated for length — the full script is available from the "
            "Data Quality screen.", s["note"]))


_SQL_COLS = 96   # fits Courier 7.2pt across the content frame


def _wrap_sql_line(line: str) -> list:
    """Break one SQL line to the page width, at a space where possible.

    A `GROUP BY` over thirty columns is a single line in the script. Left
    to itself the Paragraph cannot break it (every space is
    non-breaking), so it runs off the page and the reader loses the tail
    of the statement without any sign that it happened. Continuations are
    indented so the break is visibly a wrap, not a new statement.
    """
    line = line.rstrip()
    if len(line) <= _SQL_COLS:
        return [line]
    indent = len(line) - len(line.lstrip())
    out, rest = [], line
    while len(rest) > _SQL_COLS:
        cut = rest.rfind(" ", indent + 1, _SQL_COLS)
        if cut <= indent:
            cut = _SQL_COLS
        out.append(rest[:cut].rstrip())
        rest = " " * (indent + 4) + rest[cut:].lstrip()
    out.append(rest)
    return out


def _sql_escape(text: str) -> str:
    """Escape SQL for ReportLab's mini-HTML parser.

    `WHERE "x" < 108 OR "x" > 877` is a legal comparison and an illegal
    tag; unescaped, ReportLab swallows the rest of the line. Spaces become
    non-breaking so indentation survives — the wrapping is done by
    `_wrap_sql_line` beforehand, where the break points can be chosen.
    """
    return (str(text).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;")
            .replace(" ", "&nbsp;") if text else "")


# ══════════════════════════════════════════════════════════
#  INDUSTRY BENCHMARKS (HR domain)
# ══════════════════════════════════════════════════════════

# "Published hr ranges" reads as machine output. Acronym domains keep
# their capitalisation; the rest are title-cased.
_DOMAIN_LABELS = {
    "hr": "HR", "saas": "SaaS", "ecommerce": "e-commerce",
    "finance": "finance", "sales": "sales", "marketing": "marketing",
    "operations": "operations", "healthcare": "healthcare",
}


def _has_reference_ranges(domain, df) -> bool:
    """Whether this dataset has any metric with a published range.

    Checked before the section is added to the contents page, so the
    contents never promises a section the report does not contain.
    """
    from app.engines.industry_benchmarks import (DOMAIN_BENCHMARKS,
                                                 lookup_benchmark)
    if df is None or str(domain or "").lower() not in DOMAIN_BENCHMARKS:
        return False
    return any(lookup_benchmark(str(domain).lower(), str(c)) is not None
               for c in df.columns)


def _benchmark_section(story, s, T, domain, CW, df=None):
    """Reference ranges for this domain, and only this domain.

    What was here before was three hardcoded tables citing "Salesforce
    2024", "Gartner 2024", "Klaviyo 2024" and similar against specific
    figures — precise-looking attributions to reports whose contents
    cannot be checked from inside this app. That is the single most
    damaging thing a report can contain: a reader who looks one up and
    cannot find the number stops believing the rest of the document.
    Most rows also read "—" for the client's own value, so the table was
    largely a list of other people's numbers.

    Every range now comes from services/industry_benchmarks with a named
    source, is looked up strictly within the report's own domain, and is
    only shown for metrics this dataset actually contains. An HR report
    therefore cites HR sources and nothing else.
    """
    from app.engines.industry_benchmarks import (DOMAIN_BENCHMARKS,
                                                 lookup_benchmark)

    domain = str(domain or "").lower()
    if domain not in DOMAIN_BENCHMARKS or df is None:
        return

    def _num_mean(series):
        """Mean robust to Yes/No/True/False string encodings."""
        col = series
        if is_text_dtype(col):
            mapped = col.astype(str).str.strip().str.lower().map(
                {"yes": 1, "no": 0, "true": 1, "false": 0,
                 "y": 1, "n": 0, "1": 1, "0": 0})
            col = mapped if mapped.notna().any() else pd.to_numeric(
                col, errors="coerce")
        else:
            col = pd.to_numeric(col, errors="coerce")
        col = col.dropna()
        return float(col.mean()) if len(col) else None

    rows = []
    for c in df.columns:
        bm = lookup_benchmark(domain, str(c))
        if bm is None:
            continue
        value = _num_mean(df[c])
        if value is None:
            continue
        # A rate stored as 0-1 is the same measure as one stored as 0-100.
        shown = value
        if bm.unit == "%" and 0 <= value <= 1 and bm.high > 1:
            shown = value * 100
        if bm.unit in ("%", "x") or bm.unit.startswith("/"):
            value_str = "{:,.1f}{}".format(shown, bm.unit)
            range_str = "{:g}–{:g}{}".format(bm.low, bm.high, bm.unit)
        else:
            value_str = "{:,.1f} {}".format(shown, bm.unit).strip()
            range_str = "{:g}–{:g} {}".format(bm.low, bm.high, bm.unit).strip()

        if shown < bm.low:
            position = "below the range"
        elif shown > bm.high:
            position = "above the range"
        else:
            position = "within the range"
        rows.append([str(c), value_str, range_str, position, bm.source])

    if not rows:
        return

    _sec(story, s, T, "Performance Against Published Ranges",
         "Published {} ranges, against the figures in this dataset".format(
             _DOMAIN_LABELS.get(domain, domain.title())))
    story.append(Paragraph(
        "These are general, publicly-cited ranges, not a licensed benchmark "
        "set, and they move with sector, company size and region. They "
        "answer one question — is this figure in a plausible place — and "
        "nothing more. Where a figure sits outside a range, that is a "
        "prompt to look, not a finding in itself; the comparisons against "
        "this organisation's own distribution elsewhere in this report are "
        "the stronger evidence.", s["body"]))
    story.append(Spacer(1, 3 * mm))

    _gtable(story, T,
            ["Metric", "This dataset", "Published range", "Position", "Source"],
            rows,
            [CW * x for x in [0.22, 0.14, 0.17, 0.15, 0.32]])
    story.append(Paragraph(
        "Sources are named per row so each can be checked. No range here is "
        "attributed to a report this analysis has not drawn it from.",
        s["note"]))


# ══════════════════════════════════════════════════════════
#  ATTRITION PAGE
# ══════════════════════════════════════════════════════════

def _attrition_page(story, s, T, attrition, CW):
    if attrition is None: return
    _sec(story, s, T, "Attrition", "Who left, from where, and what it costs")

    dept = getattr(attrition, "dept_attrition", {}) or {}
    dept_p = getattr(attrition, "dept_p", None)
    dept_differs = dept_p is not None and dept_p < 0.05
    # A "highest department" is only worth a headline figure when the
    # departments really differ; otherwise it is whichever one chance put on top.
    top_dept = max(dept.items(), key=lambda kv: kv[1]) if dept and dept_differs else None
    kpis = [
        {"label": "ATTRITION RATE", "value": "{:.1f}%".format(attrition.rate),
         "sub": "{:,} of {:,} left".format(attrition.n_left, attrition.n_total),
         "color": T["negative"] if attrition.rate > 15 else T["primary"]},
        {"label": "PLANNING BAND", "value": "10–15%",
         "sub": "indicative; varies by sector"},
        {"label": "LOWEST-QUARTER SATISFACTION", "value": "{:,}".format(attrition.n_flight_risk),
         "sub": "{:.0f}% of those remaining".format(attrition.flight_risk_pct)},
    ]
    if top_dept:
        kpis.append({"label": "HIGHEST DEPARTMENT", "value": "{:.1f}%".format(top_dept[1]),
                     "sub": _clean(str(top_dept[0]))[:24]})
    _kpi_row(story, s, T, kpis, CW)

    _narrative_box(story, s, T,
                   getattr(attrition, "interpretation", ""))
    _narrative_box(story, s, T,
                   getattr(attrition, "cost_estimate", ""))

    # Drivers
    drivers = getattr(attrition, "top_drivers", [])
    if drivers:
        story.append(Paragraph("Attrition Drivers", s["h3"]))
        rows = [[d.get("factor","")[:18], d.get("type","").title(),
                 "{:.0f}%".format(d.get("impact",0)), d.get("detail","")[:65]]
                for d in drivers[:6]]
        _gtable(story, T, ["Factor","Type","Impact","Finding"],
                rows, [CW*x for x in [0.22,0.13,0.14,0.51]])

    # Dept breakdown
    dept_atr = getattr(attrition, "dept_attrition", {})
    if dept_atr:
        story.append(Paragraph("Attrition by Department", s["h3"]))
        sorted_d = sorted(dept_atr.items(), key=lambda x: x[1], reverse=True)
        rows = [[str(dept), "{:.1f}%".format(rate),
                 "above band" if rate > 15 else "within band" if rate >= 10 else "below band"]
                for dept, rate in sorted_d]
        _gtable(story, T, ["Department", "Rate", "Against the 10–15% band"],
                rows, [CW*0.50, CW*0.20, CW*0.30])
        if dept_p is not None:
            story.append(Paragraph(
                "Departments differ in attrition by more than chance (chi-square "
                "p={:.3f}).".format(dept_p) if dept_differs else
                "The differences between departments are within what chance alone "
                "produces (chi-square p={:.2f}); the ranking should not be relied "
                "on.".format(dept_p), s["note"]))


# ══════════════════════════════════════════════════════════
#  DATASET OVERVIEW
# ══════════════════════════════════════════════════════════

def _dataset_overview(story, s, T, df, profile, CW):
    _sec(story, s, T, "The Data", "What the dataset contains")

    # Identifiers get their own row and stay out of the statistics: the mean
    # of employee_id is not a figure anyone should read.
    from app.engines.domains.base import is_id_column
    id_cols  = [c for c in df.columns if is_id_column(c, df[c])]
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in id_cols]
    cat_cols = [c for c in text_columns(df) if c not in id_cols]
    dt_cols  = df.select_dtypes(include="datetime").columns.tolist()

    rows = [[kind, len(cols), _clean(", ".join(map(str, cols[:8])))]
            for kind, cols in (("Measures", num_cols), ("Categories", cat_cols),
                               ("Dates", dt_cols), ("Identifiers", id_cols)) if cols]
    _gtable(story, T, ["Kind", "Columns", "Names"], rows,
            [CW*0.20, CW*0.12, CW*0.68])

    if num_cols:
        # One row per measure, figures formatted as a reader writes them;
        # the old table printed 59333.386 and cut headers to 10 letters.
        from app.engines.chart_exporter import fmt
        story.append(Paragraph("Descriptive statistics", s["h3"]))
        rows = []
        for c in num_cols[:10]:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if v.empty:
                continue
            q = v.quantile([0.25, 0.5, 0.75])
            rows.append([_clean(str(c)), fmt(len(v)), fmt(v.mean()), fmt(q[0.5]),
                         fmt(v.min()), fmt(q[0.25]), fmt(q[0.75]), fmt(v.max())])
        _gtable(story, T, ["Measure", "Count", "Mean", "Median", "Min", "25th pct",
                           "75th pct", "Max"], rows,
                [CW * x for x in (0.24, 0.10, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11)])
        skewed = [str(c) for c in num_cols[:10]
                  if abs(float(pd.to_numeric(df[c], errors="coerce").skew() or 0)) > 1]
        if skewed:
            story.append(Paragraph(
                "{} {} skewed, so the median describes a typical record better "
                "than the mean.".format(", ".join(_clean(c) for c in skewed),
                                        "is" if len(skewed) == 1 else "are"),
                s["note"]))


# ══════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════

def _stats_section(story, s, T, stats_report, CW):
    if stats_report is None: return
    _sec(story, s, T, "Statistical Analysis",
         "How each measure is distributed, and which move together")

    col_stats = getattr(stats_report, "column_stats", {})
    rows = []
    for col, cs in list(col_stats.items())[:10]:
        if getattr(cs, "mean", None) is None:
            continue
        rows.append([_clean(str(col)),
                     "Normal" if getattr(cs, "is_normal", False) else "Not normal",
                     (getattr(cs, "skew_label", "") or "—").capitalize(),
                     str(getattr(cs, "outlier_count_iqr", 0))])
    if rows:
        _gtable(story, T, ["Measure", "Distribution", "Shape", "Outliers (1.5×IQR)"],
                rows, [CW * x for x in (0.30, 0.20, 0.32, 0.18)])

    # Correlations
    corrs = getattr(stats_report, "correlations", [])
    sig   = [c for c in corrs
             if getattr(c, "is_significant", False)
             and abs(getattr(c, "pearson_r", 0)) >= 0.15]
    if sig:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Measures that move together", s["h3"]))
        rows = [[_clean(c.col_a), _clean(c.col_b), "{:.2f}".format(c.pearson_r),
                 "{:.3f}".format(getattr(c, "p_value", 0)), c.strength.capitalize(),
                 "{:.0f}%".format(c.pearson_r ** 2 * 100)]
                for c in sig[:6]]
        _gtable(story, T,
                ["Measure", "With", "r", "p", "Strength", "Variance shared"],
                rows, [CW*x for x in [0.24, 0.24, 0.10, 0.10, 0.16, 0.16]])
        story.append(Paragraph(
            "Two measures that move together are associated; that does not show "
            "that one drives the other.", s["note"]))


# ══════════════════════════════════════════════════════════
#  BUSINESS INTELLIGENCE
# ══════════════════════════════════════════════════════════

def _bi_section(story, s, T, bi_report, CW):
    if bi_report is None: return
    _sec(story, s, T, "Business Intelligence",
         "Spread, segments and what separates the best from the rest")

    brief = getattr(bi_report, "executive_brief", "")
    if brief:
        _narrative_box(story, s, T, brief)

    # Benchmarks
    bms = getattr(bi_report, "benchmarks", [])
    if bms:
        from app.engines.chart_exporter import fmt
        story.append(Paragraph("How spread out each measure is", s["h3"]))
        rows = [[_clean(bm.column), fmt(bm.mean), fmt(bm.median),
                 fmt(bm.top_10_pct), fmt(bm.bottom_10_pct),
                 bm.benchmark_label.split("—")[0].strip()[:24]]
                for bm in bms[:6]]
        _gtable(story, T,
                ["Metric","Mean","Median","Top 10%","Bottom 10%","Variation"],
                rows, [CW*x for x in [0.22,0.12,0.12,0.12,0.13,0.29]])

    # Cohorts
    sig_c = [c for c in getattr(bi_report, "cohorts", [])
             if c.is_significant]
    if sig_c:
        story.append(Paragraph("Segments that differ", s["h3"]))
        for c in sig_c[:3]:
            story.append(Paragraph("• " + c.interpretation, s["bl"]))

    # Key insights
    ki = getattr(bi_report, "key_insights", [])
    if ki:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Observations", s["h3"]))
        for ins in ki[:5]:
            story.append(Paragraph("• " + str(ins), s["bl"]))


def _ml_section(story, s, T, ml_report, CW):
    """The model the user last trained on this dataset.

    The Reports page has always offered "ML results (if trained)"; the
    report was passed in and never written, so the box did nothing.
    """
    best = getattr(ml_report, "best_model", None)
    if best is None:
        return
    target = ml_report.target_col
    _sec(story, s, T, "Predictive Model",
         "What the best model trained on '{}' can and cannot tell you".format(target))

    if ml_report.task == "regression":
        score = "R² {:.2f} on held-out rows".format(best.test_score)
        if best.rmse is not None:
            score += " · typical error ±{:,.2f}".format(best.rmse)
    else:
        score = "{:.1f}% accuracy on held-out rows".format(best.test_score * 100)
    _narrative_box(story, s, T, _clean("{} predicting '{}' from {} column(s), {:,} rows: {}.".format(
        best.name, target, ml_report.n_features, ml_report.n_rows_used, score)))

    note = getattr(ml_report, "no_signal", "")
    # A model with no signal has nothing else worth reporting; its verdict
    # is printed once, below.
    for line in ([] if note else ml_report.insights[:4]):
        story.append(Paragraph("• " + _clean(str(line)), s["bl"]))

    imps = ml_report.feature_importance[:6]
    if imps and not note:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("What the model relies on", s["h3"]))
        _gtable(story, T, ["Column", "Share of the model", "Direction"],
                [[_clean(f.feature), "{:.0f}%".format(f.importance * 100), f.direction]
                 for f in imps],
                [CW * x for x in (0.50, 0.25, 0.25)])
        story.append(Paragraph(
            "Reliance is an association the model found, not proof that "
            "changing a column would change '{}'.".format(_clean(target)), s["bl"]))
    elif note:
        # No ranking: it would only order noise.
        story.append(Spacer(1, 2*mm))
        _narrative_box(story, s, T, _clean(note))

    # The task description ("Continuous numeric target (497 unique values)")
    # is an internal detail, not a caveat for the reader.
    reason = getattr(ml_report, "task_reason", None)
    for w in [w for w in ml_report.warnings if w and w != reason][:3]:
        story.append(Paragraph("<b>Note:</b> " + _clean(str(w)), s["bl"]))


# ══════════════════════════════════════════════════════════
#  CHART PAGE
# ══════════════════════════════════════════════════════════

def _charts_section(story, s, T, chart_data, CW):
    """All charts under one heading, each at its true proportions with the
    caption computed from its own data. Every chart used to get its own
    page and a fixed 2:1 frame, which squashed some and stretched others."""
    if not chart_data:
        return
    _sec(story, s, T, "Charts", "Each caption is computed from the figures plotted")
    for i, (title, img_bytes, caption) in enumerate(chart_data, 1):
        if not img_bytes:
            continue
        try:
            from PIL import Image as _PIL
            w, h = _PIL.open(io.BytesIO(img_bytes)).size
            width = CW
            height = CW * h / w
            if height > 105 * mm:            # keep tall charts on the page
                height = 105 * mm
                width = height * w / h
            img = Image(io.BytesIO(img_bytes), width=width, height=height)
            img.hAlign = "LEFT"
        except Exception:
            logger.warning("chart image could not be placed", exc_info=True)
            continue
        block = [Paragraph("{}. {}".format(i, _clean(title)), s["h3"]), img]
        if caption:
            block.append(Paragraph(_clean(caption), s["caption"]))
        block.append(Spacer(1, 5 * mm))
        story.append(KeepTogether(block))


# ══════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════

def _recommendations(story, s, T, actions, CW):
    _sec(story, s, T, "Recommended Next Steps")
    if not actions:
        _narrative_box(story, s, T,
                       "The data did not point to a specific action beyond the findings "
                       "above. Columns such as targets, owners or categories allow more "
                       "specific recommendations.")

    pri_map = {
        "CRITICAL":   (T["negative"],  T["critical_bg"]),
        "SHORT TERM": (T["warning"],   T["warning_bg"]),
        "LONG TERM":  (T["info"],      T["info_bg"]),
    }

    for n, action in enumerate(actions[:9], 1):
        action_str = str(action)
        # Only an explicit label sets urgency; matching the words anywhere
        # in the text tagged "...a critical supplier..." as CRITICAL.
        priority   = "NEXT STEP"
        for p in ("CRITICAL", "SHORT TERM", "LONG TERM"):
            if action_str.upper().startswith("[" + p + "]"):
                priority = p
                break

        col, bg = pri_map.get(priority, (T["accent"], T["bg_card"]))
        text    = _clean(action_str
                   .replace("[CRITICAL] ", "")
                   .replace("[SHORT TERM] ", "")
                   .replace("[LONG TERM] ", "")
                   .strip())

        card = Table([[
            Paragraph(str(n), ParagraphStyle(
                "pri", fontName=FONT_BOLD, fontSize=10,
                textColor=HexColor(col), alignment=TA_CENTER)),
            Paragraph(text, s["body"]),
        ]], colWidths=[CW*0.07, CW*0.93])
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), HexColor(bg)),
            ("LINEBEFORE",    (0,0), (0,-1),  3, HexColor(col)),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether([card, Spacer(1, 2*mm)]))

    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Each step follows from this dataset alone; check it against what the "
        "data cannot show before acting on it.", s["note"]))


# ══════════════════════════════════════════════════════════
#  APPENDIX
# ══════════════════════════════════════════════════════════

def _prepared_by_line(config: dict) -> str:
    """Who prepared the report, for the basis of preparation.

    A review deliverable is signed by the person or firm accountable for
    it. What produced the document is not the reader's concern and is
    named nowhere — the analyst's or consultancy's name is what belongs
    here, and it is the client's or freelancer's to set.
    """
    who = str(config.get("prepared_by") or "").strip()
    if not who:
        return ""
    return " Prepared by {}, who is responsible for the analysis and the " \
           "conclusions drawn from it.".format(_clean(who))


def _appendix(story, s, T, config, CW, domain: str = "general"):
    _sec(story, s, T, "Appendix: Method and Sources")

    # This section is what a reviewing analyst reads to decide whether to
    # trust the rest. It states the tests actually applied and why each was
    # chosen — it does not describe the tooling that rendered the document,
    # which tells the reader nothing about validity.
    story.append(Paragraph("A. Analytical Method", s["h3"]))
    story.append(Paragraph(
        "Every figure in this report is computed directly from the supplied "
        "dataset. No values are estimated, imputed into the findings, or "
        "carried over from other engagements.",
        s["body"]))
    story.append(Paragraph(
        "<b>Distributional testing.</b> Normality is assessed with Shapiro-Wilk "
        "(n≤5,000) and D'Agostino-Pearson, rather than assumed. The outcome "
        "determines which downstream test is used, so a non-normal column is "
        "never summarised with a statistic that presumes normality.",
        s["body"]))
    story.append(Paragraph(
        "<b>Association.</b> Pearson's r is used where both variables are "
        "approximately normal; Spearman's rank correlation otherwise. "
        "Correlations are reported with their p-value and sample size. "
        "Pairs that are mechanically related (a rate against its own "
        "numerator, a duplicated column) are excluded rather than presented "
        "as findings.",
        s["body"]))
    story.append(Paragraph(
        "<b>Group differences.</b> Two-group comparisons use Welch's t-test "
        "where the normality condition holds and Mann-Whitney U where it does "
        "not; comparisons across three or more groups use one-way ANOVA or "
        "Kruskal-Wallis on the same basis. Categorical association uses "
        "Chi-square with an expected-frequency check.",
        s["body"]))
    story.append(Paragraph(
        "<b>Outliers.</b> Flagged by the 1.5×IQR rule and cross-checked with "
        "the modified Z-score (Iglewicz &amp; Hoaglin), which is robust to "
        "skew. Outliers are reported, never silently removed — an extreme "
        "value is frequently the finding rather than an error.",
        s["body"]))
    story.append(Paragraph(
        "<b>Missing and duplicate records.</b> Completeness is measured per "
        "column and reported before any analysis. Records are not dropped "
        "to improve a result; where a test required complete cases, the "
        "excluded count is stated alongside it.",
        s["body"]))
    story.append(Paragraph(
        "<b>Judgement applied.</b> Where more than one treatment was "
        "defensible, the more conservative was taken: findings that did not "
        "survive correction for multiple testing were dropped rather than "
        "reported with a caveat, effect sizes below the level that would "
        "change a decision were left out, and no figure was carried into a "
        "conclusion that the underlying column could not support. Candidate "
        "findings withheld on that basis are counted in the findings "
        "section rather than removed silently.",
        s["body"]))
    story.append(Paragraph(
        "<b>Limitations.</b> Findings describe association within this "
        "dataset and the period it covers. They do not establish causation, "
        "and do not extrapolate beyond the observed range of each variable. "
        "Segment-level results with small denominators are marked as "
        "directional. Where a question could not be answered from the data "
        "supplied, this report says so rather than answering it from "
        "general expectation.",
        s["body"]))

    story.append(Paragraph("B. Quality Score Formula", s["h3"]))
    _gtable(story, T,
            ["Component", "Weight", "Description"],
            [["Completeness",  "60%", "% of non-missing cells"],
             ["Deduplication", "30%", "% of unique rows"],
             ["Column Health", "10%", "Avg per-column quality score"]],
            [CW*0.25, CW*0.15, CW*0.60])

    # Reference ranges are listed per detected domain. This list was
    # previously hardcoded to HR sources, so a finance or e-commerce report
    # cited SHRM attrition benchmarks and Gallup engagement data — an
    # immediate credibility failure for any reader who checks.
    # Sources come from the domain blueprint so a finance report cites
    # finance conventions and an HR report cites HR bodies — the previous
    # single list put SHRM and Gallup in the footer of every report
    # regardless of what it was about.
    _bp = blueprint_for(domain)
    _sources = list(_bp.references) or [
        "No external benchmark set applies to this dataset's domain. All "
        "comparisons in this report are internal — each metric is measured "
        "against its own distribution within the supplied data."]
    story.append(Paragraph("C. Reference Ranges & Sources", s["h3"]))
    for src in _sources:
        story.append(Paragraph("• " + src, s["bl"]))
    if _bp.reference_note:
        story.append(Paragraph(_bp.reference_note, s["note"]))



# ══════════════════════════════════════════════════════════
#  MAIN PUBLIC FUNCTION  (identical signature to original)
# ══════════════════════════════════════════════════════════

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
) -> bytes:
    """
    Build complete senior-analyst PDF report.
    Identical public API to original pdf_builder.py — drop-in replacement.
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
        auto_key  = DOMAIN_THEMES.get(domain, "Corporate Light")
        theme_name = auto_key
    T = THEMES[theme_name]
    config = dict(config)
    config["domain"]     = domain
    config["theme_name"] = theme_name

    report_title = config.get("title", "Data Analysis Report")
    client_name  = config.get("client_name", "Client")
    report_date  = datetime.now().strftime("%B %d, %Y")

    # ── Facts about the data for the cover ────────────────
    complete = 100 - df.isna().mean().mean() * 100 if len(df.columns) else 100
    kpis_cover = [
        {"label": "Records", "value": "{:,}".format(len(df))},
        {"label": "Columns", "value": str(df.shape[1])},
        {"label": "Cells filled", "value": "{:.1f}%".format(complete)},
    ]
    dates = df.select_dtypes(include="datetime")
    if not dates.empty:
        col = dates.notna().sum().idxmax()
        lo, hi = dates[col].min(), dates[col].max()
        if pd.notna(lo) and pd.notna(hi):
            kpis_cover.append({"label": "Period covered",
                               "value": "{} – {}".format(lo.strftime("%b %Y"),
                                                         hi.strftime("%b %Y"))
                               if lo.year != hi.year or lo.month != hi.month
                               else lo.strftime("%b %Y")})

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

    doc = _ReportDoc(
        content_buf, pagesize=A4,
        leftMargin=M, rightMargin=M,
        topMargin=22*mm, bottomMargin=18*mm,
        title=report_title, author=client_name,
    )
    frame = Frame(M, 18*mm, CW, H - 40*mm,
                  leftPadding=0, rightPadding=0,
                  topPadding=0,  bottomPadding=0)
    tpl   = PageTemplate(id="main", frames=[frame], onPage=lambda c,d: None)
    doc.addPageTemplates([tpl])

    s     = _styles(T)
    story = []

    # ── Assemble: sections flow; each starts a new page only when too
    # little room is left (see _sec). ─────────────────────
    _toc(story, s, T, CW)
    _basis_box(story, s, T, config)
    story.append(PageBreak())

    _exec_summary(story, s, T, executive_summary,
                  findings, risks, opportunities, CW)
    _dq_note(story, s, T, df, profile, CW)
    if cleaning_summary:
        _data_prep_section(story, s, T, cleaning_summary, CW,
                           table=config.get("source_table") or "source_table")
    if _has_reference_ranges(domain, df):
        _benchmark_section(story, s, T, domain, CW, df=df)
    _top_insights(story, s, T, top_insights, CW, domain=domain)
    if attrition:
        _attrition_page(story, s, T, attrition, CW)
    _charts_section(story, s, T, chart_data, CW)
    _dataset_overview(story, s, T, df, profile, CW)
    if stats_report:
        _stats_section(story, s, T, stats_report, CW)
    if bi_report:
        _bi_section(story, s, T, bi_report, CW)
    if getattr(ml_report, "best_model", None) is not None:
        _ml_section(story, s, T, ml_report, CW)
    _recommendations(story, s, T, recommendations, CW)
    # No forced break: a short "Next Steps" was left alone on its page.
    _appendix(story, s, T, config, CW, domain=domain)

    # ── Build PDF ─────────────────────────────────────────
    # Two passes: the first finds where each section starts, the second
    # prints those page numbers on the contents page.
    doc.multiBuild(_bind_headings(story), canvasmaker=canvas_maker)
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
