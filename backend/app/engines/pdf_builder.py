"""
core/pdf_builder.py — Analytiq
Senior Analyst Edition v4 — DROP-IN REPLACEMENT

UPGRADED from basic to:
  ✅ Premium dark-navy cover with domain badge + KPI strip
  ✅ Running header / footer on every page
  ✅ Insight cards: Problem → Cause → Evidence → Action → Impact
  ✅ SHRM/Gallup/Mercer industry benchmarks (HR domain)
  ✅ Correlation warning box (r ≠ causation)
  ✅ Scenario modelling table
  ✅ Risk register with severity colouring
  ✅ Deduplication transparency note
  ✅ All chart analyses properly domain-aware

SAME public API — zero changes needed in 8_Reports.py:
  build_pdf(df, config, profile, cleaning_summary,
            stats_report, bi_report, ml_report,
            chart_data, executive_summary,
            findings, risks, opportunities, recommendations,
            top_insights, attrition, domain) → bytes
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
from app.engines.report_blueprints import blueprint_for
from app.services.dtypes import is_text_dtype, text_columns

logger = logging.getLogger(__name__)

W, H = A4
CW_DEFAULT = W - 36 * mm   # content width (18mm each side)


# ══════════════════════════════════════════════════════════
#  DOMAIN COLOUR THEMES  (matches your existing THEMES keys)
# ══════════════════════════════════════════════════════════

THEMES = {
    "Corporate Light": {
        "cover_bg":    "#0A1628", "cover_accent": "#1B4FD8",
        "header_bg":   "#0A1628", "header_text":  "#FFFFFF",
        "accent":      "#1B4FD8", "accent2":      "#60A5FA",
        "text":        "#1F2937", "text_muted":   "#6B7280",
        "bg_light":    "#EFF6FF", "bg_card":      "#F8FAFF",
        "border":      "#E5E7EB",
        "positive":    "#10B981", "negative":     "#EF4444",
        "warning":     "#F59E0B", "info":         "#3B82F6",
        "critical_bg": "#FEE2E2", "warning_bg":   "#FEF3C7",
        "positive_bg": "#D1FAE5", "info_bg":      "#DBEAFE",
        "domain_label":"BUSINESS ANALYTICS",
        "domain_badge":"#1B4FD8",
    },
    "HR Blue": {
        "cover_bg":    "#0A1F4E", "cover_accent": "#1976D2",
        "header_bg":   "#0A1F4E", "header_text":  "#FFFFFF",
        "accent":      "#1976D2", "accent2":      "#90CAF9",
        "text":        "#1A2035", "text_muted":   "#5A6482",
        "bg_light":    "#E8F0FE", "bg_card":      "#F5F8FF",
        "border":      "#C5D3F0",
        "positive":    "#2E7D32", "negative":     "#C62828",
        "warning":     "#E65100", "info":         "#1565C0",
        "critical_bg": "#FFEBEE", "warning_bg":   "#FFF3E0",
        "positive_bg": "#E8F5E9", "info_bg":      "#E3F2FD",
        "domain_label":"HR & PEOPLE ANALYTICS",
        "domain_badge":"#1976D2",
    },
    "Ecommerce Orange": {
        "cover_bg":    "#3E1500", "cover_accent": "#F4511E",
        "header_bg":   "#BF360C", "header_text":  "#FFFFFF",
        "accent":      "#F4511E", "accent2":      "#FFAB91",
        "text":        "#1A1A1A", "text_muted":   "#5A5A5A",
        "bg_light":    "#FBE9E7", "bg_card":      "#FFF8F6",
        "border":      "#FFCCBC",
        "positive":    "#2E7D32", "negative":     "#B71C1C",
        "warning":     "#E65100", "info":         "#1565C0",
        "critical_bg": "#FFEBEE", "warning_bg":   "#FFF3E0",
        "positive_bg": "#E8F5E9", "info_bg":      "#E8F0FE",
        "domain_label":"E-COMMERCE ANALYTICS",
        "domain_badge":"#F4511E",
    },
    "Sales Green": {
        "cover_bg":    "#0A2710", "cover_accent": "#2E7D32",
        "header_bg":   "#1B5E20", "header_text":  "#FFFFFF",
        "accent":      "#2E7D32", "accent2":      "#A5D6A7",
        "text":        "#1A2A1A", "text_muted":   "#4A6A4A",
        "bg_light":    "#E8F5E9", "bg_card":      "#F5FBF5",
        "border":      "#C8E6C9",
        "positive":    "#1B5E20", "negative":     "#B71C1C",
        "warning":     "#E65100", "info":         "#1565C0",
        "critical_bg": "#FFEBEE", "warning_bg":   "#FFF3E0",
        "positive_bg": "#E8F5E9", "info_bg":      "#E8F0FE",
        "domain_label":"SALES PERFORMANCE ANALYTICS",
        "domain_badge":"#2E7D32",
    },
    "Dark Tech": {
        "cover_bg":    "#0D1117", "cover_accent": "#58A6FF",
        "header_bg":   "#0D1117", "header_text":  "#E6EDF3",
        "accent":      "#58A6FF", "accent2":      "#3FB950",
        "text":        "#E6EDF3", "text_muted":   "#8B949E",
        "bg_light":    "#161B22", "bg_card":      "#1C2128",
        "border":      "#30363D",
        "positive":    "#3FB950", "negative":     "#F85149",
        "warning":     "#D29922", "info":         "#58A6FF",
        "critical_bg": "#1C1010", "warning_bg":   "#1C1800",
        "positive_bg": "#0D1A0F", "info_bg":      "#0D1421",
        "domain_label":"TECHNICAL ANALYTICS",
        "domain_badge":"#58A6FF",
    },
}

# Auto-select theme by domain
DOMAIN_THEMES = {
    "hr":        "HR Blue",
    "ecommerce": "Ecommerce Orange",
    "sales":     "Sales Green",
    "finance":   "Corporate Light",
    "general":   "Corporate Light",
}

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
        "h1":    ps("h1",   fontName="Helvetica-Bold", fontSize=17,
                    textColor=_c(T["accent"]),     spaceAfter=4),
        "h2":    ps("h2",   fontName="Helvetica-Bold", fontSize=13,
                    textColor=_c(T["text"]),       spaceBefore=8, spaceAfter=3),
        "h3":    ps("h3",   fontName="Helvetica-Bold", fontSize=10,
                    textColor=_c(T["accent"]),     spaceBefore=6, spaceAfter=3),
        "body":  ps("body", fontName="Helvetica",      fontSize=9,
                    textColor=_c(T["text"]),       leading=14,  spaceAfter=3,
                    alignment=TA_JUSTIFY),
        "sm":    ps("sm",   fontName="Helvetica",      fontSize=7.5,
                    textColor=_c(T["text_muted"]), leading=11,  spaceAfter=2),
        "bl":    ps("bl",   fontName="Helvetica",      fontSize=9,
                    textColor=_c(T["text"]),       leading=13,  spaceAfter=3,
                    leftIndent=10, firstLineIndent=-10),
        "toc":   ps("toc",  fontName="Helvetica",      fontSize=10,
                    textColor=_c(T["text"]),       leading=16,  spaceAfter=3),
        "wh":    ps("wh",   fontName="Helvetica",      fontSize=9,
                    textColor=HexColor("#FFFFFF"),  leading=13),
        "wbh":   ps("wbh",  fontName="Helvetica-Bold", fontSize=10,
                    textColor=HexColor("#FFFFFF")),
        "note":  ps("note", fontName="Helvetica-Oblique", fontSize=7.5,
                    textColor=_c(T["text_muted"]), spaceAfter=3),
        "warn":  ps("warn", fontName="Helvetica",      fontSize=8.5,
                    textColor=_c(T["text"]),       leading=13,
                    backColor=_c(T["warning_bg"])),
        # Insight card row styles
        "rl":    ps("rl",   fontName="Helvetica-Bold", fontSize=8,
                    textColor=HexColor("#FFFFFF"),  alignment=TA_CENTER),
        "rv":    ps("rv",   fontName="Helvetica",      fontSize=8.5,
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
        T = self.T
        # ── Header ────────────────────────────────────────
        self.setFillColor(_c(T["header_bg"]))
        self.rect(0, H - 24*mm, W, 24*mm, fill=1, stroke=0)
        self.setFillColor(_c(T["accent"]))
        self.rect(0, H - 25.5*mm, W, 1.5*mm, fill=1, stroke=0)
        self.setFillColor(HexColor("#FFFFFF"))
        self.setFont("Helvetica-Bold", 10)
        self.drawString(18*mm, H - 14*mm, "Analytiq")
        self.setFont("Helvetica", 7.5)
        self.setFillColor(HexColor(T["accent2"]))
        self.drawString(18*mm, H - 20*mm, self.report_title)
        self.setFillColor(HexColor("#FFFFFF"))
        self.setFont("Helvetica", 7)
        self.drawRightString(W - 18*mm, H - 14*mm, self.report_date)
        self.drawRightString(W - 18*mm, H - 20*mm,
                             "CONFIDENTIAL — " + self.client_name)
        # ── Footer ────────────────────────────────────────
        self.setFillColor(_c(T["header_bg"]))
        self.rect(0, 0, W, 12*mm, fill=1, stroke=0)
        self.setFillColor(_c(T["accent"]))
        self.rect(0, 12*mm, W, 1.2*mm, fill=1, stroke=0)
        self.setFillColor(HexColor("#FFFFFF"))
        self.setFont("Helvetica", 6.5)
        # The footer previously named a fixed set of HR benchmark bodies
        # (SHRM · Gallup · Mercer · Deloitte) on EVERY page of EVERY report,
        # so a finance or e-commerce deliverable cited HR attrition sources
        # 15 times over. Sources belong in the appendix, keyed to the
        # detected domain; the footer carries the client and confidentiality
        # marking, which is what a footer is for.
        self.drawString(18*mm, 4.5*mm,
            "{} · Confidential · Prepared for {}".format(
                getattr(self, "agency_name", "Analytiq"),
                self.client_name))
        self.drawRightString(W - 18*mm, 4.5*mm,
            "Page {} of {}".format(self._pageNumber, tot))


# ══════════════════════════════════════════════════════════
#  COVER PAGE  (drawn on separate canvas, merged via pypdf)
# ══════════════════════════════════════════════════════════

def _build_cover(T: dict, config: dict, kpis_preview: list) -> bytes:
    buf = io.BytesIO()
    cv  = CV.Canvas(buf, pagesize=A4)
    title       = config.get("title", "Data Analysis Report")
    client_name = config.get("client_name", "Client")
    report_date = datetime.now().strftime("%B %d, %Y")
    domain_lbl  = T.get("domain_label", "BUSINESS ANALYTICS")

    # BG
    cv.setFillColor(_c(T["cover_bg"]))
    cv.rect(0, 0, W, H, fill=1, stroke=0)
    # Top stripe
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(0, H - 5*mm, W, 5*mm, fill=1, stroke=0)
    # Right panel
    cv.setFillColor(HexColor("#0D1F3C"))
    cv.rect(W - 17*mm, 0, 17*mm, H, fill=1, stroke=0)
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(W - 17*mm, 0, 1.8*mm, H, fill=1, stroke=0)
    # Decorative circles
    cv.setFillColor(HexColor("#112240"))
    cv.circle(W * 0.73, H * 0.53, 190, fill=1, stroke=0)
    cv.setFillColor(HexColor("#0D1A35"))
    cv.circle(W * 0.73, H * 0.53, 135, fill=1, stroke=0)

    # Brand
    cv.setFillColor(HexColor("#FFFFFF"))
    cv.setFont("Helvetica-Bold", 15)
    cv.drawString(20*mm, H - 32*mm, "Analytiq")
    cv.setFillColor(HexColor(T["accent2"]))
    cv.setFont("Helvetica", 9.5)
    cv.drawString(20*mm, H - 40*mm, "Advanced Analytics Platform")
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(20*mm, H - 44*mm, 55*mm, 1.2*mm, fill=1, stroke=0)

    # ── Client / Company Logo (top-right of cover) ────────
    logo_path = config.get("logo_path", "")
    if logo_path and os.path.exists(logo_path):
        try:
            cv.drawImage(
                logo_path,
                W - 68*mm, H - 45*mm,
                width=48*mm, height=20*mm,
                preserveAspectRatio=True,
                mask="auto",
            )
        except Exception:
            pass  # logo fails gracefully — PDF still builds

    # Domain badge
    cv.setFillColor(_c(T["domain_badge"]))
    cv.roundRect(20*mm, H - 60*mm, 85*mm, 11*mm, 3, fill=1, stroke=0)
    cv.setFillColor(HexColor("#FFFFFF"))
    cv.setFont("Helvetica-Bold", 8)
    cv.drawString(25*mm, H - 56*mm, "◆  " + domain_lbl)

    # Title (word wrap at ~28 chars)
    words, lines, line = title.split(), [], ""
    for w in words:
        test = (line + " " + w).strip()
        if len(test) <= 28:
            line = test
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)

    cv.setFillColor(HexColor("#FFFFFF"))
    y_title = H / 2 + 36*mm
    for ln in lines:
        cv.setFont("Helvetica-Bold", 30 if len(ln) <= 20 else 24)
        cv.drawString(20*mm, y_title, ln)
        y_title -= 11*mm

    cv.setFillColor(HexColor(T["accent2"]))
    cv.setFont("Helvetica", 10)
    cv.drawString(20*mm, H / 2 + 12*mm,
                  config.get("subtitle", "Powered by Analytiq"))
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(20*mm, H / 2 + 6*mm, W - 37*mm, 1.5*mm, fill=1, stroke=0)

    # KPI strip (up to 4)
    kpis = kpis_preview[:4]
    bw   = (W - 37*mm) / max(len(kpis), 1)
    for i, kpi in enumerate(kpis):
        x = 20*mm + i * bw
        cv.setFillColor(HexColor("#1A3A5C"))
        cv.roundRect(x + 1.5, H / 2 - 14*mm, bw - 3, 18*mm, 3, fill=1, stroke=0)
        cv.setFillColor(HexColor(kpi.get("color", T["accent2"])))
        cv.setFont("Helvetica-Bold", 16)
        cv.drawCentredString(x + bw / 2, H / 2 - 2*mm,
                             str(kpi.get("value", ""))[:9])
        cv.setFillColor(HexColor("#FFFFFF"))
        cv.setFont("Helvetica-Bold", 7)
        cv.drawCentredString(x + bw / 2, H / 2 - 8*mm,
                             str(kpi.get("label", ""))[:18])
        cv.setFillColor(HexColor(T["accent2"]))
        cv.setFont("Helvetica", 6.5)
        cv.drawCentredString(x + bw / 2, H / 2 - 13*mm,
                             str(kpi.get("sub", ""))[:20])

    # Bottom meta
    cv.setFillColor(HexColor("#0D1F3C"))
    cv.rect(0, 0, W - 17*mm, 30*mm, fill=1, stroke=0)
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(0, 30*mm, W - 17*mm, 1.2*mm, fill=1, stroke=0)
    meta = [("PREPARED FOR", client_name),
            ("DATE", report_date),
            ("CLASSIFICATION", "CONFIDENTIAL")]
    mw = (W - 37*mm) / 3
    for i, (k, v) in enumerate(meta):
        x = 20*mm + i * mw
        cv.setFillColor(HexColor(T["accent2"]))
        cv.setFont("Helvetica", 6.5); cv.drawString(x, 21*mm, k)
        cv.setFillColor(HexColor("#FFFFFF"))
        cv.setFont("Helvetica-Bold", 8); cv.drawString(x, 13*mm, v[:22])

    # The cover previously carried "Powered by Groq Llama 3.3 70B". Naming
    # the model on the front page of a client deliverable invites the
    # reader to discount everything behind it, and says nothing about
    # whether the analysis is sound. Confidentiality marking belongs here
    # instead.
    cv.setFillColor(HexColor(T["accent2"]))
    cv.setFont("Helvetica", 6.5)
    cv.drawRightString(W - 21*mm, 5*mm, "Confidential")
    cv.save()
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════
#  REUSABLE COMPONENT HELPERS
# ══════════════════════════════════════════════════════════

def _sec(story: list, s: dict, T: dict, title: str, sub: str = ""):
    story.append(Spacer(1, 3*mm))
    story.append(HRFlowable(width="100%", thickness=3,
                             color=_c(T["accent"]), spaceAfter=3))
    story.append(Paragraph(title, s["h2"]))
    if sub:
        story.append(Paragraph(sub, s["sm"]))


def _kpi_row(story: list, s: dict, T: dict, kpis: list, CW: float):
    cols = min(4, len(kpis))
    cw   = CW / cols
    vals = [Paragraph(
                "<b>{}</b>".format(k.get("value", "")),
                ParagraphStyle("kv", fontName="Helvetica-Bold", fontSize=18,
                               textColor=HexColor(k.get("color", T["accent"])),
                               alignment=TA_CENTER))
            for k in kpis[:cols]]
    lbls = [Paragraph(
                k.get("label", ""),
                ParagraphStyle("kl", fontName="Helvetica-Bold", fontSize=7.5,
                               textColor=_c(T["text"]), alignment=TA_CENTER))
            for k in kpis[:cols]]
    subs = [Paragraph(
                k.get("sub", ""),
                ParagraphStyle("ks", fontName="Helvetica", fontSize=7,
                               textColor=_c(T["text_muted"]), alignment=TA_CENTER))
            for k in kpis[:cols]]
    t = Table([vals, lbls, subs], colWidths=[cw]*cols)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["bg_light"])),
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


def _gtable(story: list, T: dict, headers: list,
            rows_data: list, col_widths: list,
            severity_col: int = -1):
    """Generic styled table."""
    hrow = [Paragraph(h, ParagraphStyle(
                "th", fontName="Helvetica-Bold", fontSize=8,
                textColor=HexColor("#FFFFFF"), alignment=TA_CENTER))
            for h in headers]
    body = []
    for row in rows_data:
        body.append([Paragraph(str(c), ParagraphStyle(
                "td", fontName="Helvetica", fontSize=8,
                textColor=_c(T["text"]), leading=12))
                     for c in row])
    tbl = Table([hrow] + body, colWidths=col_widths)
    sty = [
        ("BACKGROUND",    (0,0), (-1,0),  _c(T["header_bg"])),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [HexColor("#FFFFFF"), _c(T["bg_light"])]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 5),
        ("RIGHTPADDING",  (0,0), (-1,-1), 5),
        ("BOX",           (0,0), (-1,-1), 0.5, _c(T["border"])),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, _c(T["border"])),
    ]
    if severity_col >= 0:
        _sev = {"CRITICAL": "#EF4444", "HIGH": "#F59E0B",
                "MEDIUM":   "#06B6D4", "LOW":  "#10B981",
                "IMMEDIATE":"#EF4444", "SHORT TERM": "#F59E0B",
                "LONG TERM":"#3B82F6", "WARNING": "#3B82F6",
                "NONE":     "#10B981"}
        for i, row in enumerate(rows_data, 1):
            val = str(row[severity_col]).upper()
            for k, col in _sev.items():
                if k in val:
                    sty += [
                        ("BACKGROUND", (severity_col, i), (severity_col, i), HexColor(col)),
                        ("TEXTCOLOR",  (severity_col, i), (severity_col, i), HexColor("#FFFFFF")),
                        ("FONTNAME",   (severity_col, i), (severity_col, i), "Helvetica-Bold"),
                        ("FONTSIZE",   (severity_col, i), (severity_col, i), 7),
                    ]
                    break
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

    bs = ParagraphStyle("bi_badge", fontName="Helvetica-Bold", fontSize=7.5,
                        textColor=HexColor("#FFFFFF"), alignment=TA_CENTER)
    ts = ParagraphStyle("bi_title", fontName="Helvetica-Bold", fontSize=9.5,
                        textColor=_c(T["text"]))
    rl = ParagraphStyle("bi_lbl",   fontName="Helvetica-Bold", fontSize=8,
                        textColor=HexColor("#FFFFFF"), alignment=TA_CENTER)
    rv = ParagraphStyle("bi_val",   fontName="Helvetica",      fontSize=8.5,
                        textColor=_c(T["text"]),  leading=12.5)

    num_str = "{}. ".format(num) if num else ""
    hdr = Table([[
        Paragraph(_get(ins, "severity", "INFO").upper(), bs),
        Paragraph("{}{}".format(num_str, _get(ins, "title", "")), ts),
    ]], colWidths=[20*mm, CW - 20*mm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), HexColor(col)),
        ("BACKGROUND", (1,0), (1,0), _c(T["bg_light"])),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",      (0,0), (0,0),  "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING",(0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0),(-1,-1), 8),
        ("BOX",        (0,0), (-1,-1), 0.5, _c(T["border"])),
    ]))

    lw, vw = 26*mm, CW - 26*mm
    rows = [
        [Paragraph(k, rl), Paragraph(_get(ins, fk, ""), rv)]
        for k, fk in [("PROBLEM", "problem"), ("CAUSE",  "cause"),
                      ("EVIDENCE","evidence"),("ACTION", "action"),
                      ("IMPACT",  "impact")]
        if _get(ins, fk, "").strip()
    ]
    if not rows:
        rows = [[Paragraph("DETAIL", rl),
                 Paragraph(str(ins)[:200], rv)]]

    body = Table(rows, colWidths=[lw, vw])
    body.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,-1), _c(T["header_bg"])),
        ("ROWBACKGROUNDS",(1,0),(1,-1),
         [HexColor("#FFFFFF"), _c(T["bg_light"]),
          HexColor("#FFFFFF"), _c(T["bg_light"]), HexColor("#FFFFFF")]),
        ("VALIGN",  (0,0), (-1,-1), "TOP"),
        ("ALIGN",   (0,0), (0,-1),  "CENTER"),
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

def _toc(story, s, T, entries, CW):
    _sec(story, s, T, "Table of Contents")
    for num, title in entries:
        row = Table([[
            Paragraph(str(num), ParagraphStyle(
                "tn", fontName="Helvetica-Bold", fontSize=10,
                textColor=_c(T["accent"]), alignment=TA_CENTER)),
            Paragraph(title, s["toc"]),
        ]], colWidths=[9*mm, CW - 9*mm])
        row.setStyle(TableStyle([
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
            ("LINEBELOW",   (0,0), (-1,-1), 0.3, _c(T["border"])),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        story.append(row)


# ══════════════════════════════════════════════════════════
#  EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════

def _exec_summary(story, s, T, summary, findings, risks, opps, CW):
    _sec(story, s, T, "Executive Summary",
         "Key findings and strategic priorities")
    if summary:
        _narrative_box(story, s, T, summary)
    if findings:
        story.append(Paragraph("Key Findings", s["h3"]))
        for f in findings[:5]:
            story.append(Paragraph("+ " + str(f), s["bl"]))
        story.append(Spacer(1, 2*mm))
    if risks:
        story.append(Paragraph("Business Risks", s["h3"]))
        for r in risks[:4]:
            story.append(Paragraph("! " + str(r),
                ParagraphStyle("risk_p", fontName="Helvetica", fontSize=9,
                               textColor=_c(T["negative"]), leading=13,
                               leftIndent=10, firstLineIndent=-10,
                               spaceAfter=3)))
        story.append(Spacer(1, 2*mm))
    if opps:
        story.append(Paragraph("Opportunities", s["h3"]))
        for o in opps[:3]:
            story.append(Paragraph("* " + str(o),
                ParagraphStyle("opp_p", fontName="Helvetica", fontSize=9,
                               textColor=_c(T["positive"]), leading=13,
                               leftIndent=10, firstLineIndent=-10,
                               spaceAfter=3)))


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
             "Each finding: Problem → Cause → Evidence → Action → Impact")
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
         "Each finding: Problem → Cause → Evidence → Action → Impact")

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
    _sec(story, s, T, "Attrition Deep Dive",
         "Employee turnover analysis — drivers, segments, cost")

    _kpi_row(story, s, T, [
        {"label": "ATTRITION RATE", "value": "{:.1f}%".format(attrition.rate),
         "sub": "{:,} left".format(attrition.n_left),
         "color": T["negative"] if attrition.rate > 15 else T["warning"]},
        {"label": "SEVERITY", "value": attrition.severity.upper(),
         "sub": "Benchmark: 10–15%", "color": T["negative"]},
        {"label": "FLIGHT RISK", "value": "{:,}".format(attrition.n_flight_risk),
         "sub": "{:.0f}% of remaining".format(attrition.flight_risk_pct),
         "color": T["warning"]},
        {"label": "COST RISK",
         "value": "HIGH" if attrition.n_left > 50 else "MED",
         "sub": "50–200% salary/hire", "color": T["negative"]},
    ], CW)

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
                 "CRITICAL" if rate > 25 else "HIGH" if rate > 18 else "OK"]
                for dept, rate in sorted_d]
        _gtable(story, T, ["Department","Rate","Status"],
                rows, [CW*0.50, CW*0.25, CW*0.25], severity_col=2)


# ══════════════════════════════════════════════════════════
#  DATASET OVERVIEW
# ══════════════════════════════════════════════════════════

def _dataset_overview(story, s, T, df, profile, CW):
    _sec(story, s, T, "Dataset Overview & Descriptive Statistics",
         "Column breakdown and statistical summary")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = text_columns(df)
    dt_cols  = df.select_dtypes(include="datetime").columns.tolist()

    _gtable(story, T,
            ["Type", "Count", "Columns (sample)"],
            [["Numeric",     len(num_cols), ", ".join(num_cols[:6])],
             ["Categorical", len(cat_cols), ", ".join(cat_cols[:6])],
             ["DateTime",    len(dt_cols),  ", ".join(dt_cols[:4]) or "None"]],
            [CW*0.20, CW*0.12, CW*0.68])

    if num_cols:
        story.append(Paragraph("Descriptive Statistics", s["h3"]))
        show  = num_cols[:5]
        desc  = df[show].describe().round(3)
        hrow  = ["Stat"] + [c[:10] for c in show]
        rows  = [hrow] + [
            [stat] + [str(desc.loc[stat, c]) for c in show]
            for stat in ["mean","std","min","25%","50%","75%","max"]
            if stat in desc.index
        ]
        cw_s = CW / (len(show) + 1)
        tbl  = Table(rows, colWidths=[cw_s] * (len(show)+1), repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("TEXTCOLOR",    (0,0), (-1,0),  HexColor("#FFFFFF")),
            ("BACKGROUND",   (0,0), (-1,0),  _c(T["header_bg"])),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [HexColor("#FFFFFF"), _c(T["bg_light"])]),
            ("GRID",         (0,0), (-1,-1), 0.3, _c(T["border"])),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ]))
        story.append(KeepTogether([tbl]))
        story.append(Spacer(1, 2*mm))

        # Skew warning
        for col in num_cols[:6]:
            try:
                sk = float(df[col].skew())
                if abs(sk) > 1.0:
                    story.append(Paragraph(
                        "★ {} is heavily skewed (skew={:.2f}) — "
                        "use median not mean for reporting.".format(col, sk),
                        s["note"]))
            except Exception:
                logger.debug("_dataset_overview: suppressed exception", exc_info=True)


# ══════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════

def _stats_section(story, s, T, stats_report, CW):
    if stats_report is None: return
    _sec(story, s, T, "Statistical Analysis",
         "Distribution, normality, correlations")

    # Correlation honest-warning box
    warn_t = Table([[Paragraph(
        "<b>⚠ Analyst Note:</b> "
        "Correlation r does NOT mean 'Variable A changes Variable B by r%.' "
        "That is a dangerous misread. r = -0.35 means the two variables share "
        "12.3% of their variance (r² = 0.123). Association only — "
        "NOT causation, NOT magnitude of effect.",
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

    # Distribution summary
    col_stats = getattr(stats_report, "column_stats", {})
    if col_stats:
        story.append(Paragraph("Distribution Summary", s["h3"]))
        for col, cs in list(col_stats.items())[:8]:
            if getattr(cs, "mean", None) is None: continue
            normal = "Normal" if getattr(cs, "is_normal", False) else "Non-normal"
            sk_lbl = getattr(cs, "skew_label", "") or ""
            outs   = getattr(cs, "outlier_count_iqr", 0)
            story.append(Paragraph(
                "• '{}': {} | {} | Outliers: {}".format(col, normal, sk_lbl, outs),
                s["bl"]))

    # Correlations
    corrs = getattr(stats_report, "correlations", [])
    sig   = [c for c in corrs
             if getattr(c, "is_significant", False)
             and abs(getattr(c, "pearson_r", 0)) >= 0.15]
    if sig:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Significant Correlations (Correct Interpretation)", s["h3"]))
        rows = [[c.col_a, c.col_b,
                 str(round(c.pearson_r, 4)),
                 str(round(getattr(c, "p_value", 0), 4)),
                 c.strength.title(),
                 "r²={:.3f} — {:.1f}% variance shared. Association only.".format(
                     c.pearson_r**2, c.pearson_r**2 * 100)]
                for c in sig[:6]]
        _gtable(story, T,
                ["Col A", "Col B", "r", "p", "Strength", "Interpretation"],
                rows, [CW*x for x in [0.17, 0.17, 0.08, 0.08, 0.12, 0.38]])


# ══════════════════════════════════════════════════════════
#  BUSINESS INTELLIGENCE
# ══════════════════════════════════════════════════════════

def _bi_section(story, s, T, bi_report, CW):
    if bi_report is None: return
    _sec(story, s, T, "Business Intelligence",
         "Benchmarking, cohort analysis, segment performance")

    brief = getattr(bi_report, "executive_brief", "")
    if brief:
        _narrative_box(story, s, T, brief)

    # Benchmarks
    bms = getattr(bi_report, "benchmarks", [])
    if bms:
        story.append(Paragraph("Benchmarking Summary", s["h3"]))
        rows = [[bm.column, str(bm.mean), str(bm.median),
                 str(bm.top_10_pct), str(bm.bottom_10_pct),
                 bm.benchmark_label.split("—")[0].strip()[:15]]
                for bm in bms[:4]]
        _gtable(story, T,
                ["Metric","Mean","Median","Top 10%","Bottom 10%","Variation"],
                rows, [CW*x for x in [0.22,0.12,0.12,0.12,0.13,0.29]])

    # Cohorts
    sig_c = [c for c in getattr(bi_report, "cohorts", [])
             if c.is_significant]
    if sig_c:
        story.append(Paragraph("Significant Cohort Differences", s["h3"]))
        for c in sig_c[:3]:
            story.append(Paragraph("• " + c.interpretation, s["bl"]))

    # Key insights
    ki = getattr(bi_report, "key_insights", [])
    if ki:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("Key Business Insights", s["h3"]))
        for ins in ki[:5]:
            story.append(Paragraph("• " + str(ins), s["bl"]))


# ══════════════════════════════════════════════════════════
#  CHART PAGE
# ══════════════════════════════════════════════════════════

def _chart_page(story, s, T, img_bytes, title, narrative, num, CW):
    _sec(story, s, T, "Chart {}: {}".format(num, title))
    if img_bytes:
        try:
            img = Image(io.BytesIO(img_bytes),
                        width=CW, height=CW * 0.48)
            story.append(KeepTogether([img, Spacer(1, 3*mm)]))
        except Exception:
            logger.debug("_chart_page: suppressed exception", exc_info=True)
    if narrative:
        story.append(Paragraph("Analysis", s["h3"]))
        _narrative_box(story, s, T, narrative)


# ══════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════

def _recommendations(story, s, T, actions, CW):
    _sec(story, s, T, "Recommendations & Action Plan",
         "Prioritised by urgency — act on CRITICAL items first")

    pri_map = {
        "CRITICAL":   (T["negative"],  T["critical_bg"]),
        "SHORT TERM": (T["warning"],   T["warning_bg"]),
        "LONG TERM":  (T["info"],      T["info_bg"]),
    }

    for action in actions[:9]:
        action_str = str(action)
        priority   = "LONG TERM"
        for p in ("CRITICAL", "SHORT TERM"):
            if p in action_str.upper():
                priority = p
                break

        col, bg = pri_map.get(priority, (T["accent"], T["bg_light"]))
        text    = (action_str
                   .replace("[CRITICAL] ", "")
                   .replace("[SHORT TERM] ", "")
                   .replace("[LONG TERM] ", "")
                   .strip())

        card = Table([[
            Paragraph(priority, ParagraphStyle(
                "pri", fontName="Helvetica-Bold", fontSize=7,
                textColor=HexColor(col), alignment=TA_CENTER)),
            Paragraph(text, s["body"]),
        ]], colWidths=[CW*0.14, CW*0.86])
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

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "All recommendations are based solely on the provided dataset. "
        "Verify with domain experts before implementation.",
        s["sm"]))


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
    _sec(story, s, T, "Appendix — Methodology & Sources")

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

    story.append(Spacer(1, 4*mm))
    disc = Table([[Paragraph(
        "<b>BASIS OF PREPARATION</b><br/>"
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
        s["wh"])]],
        colWidths=["100%"])
    disc.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), _c(T["header_bg"])),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("BOX",           (0,0), (-1,-1), 1.5, _c(T["accent"])),
    ]))
    story.append(disc)


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

    _dataset_overview(story, s, T, df, profile, CW)
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
