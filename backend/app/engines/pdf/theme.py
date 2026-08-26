"""
engines/pdf/theme.py — palette, paragraph styles, page furniture, cover.

Everything that decides how the report *looks*: the theme table, the
derived ParagraphStyles, the canvas that draws the running header and
footer, and the cover page.
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

W, H = A4
CW_DEFAULT = W - 36 * mm   # content width (18mm each side)

# ── Typography ────────────────────────────────────────────
# ReportLab's built-in Helvetica renders correctly but reads as a
# PDF-library default rather than a designed document, which is the wrong
# first impression for a report a client is paying for. Carlito and
# Caladea are metric-compatible with Calibri and Cambria — the faces most
# consulting templates are actually built on — and ship with the app.
#
# The health report already registered Carlito, but no font files were
# ever committed, so it logged a warning and silently fell back. Both
# reports now find them.
FONT_BODY = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"
FONT_SERIF = "Times-Roman"
FONT_SERIF_BOLD = "Times-Bold"

# theme.py sits at backend/app/engines/pdf/, so the backend root is four
# levels up: pdf -> engines -> app -> backend.
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_FONT_DIR = os.path.join(_BACKEND_ROOT, "assets", "fonts")


def register_premium_fonts() -> bool:
    """Register the shipped typefaces. Idempotent, and safe to fail.

    Returns True when the sans family registered. A missing font must
    degrade to Helvetica rather than break the report, so every failure
    here is a warning and not an exception.
    """
    global FONT_BODY, FONT_BOLD, FONT_ITALIC, FONT_SERIF, FONT_SERIF_BOLD
    from reportlab.pdfbase import pdfmetrics as _pm
    from reportlab.pdfbase.ttfonts import TTFont as _TTF

    wanted = [
        ("AQ-Sans", "Carlito-Regular.ttf"),
        ("AQ-Sans-Bold", "Carlito-Bold.ttf"),
        ("AQ-Sans-Italic", "Carlito-Italic.ttf"),
        ("AQ-Serif", "Caladea-Regular.ttf"),
        ("AQ-Serif-Bold", "Caladea-Bold.ttf"),
    ]
    done = set(_pm.getRegisteredFontNames())
    for alias, fname in wanted:
        if alias in done:
            continue
        path = os.path.join(_FONT_DIR, fname)
        if not os.path.exists(path):
            logger.warning("report font missing at %s — falling back to "
                           "Helvetica", path)
            continue
        try:
            _pm.registerFont(_TTF(alias, path))
            done.add(alias)
        except Exception:
            logger.warning("could not register %s", alias, exc_info=True)

    if "AQ-Sans" in done:
        FONT_BODY = "AQ-Sans"
    if "AQ-Sans-Bold" in done:
        FONT_BOLD = "AQ-Sans-Bold"
    if "AQ-Sans-Italic" in done:
        FONT_ITALIC = "AQ-Sans-Italic"
    if "AQ-Serif" in done:
        FONT_SERIF = "AQ-Serif"
    if "AQ-Serif-Bold" in done:
        FONT_SERIF_BOLD = "AQ-Serif-Bold"
    return FONT_BODY != "Helvetica"


register_premium_fonts()


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

# Auto-select theme by domain. Derived from the domain registry so a
# newly registered domain gets its theme automatically instead of
# silently falling back to Corporate Light.
def _domain_theme(domain: str) -> str:
    from app.engines.domains.registry import theme_for
    return theme_for(domain)

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
        # Default to the registered body face so a style declared without
        # one cannot silently reintroduce Helvetica into an otherwise
        # consistently-set document.
        kw.setdefault("fontName", FONT_BODY)
        return ParagraphStyle(name, **kw)

    return {
        # Exhibit counter. Consulting reports number every table and figure
        # so the text can refer to "Exhibit 4" instead of "the table above",
        # which stops meaning anything once a page break moves it. Carried
        # on the style dict because that is already threaded through every
        # section function; one counter per build_pdf call.
        "_exhibit": {"n": 0},
        "h1":    ps("h1",   fontName=FONT_SERIF_BOLD, fontSize=17,
                    textColor=_c(T["accent"]),     spaceAfter=4),
        "h2":    ps("h2",   fontName=FONT_SERIF_BOLD, fontSize=13,
                    textColor=_c(T["text"]),       spaceBefore=8, spaceAfter=3),
        "h3":    ps("h3",   fontName=FONT_BOLD, fontSize=10,
                    textColor=_c(T["accent"]),     spaceBefore=6, spaceAfter=3),
        "body":  ps("body", fontName=FONT_BODY,      fontSize=9,
                    textColor=_c(T["text"]),       leading=14,  spaceAfter=3,
                    alignment=TA_JUSTIFY),
        "sm":    ps("sm",   fontName=FONT_BODY,      fontSize=7.5,
                    textColor=_c(T["text_muted"]), leading=11,  spaceAfter=2),
        "bl":    ps("bl",   fontName=FONT_BODY,      fontSize=9,
                    textColor=_c(T["text"]),       leading=13,  spaceAfter=3,
                    leftIndent=10, firstLineIndent=-10),
        "toc":   ps("toc",  fontName=FONT_BODY,      fontSize=10,
                    textColor=_c(T["text"]),       leading=16,  spaceAfter=3),
        "wh":    ps("wh",   fontName=FONT_BODY,      fontSize=9,
                    textColor=HexColor("#FFFFFF"),  leading=13),
        "wbh":   ps("wbh",  fontName=FONT_BOLD, fontSize=10,
                    textColor=HexColor("#FFFFFF")),
        "note":  ps("note", fontName=FONT_ITALIC, fontSize=7.5,
                    textColor=_c(T["text_muted"]), spaceAfter=3),
        "warn":  ps("warn", fontName=FONT_BODY,      fontSize=8.5,
                    textColor=_c(T["text"]),       leading=13,
                    backColor=_c(T["warning_bg"])),
        # Insight card row styles
        "rl":    ps("rl",   fontName=FONT_BOLD, fontSize=8,
                    textColor=HexColor("#FFFFFF"),  alignment=TA_CENTER),
        "rv":    ps("rv",   fontName=FONT_BODY,      fontSize=8.5,
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
        self.setFont(FONT_BOLD, 10)
        self.drawString(18*mm, H - 14*mm, "Analytiq")
        self.setFont(FONT_BODY, 7.5)
        self.setFillColor(HexColor(T["accent2"]))
        self.drawString(18*mm, H - 20*mm, self.report_title)
        self.setFillColor(HexColor("#FFFFFF"))
        self.setFont(FONT_BODY, 7)
        self.drawRightString(W - 18*mm, H - 14*mm, self.report_date)
        self.drawRightString(W - 18*mm, H - 20*mm,
                             "CONFIDENTIAL — " + self.client_name)
        # ── Footer ────────────────────────────────────────
        self.setFillColor(_c(T["header_bg"]))
        self.rect(0, 0, W, 12*mm, fill=1, stroke=0)
        self.setFillColor(_c(T["accent"]))
        self.rect(0, 12*mm, W, 1.2*mm, fill=1, stroke=0)
        self.setFillColor(HexColor("#FFFFFF"))
        self.setFont(FONT_BODY, 6.5)
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
    cv.setFont(FONT_BOLD, 15)
    cv.drawString(20*mm, H - 32*mm, "Analytiq")
    cv.setFillColor(HexColor(T["accent2"]))
    cv.setFont(FONT_BODY, 9.5)
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
    cv.setFont(FONT_BOLD, 8)
    # Carlito has no U+25C6 diamond; U+25CF is present and reads
    # the same at badge size. A missing glyph renders as a box.
    cv.drawString(25*mm, H - 56*mm, "\u25CF  " + domain_lbl)

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
        cv.setFont(FONT_SERIF_BOLD, 30 if len(ln) <= 20 else 24)
        cv.drawString(20*mm, y_title, ln)
        y_title -= 11*mm

    cv.setFillColor(HexColor(T["accent2"]))
    cv.setFont(FONT_BODY, 10)
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
        cv.setFont(FONT_BOLD, 16)
        cv.drawCentredString(x + bw / 2, H / 2 - 2*mm,
                             str(kpi.get("value", ""))[:9])
        cv.setFillColor(HexColor("#FFFFFF"))
        cv.setFont(FONT_BOLD, 7)
        cv.drawCentredString(x + bw / 2, H / 2 - 8*mm,
                             str(kpi.get("label", ""))[:18])
        cv.setFillColor(HexColor(T["accent2"]))
        cv.setFont(FONT_BODY, 6.5)
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
        cv.setFont(FONT_BODY, 6.5); cv.drawString(x, 21*mm, k)
        cv.setFillColor(HexColor("#FFFFFF"))
        cv.setFont(FONT_BOLD, 8); cv.drawString(x, 13*mm, v[:22])

    # The cover previously carried "Powered by Groq Llama 3.3 70B". Naming
    # the model on the front page of a client deliverable invites the
    # reader to discount everything behind it, and says nothing about
    # whether the analysis is sound. Confidentiality marking belongs here
    # instead.
    cv.setFillColor(HexColor(T["accent2"]))
    cv.setFont(FONT_BODY, 6.5)
    cv.drawRightString(W - 21*mm, 5*mm, "Confidential")
    cv.save()
    buf.seek(0)
    return buf.read()
