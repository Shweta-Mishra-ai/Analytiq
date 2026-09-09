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


from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
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

# One product, one look. These five entries used to be five different
# liveries: navy-and-blue for HR, a deep orange cover for e-commerce, a
# green one for sales. A client who received two Analytiq reports
# received what looked like reports from two companies, and the green
# and orange were never checked against the blue chart palette they had
# to sit beside — which is where the mismatched greens came from.
#
# The keys stay, because domains, decks and the report engine address
# themes by name. What changes is that every light theme now resolves to
# the same checked palette; the only thing a domain still contributes is
# the words on the cover badge. "Dark Tech" remains genuinely different
# because a dark surface is a different surface, not a different brand,
# and its steps are the dark half of the same six hues.
from app.engines.palette import (CATEGORICAL_DARK, PRINT, STATUS_DARK,
                                 STATUS_LIGHT)


def _light_theme(domain_label: str) -> dict:
    """The house style, with only the cover badge wording varying."""
    return {
        "cover_bg":    PRINT["ground"],   "cover_accent": PRINT["accent"],
        "header_bg":   PRINT["ground"],   "header_text":  PRINT["ground_text"],
        "accent":      PRINT["accent"],   "accent2":      PRINT["accent_soft"],
        "text":        PRINT["ink"],      "text_muted":   PRINT["ink_soft"],
        "bg_light":    PRINT["surface_tint"], "bg_card":  PRINT["surface_alt"],
        "border":      PRINT["rule"],
        "positive":    STATUS_LIGHT["good"],    "negative": STATUS_LIGHT["critical"],
        "warning":     STATUS_LIGHT["warning"], "info":     PRINT["accent"],
        # Tints for callout backgrounds, mixed from the status hues
        # towards white so a card never fights the text on it.
        "critical_bg": "#FBF0EF", "warning_bg":  "#FBF5E8",
        "positive_bg": "#ECF6F2", "info_bg":     PRINT["surface_tint"],
        "domain_label": domain_label,
        "domain_badge": PRINT["accent"],
    }


THEMES = {
    "Corporate Light":  _light_theme("BUSINESS ANALYTICS"),
    "HR Blue":          _light_theme("WORKFORCE ANALYTICS"),
    "Ecommerce Orange": _light_theme("COMMERCE ANALYTICS"),
    "Sales Green":      _light_theme("REVENUE ANALYTICS"),
    "Dark Tech": {
        "cover_bg":    "#0B0F14", "cover_accent": CATEGORICAL_DARK[0],
        "header_bg":   "#0B0F14", "header_text":  "#E9EEF5",
        "accent":      CATEGORICAL_DARK[0], "accent2": "#8FB4DC",
        "text":        "#E9EEF5", "text_muted":   "#98A3B2",
        "bg_light":    "#141A21", "bg_card":      "#1A212A",
        "border":      "#2A323C",
        "positive":    STATUS_DARK["good"],    "negative": STATUS_DARK["critical"],
        "warning":     STATUS_DARK["warning"], "info":     CATEGORICAL_DARK[0],
        "critical_bg": "#1E1312", "warning_bg":   "#1D1808",
        "positive_bg": "#0E1A16", "info_bg":      "#101922",
        "domain_label":"TECHNICAL ANALYTICS",
        "domain_badge": CATEGORICAL_DARK[0],
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

def _fit_text(text, limit: int) -> str:
    """Shorten on a word boundary, with a mark, never mid-word."""
    from app.engines.present import truncate
    return truncate(text or "", limit)


def _draw_fitted(cv, x: float, y: float, text: str, max_width: float,
                 font: str, size: float, min_size: float = 5.5) -> None:
    """Draw `text` at `x, y`, shrinking the font until it fits `max_width`.

    Only if it still will not fit at the smallest readable size does it
    get shortened — and then on a word boundary. Names on a cover page
    are worth a point of type size.
    """
    s = str(text or "")
    while size > min_size and cv.stringWidth(s, font, size) > max_width:
        size -= 0.25
    if cv.stringWidth(s, font, size) > max_width:
        # Binary-search the longest word-boundary form that fits.
        lo, hi = 4, len(s)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cv.stringWidth(_fit_text(s, mid), font, size) <= max_width:
                lo = mid
            else:
                hi = mid - 1
        s = _fit_text(s, lo)
    cv.setFont(font, size)
    cv.drawString(x, y, s)


def _draw_fitted_centred(cv, cx: float, y: float, text: str,
                         max_width: float, font: str, size: float,
                         min_size: float = 6.0) -> None:
    """Centred equivalent of _draw_fitted."""
    s = str(text or "")
    while size > min_size and cv.stringWidth(s, font, size) > max_width:
        size -= 0.25
    cv.setFont(font, size)
    cv.drawCentredString(cx, y, s)


class _ReportCanvas(CV.Canvas):
    """Draws premium header + footer on every content page."""

    def __init__(self, fn, T, report_title="", client_name="",
                 report_date="", confidential=False, agency_name="Analytiq",
                 **kw):
        super().__init__(fn, **kw)
        self._sp          = []
        self.T            = T
        # Truncated on a word boundary with a mark, not sliced: a title
        # cut mid-word reads as a bug on every page of the document.
        self.report_title = _fit_text(report_title, 60)
        self.client_name  = client_name
        self.report_date  = report_date
        # Stamping CONFIDENTIAL on a report nobody marked confidential
        # devalues the marking on the ones that are.
        self.confidential = bool(confidential)
        self.agency_name  = agency_name or "Analytiq"

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
        right = ("CONFIDENTIAL — " + self.client_name
                 if self.confidential else self.client_name)
        self.drawRightString(W - 18*mm, H - 20*mm, _fit_text(right, 48))
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
        parts = [self.agency_name]
        if self.confidential:
            parts.append("Confidential")
        parts.append("Prepared for {}".format(self.client_name))
        self.drawString(18*mm, 4.5*mm, _fit_text(" · ".join(parts), 90))
        self.drawRightString(W - 18*mm, 4.5*mm,
            "Page {} of {}".format(self._pageNumber, tot))


# ══════════════════════════════════════════════════════════
#  COVER PAGE  (drawn on separate canvas, merged via pypdf)
# ══════════════════════════════════════════════════════════

def _draw_cover_art(cv, T: dict, art: dict) -> None:
    """Paint the artwork, then the caption that says what it is.

    Both, always, in one function: the caption is the whole defence of
    this feature, so it is not possible to draw the image without it.
    Failure is swallowed to the flat colour — a decorative background is
    never worth losing a report over.
    """
    try:
        from reportlab.lib.utils import ImageReader
        reader = ImageReader(io.BytesIO(art["image"]))
        cv.saveState()
        # Dimmed hard: this sits behind the title block, and cover text
        # staying readable matters more than the picture does.
        cv.setFillAlpha(0.30)
        cv.drawImage(reader, 0, 0, width=W, height=H,
                     preserveAspectRatio=True, anchor="c", mask="auto")
        cv.restoreState()

        caption = art.get("caption") or ""
        if caption:
            cv.setFillColor(HexColor("#8AA0BF"))
            cv.setFont(FONT_BODY, 6)
            cv.drawString(14*mm, 8*mm, caption)
    except Exception:
        logger.warning("cover artwork could not be drawn — using the flat "
                       "cover colour", exc_info=True)


def _build_cover(T: dict, config: dict, kpis_preview: list,
                 cover_art: dict = None) -> bytes:
    buf = io.BytesIO()
    cv  = CV.Canvas(buf, pagesize=A4)
    title       = config.get("title", "Data Analysis Report")
    client_name = config.get("client_name", "Client")
    report_date = datetime.now().strftime("%B %d, %Y")
    # The domain's own band wins over the theme's. Themes are shared, so
    # taking it from the theme headed an education report WORKFORCE
    # ANALYTICS and a logistics one REVENUE ANALYTICS.
    domain_lbl  = (config.get("cover_label")
                   or T.get("domain_label", "BUSINESS ANALYTICS"))

    # BG
    cv.setFillColor(_c(T["cover_bg"]))
    cv.rect(0, 0, W, H, fill=1, stroke=0)

    # Optional generated artwork, behind everything and heavily dimmed.
    # This is the only surface in the entire deliverable where a
    # generated image may appear — see ai/imagery.py for the argument.
    # It is off unless someone assigned a model to the task, and the
    # caption below is not optional.
    if cover_art and cover_art.get("image"):
        _draw_cover_art(cv, T, cover_art)
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
    y_title = H / 2 + 20*mm
    for ln in lines:
        cv.setFont(FONT_SERIF_BOLD, 30 if len(ln) <= 20 else 24)
        cv.drawString(20*mm, y_title, ln)
        y_title -= 11*mm

    cv.setFillColor(HexColor(T["accent2"]))
    cv.setFont(FONT_BODY, 10)
    cv.drawString(20*mm, H / 2 - 8*mm,
                  config.get("subtitle", "Powered by Analytiq"))
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(20*mm, H / 2 - 14*mm, W - 37*mm, 1.5*mm, fill=1, stroke=0)

    # KPI strip (up to 4)
    kpis = kpis_preview[:4]
    bw   = (W - 37*mm) / max(len(kpis), 1)
    # The strip sat at the vertical midpoint and the meta bar at the foot,
    # leaving a third of the cover as an empty navy band. Dropped to just
    # above the meta bar it gives the page three bands — identity, title,
    # figures — instead of two and a hole.
    strip_y = 46*mm
    for i, kpi in enumerate(kpis):
        x = 20*mm + i * bw
        cv.setFillColor(HexColor("#1A3A5C"))
        cv.roundRect(x + 1.5, strip_y - 6*mm, bw - 3, 22*mm, 3,
                     fill=1, stroke=0)
        cv.setFillColor(HexColor(kpi.get("color", T["accent2"])))
        # Shrink rather than slice: "100 / 100" is nine characters and
        # any longer value was losing its tail mid-number.
        _draw_fitted_centred(cv, x + bw / 2, strip_y + 9*mm,
                             str(kpi.get("value", "")), bw - 8*mm,
                             FONT_BOLD, 16, min_size=9)
        cv.setFillColor(HexColor("#FFFFFF"))
        cv.setFont(FONT_BOLD, 7)
        cv.drawCentredString(x + bw / 2, strip_y + 3.5*mm,
                             _fit_text(kpi.get("label", ""), 18))
        cv.setFillColor(HexColor(T["accent2"]))
        cv.setFont(FONT_BODY, 6.5)
        cv.drawCentredString(x + bw / 2, strip_y - 1.5*mm,
                             _fit_text(kpi.get("sub", ""), 20))

    # Bottom meta
    cv.setFillColor(HexColor("#0D1F3C"))
    cv.rect(0, 0, W - 17*mm, 30*mm, fill=1, stroke=0)
    cv.setFillColor(_c(T["cover_accent"]))
    cv.rect(0, 30*mm, W - 17*mm, 1.2*mm, fill=1, stroke=0)
    meta = [("PREPARED FOR", client_name), ("DATE", report_date)]
    prepared_by = str(config.get("prepared_by", "")).strip()
    if config.get("confidential"):
        meta.append(("CLASSIFICATION", "CONFIDENTIAL"))
    elif prepared_by:
        # A dash under "CLASSIFICATION" is a field admitting it has
        # nothing to say. The name of whoever is accountable for the
        # analysis is worth more space on a cover than that.
        meta.append(("PREPARED BY", prepared_by))
    mw = (W - 37*mm) / max(len(meta), 1)
    for i, (k, v) in enumerate(meta):
        x = 20*mm + i * mw
        cv.setFillColor(HexColor(T["accent2"]))
        cv.setFont(FONT_BODY, 6.5); cv.drawString(x, 21*mm, k)
        cv.setFillColor(HexColor("#FFFFFF"))
        # Shrink to fit rather than slice. "Northwind Manufacturing"
        # became "Northwind Manufacturin" on the cover of a client
        # deliverable — the client's own name, misspelled, in the first
        # thing they read.
        _draw_fitted(cv, x, 13*mm, str(v), mw - 6*mm, FONT_BOLD, 8)

    # The cover previously carried "Powered by Groq Llama 3.3 70B". Naming
    # the model on the front page of a client deliverable invites the
    # reader to discount everything behind it, and says nothing about
    # whether the analysis is sound. Confidentiality marking belongs here
    # instead.
    if config.get("confidential"):
        cv.setFillColor(HexColor(T["accent2"]))
        cv.setFont(FONT_BODY, 6.5)
        cv.drawRightString(W - 21*mm, 5*mm, "Confidential")
    cv.save()
    buf.seek(0)
    return buf.read()
