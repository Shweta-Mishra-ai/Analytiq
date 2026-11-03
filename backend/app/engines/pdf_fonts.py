"""
engines/pdf_fonts.py — the typefaces every generated PDF uses.

ReportLab's built-in Helvetica covers Latin-1 only: a rupee sign rendered
as a black box, as did Greek, Cyrillic and most accented Latin beyond
Western Europe. Carlito (SIL Open Font License, bundled in
app/assets/fonts with its licence) covers those and is embedded in each
PDF, so the report looks the same on every machine.

Hindi (Devanagari), Arabic and Chinese/Japanese need fonts Carlito does
not have, so the Noto faces for them are bundled too (also OFL), and
`script_markup` switches to them for each run of such text. Devanagari
conjuncts are shaped by HarfBuzz (ReportLab does this when uharfbuzz is
installed and the paragraph has `shaping` on, which every paragraph does
here). Arabic is joined and put in right-to-left order with
arabic-reshaper and python-bidi, because ReportLab's own right-to-left
support depends on rlbidi, which is not published on PyPI.

Not covered: Korean (Hangul), Thai and other scripts with no bundled
font; they still render as boxes. The CJK face is regular weight only.

If the font files are missing the PDFs still build, in Helvetica, and the
fallback is logged once.
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "fonts")

_FACES = {
    "regular": ("Report", "Carlito-Regular.ttf", "Helvetica"),
    "bold": ("Report-Bold", "Carlito-Bold.ttf", "Helvetica-Bold"),
    "italic": ("Report-Italic", "Carlito-Italic.ttf", "Helvetica-Oblique"),
    "bold_italic": ("Report-BoldItalic", "Carlito-BoldItalic.ttf", "Helvetica-BoldOblique"),
}

# Script → (family alias, regular file, bold file or None).
_SCRIPT_FACES = {
    "deva": ("ReportDeva", "NotoSansDevanagari-Regular.ttf", "NotoSansDevanagari-Bold.ttf"),
    "arab": ("ReportArab", "NotoSansArabic-Regular.ttf", "NotoSansArabic-Bold.ttf"),
    "cjk": ("ReportCJK", "NotoSansSC-Regular.ttf", None),
}

# Character ranges per script. Runs of spaces and punctuation between two
# characters of the same script stay in that run, so "विभाग प्रबंधक" is one
# run and is shaped (and, for Arabic, ordered) as a phrase.
_RANGES = {
    "deva": r"\u0900-\u097f\ua8e0-\ua8ff\u1cd0-\u1cff",
    "arab": r"\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff",
    "cjk": r"\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef",
}
_RUN = re.compile("|".join(
    r"(?P<{0}>[{1}](?:[\s\u200c\u200d.,:;\-]*[{1}])*)".format(name, chars)
    for name, chars in _RANGES.items()))


def _register() -> dict:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping

    names = {}
    for face, (alias, filename, fallback) in _FACES.items():
        path = os.path.join(FONT_DIR, filename)
        try:
            pdfmetrics.registerFont(TTFont(alias, path))
            names[face] = alias
        except Exception:
            logger.warning("PDF font %s unavailable (%s) — using %s", filename, path, fallback)
            names[face] = fallback
    if all(names[f] == _FACES[f][0] for f in _FACES):
        # Lets <b> and <i> inside a Paragraph pick the right face.
        addMapping("Report", 0, 0, names["regular"])
        addMapping("Report", 1, 0, names["bold"])
        addMapping("Report", 0, 1, names["italic"])
        addMapping("Report", 1, 1, names["bold_italic"])
    return names


def _register_scripts() -> dict:
    """Script → font family alias, for the scripts whose fonts loaded."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping

    families = {}
    for script, (alias, regular, bold) in _SCRIPT_FACES.items():
        try:
            pdfmetrics.registerFont(TTFont(alias, os.path.join(FONT_DIR, regular)))
            bold_alias = alias
            if bold:
                bold_alias = alias + "-Bold"
                pdfmetrics.registerFont(TTFont(bold_alias, os.path.join(FONT_DIR, bold)))
            # <b> inside a run picks the bold face; there is no italic, so
            # italic maps to the upright faces rather than failing.
            addMapping(alias, 0, 0, alias)
            addMapping(alias, 1, 0, bold_alias)
            addMapping(alias, 0, 1, alias)
            addMapping(alias, 1, 1, bold_alias)
            families[script] = alias
        except Exception:
            logger.warning("PDF font for %s text unavailable — it will render as boxes",
                           script, exc_info=True)
    return families


_NAMES = _register()
_SCRIPTS = _register_scripts()

# New canvases start in this font, so pages no longer carry an unused
# Helvetica reference alongside the embedded face.
from reportlab import rl_config as _rl_config  # noqa: E402
_rl_config.canvas_basefontname = _NAMES["regular"]

# Devanagari needs HarfBuzz shaping, or its conjuncts fall apart into
# letters with a visible virama ("क्षेत्र" drawn as क् षे त् र). Shaping is
# switched on only for paragraphs that contain Devanagari: turned on for
# every paragraph, it broke ReportLab's table of contents (its dot-leader
# callbacks) and with it every report.
_DEVANAGARI = re.compile("[" + _RANGES["deva"] + "]")


def _shape_when_needed():
    from reportlab.platypus import paragraph as _p
    original = _p.Paragraph.__init__
    if getattr(original, "_shapes_devanagari", False):
        return

    def __init__(self, text, style=None, *args, **kwargs):
        if style is not None and not getattr(style, "shaping", 0) \
                and isinstance(text, str) and _DEVANAGARI.search(text):
            from reportlab.lib.styles import ParagraphStyle
            style = ParagraphStyle(style.name + "-shaped", parent=style, shaping=1)
        original(self, text, style, *args, **kwargs)

    __init__._shapes_devanagari = True
    _p.Paragraph.__init__ = __init__


_UNSHAPED = (_SCRIPT_FACES["arab"][0], _SCRIPT_FACES["cjk"][0])


def _shape_per_font():
    """Shape a mixed-font word one font at a time.

    ReportLab shapes each word (text between spaces) with the font of its
    first piece. A word that changes font part-way — "कर्मचारी_आईडी", or
    "arabic_مبيعات.csv" in a paragraph that also holds Hindi — asked that
    font for glyphs it does not have, and the build failed with
    "'TTFont' object has no attribute 'hbAddPrivate'". Each same-font piece
    is shaped on its own here and the pieces joined back into one word.
    """
    from reportlab.pdfbase import pdfmetrics, ttfonts
    from reportlab.platypus import paragraph as _para
    from reportlab.pdfgen import canvas as _canvas
    original = ttfonts.shapeFragWord
    if getattr(original, "_per_font", False) or not ttfonts.uharfbuzz:
        return

    def width(pairs):
        total = 0.0
        for f, t in pairs:
            if t and not hasattr(f, "cbDefn"):
                total += pdfmetrics.stringWidth(t, f.fontName, f.fontSize)
        return total

    def shape(w, *args, **kwargs):
        if isinstance(w, ttfonts.ShapedFragWord):
            return w
        pairs = list(w[1:])
        fonts = {f.fontName for f, _ in pairs if not hasattr(f, "cbDefn")}
        if len(fonts) <= 1:
            if fonts and next(iter(fonts)).startswith(_UNSHAPED):
                return w
            return original(w, *args, **kwargs)
        groups, current, name = [], [], None
        for f, t in pairs:
            fname = None if hasattr(f, "cbDefn") else f.fontName
            if current and fname != name:
                groups.append(current)
                current = []
            current.append((f, t))
            name = fname
        if current:
            groups.append(current)
        out = ttfonts.makeShapedFragWord(w)([])
        total, changed = 0.0, False
        for g in groups:
            sub = w.__class__([width(g)] + g)
            # Arabic runs are already joined and in visual order, and HarfBuzz
            # would reverse them again; CJK needs no shaping. Only Devanagari
            # (and Latin) pieces go through it.
            skip = hasattr(g[0][0], "cbDefn") or g[0][0].fontName.startswith(_UNSHAPED)
            shaped = sub if skip else original(sub, *args, **kwargs)
            if shaped is not sub:
                changed = True
            total += shaped[0]
            out.extend(shaped[1:])
        if not changed:
            return w
        out.insert(0, total)
        return out

    shape._per_font = True
    ttfonts.shapeFragWord = shape
    _para.shapeFragWord = shape
    _canvas.shapeFragWord = shape


_shape_when_needed()
_shape_per_font()

FONT_REG = _NAMES["regular"]
FONT_BOLD = _NAMES["bold"]
FONT_ITALIC = _NAMES["italic"]
FONT_BOLD_ITALIC = _NAMES["bold_italic"]


def _visual_arabic(run: str) -> str:
    """Arabic letters joined into their contextual forms and set out in
    the left-to-right order the PDF draws them in."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(run))
    except Exception:
        logger.warning("Arabic shaping unavailable — text drawn unjoined", exc_info=True)
        return run


def has_other_script(text: str) -> bool:
    return bool(_RUN.search(str(text)))


def script_markup(text: str) -> str:
    """Paragraph markup with each run of Hindi, Arabic or CJK text set in a
    font that has it. `text` is already escaped for ReportLab; the
    characters matched are never part of an entity or tag."""
    if not _SCRIPTS:
        return text

    def wrap(m: re.Match) -> str:
        script = m.lastgroup
        family = _SCRIPTS.get(script)
        run = m.group(0)
        if family is None:
            return run
        if script == "arab":
            run = _visual_arabic(run)
        return '<font name="{}">{}</font>'.format(family, run)

    return _RUN.sub(wrap, text)


def _pieces(text: str, font: str):
    """(piece, font) for each run: other scripts in a font that has them,
    the rest in `font`; Arabic runs joined and in visual order."""
    text = str(text)
    pos, out = 0, []
    for m in _RUN.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], font))
        run = m.group(0)
        family = _SCRIPTS.get(m.lastgroup)
        if m.lastgroup == "arab" and family:
            run = _visual_arabic(run)
        out.append((run, family or font))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], font))
    return out


def _deva_line(piece: str, face: str, size: float, color=None):
    """A one-line shaped paragraph for a Devanagari piece, and its width.

    ReportLab only shapes text drawn straight on the canvas through its own
    bidi package (rlbidi), which is not on PyPI; paragraphs shape without
    it, so canvas Devanagari is drawn as a one-line paragraph."""
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from xml.sax.saxutils import escape
    style = ParagraphStyle("deva-line", fontName=face, fontSize=size,
                           leading=size * 1.25, shaping=1,
                           **({"textColor": color} if color is not None else {}))
    para = Paragraph(escape(piece), style)
    _, height = para.wrap(100000, 1000)
    widths = para.getActualLineWidths0()
    return para, height, (widths[0] if widths else 0.0)


def _piece_width(piece: str, face: str, size: float) -> float:
    """Width of one piece as drawn: Devanagari is measured after shaping,
    which changes its width (conjuncts are narrower than their letters)."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    if face == _SCRIPTS.get("deva"):
        return _deva_line(piece, face, size)[2]
    return stringWidth(piece, face, size)


def mixed_width(text: str, font: str, size: float) -> float:
    """stringWidth for text that may mix scripts."""
    return sum(_piece_width(p, f, size) for p, f in _pieces(text, font))


def draw_mixed(canvas, x: float, y: float, text: str, font: str, size: float,
               align: str = "left") -> float:
    """drawString for text that may mix scripts: each run in a font that
    has it, Latin in `font`. `align` is left, right or centre around x.
    Returns the width drawn."""
    width = mixed_width(text, font, size)
    cursor = x - width if align == "right" else x - width / 2 if align == "centre" else x
    deva = _SCRIPTS.get("deva")
    for piece, face in _pieces(text, font):
        if face == deva:
            # Shaped, as in paragraphs; drawn unshaped its vowel signs sit
            # in the wrong place ("रिपोर्ट" read as रपिोर्ट). The paragraph's
            # first baseline is one font size below its top.
            para, height, w = _deva_line(piece, face, size,
                                         getattr(canvas, "_fillColorObj", None))
            para.drawOn(canvas, cursor, y - (height - size))
            cursor += w
            continue
        canvas.setFont(face, size)
        canvas.drawString(cursor, y, piece)
        cursor += _piece_width(piece, face, size)
    canvas.setFont(font, size)
    return width


def matplotlib_fallbacks() -> list:
    """Font family names to list after the main chart font, so chart labels
    in Hindi, Arabic or CJK draw instead of showing boxes. Matplotlib 3.11
    shapes and orders them itself."""
    from matplotlib import font_manager
    names = []
    for script, (_, regular, _) in _SCRIPT_FACES.items():
        path = os.path.join(FONT_DIR, regular)
        if not os.path.exists(path):
            continue
        try:
            font_manager.fontManager.addfont(path)
            names.append(font_manager.FontProperties(fname=path).get_name())
        except Exception:
            logger.warning("chart font %s unavailable", regular, exc_info=True)
    return names
