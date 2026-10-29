"""
engines/pdf_fonts.py — the typeface every generated PDF uses.

ReportLab's built-in Helvetica covers Latin-1 only: a rupee sign rendered
as a black box, as did Greek, Cyrillic and most accented Latin beyond
Western Europe. Carlito (SIL Open Font License, bundled in
app/assets/fonts with its licence) covers those and is embedded in each
PDF, so the report looks the same on every machine.

Not covered: Devanagari, CJK and Arabic text still cannot render — Carlito
has no glyphs for them, and ReportLab does not perform the glyph shaping
Devanagari and Arabic need, so adding a font alone would not fix them.

If the font files are missing the PDFs still build, in Helvetica, and the
fallback is logged once.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "assets", "fonts")

_FACES = {
    "regular": ("Report", "Carlito-Regular.ttf", "Helvetica"),
    "bold": ("Report-Bold", "Carlito-Bold.ttf", "Helvetica-Bold"),
    "italic": ("Report-Italic", "Carlito-Italic.ttf", "Helvetica-Oblique"),
    "bold_italic": ("Report-BoldItalic", "Carlito-BoldItalic.ttf", "Helvetica-BoldOblique"),
}


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


_NAMES = _register()

# New canvases start in this font, so pages no longer carry an unused
# Helvetica reference alongside the embedded face.
from reportlab import rl_config as _rl_config  # noqa: E402
_rl_config.canvas_basefontname = _NAMES["regular"]

FONT_REG = _NAMES["regular"]
FONT_BOLD = _NAMES["bold"]
FONT_ITALIC = _NAMES["italic"]
FONT_BOLD_ITALIC = _NAMES["bold_italic"]
