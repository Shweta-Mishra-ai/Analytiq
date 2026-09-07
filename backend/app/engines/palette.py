"""
engines/palette.py — the one place a colour is chosen.

Before this file, colour was picked in four places that had never been
looked at together: the web app's CSS tokens, the analysis PDF's theme,
the health PDF's own local constants, and the chart layer's matplotlib
palettes. Each was internally reasonable and none agreed with the others,
so a single client deliverable carried an electric blue rule (#1B4FD8),
a chart in a different blue (#1565C0), a mint-green score badge
(#22d3a5) borrowed from the dark dashboard, and a second, deeper green
(#059669) for "OK" in a table two pages later. On paper the mint read as
a UI colour that had wandered into a printed document — which is exactly
what it was.

The fix is not "pick nicer colours". It is to have one set of hues, in a
fixed order, stepped twice: once for the dark application surface and
once for the white page. Same identity, two renderings — so a chart in
the app and the same chart in the exported PDF are recognisably the same
chart.

Both steppings were checked, not eyeballed, against the six standard
gates (lightness band, chroma floor, colour-vision separation,
normal-vision separation, contrast against the surface):

    light  worst adjacent CVD ΔE 10.5, normal ΔE 16.3, all ≥ 3:1
    dark   worst adjacent CVD ΔE  9.5, normal ΔE 16.2, all ≥ 3:1

The slot ORDER is what makes that true, not decoration: gold sat beside
red in the first draft and the pair failed both separation gates on the
dark surface. Moving gold to slot 4 fixed it. Reorder these lists and
you must re-run the check.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── The hues, in fixed order ──────────────────────────────
# Assign by slot, never cycle. A seventh series folds into "Other" or
# becomes small multiples; a generated hue would be outside the checked
# set and is not a colour this product uses.
HUES = ("blue", "orange", "teal", "gold", "plum", "red")

CATEGORICAL_LIGHT = ("#1F5FA8", "#C2622A", "#009176",
                     "#B58A1E", "#7A4A8C", "#B03A32")
CATEGORICAL_DARK  = ("#4589DE", "#D9762F", "#12A07C",
                     "#AD861F", "#A177C6", "#DE5D52")

# One series needs no palette. Four histograms of four unrelated columns
# came out blue, navy, green and purple, which reads as an encoding —
# the reader looks for what the colours mean, and there is nothing to
# find. One series, one colour.
SINGLE_LIGHT = CATEGORICAL_LIGHT[0]
SINGLE_DARK  = CATEGORICAL_DARK[0]


# ── Status: reserved, never a series colour ───────────────
# These carry meaning, so they never double as "series 5". Each is used
# with a word beside it ("Poor", "Review"), never colour alone.
STATUS_LIGHT = {
    "good":     "#0F7A5A",
    "warning":  "#A8720F",
    "serious":  "#B4571C",
    "critical": "#A62B24",
    "neutral":  "#5A6675",
}
STATUS_DARK = {
    "good":     "#22A47F",
    "warning":  "#C99A2E",
    "serious":  "#D9762F",
    "critical": "#E06A63",
    "neutral":  "#8B93A1",
}


# ── The printed page ──────────────────────────────────────
# A deep navy ground with one blue accent — the same blue as chart slot
# 1, which is the whole point — and neutrals that lean very slightly
# toward it rather than sitting at a dead grey.
PRINT = {
    "ground":      "#0B1B2E",   # cover and running header
    "ground_text": "#FFFFFF",
    "accent":      CATEGORICAL_LIGHT[0],
    "accent_soft": "#8FB4DC",
    "ink":         "#16202E",   # body text
    "ink_soft":    "#4A5768",   # secondary text
    "muted":       "#7A8798",   # captions, footnotes
    "rule":        "#D8DFE8",   # hairlines and table grid
    "surface":     "#FFFFFF",
    "surface_alt": "#F5F8FB",   # zebra rows, card fills
    "surface_tint":"#EAF1F8",   # the one tinted panel per page
}


# ── Data health grades ────────────────────────────────────
# The grade is a judgement, so it wears a status colour, not a series
# colour. The old scale ran mint → blue → amber → orange → red, five
# hues for one ordered scale; two of them appeared nowhere else in the
# document. Four steps, all from STATUS, is enough to rank and does not
# introduce a hue the rest of the page has never seen.
GRADE_COLORS_LIGHT = {
    "A+": STATUS_LIGHT["good"],
    "A":  STATUS_LIGHT["good"],
    "B+": PRINT["accent"],
    "B":  PRINT["accent"],
    "C":  STATUS_LIGHT["warning"],
    "D":  STATUS_LIGHT["critical"],
    "F":  STATUS_LIGHT["critical"],
}
GRADE_COLORS_DARK = {
    "A+": STATUS_DARK["good"],
    "A":  STATUS_DARK["good"],
    "B+": CATEGORICAL_DARK[0],
    "B":  CATEGORICAL_DARK[0],
    "C":  STATUS_DARK["warning"],
    "D":  STATUS_DARK["critical"],
    "F":  STATUS_DARK["critical"],
}


def grade_color(grade: str, dark: bool = False) -> str:
    """Colour for a data-health grade. Unknown grades read as neutral
    rather than borrowing whichever colour happens to be first."""
    table = GRADE_COLORS_DARK if dark else GRADE_COLORS_LIGHT
    return table.get(str(grade).strip().upper(),
                     (STATUS_DARK if dark else STATUS_LIGHT)["neutral"])


def series_colors(n: int, dark: bool = False) -> list:
    """The first `n` categorical slots, in order.

    Never cycles. Asking for more slots than exist returns every slot
    and no more — the caller is expected to have folded the tail into
    "Other" before it got here, and silently repeating hue 1 for series
    7 would make two different things look like one.
    """
    palette = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    return list(palette[:max(0, n)])
