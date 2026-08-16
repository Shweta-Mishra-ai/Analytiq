"""
How the document is laid out, as opposed to what it says.

Two defects a reader notices before reading a word:

1. **Near-empty pages.** Every section forced a page break regardless of
   how short it was, so the Data Quality note sat alone on an otherwise
   blank page and a fifth of the document was whitespace. That reads as
   padding — the impression is that there was not enough to say.
2. **A dark theme drawn on white paper.** The theme's body text is
   #E6EDF3. Nothing painted the page, so the text was near-invisible and
   only the elements carrying their own fill were readable. It was not a
   dark theme; it was a broken light one.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pypdf
import pytest

from app.engines.pdf_builder import THEMES


@pytest.fixture()
def df():
    rng = np.random.default_rng(400)
    n = 400
    return pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=n, freq="D"),
        "category": rng.choice(["Ops", "Retail", "Trade"], n),
        "revenue": rng.normal(20_000, 3_000, n).round(2),
        "cost": rng.normal(12_000, 2_000, n).round(2),
    })


def _build(df, theme=""):
    from app.engines.data_profiler import profile_dataset
    from app.engines.insights_builder import build_top_insights
    from app.engines.pdf_builder import build_pdf
    from app.engines.story_engine import detect_domain, generate_story

    domain, _c = detect_domain(df)
    story = generate_story(df)
    insights = build_top_insights(df=df, domain=domain, story_obj=story,
                                  attrition=None, avg_salary_k=60.0)
    return build_pdf(
        df=df,
        config={"title": "Review", "subtitle": "", "client_name": "Acme",
                "confidential": True, "theme_name": theme, "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None,
        stats_report=None, bi_report=None, ml_report=None, chart_data=[],
        executive_summary=story.executive_summary,
        findings=story.key_findings, risks=story.business_risks,
        opportunities=story.opportunities,
        recommendations=story.recommended_actions, top_insights=insights,
        attrition=None, domain=domain)


# ══════════════════════════════════════════════════════════
#  No page is mostly empty
# ══════════════════════════════════════════════════════════

def test_no_page_is_nearly_blank(df):
    """A page carrying two lines is a page the reader turns past
    wondering what went wrong."""
    reader = pypdf.PdfReader(io.BytesIO(_build(df)))
    thin = []
    for i, page in enumerate(reader.pages):
        if i == 0:      # the cover is deliberately sparse
            continue
        text = (page.extract_text() or "").strip()
        if 0 < len(text) < 200:
            thin.append((i, len(text)))
    assert not thin, "pages with almost nothing on them: {}".format(thin)


def test_the_document_is_not_padded_with_page_breaks(df):
    """The same content should not occupy noticeably more pages than it
    needs."""
    reader = pypdf.PdfReader(io.BytesIO(_build(df)))
    total_chars = sum(len((p.extract_text() or "").strip())
                      for p in reader.pages)
    # Roughly: a page of this report holds ~1,500 characters of body text.
    # Allowing a generous factor still catches one-section-per-page.
    assert len(reader.pages) <= max(4, total_chars / 700), (
        "{} pages for {} characters".format(len(reader.pages), total_chars))


# ══════════════════════════════════════════════════════════
#  Every theme is usable
# ══════════════════════════════════════════════════════════

def test_every_theme_declares_its_paper_and_row_colours():
    """Row backgrounds were hardcoded white, so a dark theme drew white
    rows under near-white text."""
    for name, theme in THEMES.items():
        assert "page_bg" in theme, name
        assert "row_bg" in theme, name


def test_no_theme_puts_its_text_the_same_colour_as_its_paper():
    """The check that would have caught the dark theme immediately."""
    def _lum(hex_colour):
        h = hex_colour.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255

    for name, theme in THEMES.items():
        for surface in ("page_bg", "row_bg", "bg_light"):
            contrast = abs(_lum(theme["text"]) - _lum(theme[surface]))
            assert contrast > 0.35, (
                "{}: '{}' text on '{}' {} is unreadable "
                "(luminance gap {:.2f})".format(
                    name, theme["text"], theme[surface], surface, contrast))


@pytest.mark.parametrize("theme", sorted(THEMES))
def test_every_theme_renders(df, theme):
    reader = pypdf.PdfReader(io.BytesIO(_build(df, theme=theme)))
    assert len(reader.pages) >= 3
    text = " ".join((p.extract_text() or "") for p in reader.pages)
    assert "Executive Summary" in text


def test_the_dark_theme_paints_the_page(df):
    """Without a painted page the body text is drawn near-white on
    white."""
    import pypdfium2 as pdfium

    pdf = _build(df, theme="Dark Tech")
    doc = pdfium.PdfDocument(io.BytesIO(pdf))
    # Sample the middle of a body page, away from header and footer.
    image = doc[2].render(scale=1).to_pil().convert("RGB")
    w, h = image.size
    pixel = image.getpixel((w // 2, h // 2))
    assert sum(pixel) / 3 < 90, \
        "page centre is {} — the dark theme is still on white paper".format(pixel)


def test_a_light_theme_leaves_the_page_white(df):
    import pypdfium2 as pdfium

    pdf = _build(df, theme="Corporate Light")
    doc = pdfium.PdfDocument(io.BytesIO(pdf))
    image = doc[2].render(scale=1).to_pil().convert("RGB")
    w, h = image.size
    pixel = image.getpixel((w // 2, h // 2))
    assert sum(pixel) / 3 > 200, pixel
