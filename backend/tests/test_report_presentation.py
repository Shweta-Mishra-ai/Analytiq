"""
How the report looks, judged the way a reader judges it.

Two things were wrong. The document rendered in ReportLab's built-in
Helvetica — correct, but reading as a library default rather than a
designed deliverable — because the font files the code looked for had
never been committed, so registration logged a warning and fell back
silently. And it lacked the conventions that let a reader navigate a
professional report: numbered exhibits, source lines, and definitions for
the terms it uses.
"""
import io
import re

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics

from app.engines.pdf.theme import (
    FONT_BODY, FONT_BOLD, FONT_ITALIC, FONT_SERIF_BOLD,
    register_premium_fonts,
)
from app.engines.pdf_builder import build_pdf

CONFIG = {"title": "Presentation", "client_name": "Acme", "subtitle": "",
          "confidential": True, "theme_name": "", "logo_path": None,
          "prepared_by": "", "source_table": "src"}


@pytest.fixture(scope="module")
def report():
    from app.engines.chart_exporter import generate_all_charts
    from app.engines.predictive import (
        compute_drivers, find_binary_target, find_top_cluster)

    rng = np.random.default_rng(11)
    n = 900
    ot = rng.choice(["Yes", "No"], n, p=[.35, .65])
    ten = rng.integers(1, 20, n)
    p = (.05 + .32 * (ot == "Yes") + .24 * (ten <= 2)).clip(0, .92)
    df = pd.DataFrame({
        "employee_id": np.arange(n),
        "Attrition": np.where(rng.random(n) < p, "Yes", "No"),
        "OverTime": ot, "YearsAtCompany": ten,
        "JobRole": rng.choice(["Sales Rep", "Engineer", "Manager"], n),
        "MonthlyIncome": rng.uniform(2500, 19000, n).round(2),
    })
    target = find_binary_target(df)
    charts = [(t, b, "Chart narrative.")
              for t, b, _spec in generate_all_charts(df, "Corporate Light", 3) if b]
    pdf = build_pdf(df=df, config=dict(CONFIG), domain="hr",
                    chart_data=charts,
                    predictive=compute_drivers(df, target),
                    top_cluster=find_top_cluster(df, target))
    reader = PdfReader(io.BytesIO(pdf))
    text = re.sub(r"\s+", " ",
                  "\n".join((p.extract_text() or "") for p in reader.pages))
    return pdf, reader, text


# ── typography ────────────────────────────────────────────

def test_the_shipped_typefaces_register():
    assert register_premium_fonts() is True
    assert FONT_BODY != "Helvetica"
    assert FONT_BOLD != "Helvetica-Bold"


def test_registration_is_idempotent():
    """theme.py registers on import; calling again must not raise."""
    assert register_premium_fonts() is True
    assert register_premium_fonts() is True


def test_the_document_embeds_its_fonts(report):
    _pdf, reader, _text = report
    embedded = set()
    for page in reader.pages:
        for f in (page.get("/Resources", {}).get("/Font", {}) or {}).values():
            embedded.add(str(f.get_object().get("/BaseFont", "")))
    assert any("Carlito" in f for f in embedded), \
        f"body face not embedded: {sorted(embedded)}"
    assert any("Caladea" in f for f in embedded), \
        f"display face not embedded: {sorted(embedded)}"


def test_no_glyphs_are_actually_drawn_in_a_fallback_face(report):
    """ReportLab declares Helvetica in every page's resources whether or
    not it is used. What matters is that nothing selects it."""
    _pdf, reader, _text = report
    for i, page in enumerate(reader.pages, 1):
        fonts = (page.get("/Resources", {}).get("/Font", {}) or {})
        fallback = {k for k, v in fonts.items()
                    if "Helvetica" in str(v.get_object().get("/BaseFont", ""))}
        if not fallback:
            continue
        content = page.get_contents().get_data().decode("latin-1", "replace")
        selected = set(re.findall(r"/(\w+)\s+[\d.]+\s+Tf", content))
        assert not (fallback & selected), \
            f"page {i} draws text in a fallback face"


@pytest.mark.parametrize("char,name", [
    ("●", "badge bullet"),
    ("▪", "importance bar"),
    ("—", "em dash"),
    ("·", "middle dot"),
    ("→", "arrow"),
])
def test_every_symbol_the_report_uses_exists_in_the_font(char, name):
    """A glyph the face does not carry renders as a hollow box. Carlito has
    no U+25C6 diamond or U+2588 full block, both of which the report used
    before the typeface changed."""
    face = pdfmetrics.getFont(FONT_BODY).face
    assert face.charToGlyph.get(ord(char)) is not None, \
        f"{name} (U+{ord(char):04X}) is missing from {FONT_BODY}"


@pytest.mark.parametrize("char", ["◆", "█"])
def test_the_report_no_longer_uses_glyphs_the_font_lacks(char):
    import pathlib
    pdf_dir = pathlib.Path("app/engines/pdf")
    for path in pdf_dir.glob("*.py"):
        body = path.read_text()
        # Allow the character inside a comment explaining its absence.
        code = "\n".join(ln for ln in body.splitlines()
                         if not ln.strip().startswith("#"))
        assert char not in code, f"{path.name} still uses U+{ord(char):04X}"


# ── exhibits ──────────────────────────────────────────────

def test_tables_and_figures_are_numbered(report):
    _pdf, _reader, text = report
    numbers = [int(n) for n in re.findall(r"Exhibit (\d+)", text)]
    assert numbers, "nothing is labelled as an exhibit"
    assert len(numbers) >= 4


def test_exhibit_numbers_run_in_order_without_gaps(report):
    """A reader who sees Exhibit 3 and then Exhibit 5 goes looking for the
    one that is missing."""
    _pdf, _reader, text = report
    numbers = [int(n) for n in re.findall(r"Exhibit (\d+)", text)]
    unique = sorted(set(numbers))
    assert unique == list(range(1, len(unique) + 1)), \
        f"exhibit numbering is not contiguous: {unique}"


def test_every_exhibit_carries_a_source_line(report):
    """The line that separates a figure computed from the client's data
    from one carried in from somewhere else."""
    _pdf, _reader, text = report
    n_exhibits = len(set(re.findall(r"Exhibit (\d+)", text)))
    assert text.count("Source:") >= n_exhibits


def test_the_counter_restarts_for_each_report():
    """A per-build counter held on the style dict; two reports in one
    process must both begin at Exhibit 1."""
    df = pd.DataFrame({"revenue": np.linspace(10, 900, 200),
                       "region": ["N", "S"] * 100})
    texts = []
    for _ in range(2):
        pdf = build_pdf(df=df, config=dict(CONFIG), domain="sales")
        texts.append("\n".join((p.extract_text() or "")
                               for p in PdfReader(io.BytesIO(pdf)).pages))
    for t in texts:
        assert "Exhibit 1" in t


# ── glossary ──────────────────────────────────────────────

def test_the_report_defines_the_terms_it_uses(report):
    _pdf, _reader, text = report
    assert "Glossary" in text
    for term in ("AUC", "Base rate", "Lift", "Target leakage"):
        assert term in text, f"{term} is used but never defined"


def test_definitions_say_what_the_number_lets_you_do(report):
    """A glossary that restates the method teaches nothing."""
    _pdf, _reader, text = report
    assert "no better than shuffling" in text or "perfect order" in text
