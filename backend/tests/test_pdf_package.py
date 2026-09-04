"""
Structure and public surface of the PDF builder package.

pdf_builder.py had grown to 1,664 lines, so every report change touched
the same module. It is now app/engines/pdf/, split by responsibility.
These tests pin the split: the public API is unchanged, the shim still
works for existing importers, and no single module is allowed to grow
back into the monolith.
"""
import importlib
import inspect
import io

import pandas as pd
import pytest

MODULES = [
    "app.engines.pdf",
    "app.engines.pdf.theme",
    "app.engines.pdf.primitives",
    "app.engines.pdf.narrative_sections",
    "app.engines.pdf.data_sections",
    "app.engines.pdf.domain_sections",
    "app.engines.pdf.predictive_sections",
    "app.engines.pdf.builder",
]

# Comfortably above today's largest module, far below the 1,664-line
# monolith. A module past this has taken on more than one job.
MAX_MODULE_LINES = 900


@pytest.mark.parametrize("mod", MODULES)
def test_every_module_imports_cleanly(mod):
    assert importlib.import_module(mod) is not None


@pytest.mark.parametrize("mod", MODULES)
def test_no_module_grows_back_into_a_monolith(mod):
    m = importlib.import_module(mod)
    n = len(open(m.__file__).read().splitlines())
    assert n <= MAX_MODULE_LINES, (
        f"{mod} is {n} lines. Split it rather than letting the package "
        f"collapse back into one file.")


def test_build_pdf_is_the_same_object_through_every_path():
    """The shim must not shadow the real builder with a copy."""
    from app.engines.pdf_builder import build_pdf as via_shim
    from app.engines.pdf import build_pdf as via_package
    from app.engines.pdf.builder import build_pdf as direct
    assert via_shim is via_package is direct


ORIGINAL_PARAMS = [
    "df", "config", "profile", "cleaning_summary", "stats_report",
    "bi_report", "ml_report", "chart_data", "executive_summary",
    "findings", "risks", "opportunities", "recommendations",
    "top_insights", "attrition", "domain",
]


def test_original_parameters_keep_their_names_and_order():
    """Callers pass these by keyword and by position. The API may grow, but
    not shift underneath anyone."""
    from app.engines.pdf_builder import build_pdf
    params = list(inspect.signature(build_pdf).parameters)
    assert params[:len(ORIGINAL_PARAMS)] == ORIGINAL_PARAMS


def test_every_added_parameter_is_optional():
    """A new section must never make an existing call site invalid."""
    from app.engines.pdf_builder import build_pdf
    sig = inspect.signature(build_pdf)
    for name, param in list(sig.parameters.items())[len(ORIGINAL_PARAMS):]:
        assert param.default is not inspect.Parameter.empty, \
            f"{name} was added without a default — existing callers break"


def test_the_shim_exports_exactly_what_callers_import():
    """Derived from the codebase, not from a hand-written list.

    The list this replaces was written when the shim was created and
    asserted a dozen names, several of which nothing had ever imported
    from here — so it locked in a surface that was wrong the day it was
    written and would have gone on being wrong. Reading the imports
    instead means the shim can shrink as callers move to the package,
    and cannot silently stop exporting something that is still in use.
    """
    import re
    from pathlib import Path

    import app.engines.pdf_builder as shim

    root = Path(__file__).resolve().parent.parent
    wanted = set()
    pattern = re.compile(
        r"from\s+app\.engines\.pdf_builder\s+import\s+([^\n(]+)")
    for path in list((root / "app").rglob("*.py")) + list((root / "tests").rglob("*.py")):
        if path.name == "pdf_builder.py":
            continue
        for names in pattern.findall(path.read_text(encoding="utf-8")):
            for name in names.split(","):
                name = name.strip().split(" as ")[0].strip()
                if name:
                    wanted.add(name)

    assert wanted, "no caller imports from the shim — it can be deleted"
    missing = sorted(n for n in wanted if not hasattr(shim, n))
    assert not missing, f"the shim no longer exports: {', '.join(missing)}"

    # And nothing beyond that: a shim that forwards everything is a
    # second public API by accident.
    import types
    exported = {n for n in vars(shim)
                if not n.startswith("__")
                and not isinstance(getattr(shim, n), types.ModuleType)}
    extra = sorted(exported - wanted)
    assert not extra, (
        "the shim re-exports names nothing imports from it: "
        + ", ".join(extra))
def test_report_still_builds_end_to_end():
    from app.engines.pdf_builder import build_pdf
    from pypdf import PdfReader

    df = pd.DataFrame({
        "region": ["North", "South"] * 40,
        "revenue": list(range(100, 180)),
        "units": list(range(1, 81)),
    })
    pdf = build_pdf(
        df=df,
        config={"title": "Structure Check", "client_name": "Test",
                "subtitle": "", "confidential": True, "theme_name": "",
                "logo_path": None, "prepared_by": "", "source_table": "src"},
        domain="sales",
    )
    assert pdf[:5] == b"%PDF-"
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) >= 3
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    assert "Structure Check" in text


def test_theme_lookup_resolves_for_every_registered_domain():
    """Themes are selected through the registry; a domain whose theme is
    missing falls back mid-build rather than failing loudly."""
    from app.engines.pdf_builder import THEMES, _domain_theme
    from app.engines.domains.registry import REGISTRY
    for key in REGISTRY:
        assert _domain_theme(key) in THEMES


# ══════════════════════════════════════════════════════════
#  A SECTION HEADING MUST NEVER BE ALONE ON A PAGE
# ══════════════════════════════════════════════════════════

def _story_pages(story):
    """Render a story and return the text of each page."""
    import io

    import pypdf
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4).build(list(story))
    reader = pypdf.PdfReader(io.BytesIO(buf.getvalue()))
    return [(p.extract_text() or "").strip() for p in reader.pages]


def _findings_block(words: int, nest_the_card: bool):
    """The shape `_top_insights` builds: a section banner, a group
    heading, and the first finding card, kept together."""
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                    Spacer, Table)

    from app.engines.pdf.primitives import KeepWholeIfItFits

    st = getSampleStyleSheet()
    hdr = Table([[Paragraph("CRITICAL", st["Normal"]),
                  Paragraph("1. Satisfaction averages 43% of scale",
                            st["Normal"])]], colWidths=[20 * mm, 150 * mm])
    body = Table([[Paragraph(k, st["Normal"]), Paragraph("x " * words,
                                                         st["Normal"])]
                  for k in ("PROBLEM", "CAUSE", "EVIDENCE", "ACTION",
                            "IMPACT")], colWidths=[26 * mm, 144 * mm])
    parts = [hdr, body, Spacer(1, 4 * mm)]

    banner = [Spacer(1, 3 * mm),
              Paragraph("Workforce Analytics Review — Findings",
                        st["Heading1"]),
              Paragraph("Each finding: Problem → Cause → Evidence → "
                        "Action → Impact", st["Normal"])]
    head = [Spacer(1, 2 * mm),
            Paragraph("Engagement & Satisfaction", st["Heading2"]),
            Paragraph("What the survey and behavioural measures show.",
                      st["Normal"])]
    first = [KeepTogether(parts)] if nest_the_card else parts
    return [Paragraph("the previous section", st["Normal"]), PageBreak(),
            KeepWholeIfItFits(banner + head + first)]


@pytest.mark.parametrize("words", [50, 200, 400, 440, 700, 1200, 2000])
def test_the_findings_heading_is_never_stranded_alone_on_a_page(words):
    """A reader of a generated report turned to a page carrying nothing
    but "Workforce Analytics Review — Findings" and its subtitle; the
    findings themselves were overleaf.

    `KeepTogether` is unconditional. Once the banner, the group heading
    and the first card together ran past one page, ReportLab dissolved
    the outer block, placed the small heading flowables, then found the
    card — atomic, because it was wrapped in a `KeepTogether` of its own
    — too tall for what was left and pushed it to the next page. The
    heading stayed behind on a page of its own.
    """
    pages = _story_pages(_findings_block(words, nest_the_card=False))
    heading = next(i for i, t in enumerate(pages)
                   if "Workforce Analytics Review" in t)
    assert "PROBLEM" in pages[heading], (
        "the section heading is alone on page {} — the first finding "
        "landed on the next page".format(heading + 1))


@pytest.mark.parametrize("words", [440, 700, 1200])
def test_nesting_the_card_is_what_stranded_the_heading(words):
    """Pins the cause, so a future refactor that re-wraps the first card
    in its own `KeepTogether` fails here rather than in a client's PDF."""
    pages = _story_pages(_findings_block(words, nest_the_card=True))
    heading = next(i for i, t in enumerate(pages)
                   if "Workforce Analytics Review" in t)
    assert "PROBLEM" not in pages[heading]


def test_a_card_that_fits_is_still_kept_whole():
    """The fix must not cost the behaviour it replaced: a card small
    enough for one page is never cut across a page break."""
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, Spacer, Table

    from app.engines.pdf.primitives import KeepWholeIfItFits

    st = getSampleStyleSheet()
    for filler in (430, 500, 600, 700):
        hdr = Table([[Paragraph("HIGH", st["Normal"]),
                      Paragraph("2. A finding", st["Normal"])]],
                    colWidths=[20 * mm, 150 * mm])
        body = Table([[Paragraph(k, st["Normal"]),
                       Paragraph("x " * 120, st["Normal"])]
                      for k in ("PROBLEM", "CAUSE", "EVIDENCE", "ACTION",
                                "IMPACT")], colWidths=[26 * mm, 144 * mm])
        pages = _story_pages([
            Paragraph("filler " * filler, st["Normal"]),
            KeepWholeIfItFits([hdr, body, Spacer(1, 4 * mm)])])
        starts = [i for i, t in enumerate(pages) if "PROBLEM" in t]
        ends = [i for i, t in enumerate(pages) if "IMPACT" in t]
        assert starts == ends, (
            "filler={}: the card was split across a page break".format(filler))
