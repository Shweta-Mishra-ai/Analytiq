"""
engines/pdf — the report builder, split by responsibility.

  theme.py              palette, styles, canvas furniture, cover
  primitives.py         headings, KPI strips, tables, insight cards
  narrative_sections.py the sections that make an argument
  data_sections.py      the sections that show the numbers
  domain_sections.py    sections that apply to some domains only
  builder.py            assembly

Split out of a single 1,664-line module. The public entry point is
build_pdf, re-exported here and from app.engines.pdf_builder so existing
imports keep working.
"""
from app.engines.pdf.builder import build_pdf
# _domain_theme is re-exported for the compatibility shim, which
# forwards it for deck_builder; the redundant alias tells the
# linter this is deliberate rather than a leftover.
from app.engines.pdf.theme import THEMES, HR_BENCHMARKS
from app.engines.pdf.theme import _domain_theme as _domain_theme

__all__ = ["build_pdf", "THEMES", "HR_BENCHMARKS"]
