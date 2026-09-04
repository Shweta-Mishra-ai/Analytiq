"""
engines/pdf_builder.py — compatibility shim.

The report builder lives in app/engines/pdf/, split by responsibility
(theme, primitives, narrative sections, data sections, domain sections,
assembly). It had grown to 1,664 lines in one module, which made every
report change a change to the same file.

This module exists so `from app.engines.pdf_builder import build_pdf`
keeps working — the API and a dozen tests still use that path. New code
should import from app.engines.pdf instead.

It used to re-export around forty names, including private helpers that
nothing outside the package ever imported. A shim that forwards
everything is not a compatibility layer, it is a second public API by
accident, and it kept the old module's whole surface alive long after
the split was supposed to have narrowed it. What remains is the list
that something actually imports from here — checked, not assumed.
"""
from app.engines.pdf.builder import build_pdf                    # noqa: F401
from app.engines.pdf.theme import THEMES, _domain_theme          # noqa: F401
from app.engines.pdf.narrative_sections import _domain_label     # noqa: F401
# The SQL wrapping moved to `lineage` when the data-preparation section
# and its script were split apart; the old path keeps working.
from app.engines.pdf.lineage import (                            # noqa: F401
    _SQL_COLS, _wrap_sql_line,
)

# HR_BENCHMARKS is deliberately not here. It is part of the pdf
# package's public API (app.engines.pdf exports it) but nothing has ever
# imported it through this old path, and forwarding it would be keeping
# a compatibility promise nobody asked for.
__all__ = ["build_pdf", "THEMES"]
