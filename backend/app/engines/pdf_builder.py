"""
engines/pdf_builder.py — compatibility shim.

The report builder now lives in app/engines/pdf/ , split by
responsibility (theme, primitives, narrative sections, data sections,
domain sections, assembly). It had grown to 1,664 lines in one module,
which made every report change a change to the same file.

This module keeps working for existing importers — `from
app.engines.pdf_builder import build_pdf, THEMES` is used by the API, the
domain registry's theme lookup, and the test suite. New code should
import from app.engines.pdf instead.
"""
from app.engines.pdf.builder import build_pdf
from app.engines.pdf.theme import (
    THEMES, HR_BENCHMARKS, W, H, CW_DEFAULT,
    FONT_BODY, FONT_BOLD, FONT_ITALIC,
    _c, _styles, _ReportCanvas, _build_cover, _domain_theme,
)
from app.engines.pdf.primitives import (
    _sec, _kpi_row, _narrative_box, _gtable, _insight_card, _toc, _clean,
)
from app.engines.pdf.narrative_sections import (
    _exec_summary, _top_insights, _dq_note, _readiness_block,
    _benchmark_section, _attrition_page, _domain_label,
    _has_reference_ranges,
)
from app.engines.pdf.predictive_sections import (
    _decision_table, _leakage_note, _model_note, _predictive_section,
)
from app.engines.pdf.data_sections import (
    _SQL_COLS,
    _data_prep_section, _wrap_sql_line, _sql_escape, _dataset_overview,
    _stats_section, _bi_section, _chart_page, _recommendations,
)
from app.engines.pdf.domain_sections import _appendix, _prepared_by_line

__all__ = ["build_pdf", "THEMES", "HR_BENCHMARKS"]
