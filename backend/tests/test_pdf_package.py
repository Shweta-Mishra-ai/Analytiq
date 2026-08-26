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


@pytest.mark.parametrize("name", [
    "THEMES", "HR_BENCHMARKS", "_SQL_COLS", "_wrap_sql_line", "_sql_escape",
    "_build_cover", "_styles", "_domain_theme", "_domain_label",
    "_benchmark_section", "_exec_summary", "_top_insights", "_appendix",
])
def test_shim_still_exports_what_callers_import(name):
    """The API, the domain registry and the test suite import these from
    pdf_builder. The shim exists so they keep working."""
    import app.engines.pdf_builder as shim
    assert hasattr(shim, name), f"pdf_builder no longer exports {name}"


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
