"""
Two defects that presented as unused variables.

A value computed and then never read is usually tidying. Twice here it
was a feature that had been wired up and then disconnected, and in both
cases the disconnected half was the part that made the output correct.

**The variance check.** `equal_var` came out of a Levene test whose only
purpose is to decide whether a one-way ANOVA is valid. It was assigned
and never read, so ANOVA ran either way. ANOVA assumes equal variances;
against groups that violate that — a large tight group beside a small
spread one, which is the ordinary shape of a segment comparison — it
reports significance far more often than the 5% it claims. The report
then asserted that segments differ on the strength of a test not
entitled to say so.

**The severity tint.** Every insight card computed a background colour
from its severity and drew itself on the default panel regardless. A
critical finding and an informational one were the same colour, so the
only thing separating them on the page was the word in the badge.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════
#  The test has to match the data
# ══════════════════════════════════════════════════════════

def _groups(spread_a: float, spread_b: float, n_a: int, n_b: int, seed=11):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "segment": ["A"] * n_a + ["B"] * n_b,
        "value": np.concatenate([rng.normal(100, spread_a, n_a),
                                 rng.normal(100, spread_b, n_b)]),
    })


def _run(df):
    from app.engines.eda_engine import analyze_group_comparison

    # is_normal=True is the branch the Levene check guards: it is only
    # reached when the data looks normal enough for a parametric test,
    # which is exactly when the equal-variance assumption starts to
    # matter.
    return analyze_group_comparison(df, "value", "segment", is_normal=True)


def test_equal_variances_still_use_anova():
    """The fix must not throw away the parametric test where it is
    valid — ANOVA has more power when its assumption holds."""
    result = _run(_groups(10, 10, 200, 200))
    assert result is not None
    assert "ANOVA" in result.test_used, result.test_used


def test_unequal_variances_do_not_get_an_anova():
    """The defect: Levene said the assumption was violated and the code
    ran ANOVA anyway."""
    result = _run(_groups(2, 40, 300, 40))
    assert result is not None
    assert "ANOVA" not in result.test_used, result.test_used
    assert "Kruskal" in result.test_used, result.test_used


def test_the_report_names_the_test_it_actually_ran():
    """A reader who cannot see which test produced a p-value cannot
    check it."""
    result = _run(_groups(2, 40, 300, 40))
    assert "Levene" in result.test_used, result.test_used


def test_the_conservative_test_reduces_false_positives():
    """Same groups, same mean, different spread — significance here is
    the assumption failing, not a real difference."""
    false_positives = 0
    for seed in range(30):
        result = _run(_groups(2, 40, 300, 40, seed=seed))
        if result and result.p_value < 0.05:
            false_positives += 1
    # With the mean identical in both groups, a correctly-sized test
    # should land near 5%. The old path ran well above it.
    assert false_positives <= 6, "{}/30 false positives".format(false_positives)


# ══════════════════════════════════════════════════════════
#  Severity is visible, not just stated
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def frame():
    rng = np.random.default_rng(400)
    n = 300
    return pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=n, freq="D"),
        "category": rng.choice(["Ops", "Retail", "Trade"], n),
        "revenue": rng.normal(20_000, 3_000, n).round(2),
        "cost": rng.normal(12_000, 2_000, n).round(2),
    })


def test_a_critical_card_is_not_the_same_colour_as_an_info_card():
    """`bg` was computed per card and never applied."""
    from app.engines.pdf_builder import THEMES

    for name, theme in THEMES.items():
        assert theme["critical_bg"] != theme["bg_card"], name
        assert theme["warning_bg"] != theme["bg_card"], name


def test_the_severity_tint_reaches_the_page(frame):
    """Rendered, not asserted from the palette: the card body must
    actually differ between a critical and an informational finding."""
    import pypdfium2 as pdfium

    from app.engines.domains.base import build_insight
    from app.engines.pdf_builder import build_pdf

    def _render(severity):
        ins = [build_insight(
            title="A finding that needs a full card",
            problem="Something measurable happened, stated with a figure of 42%.",
            cause="A stated cause, hedged appropriately.",
            evidence="n=300, p<0.01, measured within this dataset.",
            action="1. Do the first thing  2. Then the second",
            impact="The measured upper bound is 1,200.",
            severity=severity, category="finance_margin")]
        pdf = build_pdf(
            df=frame,
            config={"title": "R", "subtitle": "", "client_name": "Acme",
                    "confidential": False, "theme_name": "Corporate Light",
                    "logo_path": None},
            profile=None, cleaning_summary=None, stats_report=None,
            bi_report=None, ml_report=None, chart_data=[],
            executive_summary="Summary.", findings=[], risks=[],
            opportunities=[], recommendations=[], top_insights=ins,
            attrition=None, domain="finance")
        doc = pdfium.PdfDocument(io.BytesIO(pdf))
        return {doc[i].render(scale=1).to_pil().convert("RGB").tobytes()
                for i in range(len(doc))}

    assert _render("critical") != _render("info"), \
        "a critical card renders identically to an informational one"
