"""
The checklist has to be able to fail.

A review page that says "Passed" whatever it is given is worse than no
review page: it tells the reader a check was made when none was, and it
is the first thing an experienced reader tests. So the tests here are
mostly about the failing side — a report with an invented target, an
asserted cause, a forecast, a contradiction or a named tool must come
back with an exception, and the exception must reach the printed
document rather than being swallowed on the way.

The rules are stated twice on purpose. `tests/test_report_standard.py`
holds its own copy of the patterns and asserts them against four
domains; if the checklist imported its rules from the code it checks, it
would pass by construction. `docs/QUALITY_STANDARD.md` is the statement
both answer to.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.engines.report_signoff import (EXCEPTION, NA, PASSED, exceptions,
                                        run_checks, summary_line)


class _Insight:
    def __init__(self, title="T", problem="p", cause="c", evidence="12 rows",
                 action="a", impact="i", severity="warning"):
        self.title = title
        self.problem = problem
        self.cause = cause
        self.evidence = evidence
        self.action = action
        self.impact = impact
        self.severity = severity


def _clean(**over):
    base = dict(
        executive_summary="Revenue of 1.2m is 6% below the 1.3m budget.",
        findings=["Margin held at 38.4% across 30 periods."],
        risks=["Two cost centres carry 61% of spend."],
        opportunities=["Closing the gap to the median is worth 84k."],
        recommendations=["Review the two cost centres above 20% of spend."],
        insights=[_Insight(severity="warning"), _Insight(severity="info")],
        critical_issues=[],
        prepared_by="",
        missing_pct=0.4,
        stats_report=None,
    )
    base.update(over)
    return run_checks(**base)


def _outcome(checks, fragment):
    return next(c for c in checks if fragment in c.rule).outcome


# ══════════════════════════════════════════════════════════
#  A clean report passes
# ══════════════════════════════════════════════════════════

def test_a_clean_report_raises_nothing():
    assert exceptions(_clean()) == []


def test_the_summary_counts_what_did_not_apply():
    """"All 11 checks passed" beside a table of 12 rows is the kind of
    arithmetic a reader notices."""
    checks = _clean()
    line = summary_line(checks)
    assert str(len(checks)) in line, line
    if any(c.outcome == NA for c in checks):
        assert "did not apply" in line, line


def test_every_check_says_what_it_was_based_on():
    for check in _clean():
        assert check.basis.strip(), check.rule
        assert check.outcome in (PASSED, EXCEPTION, NA), check


# ══════════════════════════════════════════════════════════
#  Each rule can fail
# ══════════════════════════════════════════════════════════

def test_a_finding_without_a_figure_is_an_exception():
    checks = _clean(findings=["Revenue distribution analysis shows high "
                              "variability."])
    assert _outcome(checks, "carries a figure") == EXCEPTION


def test_an_insight_with_no_traceable_evidence_is_an_exception():
    checks = _clean(insights=[_Insight(evidence="observed in the data")])
    assert _outcome(checks, "carries a figure") == EXCEPTION


def test_an_asserted_cause_is_an_exception():
    checks = _clean(findings=["Attrition rose because of the pay freeze."])
    assert _outcome(checks, "No cause is asserted") == EXCEPTION


def test_the_hedge_is_not_mistaken_for_the_claim():
    """"What is driving this is not identifiable from the data supplied"
    is the sentence the standard asks for, not the one it bans."""
    checks = _clean(findings=[
        "Attrition rose 4 points; what is driving it is not identifiable "
        "from the columns supplied."])
    assert _outcome(checks, "No cause is asserted") == PASSED


def test_a_forecast_is_an_exception():
    checks = _clean(findings=["Attrition will reach 25% by December."])
    assert _outcome(checks, "future fact") == EXCEPTION


def test_an_invented_target_is_an_exception():
    checks = _clean(findings=["Average rating is 3.6 (Target 4.0+)."])
    assert _outcome(checks, "names who sets it") == EXCEPTION


def test_scientific_notation_is_an_exception():
    checks = _clean(executive_summary="Total revenue was 7.72e+04.")
    assert _outcome(checks, "written for a reader") == EXCEPTION


def test_float_noise_is_an_exception():
    checks = _clean(findings=["Mean spend was 18420.000000001 per order."])
    assert _outcome(checks, "written for a reader") == EXCEPTION


def test_urgency_without_a_critical_finding_is_an_exception():
    checks = _clean(recommendations=["[CRITICAL] Review pricing."],
                    critical_issues=[])
    assert _outcome(checks, "Urgency is earned") == EXCEPTION


def test_urgency_with_a_critical_finding_passes():
    checks = _clean(recommendations=["[CRITICAL] Review pricing."],
                    critical_issues=[_Insight(severity="critical")])
    assert _outcome(checks, "Urgency is earned") == PASSED


def test_a_contradiction_is_an_exception():
    checks = _clean(insights=[_Insight(title="Gross Margin Healthy"),
                              _Insight(title="Gross Margin Down 8 Points")])
    assert _outcome(checks, "contradicts another") == EXCEPTION


def test_naming_the_tooling_is_an_exception():
    """The person delivering the work signs it. What produced the
    document is not the reader's concern and is named nowhere."""
    checks = _clean(findings=["This analysis was generated by an LLM."])
    assert _outcome(checks, "preparer is accountable") == EXCEPTION


def test_the_preparer_is_named_when_the_client_set_one():
    checks = _clean(prepared_by="R. Mehta, Data Consultancy")
    row = next(c for c in checks if "preparer is accountable" in c.rule)
    assert "R. Mehta" in row.basis, row.basis


def test_an_empty_section_is_an_exception():
    checks = _clean(recommendations=[])
    assert _outcome(checks, "heading with nothing under it") == EXCEPTION


# ══════════════════════════════════════════════════════════
#  Multiple testing is re-checked, not restated
# ══════════════════════════════════════════════════════════

class _Corr:
    def __init__(self, p, significant):
        self.p_value = p
        self.is_significant = significant


class _Stats:
    def __init__(self, correlations):
        self.correlations = correlations


def test_no_tests_run_is_not_a_pass():
    """Saying "Passed" for a check that never ran is the failure mode
    this whole page exists to avoid."""
    assert _outcome(_clean(stats_report=None), "multiple testing") == NA
    assert _outcome(_clean(stats_report=_Stats([])), "multiple testing") == NA


def test_a_correction_that_holds_passes():
    stats = _Stats([_Corr(0.0001, True), _Corr(0.4, False),
                    _Corr(0.6, False), _Corr(0.9, False)])
    assert _outcome(_clean(stats_report=stats), "multiple testing") == PASSED


def test_a_finding_that_only_survives_uncorrected_is_an_exception():
    """Twenty pairs at p<0.05 produce one significant result by chance.
    Flagged on the raw p-value, it reaches the report with the same
    confidence as a real one."""
    corrs = [_Corr(0.04, True)] + [_Corr(0.5 + i * 0.02, False)
                                   for i in range(19)]
    checks = _clean(stats_report=_Stats(corrs))
    assert _outcome(checks, "multiple testing") == EXCEPTION


# ══════════════════════════════════════════════════════════
#  It reaches the page
# ══════════════════════════════════════════════════════════

def _pdf(**over):
    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf_builder import build_pdf
    from app.engines.story_engine import generate_story

    rng = np.random.default_rng(5)
    rows = []
    for i, m in enumerate(pd.date_range("2023-01-31", periods=24, freq="ME")):
        for cc in ("Retail", "Wholesale"):
            rev = rng.normal(8e5, 4e4) * (1 + i * 0.015)
            rows.append({"period": m, "cost_centre": cc,
                         "revenue": round(rev, 2),
                         "cogs": round(rev * 0.62, 2),
                         "opex": round(rev * 0.18, 2)})
    df = pd.DataFrame(rows)
    story = generate_story(df)
    kwargs = dict(
        df=df,
        config={"title": "Review", "subtitle": "", "client_name": "Acme",
                "confidential": False, "theme_name": "", "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None, stats_report=None,
        bi_report=None, ml_report=None, chart_data=[],
        executive_summary=story.executive_summary,
        findings=story.key_findings, risks=story.business_risks,
        opportunities=story.opportunities,
        recommendations=story.recommended_actions,
        top_insights=story.top_insights, attrition=None, domain=story.domain)
    kwargs.update(over)
    return build_pdf(**kwargs)


def _text(pdf: bytes) -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(io.BytesIO(pdf))
    return " ".join((doc[i].get_textpage().get_text_range() or "")
                    for i in range(len(doc)))


@pytest.fixture(scope="module")
def clean_report():
    return _text(_pdf())


def test_the_document_prints_the_checklist(clean_report):
    assert "Review Checklist" in clean_report


def test_the_checklist_names_each_rule(clean_report):
    for fragment in ("carries a figure", "future fact", "Urgency is earned",
                     "Bars are drawn from zero"):
        assert fragment in clean_report, fragment


def test_an_exception_is_printed_rather_than_hidden():
    """A report that discloses one untraceable finding is worth more
    than one that quietly dropped it."""
    text = _text(_pdf(findings=["Revenue analysis shows high variability."]))
    assert "Exception" in text, "the exception did not reach the page"


def test_a_broken_checklist_does_not_take_the_report_down(monkeypatch):
    """The checklist is a page in an appendix. Failing to produce it is
    not a reason to lose the analysis in front of it."""
    from app.engines import report_signoff

    def boom(**_kwargs):
        raise RuntimeError("checklist exploded")

    monkeypatch.setattr(report_signoff, "run_checks", boom)
    text = _text(_pdf())
    assert "Methodology" in text
    assert "Review Checklist" not in text
