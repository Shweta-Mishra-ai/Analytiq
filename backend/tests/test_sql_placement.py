"""Where the SQL lineage belongs.

A report is read in order. What sits early is what the reader is told
matters, and two pages of `DELETE ... USING (SELECT MIN(ctid) ...)`
arriving before the first finding says the wrong thing about the
document — however good the SQL is.
"""
import io

import pandas as pd
import pypdf
import pytest

from app.engines.data_cleaner import CleaningPolicy, auto_clean, \
    get_cleaning_summary


@pytest.fixture()
def cleaned():
    """A frame with something to clean, and the summary it produces."""
    df = pd.DataFrame({
        "EmpID": [f"E{i:04d}" for i in range(200)],
        "Attrition": ["Yes", "No"] * 100,
        "Constant": [1] * 200,
        "Region": ["North", "South", "East", "West"] * 50,
        "Salary": list(range(30_000, 30_200)),
        "Tenure": list(range(200)),
    })
    frame, report = auto_clean(df, CleaningPolicy())
    return frame, get_cleaning_summary(report)


def _report_pages(df, summary):
    from app.engines.pdf_builder import build_pdf
    pdf = build_pdf(
        df=df,
        config={"title": "Placement Check", "client_name": "Test",
                "subtitle": "", "confidential": True, "theme_name": "",
                "logo_path": None, "prepared_by": "",
                "source_table": "hr_analytics"},
        cleaning_summary=summary,
        domain="hr",
    )
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    return [(p.extract_text() or "") for p in reader.pages]


# The contents page lists every section heading, so a naive search finds
# "Data Preparation" there and reports it as page 2 for both the section
# and the appendix. Sections are located from the body onwards.
_BODY_STARTS_AT = 2      # cover, contents


def _first_page_matching(pages, needle):
    for i in range(_BODY_STARTS_AT, len(pages)):
        if needle in pages[i]:
            return i
    return None


def test_no_sql_appears_before_the_appendix(cleaned):
    """It ran to two full pages at pages five and six of nineteen."""
    df, summary = cleaned
    pages = _report_pages(df, summary)

    appendix = _first_page_matching(pages, "Appendix — Methodology")
    assert appendix is not None, "the appendix is missing"

    for i, text in enumerate(pages[:appendix]):
        for statement in ("ALTER TABLE", "DELETE FROM", "SELECT MIN("):
            assert statement not in text, (
                "{} is on page {}, ahead of the appendix on page {}".format(
                    statement, i + 1, appendix + 1))


def test_the_sql_is_still_in_the_report(cleaned):
    """Moved, not dropped. A data team needs it to verify each step or
    apply the same treatment upstream."""
    df, summary = cleaned
    pages = _report_pages(df, summary)
    appendix_on = "\n".join(pages[_first_page_matching(
        pages, "Appendix — Methodology"):])
    assert "D. Data Preparation — Equivalent SQL" in appendix_on
    assert "ALTER TABLE" in appendix_on


def test_data_preparation_says_where_the_sql_went(cleaned):
    """A reader who wants the script must be able to find it."""
    df, summary = cleaned
    pages = _report_pages(df, summary)
    prep = _first_page_matching(pages, "Data Preparation")
    assert prep is not None
    assert "Appendix D" in pages[prep]


def test_the_table_of_changes_stays_in_the_body(cleaned):
    """What changed between the file supplied and the figures reported is
    front matter, not an appendix: without it the reader has to take the
    whole report on trust."""
    df, summary = cleaned
    pages = _report_pages(df, summary)
    prep = _first_page_matching(pages, "Data Preparation")
    appendix = _first_page_matching(pages, "Appendix — Methodology")
    assert prep < appendix, "the change table was moved out of the body"
    assert "Treatment" in pages[prep], "the change table is not on that page"


def test_a_report_without_cleaning_still_builds(cleaned):
    """No cleaning summary means no SQL section and no dangling
    cross-reference to one."""
    df, _ = cleaned
    pages = _report_pages(df, None)
    joined = "\n".join(pages)
    assert "Appendix — Methodology" in joined
    assert "Equivalent SQL" not in joined
    assert "Appendix D" not in joined
