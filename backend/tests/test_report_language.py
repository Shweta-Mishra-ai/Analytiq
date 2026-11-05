"""
The report language option: a Hindi report is written in Hindi throughout.

Report text is translated where it is printed (app/i18n), so these tests
read the finished PDFs. With Hindi data — Hindi column names and values —
any Latin word left in the text is report wording that escaped
translation. With the English-named HR data the check is the translation
layer's own record of text it could not translate.

An English report is the default and must be unchanged by all of this.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import pypdf
import pytest

from app import i18n

# Symbols and names a Hindi report keeps as they are: statistical symbols,
# the brand on the cover (its "ti" ligature extracts as "Analy"), the
# file name's extension, and the SQL the data-preparation section shows.
_KEPT = {"Analytiq", "Analy", "csv", "SQL", "p", "r", "n", "R", "H", "U", "t", "q", "Z", "F", "x"}


def _hindi_df(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "विभाग": rng.choice(["बिक्री", "लेखा", "संचालन", "विपणन"], n),
        "शहर": rng.choice(["दिल्ली", "मुंबई", "पुणे"], n),
        "वेतन": rng.normal(50_000, 9_000, n).round(0),
        "आयु": rng.integers(21, 60, n),
        "अंक": rng.normal(70, 12, n).round(1),
        "तिथि": pd.date_range("2023-01-01", periods=n, freq="D"),
    })


def _upload(client, df, name="डेटा.csv"):
    r = client.post("/api/datasets/upload",
                    files={"file": (name, io.BytesIO(df.to_csv(index=False).encode()),
                                    "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["meta"]["dataset_id"]


def _text(pdf: bytes) -> str:
    return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)


def _latin_words(text: str) -> set:
    return {w for w in re.findall(r"[A-Za-z]+", text) if w not in _KEPT}


@pytest.mark.parametrize("kind", ["pdf", "health-pdf"])
def test_a_hindi_report_of_hindi_data_has_no_english_words(client, kind):
    ds = _upload(client, _hindi_df())
    r = client.post(f"/api/reports/{ds}/{kind}", json={"language": "hi"})
    assert r.status_code == 200, r.text
    assert _latin_words(_text(r.content)) == set()


def test_every_report_sentence_from_hr_data_is_translated(client, hr_csv_bytes):
    r = client.post("/api/datasets/upload",
                    files={"file": ("hr.csv", io.BytesIO(hr_csv_bytes), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    client.post(f"/api/ml/{ds}/train", json={"target": "attrition"})
    i18n.clear_misses()
    for kind, body in (("pdf", {"language": "hi", "include_ml": True}),
                       ("health-pdf", {"language": "hi"})):
        r = client.post(f"/api/reports/{ds}/{kind}", json=body)
        assert r.status_code == 200, r.text
    assert i18n.misses() == {}


def test_the_data_is_printed_as_uploaded():
    """Column names and values are the user's: never translated, even when
    a catalog entry has the same words."""
    with i18n.use_language("hi", ["High", "Revenue"]):
        assert i18n.tr("High") == "High"
        assert i18n.tr("Revenue") == "Revenue"
    with i18n.use_language("hi"):
        assert i18n.tr("High") != "High"


def test_english_is_unchanged():
    with i18n.use_language("en"):
        assert i18n.tr("Executive Summary") == "Executive Summary"
        assert i18n.tr("Average {measure}", measure="salary") == "Average salary"
    assert i18n.tr("Key Findings") == "Key Findings"         # no context: English


def test_an_english_report_reads_the_same_with_and_without_the_option(client, hr_csv_bytes):
    r = client.post("/api/datasets/upload",
                    files={"file": ("hr.csv", io.BytesIO(hr_csv_bytes), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    a = _text(client.post(f"/api/reports/{ds}/pdf", json={}).content)
    b = _text(client.post(f"/api/reports/{ds}/pdf", json={"language": "en"}).content)
    assert a == b and "Executive Summary" in a


def test_an_unknown_language_is_refused(client, uploaded_dataset_id):
    r = client.post(f"/api/reports/{uploaded_dataset_id}/pdf", json={"language": "xx"})
    assert r.status_code == 422


def test_the_catalog_is_consistent():
    """Every translation keeps its template's placeholders (the catalog
    refuses to load otherwise) and is written in Devanagari, apart from the
    few names kept on purpose."""
    cat = i18n.catalog("hi")
    allowed = {"SQL", "Analytiq", "Unknown", "revenue", "cost", "profit", "budget",
               "actual", "period", "category"}
    for en, hi in cat.exact.items():
        words = set(re.findall(r"[A-Za-z]{2,}", hi)) - allowed
        assert not words, (en, hi)
    for tpl in cat.by_english.values():
        words = set(re.findall(r"[A-Za-z]{2,}", re.sub(r"\{\w+(?::t)?\}", "", tpl.target)))
        assert not words - allowed, (tpl.english, tpl.target)


def test_a_sentence_template_does_not_swallow_the_next_sentence():
    with i18n.use_language("hi"):
        out = i18n.tr("Prepared for Client on 3 November 2026. All figures derive solely "
                      "from the dataset supplied for this engagement and describe the "
                      "period it covers.")
    assert not re.search(r"[A-Za-z]{2,}", out), out


def test_numbered_steps_are_translated_one_by_one():
    with i18n.use_language("hi"):
        out = i18n.tr("1. Audit COGS line items  2. Identify the top cost drivers  "
                      "3. Compare against your own historical periods  4. Review pricing")
    assert not re.search(r"[A-Za-z]{2,}", out), out
    assert out.startswith("1. ")


def test_known_column_names_are_printed_in_the_report_language(client, hr_csv_bytes):
    """English column names the dictionary knows are written in Hindi in a
    Hindi report; the data's values are not touched."""
    r = client.post("/api/datasets/upload",
                    files={"file": ("hr.csv", io.BytesIO(hr_csv_bytes), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    text = _text(client.post(f"/api/reports/{ds}/pdf", json={"language": "hi"}).content)
    for name in ("salary", "department", "satisfaction", "attrition"):
        assert name not in text.lower(), name


def test_a_column_name_that_is_also_a_value_is_left_alone():
    import pandas as pd
    df = pd.DataFrame({"salary": [1, 2], "team": ["salary", "ops"]})
    with i18n.use_language("hi", **i18n.data_context(df)):
        assert i18n.column_name("salary") == "salary"       # also a value
        assert i18n.column_name("team") == "टीम"


def test_the_hindi_contents_page_lists_the_sections(client):
    """Devanagari headings are set in an "h2-shaped" copy of the style;
    the contents page matched only "h2" and printed ReportLab's
    placeholder (a single entry on page 0) instead of the sections."""
    ds = _upload(client, _hindi_df())
    r = client.post(f"/api/reports/{ds}/pdf", json={"language": "hi"})
    assert r.status_code == 200, r.text
    import pypdf
    contents = pypdf.PdfReader(io.BytesIO(r.content)).pages[1].extract_text()
    pages = re.findall(r"(\d+)\s*$", contents, flags=re.M)
    assert len([p for p in pages if int(p) > 1]) >= 4, contents


def test_english_carried_through_a_placeholder_is_recorded():
    """A placeholder matches any text, so a sentence could come out half
    English ("16.4 मिलियन over Dec 2009 to Dec 2011") with no miss."""
    with i18n.use_language("hi"):
        i18n.clear_misses()
        i18n.tr("Exit rate by department: rose sharply over the winter months.")
        assert i18n.misses()
        i18n.clear_misses()
        i18n.tr("Exit rate by department: 12%.")
        assert not i18n.misses()
