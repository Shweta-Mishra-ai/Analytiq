"""
The gate that runs before analysis: is this data fit to analyse?

Every check here corresponds to a way the app would previously produce
arithmetically correct output describing something other than the
client's business — a revenue column stored as text is silently absent
from every statistic, and a repeated identity key doubles every sum.
Neither raised, neither was mentioned, and both produce a report that
looks finished.

The false-positive tests matter as much as the detections. A gate that
flags clean data is a gate people learn to click past.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.readiness import (assess_readiness, find_personal_data,
                                   readiness_payload)


@pytest.fixture()
def clean_df():
    rng = np.random.default_rng(70)
    n = 300
    return pd.DataFrame({
        "order_id": range(n),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "region": rng.choice(["North", "South", "East"], n),
        "revenue": rng.normal(5_000, 900, n).round(2),
        "units": rng.integers(1, 40, n),
    })


def _issue_columns(report, blocking=True):
    src = report.blockers if blocking else report.advisories
    return {i.column for i in src}


# ══════════════════════════════════════════════════════════
#  Clean data passes
# ══════════════════════════════════════════════════════════

def test_clean_data_is_ready(clean_df):
    rep = assess_readiness(clean_df)
    assert rep.ready is True, [
        (i.column, i.issue) for i in rep.blockers]
    assert rep.blockers == []


def test_a_ready_summary_states_the_shape(clean_df):
    rep = assess_readiness(clean_df)
    assert "300" in rep.summary and "ready" in rep.summary.lower()


def test_clean_data_raises_no_personal_data_flag(clean_df):
    assert find_personal_data(clean_df) == []


# ══════════════════════════════════════════════════════════
#  Blockers — analysis would be wrong, not just untidy
# ══════════════════════════════════════════════════════════

def test_numbers_stored_as_text_block_analysis(clean_df):
    """The column is silently excluded from every statistic, and the
    report says nothing about the omission."""
    df = clean_df.copy()
    df["revenue"] = df["revenue"].map(lambda v: f"${v:,.2f}")
    rep = assess_readiness(df)
    assert rep.ready is False
    assert "revenue" in _issue_columns(rep)
    issue = next(i for i in rep.blockers if i.column == "revenue")
    assert "excluded" in issue.consequence.lower()


def test_dates_stored_as_text_block_analysis(clean_df):
    df = clean_df.copy()
    df["order_date"] = df["order_date"].astype(str)
    rep = assess_readiness(df)
    assert "order_date" in _issue_columns(rep)


def test_exact_duplicate_rows_block_analysis(clean_df):
    df = pd.concat([clean_df, clean_df.head(30)], ignore_index=True)
    rep = assess_readiness(df)
    assert rep.ready is False
    assert any("duplicate" in i.issue for i in rep.blockers)


def test_a_repeated_identity_key_blocks_analysis():
    """Exact-duplicate removal does not catch this — the other columns
    differ — and every total counts the entity twice."""
    rng = np.random.default_rng(71)
    n = 200
    df = pd.DataFrame({
        "customer_id": [f"C{i % 120}" for i in range(n)],
        "revenue": rng.normal(500, 90, n).round(2),
        "region": rng.choice(["N", "S"], n),
    })
    rep = assess_readiness(df)
    assert "customer_id" in _issue_columns(rep)


def test_a_mostly_empty_column_blocks_analysis(clean_df):
    df = clean_df.copy()
    df.loc[df.index[:200], "revenue"] = np.nan
    rep = assess_readiness(df)
    assert "revenue" in _issue_columns(rep)


def test_too_few_rows_blocks_analysis():
    rng = np.random.default_rng(72)
    df = pd.DataFrame({"value": rng.normal(0, 1, 12),
                       "group": ["a", "b"] * 6})
    rep = assess_readiness(df)
    assert rep.ready is False
    assert any("rows" in i.issue for i in rep.blockers)


def test_a_dataset_with_no_measure_blocks_analysis():
    """Labels and identifiers only — there is nothing to average or
    trend, and the app would still produce a report."""
    df = pd.DataFrame({
        "record_id": range(100),
        "status": ["open", "closed"] * 50,
        "owner_team": ["a", "b", "c", "d"] * 25,
    })
    rep = assess_readiness(df)
    assert any("measurable" in i.issue for i in rep.blockers), [
        i.issue for i in rep.blockers]


def test_an_empty_frame_is_not_ready():
    rep = assess_readiness(pd.DataFrame())
    assert rep.ready is False
    assert "empty" in rep.summary.lower()


# ══════════════════════════════════════════════════════════
#  Advisories — worth fixing, analysis still valid
# ══════════════════════════════════════════════════════════

def test_a_constant_column_is_an_advisory_not_a_blocker(clean_df):
    df = clean_df.copy()
    df["currency"] = "GBP"
    rep = assess_readiness(df)
    assert rep.ready is True
    assert "currency" in _issue_columns(rep, blocking=False)


def test_an_empty_column_is_reported_once(clean_df):
    """Reported both as "entirely empty" and "holds a single value", it
    reads as two problems with one column."""
    df = clean_df.copy()
    df["notes"] = None
    rep = assess_readiness(df)
    notes_issues = [i for i in rep.issues if "notes" in i.column]
    assert len(notes_issues) == 1, [i.issue for i in notes_issues]


def test_mixed_numbers_and_text_is_an_advisory(clean_df):
    df = clean_df.copy()
    vals = df["revenue"].astype(str).tolist()
    for i in range(0, len(vals), 2):
        vals[i] = "not recorded"
    df["revenue_raw"] = vals
    rep = assess_readiness(df)
    assert any("mixed" in i.issue for i in rep.advisories), [
        i.issue for i in rep.issues]


# ══════════════════════════════════════════════════════════
#  Personal data
# ══════════════════════════════════════════════════════════

def test_email_content_is_detected_whatever_the_column_is_called():
    df = pd.DataFrame({"contact_ref": [f"user{i}@corp.com" for i in range(60)]})
    assert find_personal_data(df) == ["contact_ref"]


def test_phone_content_is_detected():
    df = pd.DataFrame({"ref": [f"+44 7700 9000{i:02d}" for i in range(60)]})
    assert find_personal_data(df) == ["ref"]


def test_a_date_column_is_not_mistaken_for_a_phone_number():
    """Digits, dashes and the right length. Flagging every date column
    buries the disclosure that matters."""
    df = pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=100).astype(str)})
    assert find_personal_data(df) == []


def test_an_order_number_is_not_mistaken_for_a_phone_number():
    """"2024000012" is a ten-digit order number and a plausible phone
    number; nothing in the value separates them, so a bare digit run is
    not treated as contact data."""
    df = pd.DataFrame({"reference": [f"{2024_000_000 + i}" for i in range(60)]})
    assert find_personal_data(df) == []


def test_a_phone_column_stored_as_bare_digits_is_caught_by_its_header():
    """The value shape cannot prove it, but the header can."""
    df = pd.DataFrame({"mobile_number": [f"{7700_900_000 + i}" for i in range(60)]})
    assert find_personal_data(df) == ["mobile_number"]


def test_headers_that_name_personal_data_are_flagged():
    df = pd.DataFrame({
        "employee_name": ["A"] * 40,
        "date_of_birth": ["1990-01-01"] * 40,
        "revenue": [1.0] * 40,
    })
    found = find_personal_data(df)
    assert "employee_name" in found and "date_of_birth" in found
    assert "revenue" not in found


def test_personal_data_appears_in_the_summary():
    df = pd.DataFrame({
        "email": [f"u{i}@corp.com" for i in range(60)],
        "revenue": np.linspace(1, 100, 60),
        "region": ["a", "b"] * 30,
    })
    rep = assess_readiness(df)
    assert "personal data" in rep.summary.lower()
    assert "email" in rep.summary


# ══════════════════════════════════════════════════════════
#  Payload
# ══════════════════════════════════════════════════════════

def test_payload_separates_blockers_from_advisories(clean_df):
    df = clean_df.copy()
    df["revenue"] = df["revenue"].map(lambda v: f"${v:,.2f}")
    df["currency"] = "GBP"
    payload = readiness_payload(assess_readiness(df))
    assert payload["ready"] is False
    assert payload["blockers"] and payload["advisories"]
    assert all("consequence" in b and "fix" in b for b in payload["blockers"])


def test_payload_is_json_serialisable(clean_df):
    import json
    json.dumps(readiness_payload(assess_readiness(clean_df)))


# ══════════════════════════════════════════════════════════
#  It reaches the report
# ══════════════════════════════════════════════════════════

def test_the_client_report_states_fitness_for_analysis(clean_df):
    import io

    import pypdf

    from app.engines.chart_exporter import generate_all_charts
    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf_builder import build_pdf
    from app.engines.story_engine import detect_domain, generate_story

    df = clean_df.copy()
    df["revenue"] = df["revenue"].map(lambda v: f"${v:,.2f}")   # a blocker
    domain, _ = detect_domain(df)
    story = generate_story(df)
    pdf = build_pdf(
        df=df,
        config={"title": "Q3", "subtitle": "", "client_name": "Acme",
                "confidential": True, "theme_name": "", "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None,
        stats_report=None, bi_report=None, ml_report=None,
        chart_data=[(t, b, "") for t, b in generate_all_charts(df, max_charts=1)],
        executive_summary=story.executive_summary, findings=story.key_findings,
        risks=story.business_risks, opportunities=story.opportunities,
        recommendations=story.recommended_actions, top_insights=[],
        attrition=None, domain=domain,
    )
    text = "\n".join((p.extract_text() or "")
                     for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)
    assert "Fitness for Analysis" in text
    assert "not ready" in text.lower()
