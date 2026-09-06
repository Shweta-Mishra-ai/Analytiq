"""
A grade is a judgement, and a judgement about ruined data has a floor.

The health score is deliberately the same number the Main Report prints
(compute_health takes it straight from profile_dataset), so it must not be
tampered with here — two different scores for one file is exactly the
client-visible contradiction the shared score exists to prevent.

The grade is a different thing. It weights completeness at 60%, so a file
whose 2,000 rows are 50 records copied forty times still scores 71 and used
to be graded "B+ — Good" on a client-facing PDF. Every average, count and
correlation in that report describes the same fifty records over and over.
These tests pin the floor: the score is untouched, the grade is capped, and
the reason is returned in words so the two can be reconciled by a reader.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.health_engine import compute_health
from app.engines.data_profiler import profile_dataset


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "amount":   rng.normal(500, 120, 2000),
        "quantity": rng.integers(1, 40, 2000),
        "region":   rng.choice(["north", "south", "east", "west"], 2000),
    })


def test_healthy_data_is_not_capped(clean):
    h = compute_health(clean)
    assert h["blocking_defect"] == ""
    assert h["grade"] in {"A+", "A"}


def test_a_file_that_is_mostly_duplicates_cannot_grade_well(clean):
    """The defect that started this: 71/100 reported as 'B+ Good'."""
    dupes = pd.concat([clean.head(50)] * 40, ignore_index=True)
    h = compute_health(dupes)

    assert h["dup_pct"] > 90
    assert h["grade"] == "D", "a file of copies must not grade above the floor"
    assert h["label"] == "Poor"

    why = h["blocking_defect"]
    assert "duplicate" in why
    # The reason must carry the number that makes it a reason, not just
    # an adjective: 50 real records behind 2,000 rows.
    assert "50 distinct records" in why
    assert "2,000" in why


def test_the_cap_does_not_touch_the_score(clean):
    """The score stays the Main Report's score — capping only the grade.

    If this ever fails, the health PDF and the analysis PDF a client
    receives together will print two different quality scores for one file.
    """
    dupes = pd.concat([clean.head(50)] * 40, ignore_index=True)
    h = compute_health(dupes)
    assert h["score"] == int(round(float(
        profile_dataset(dupes).overall_quality_score)))


def test_a_file_with_no_variation_is_told_the_cause_not_the_symptom():
    """A constant file is also ~100% duplicates; say the useful thing.

    Both faults are true of this frame, and the duplicate branch used to
    win because it was checked first — reporting "100% of the rows are
    exact duplicates — this file describes 1 distinct records" (also
    ungrammatical) instead of naming why there is nothing to analyse.
    """
    const = pd.DataFrame({"plan": ["pro"] * 2000, "fee": [49.0] * 2000})
    h = compute_health(const)

    assert h["grade"] == "D"
    assert "No column in this file varies" in h["blocking_defect"]
    assert "duplicate" not in h["blocking_defect"]


def test_the_duplicate_count_can_never_be_one_record():
    """The count in the duplicate reason is always a real plural.

    A file with exactly one distinct row is a file whose every column is
    constant, and the no-variation check now returns before the duplicate
    branch is reached — which is what makes "1 distinct records" (the
    ungrammatical output that exposed the ordering bug) unreachable rather
    than merely papered over with a singular form.
    """
    identical = pd.concat(
        [pd.DataFrame({"a": [3.0], "b": ["x"], "c": [1]})] * 500,
        ignore_index=True)
    why = compute_health(identical)["blocking_defect"]
    assert "No column in this file varies" in why
    assert "distinct record" not in why

    # And when a column does vary, at least two records survive, so the
    # plural in the duplicate sentence is always correct.
    nearly = identical.copy()
    nearly.loc[0, "a"] = 9.0
    why2 = compute_health(nearly)["blocking_defect"]
    assert "duplicate" in why2
    assert "2 distinct records" in why2


def test_ordinary_imperfect_data_is_left_alone(clean):
    """The cap is for ruin, not for imperfection — half a column missing
    is a normal finding and must not be turned into a capped grade."""
    holed = clean.copy()
    holed.loc[holed.sample(1000, random_state=1).index, "amount"] = np.nan
    h = compute_health(holed)

    assert h["blocking_defect"] == ""
    assert h["grade"] not in {"D", "C"}


@pytest.mark.parametrize("df,label", [
    (pd.DataFrame(), "no rows and no columns"),
    (pd.DataFrame({"a": [1.0], "b": ["x"]}).iloc[0:0], "columns but no rows"),
])
def test_an_empty_file_scores_zero_rather_than_a_hundred(df, label):
    """Nothing missing and nothing duplicated used to read as perfect.

    completeness and dedup both came out at 100 for a file containing no
    data at all, so it scored in the nineties and graded A — and on a
    column-less frame profile_dataset raised ZeroDivisionError, which
    compute_health swallowed into a completeness-only score of 100.
    """
    assert profile_dataset(df).overall_quality_score == 0.0

    h = compute_health(df)
    assert h["score"] == 0, label
    assert h["grade"] == "D"
    assert "no data to assess" in h["blocking_defect"]


def test_the_reason_reaches_the_report_payload(clean):
    """The key must be present on every result, capped or not, so the UI
    and the PDF can render it without guarding for its absence."""
    for frame in (clean, pd.concat([clean.head(50)] * 40, ignore_index=True)):
        assert "blocking_defect" in compute_health(frame)


def test_the_health_pdf_states_the_reason_beside_the_grade(clean):
    """A 71 printed beside a D, with no explanation, is its own
    contradiction — the PDF must say which to believe and why."""
    from app.engines.health_pdf_builder import build_health_pdf
    from app.engines.health_engine import build_full_insights

    dupes = pd.concat([clean.head(50)] * 40, ignore_index=True)
    health = compute_health(dupes)
    pdf = build_health_pdf(dupes, "general", health,
                           build_full_insights(dupes, "general"), "sales.csv")
    assert pdf and len(pdf) > 1000

    import fitz
    doc = fitz.open(stream=pdf, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    assert "Why this grade is capped" in text
    assert "exact duplicates" in text
