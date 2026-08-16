"""
The last check before a finding is printed.

Everything upstream already tries to be careful — multiplicity
correction, effect floors, hedged causes. This layer assumes that can
still fail and checks the finished text of each finding against what a
reader can actually verify.

Three things get removed, and the reasoning for each is the same: the
finding costs the report more than it is worth. An unquantified claim
cannot be checked. A future stated as fact will be wrong on a schedule.
A causal verb the data has not earned is the sentence a client's own
analyst will quote back when they reject the whole document.

The tests below also pin what must NOT be removed. Over-filtering is its
own failure: a reader can skip a dull finding and cannot detect a missing
one.
"""
from __future__ import annotations

import pytest

from app.engines.insight_guard import guard_insights, withheld_note


class _Ins:
    def __init__(self, title="", problem="", cause="", evidence="",
                 action="", impact=""):
        self.title = title
        self.problem = problem
        self.cause = cause
        self.evidence = evidence
        self.action = action
        self.impact = impact
        self.category = "general"
        self.severity = "high"


def _kept_titles(insights):
    return [i.title for i in guard_insights(insights).kept]


# ══════════════════════════════════════════════════════════
#  Unquantified claims
# ══════════════════════════════════════════════════════════

def test_a_finding_with_no_number_is_withheld():
    """"Revenue performance is concerning" is an opinion. The reader
    cannot check it and cannot act on it."""
    result = guard_insights([
        _Ins(title="Revenue performance is concerning",
             problem="Performance has been weak.",
             evidence="Observed across the dataset.")])
    assert result.kept == []
    assert "no figure" in result.dropped[0][1]


def test_a_quantified_finding_survives():
    kept = _kept_titles([
        _Ins(title="Margin fell 8.2 points",
             problem="Margin moved from 22.1% to 13.9%.",
             evidence="n=240 periods, p=0.003")])
    assert kept == ["Margin fell 8.2 points"]


def test_a_number_anywhere_in_the_evidence_is_enough():
    """The headline may be qualitative if the evidence carries the
    figure — it is the checkability that matters, not the placement."""
    kept = _kept_titles([
        _Ins(title="Attrition is concentrated in one department",
             problem="One department accounts for most exits.",
             evidence="Sales 60% vs company 27.8%, n=800")])
    assert len(kept) == 1


# ══════════════════════════════════════════════════════════
#  Futures stated as fact
# ══════════════════════════════════════════════════════════

def test_a_prediction_stated_as_fact_is_withheld():
    result = guard_insights([
        _Ins(title="Attrition will reach 35% next quarter",
             problem="Attrition is 27.8% and will reach 35% next quarter.",
             evidence="Current rate 27.8%, n=800")])
    assert result.kept == []
    assert "projection" in result.dropped[0][1]


def test_a_projection_that_names_itself_survives():
    """A straight-line extrapolation is a legitimate thing to show and an
    illegitimate thing to assert. The difference is whether the sentence
    says which it is."""
    kept = _kept_titles([
        _Ins(title="Attrition at 27.8%",
             problem="Attrition is 27.8%.",
             impact="If the same rate held, roughly 220 more people would "
                    "leave over a year — a straight-line projection, not a "
                    "forecast.",
             evidence="n=800")])
    assert len(kept) == 1


def test_a_guarantee_is_withheld():
    result = guard_insights([
        _Ins(title="Fixing onboarding guarantees a 10% improvement",
             problem="Onboarding is weak.",
             evidence="Score 42 out of 100, n=300")])
    assert result.kept == []


# ══════════════════════════════════════════════════════════
#  Causation the data has not earned
# ══════════════════════════════════════════════════════════

def test_an_unhedged_causal_claim_is_withheld():
    result = guard_insights([
        _Ins(title="Low pay drives 42% of exits",
             problem="Exits are concentrated in the lowest pay quartile.",
             cause="Low pay causes the attrition in this population.",
             evidence="42% of exits in Q1 pay band, n=800")])
    assert result.kept == []
    assert "cause" in result.dropped[0][1]


def test_a_causal_word_with_its_limits_stated_survives():
    kept = _kept_titles([
        _Ins(title="Exits concentrate in the lowest pay quartile (42%)",
             problem="42% of exits sit in the lowest pay band.",
             cause="Pay may be driving this, but the association is not "
                   "evidence of cause — the data holds no manager, workload "
                   "or progression field to test against.",
             evidence="42% vs 25% expected, p=0.004, n=800")])
    assert len(kept) == 1


def test_association_language_is_never_treated_as_causal():
    kept = _kept_titles([
        _Ins(title="Tenure correlates with satisfaction (r=0.42)",
             problem="Satisfaction rises with tenure.",
             cause="The two are correlated; direction is not established.",
             evidence="r=0.42, p<0.001, n=800")])
    assert len(kept) == 1


# ══════════════════════════════════════════════════════════
#  Not over-filtering
# ══════════════════════════════════════════════════════════

def test_a_dull_but_supported_finding_is_kept():
    """Removing a boring finding is a worse error than printing one — the
    reader can skip it, and cannot detect one that was dropped."""
    kept = _kept_titles([
        _Ins(title="Median order value is 82.40",
             problem="Half of orders fall below 82.40.",
             evidence="n=1,200 orders")])
    assert len(kept) == 1


def test_the_guard_survives_a_malformed_finding():
    class _Broken:
        category = "general"

        def __getattr__(self, name):
            raise RuntimeError("boom")

    result = guard_insights([_Broken()])
    assert len(result.kept) == 1, "a broken item was silently dropped"


def test_an_empty_list_is_handled():
    assert guard_insights([]).kept == []
    assert guard_insights(None).kept == []


def test_dict_findings_are_supported():
    result = guard_insights([
        {"title": "Margin fell 8.2 points", "problem": "22.1% to 13.9%",
         "evidence": "n=240"}])
    assert len(result.kept) == 1


# ══════════════════════════════════════════════════════════
#  Saying what was withheld
# ══════════════════════════════════════════════════════════

def test_the_withheld_note_states_the_count_and_the_reason():
    """A reader told that two findings were held back for lack of
    evidence trusts the rest more, not less."""
    result = guard_insights([
        _Ins(title="Things look bad", problem="No numbers here."),
        _Ins(title="Revenue will double next year",
             problem="Revenue is 1.2m and will double next year.",
             evidence="n=100")])
    note = withheld_note(result)
    assert note.startswith("2 candidate finding")
    assert "evidence standard" in note


def test_no_note_when_nothing_was_withheld():
    result = guard_insights([
        _Ins(title="Margin is 22.1%", problem="Margin 22.1%.",
             evidence="n=240")])
    assert withheld_note(result) == ""


# ══════════════════════════════════════════════════════════
#  It runs before anything is printed
# ══════════════════════════════════════════════════════════

def test_the_report_does_not_print_an_unsupported_finding():
    import io

    import numpy as np
    import pandas as pd
    import pypdf

    from app.engines.data_profiler import profile_dataset
    from app.engines.pdf_builder import build_pdf

    rng = np.random.default_rng(110)
    n = 200
    df = pd.DataFrame({
        "period": pd.date_range("2024-01-01", periods=n, freq="D"),
        "revenue": rng.normal(10_000, 1_200, n).round(2),
        "category": rng.choice(["A", "B"], n),
    })
    bad = _Ins(title="Revenue will collapse next quarter",
               problem="Revenue is 10,000 and will collapse next quarter.",
               evidence="n=200")
    good = _Ins(title="Revenue averages 10,000 per day",
                problem="Mean daily revenue is 10,000.",
                evidence="n=200, SD 1,200")
    pdf = build_pdf(
        df=df,
        config={"title": "R", "subtitle": "", "client_name": "Acme",
                "confidential": False, "theme_name": "", "logo_path": None},
        profile=profile_dataset(df), cleaning_summary=None,
        stats_report=None, bi_report=None, ml_report=None, chart_data=[],
        executive_summary="", findings=[], risks=[], opportunities=[],
        recommendations=[], top_insights=[bad, good],
        attrition=None, domain="finance")
    text = "\n".join((p.extract_text() or "")
                     for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)
    assert "will collapse next quarter" not in text
    assert "Revenue averages 10,000 per day" in text
    assert "withheld" in text


def test_advisory_will_is_not_treated_as_a_forecast():
    """"Review will require the stage history" is advice about the next
    step, not a claim about an outcome. Filtering it would strip the
    action out of findings that are otherwise sound."""
    kept = _kept_titles([
        _Ins(title="Cycle length differs by 67 days",
             problem="Won deals close in 28 days, lost in 95.",
             action="Confirming the cause will require the stage history.",
             evidence="p<0.001, n=1,080")])
    assert len(kept) == 1


def test_an_unlisted_prediction_verb_is_still_caught():
    """An enumeration of verbs missed "will collapse"; the next dataset
    would produce another one nobody listed."""
    for verb in ("collapse", "plummet", "double", "halve", "recover"):
        result = guard_insights([
            _Ins(title=f"Revenue will {verb} next quarter",
                 problem=f"Revenue is 1.2m and will {verb} next quarter.",
                 evidence="n=100")])
        assert result.kept == [], verb
