"""
The finance analyses a reviewer would expect and did not find: cost
structure, break-even, margin stability, loss-making segments, and a
first-digit conformity check.

Each is tested twice — once on data with the effect planted, where the
right answer is known in advance and can be checked against the number
the engine reports, and once on data where the effect is absent, where
reporting anything at all would be the failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.domains.finance import (
    _detect_finance_cols,
    _finance_benford,
    _finance_cost_structure,
    _finance_loss_making_segments,
    _finance_margin_volatility,
    _insights_finance,
)


def _run(fn, df):
    """Call one analysis in isolation and return its output buckets."""
    insights, findings, risks, opps = [], [], [], []
    fn(df, _detect_finance_cols(df), {}, insights, findings, risks, opps)
    return {"insights": insights, "findings": findings,
            "risks": risks, "opportunities": opps}


def _text(out) -> str:
    parts = list(out["findings"]) + list(out["risks"]) + list(out["opportunities"])
    for i in out["insights"]:
        parts += [i.title, i.problem, i.cause, i.evidence, i.action, i.impact]
    return " ".join(parts)


# ══════════════════════════════════════════════════════════
#  Cost structure and break-even
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def known_cost_structure():
    """Fixed cost 20,000 per record, variable rate 0.60, so contribution
    margin 40% and break-even revenue 50,000. Mean revenue 80,000 leaves
    60% headroom."""
    rng = np.random.default_rng(31)
    n = 240
    revenue = rng.uniform(40_000, 120_000, n).round(2)
    cost = (20_000 + 0.60 * revenue + rng.normal(0, 1_500, n)).round(2)
    return pd.DataFrame({
        "period": pd.date_range("2023-01-01", periods=n, freq="D"),
        "category": rng.choice(["Ops", "Retail", "Trade"], n),
        "revenue": revenue,
        "cost": cost,
    })


def test_cost_structure_recovers_the_planted_split(known_cost_structure):
    out = _run(_finance_cost_structure, known_cost_structure)
    text = _text(out)
    assert out["findings"], "no cost structure reported on a clean linear structure"
    assert "60%" in text or "61%" in text or "59%" in text, \
        "variable rate not recovered: {}".format(text[:400])
    assert "40%" in text or "39%" in text or "41%" in text, \
        "contribution margin not recovered"


def test_break_even_is_arithmetically_consistent(known_cost_structure):
    """Break-even = fixed / contribution margin. 20,000 / 0.40 = 50,000."""
    out = _run(_finance_cost_structure, known_cost_structure)
    be_line = [f for f in out["findings"] if "Break-even" in f]
    assert be_line, "break-even not reported"
    import re
    m = re.search(r"Break-even revenue: ([\d,]+)", be_line[0])
    assert m, be_line[0]
    reported = float(m.group(1).replace(",", ""))
    assert reported == pytest.approx(50_000, rel=0.08), \
        "break-even {} is not the planted 50,000".format(reported)


def test_healthy_headroom_is_an_opportunity_not_a_risk(known_cost_structure):
    out = _run(_finance_cost_structure, known_cost_structure)
    assert out["opportunities"], "no operating-leverage note on a profitable structure"
    assert not out["risks"], "flagged risk on a business trading 60% above break-even"


def test_thin_headroom_is_flagged():
    """Same structure, but mean revenue barely above break-even."""
    rng = np.random.default_rng(32)
    n = 200
    revenue = rng.uniform(48_000, 56_000, n).round(2)
    cost = (20_000 + 0.60 * revenue + rng.normal(0, 400, n)).round(2)
    df = pd.DataFrame({"revenue": revenue, "cost": cost})
    out = _run(_finance_cost_structure, df)
    assert out["risks"], "no risk raised at break-even"
    assert out["insights"], "no insight raised at break-even"
    assert out["insights"][0].severity in ("critical", "high")


def test_no_cost_structure_claimed_when_cost_does_not_track_revenue():
    """Cost independent of revenue has no fixed/variable split to recover.
    An intercept and a slope still come back from the regression; printing
    a break-even from them would be arithmetic on noise."""
    rng = np.random.default_rng(33)
    n = 300
    df = pd.DataFrame({
        "revenue": rng.uniform(10_000, 90_000, n).round(2),
        "cost": rng.uniform(10_000, 90_000, n).round(2),
    })
    out = _run(_finance_cost_structure, df)
    assert not out["findings"] and not out["insights"], \
        "invented a cost structure from unrelated columns: {}".format(_text(out)[:300])


def test_no_cost_structure_below_minimum_sample():
    rng = np.random.default_rng(34)
    n = 20
    revenue = rng.uniform(40_000, 120_000, n)
    df = pd.DataFrame({"revenue": revenue,
                       "cost": 20_000 + 0.6 * revenue + rng.normal(0, 500, n)})
    out = _run(_finance_cost_structure, df)
    assert not out["findings"], "reported a cost structure from 20 records"


# ══════════════════════════════════════════════════════════
#  Margin stability
# ══════════════════════════════════════════════════════════

def _margin_frame(margins, seed=35):
    """One row per period, engineered so each period lands on its margin."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, m in enumerate(margins):
        rev = float(rng.uniform(80_000, 120_000))
        rows.append({"period": f"P{i:02d}", "revenue": round(rev, 2),
                     "cost": round(rev * (1 - m / 100), 2)})
    return pd.DataFrame(rows)


def test_volatile_margin_is_reported_with_its_range():
    df = _margin_frame([40, 4, 35, 8, 38, 6, 33, 9])
    out = _run(_finance_margin_volatility, df)
    assert out["risks"], "an average built from 4% and 40% periods was not flagged"
    text = _text(out)
    assert "CV" in text, "no coefficient of variation reported"
    assert "4.0%" in text or "4%" in text, "the weakest period is not named"


def test_stable_margin_is_offered_as_a_planning_assumption():
    df = _margin_frame([22.1, 21.6, 22.4, 22.0, 21.8, 22.3, 22.2, 21.9])
    out = _run(_finance_margin_volatility, df)
    assert out["opportunities"], "a stable margin produced no note"
    assert not out["risks"], "flagged a margin holding within 1pp as unstable"


def test_period_labels_are_written_as_dates_not_timestamps():
    """"Weakest period: 2023-07-01 00:00:00" prints a midnight nobody
    supplied, and reads as machine output in a client report."""
    rng = np.random.default_rng(43)
    n = 24
    rev = rng.uniform(80_000, 120_000, n)
    margins = np.where(np.arange(n) % 2 == 0, 0.40, 0.05)
    df = pd.DataFrame({
        "period": pd.date_range("2023-01-01", periods=n, freq="MS"),
        "revenue": rev.round(2),
        "cost": (rev * (1 - margins)).round(2),
    })
    out = _run(_finance_margin_volatility, df)
    text = _text(out)
    assert "00:00:00" not in text, "a raw timestamp reached the report text"
    assert out["risks"], "the alternating margin was not flagged"


def test_margin_volatility_needs_enough_periods():
    """Three periods cannot establish a pattern."""
    df = _margin_frame([40, 5, 30])
    out = _run(_finance_margin_volatility, df)
    assert not out["findings"], "drew a volatility conclusion from 3 periods"


# ══════════════════════════════════════════════════════════
#  Loss-making segments
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def segment_loss_df():
    """Two profitable segments and one that loses money — a group total
    that is positive and hides it."""
    rng = np.random.default_rng(36)
    rows = []
    for seg, margin, n in (("Retail", 0.30, 200), ("Trade", 0.25, 150),
                           ("Wholesale", -0.20, 120)):
        rev = rng.uniform(1_000, 5_000, n)
        rows.append(pd.DataFrame({
            "category": seg,
            "revenue": rev.round(2),
            "cost": (rev * (1 - margin)).round(2),
        }))
    return pd.concat(rows, ignore_index=True)


def test_loss_making_segment_is_named_and_sized(segment_loss_df):
    out = _run(_finance_loss_making_segments, segment_loss_df)
    text = _text(out)
    assert out["findings"], "a segment trading 20% below cost was not reported"
    assert "Wholesale" in text, "the loss-making segment is not named"
    assert out["risks"], "no risk raised for a segment trading below cost"


def test_loss_drag_is_stated_against_the_profitable_segments(segment_loss_df):
    """"Loses 180,000" means nothing without the scale it is measured
    against."""
    out = _run(_finance_loss_making_segments, segment_loss_df)
    assert any("%" in f for f in out["findings"]), \
        "the drag is reported in absolute terms only"


def test_all_profitable_segments_produce_no_loss_finding():
    rng = np.random.default_rng(37)
    rows = []
    for seg in ("Retail", "Trade", "Wholesale"):
        rev = rng.uniform(1_000, 5_000, 150)
        rows.append(pd.DataFrame({"category": seg, "revenue": rev.round(2),
                                  "cost": (rev * 0.7).round(2)}))
    df = pd.concat(rows, ignore_index=True)
    out = _run(_finance_loss_making_segments, df)
    assert not out["findings"], "reported a loss in an all-profitable book"


def test_immaterial_loss_is_not_raised():
    """A segment losing 0.2% of group profit is a rounding error, and
    raising it as a risk trains the reader to ignore the section."""
    rng = np.random.default_rng(38)
    big = pd.DataFrame({"category": "Retail",
                        "revenue": rng.uniform(9_000, 11_000, 400).round(2)})
    big["cost"] = (big["revenue"] * 0.7).round(2)
    tiny = pd.DataFrame({"category": "Trial", "revenue": [100.0] * 5})
    tiny["cost"] = [102.0] * 5
    df = pd.concat([big, tiny], ignore_index=True)
    out = _run(_finance_loss_making_segments, df)
    assert not out["risks"], "raised a risk over a 10-unit loss"


# ══════════════════════════════════════════════════════════
#  Benford first-digit conformity
# ══════════════════════════════════════════════════════════

def _benford_sample(n, rng):
    """Values whose leading digits follow Benford — a log-uniform spread
    over several orders of magnitude does this by construction."""
    return np.power(10, rng.uniform(2, 6, n)).round(2)


def test_benford_conforming_data_is_reported_as_conforming():
    rng = np.random.default_rng(39)
    df = pd.DataFrame({"revenue": _benford_sample(4000, rng)})
    out = _run(_finance_benford, df)
    assert out["findings"], "no conformity statement produced"
    assert "conformity" in _text(out)
    assert not out["insights"], "flagged conforming data as deviating"


def test_benford_deviation_is_flagged_without_alleging_anything():
    """Values concentrated on a threshold — the classic approval-limit
    pattern. It must be reported as a place to look, never as a finding of
    manipulation."""
    rng = np.random.default_rng(40)
    natural = _benford_sample(2000, rng)
    clustered = rng.uniform(4800, 4999, 2500).round(2)   # forced leading 4
    df = pd.DataFrame({"revenue": np.concatenate([natural, clustered])})
    out = _run(_finance_benford, df)
    assert out["insights"], "a heavy digit cluster produced no insight"
    ins = out["insights"][0]
    blob = " ".join([ins.title, ins.problem, ins.cause, ins.action, ins.impact]).lower()
    for word in ("fraud", "manipulation", "falsified", "fabricated", "misconduct"):
        assert word not in blob, \
            "the digit test alleges {!r} from a distribution alone".format(word)
    assert "cannot distinguish" in blob or "not, on its own" in blob, \
        "no statement of what the test cannot establish"


def test_benford_is_skipped_on_a_narrow_value_range():
    """Prices between 90 and 110 fail Benford while being entirely
    legitimate; running the test there produces a false alarm."""
    rng = np.random.default_rng(41)
    df = pd.DataFrame({"revenue": rng.uniform(90, 110, 2000).round(2)})
    out = _run(_finance_benford, df)
    assert not out["findings"], "ran the digit test on a one-decade range"


def test_benford_is_skipped_on_a_small_population():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"revenue": _benford_sample(120, rng)})
    out = _run(_finance_benford, df)
    assert not out["findings"], "ran the digit test on 120 records"


# ══════════════════════════════════════════════════════════
#  The orchestrator wires them in
# ══════════════════════════════════════════════════════════

def test_full_finance_run_includes_the_new_analyses(known_cost_structure):
    df = known_cost_structure.copy()
    out = _insights_finance(df, {}, [])
    blob = " ".join(out["findings"] + out["risks"] + out["opportunities"])
    assert "Break-even" in blob, "cost structure not reached by the orchestrator"
    assert "Margin averages" in blob or "Margin is stable" in blob, \
        "margin stability not reached by the orchestrator"


def test_finance_findings_all_carry_a_number(known_cost_structure):
    """A finding without a figure is an opinion."""
    import re
    out = _insights_finance(known_cost_structure, {}, [])
    for f in out["findings"]:
        assert re.search(r"\d", f), "finding states no figure: {!r}".format(f)
