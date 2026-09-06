"""
Analysis that restates its own input, and calls the restatement a finding.

Every ranking engine in the app sorts candidates by strength of
association, so a column computed from the thing being explained wins
outright — perfect fit, vanishing p-value, largest effect. The output is
the most convincing-looking the engine can produce and says nothing:

    Deep Analysis → root cause:
        "Top driver: 'revenue_k' — the low group averages 0.14 against
         0.62 for the high group. Bring to high-performer level."

    Deep Analysis → cohorts:
        "Investigate what '700+' does differently to achieve 596.9%
         higher 'revenue'."

Both were real output. Low revenue is not caused by low revenue in
thousands, and the 700+ band earns more because that is what puts a row
in it. These tests pin the guards, and — just as importantly — pin that
genuine drivers and genuine cohort differences still come through, since
a guard that silences real findings is the worse failure.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines.bi.cohort import analyze_cohort
from app.engines.bi.root_cause import analyze_root_cause
from app.services.stat_guards import is_restatement


# ══════════════════════════════════════════════════════════
#  The primitive
# ══════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def money() -> np.ndarray:
    return np.random.default_rng(4).gamma(3, 200, 600)


def test_a_copy_is_a_restatement(money):
    assert is_restatement(money, money.copy()) is True


def test_a_unit_change_is_a_restatement(money):
    """The one that shipped: revenue and revenue_k."""
    assert is_restatement(money, money / 1000.0) is True


def test_a_monotone_transform_is_a_restatement(money):
    """A log or a rank reorders nothing, so it explains nothing."""
    assert is_restatement(money, np.log(money + 1)) is True
    assert is_restatement(money, pd.Series(money).rank()) is True


def test_a_strong_but_real_relationship_is_not(money):
    """The threshold is extreme on purpose. r≈0.96 is a finding."""
    noisy = money + np.random.default_rng(9).normal(0, 120, len(money))
    assert abs(np.corrcoef(money, noisy)[0, 1]) > 0.9
    assert is_restatement(money, noisy) is False


def test_an_unrelated_column_is_not(money):
    other = np.random.default_rng(12).normal(0, 1, len(money))
    assert is_restatement(money, other) is False


@pytest.mark.parametrize("a,b", [
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),            # too few paired rows
    ([5.0] * 100, [5.0] * 100),                     # no variation
    ([1.0, None, 3.0], [None, 2.0, None]),          # no overlap at all
])
def test_an_unclear_case_is_reported_not_hidden(a, b):
    """When the test cannot tell, the finding survives. Suppressing on a
    guess is how a guard starts eating real analysis."""
    assert is_restatement(pd.Series(a), pd.Series(b)) is False


# ══════════════════════════════════════════════════════════
#  Root cause
# ══════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def orders() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 800
    qty = rng.integers(1, 20, n)
    price = rng.normal(50, 12, n).round(2)
    rev = qty * price
    return pd.DataFrame({
        "revenue":       rev,
        "revenue_k":     rev / 1000.0,
        "total_revenue": rev,
        "quantity":      qty,
        "unit_price":    price,
        "region":        rng.choice(list("ABCD"), n),
    })


def test_the_top_driver_is_not_the_target_rewritten(orders):
    r = analyze_root_cause(orders, "revenue")
    assert r.top_driver == "quantity"

    factors = {d["factor"] for d in r.drivers}
    assert "revenue_k" not in factors
    assert "total_revenue" not in factors


def test_no_recommendation_asks_for_the_target_to_be_raised(orders):
    """"Bring revenue_k to 0.62" is an instruction to have more revenue."""
    r = analyze_root_cause(orders, "revenue")
    text = " ".join(r.recommendations) + r.interpretation
    assert "revenue_k" not in text
    assert "total_revenue" not in text
    assert "quantity" in text


def test_the_real_drivers_still_come_through(orders):
    """The guard must cost the analysis nothing that was worth having."""
    r = analyze_root_cause(orders, "revenue")
    factors = {d["factor"] for d in r.drivers}
    assert "quantity" in factors
    assert "unit_price" in factors


# ══════════════════════════════════════════════════════════
#  Cohorts
# ══════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def banded() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 900
    region = rng.choice(list("ABCD"), n)
    # Region A really does earn more — a finding the guard must not touch.
    rev = rng.gamma(3, 200, n) * np.where(region == "A", 1.6, 1.0)
    return pd.DataFrame({
        "revenue": rev,
        "region":  region,
        "revenue_band": pd.cut(
            rev, bins=[-1, 200, 400, 700, 1e9],
            labels=["0-200", "200-400", "400-700", "700+"]).astype(str),
    })


def test_a_band_of_the_metric_is_not_offered_as_a_difference_to_act_on(banded):
    c = analyze_cohort(banded, "revenue_band", "revenue")

    assert "ranges of revenue itself" in c.interpretation
    assert "shape of the distribution" in c.interpretation

    recs = " ".join(c.recommendations)
    assert "does differently" not in recs, (
        "the 700+ band does nothing differently — it is defined by the "
        "very number being compared")
    assert "not derived from it" in recs


def test_the_numbers_are_still_returned_for_a_binned_cohort(banded):
    """The ranking of a binned column is a legitimate distribution
    summary and the UI charts it. Only the causal claim is withdrawn."""
    c = analyze_cohort(banded, "revenue_band", "revenue")
    assert len(c.cohorts) == 4
    assert c.best_cohort == "700+"
    assert c.gap > 0
    assert c.is_significant is True


def test_a_genuine_cohort_difference_is_untouched(banded):
    c = analyze_cohort(banded, "region", "revenue")
    assert c.best_cohort == "A"
    assert "ranges of" not in c.interpretation
    assert "does differently" in " ".join(c.recommendations)


# ══════════════════════════════════════════════════════════
#  Segment health, and which way "better" runs
# ══════════════════════════════════════════════════════════
#
# The worst of the three. Health was scored as "bigger numbers", so on a
# dataset where region A had the highest revenue, half the churn and 40%
# lower support cost, the app ranked it LAST of four, listed its low churn
# and low cost as weaknesses, and printed:
#
#     "Improve 'churn_rate' from 0.04 to dataset average 0.08
#      — 46.7% improvement opportunity."
#
# That is the product advising a business to churn more customers.

from app.engines.bi.segments import analyze_segment_health


@pytest.fixture(scope="module")
def regions() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    n = 600
    region = rng.choice(list("ABCD"), n)
    good = np.where(region == "A", 1.0, 0.0)
    return pd.DataFrame({
        "region":       region,
        "revenue":      rng.gamma(3, 200, n) * np.where(good, 1.4, 1.0),
        "churn_rate":   rng.beta(2, 20, n) * np.where(good, 0.5, 1.0),
        "support_cost": rng.gamma(2, 30, n) * np.where(good, 0.6, 1.0),
    })


METRICS = ["revenue", "churn_rate", "support_cost"]


def test_the_best_segment_ranks_first(regions):
    """A wins on all three measures. It used to come last."""
    out = analyze_segment_health(regions, "region", METRICS)
    assert out[0].segment_name == "A"
    assert out[0].health_score > out[-1].health_score


def test_low_churn_and_low_cost_are_strengths(regions):
    a = next(s for s in analyze_segment_health(regions, "region", METRICS)
             if s.segment_name == "A")
    assert set(a.strengths) == {"revenue", "churn_rate", "support_cost"}
    assert a.weaknesses == []


def test_nothing_recommends_getting_worse(regions):
    """No segment may be told to move a lower-is-better metric up.

    Only the improvement sentence is in scope: "already leading in
    churn_rate" is the right thing to say to the segment that churns
    least, and says nothing about a direction to move in.
    """
    for s in analyze_segment_health(regions, "region", METRICS):
        opp = s.opportunity
        if not (opp.startswith("Raise") or opp.startswith("Bring down")):
            continue
        if "churn_rate" in opp or "support_cost" in opp:
            assert opp.startswith("Bring down"), opp
        if "revenue" in opp:
            assert opp.startswith("Raise"), opp
        # The old wording, verbatim, on the metric it was wrong about.
        assert "improvement opportunity" not in opp


def test_a_lower_is_better_metric_is_ranked_the_right_way_round(regions):
    """`rank` counts down from the largest mean, so on cost the top of
    that list is the worst segment until it is flipped."""
    a = next(s for s in analyze_segment_health(regions, "region", METRICS)
             if s.segment_name == "A")
    assert a.metrics["support_cost"]["rank"] == 1
    assert a.metrics["support_cost"]["status"] == "top"
    assert a.metrics["support_cost"]["direction"] == "lower"
    # The raw figure is still the raw figure — only the judgement flips.
    assert a.metrics["support_cost"]["vs_avg"] < 0


def test_a_metric_of_unknown_direction_is_not_judged(regions):
    """'metric_one' could be good high or good low; the name does not say,
    so it is reported and not scored — rather than assumed."""
    renamed = regions.rename(columns={
        "revenue": "metric_one", "churn_rate": "metric_two",
        "support_cost": "metric_three"})
    out = analyze_segment_health(
        renamed, "region", ["metric_one", "metric_two", "metric_three"])

    assert out, "the segments are still returned"
    for s in out:
        assert s.scored is False
        assert s.strengths == [] and s.weaknesses == []
        assert "cannot be called healthier" in s.opportunity
        assert s.metrics["metric_one"]["direction"] == "unknown"
        assert s.metrics["metric_one"]["favourable_pct"] is None


def test_a_known_direction_still_scores(regions):
    """The unknown-direction rule must not switch scoring off wholesale."""
    for s in analyze_segment_health(regions, "region", METRICS):
        assert s.scored is True


# ══════════════════════════════════════════════════════════
#  The whole BI run
# ══════════════════════════════════════════════════════════
#
# Each guard closes one door. The key-insights list is where a tautology
# suppressed in one engine walks back in through another: the cohort
# panel disclaimed the revenue-band gap, and the same gap was still
# printed at the top of "Key insights" with no disclaimer attached.

from app.engines.bi.runner import run_bi


@pytest.fixture(scope="module")
def bi_report():
    rng = np.random.default_rng(5)
    n = 900
    region = rng.choice(list("ABCD"), n)
    rev = rng.gamma(3, 200, n) * np.where(region == "A", 1.6, 1.0)
    df = pd.DataFrame({
        "revenue": rev,
        "region":  region,
        "units":   rng.integers(1, 20, n),
        "revenue_band": pd.cut(
            rev, bins=[-1, 200, 400, 700, 1e9],
            labels=["0-200", "200-400", "400-700", "700+"]).astype(str),
    })
    return run_bi(df)


def test_no_key_insight_is_a_restatement_of_its_own_subject(bi_report):
    joined = " ".join(bi_report.key_insights) + bi_report.executive_brief
    assert "Revenue Band" not in joined
    assert "revenue_band" not in joined


def test_the_real_finding_leads_instead(bi_report):
    """Region A genuinely earns 60% more. That is the headline."""
    assert bi_report.key_insights
    assert "Region" in bi_report.key_insights[0]
    assert any("A averages" in k for k in bi_report.key_insights)


def test_a_band_of_the_target_is_not_a_categorical_driver(bi_report):
    """Chi-square finds a crushing association between revenue and the
    bands cut from revenue — because that is how they were cut."""
    for rc in bi_report.root_causes:
        assert rc.top_driver != "revenue_band"
        assert "revenue_band" not in {d["factor"] for d in rc.drivers}


def test_no_verdict_on_segments_that_were_never_scored():
    """"None stands out as needing attention before the others" is a
    finding. With every segment holding the placeholder 50 — because no
    metric had a direction its name settles — nothing was compared, and
    the spread of 0 between placeholders was being read as agreement."""
    rng = np.random.default_rng(5)
    n = 900
    region = rng.choice(list("ABCD"), n)
    val = rng.gamma(3, 200, n) * np.where(region == "A", 1.6, 1.0)
    df = pd.DataFrame({
        "metric_one": val,
        "region":     region,
        "metric_two": rng.integers(1, 20, n),
    })
    report = run_bi(df)

    joined = " ".join(report.key_insights)
    assert "healthiest segment" not in joined
    assert "none stands out" not in joined
    # The real cohort findings are untouched.
    assert any("Region splits" in k for k in report.key_insights)


def test_one_measurement_does_not_occupy_two_analysis_slots():
    """Every BI analysis takes the first two or four numeric columns. A
    file carrying revenue and revenue_k spent two of them on one number
    and printed the finding twice — while a column with something else to
    say was never looked at."""
    df = pd.DataFrame({
        "revenue":    np.random.default_rng(2).gamma(3, 200, 600),
        "quantity":   np.random.default_rng(3).integers(1, 20, 600),
        "region":     np.random.default_rng(4).choice(list("ABCD"), 600),
    })
    df["revenue_k"] = df["revenue"] / 1000.0
    df = df[["revenue", "revenue_k", "quantity", "region"]]

    report = run_bi(df)
    analysed = {b.column for b in report.benchmarks}
    assert "revenue" in analysed
    assert "revenue_k" not in analysed
    # The slot freed by the duplicate goes to a column that has its own
    # story, rather than to the same one twice.
    assert "quantity" in analysed

    joined = " ".join(report.key_insights)
    assert "Revenue K" not in joined


def test_a_symmetric_driver_pair_is_reported_once():
    """Root cause is symmetric: run it on revenue and on quantity and the
    same fact comes back both ways round, as two separate-looking
    insights."""
    rng = np.random.default_rng(3)
    n = 800
    qty = rng.integers(1, 20, n)
    df = pd.DataFrame({
        "revenue":  qty * rng.normal(50, 12, n),
        "quantity": qty,
        "region":   rng.choice(list("ABCD"), n),
    })
    lines = [k for k in run_bi(df).key_insights if k.startswith("What separates")]
    assert len(lines) == 1, lines


def test_the_guard_does_not_cost_the_run_its_speed():
    """The check is O(pairs) and Spearman ranks its input, so run on
    every pair at full length it took a 50,000 x 40 BI run from 0.5s to
    16s — a guard nobody would keep. Sampling and an early stop put it
    back; this pins that it stays back."""
    import time
    rng = np.random.default_rng(1)
    n, k = 50_000, 40
    df = pd.DataFrame({"num_%d" % i: rng.normal(0, 1, n) for i in range(k)})
    df["region"] = rng.choice(list("ABCDE"), n)

    started = time.perf_counter()
    run_bi(df)
    elapsed = time.perf_counter() - started
    # Measured at 0.9s against a 0.5s baseline without the guard; the
    # bound is loose enough for a slow machine and tight enough to catch
    # a return to full-length pairwise checking.
    assert elapsed < 5.0, "BI run took %.1fs" % elapsed


@pytest.mark.parametrize("n", [20_000, 200_000])
def test_sampling_does_not_blunt_the_check(n):
    """The sample is a stride, not a head slice, so a file sorted by the
    column under test is still covered end to end."""
    rng = np.random.default_rng(1)
    v = np.sort(rng.gamma(3, 200, n))
    assert is_restatement(v, v / 1000) is True
    assert is_restatement(v, np.log(v + 1)) is True
    assert is_restatement(v, v + rng.normal(0, 120, n)) is False
