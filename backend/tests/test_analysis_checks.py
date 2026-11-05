"""
Checks a careful analyst runs before sharing: a month that breaks from the
ones before it is flagged, and a gap that reverses inside every group of
another column is reported as a mix effect, not as a segment gap.
"""
from __future__ import annotations

import io

from _check_data import daily_sales, simpson_df, spike_df
from app.engines.domains.general_depth import unusual_months
from app.engines.story_engine import generate_story


def test_a_month_far_from_the_six_before_it_is_flagged():
    r = generate_story(spike_df())
    assert any("Sep 2024" in f and "broke from" in f for f in r.key_findings)
    assert any("Sep 2024" in i.title for i in r.top_insights)


def test_the_months_after_a_spike_are_not_flagged_as_a_rebound():
    r = generate_story(spike_df())
    assert not any("Oct 2024" in f for f in r.key_findings)


def test_steady_months_raise_nothing():
    r = generate_story(spike_df(spike_month=None))
    assert not any("broke from" in f for f in r.key_findings)


def _flags(df):
    findings = []
    unusual_months(df, [], findings, [], [])
    return findings


def test_noise_rarely_raises_a_month():
    # Calibrated to about 5% of series with nothing in them; a six-month
    # MAD as the scale flagged 88%.
    hits = sum(bool(_flags(daily_sales(seed=s, per_day=3))) for s in range(40))
    assert hits <= 3, hits


def test_a_seasonal_peak_is_not_an_unusual_month():
    hits = sum(bool(_flags(daily_sales(seed=s, season=0.4))) for s in range(20))
    assert hits <= 1, hits


def test_with_two_years_a_month_is_judged_against_the_year_before():
    out = _flags(daily_sales(seed=2, season=0.4, spike=(2024, 7)))
    assert out and "year-on-year" in out[0] and "Jul 2024" in out[0]


def test_months_of_a_few_records_are_not_tested():
    assert not _flags(daily_sales(seed=1, per_day=0.3, spike=(2024, 7), factor=5))


def test_a_reversing_gap_is_called_a_mix_effect_not_a_segment_gap():
    r = generate_story(simpson_df())
    assert any("mix effect" in f and "'Plan'" in f for f in r.key_findings)
    titles = [i.title for i in r.top_insights]
    assert any("reverses within 'Plan'" in t for t in titles)
    # A is never named as the stronger group on its own.
    assert not any("'Group' Segments Differ" in t for t in titles)


def test_hindi_reports_carry_the_new_checks_translated(client):
    from app import i18n
    for df in (spike_df(), simpson_df(), daily_sales(seed=2, season=0.4, spike=(2024, 7))):
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post("/api/datasets/upload", files={"file": ("d.csv", buf.getvalue(), "text/csv")})
        ds = r.json()["meta"]["dataset_id"]
        i18n.clear_misses()
        for kind in ("pdf", "health-pdf"):
            assert client.post(f"/api/reports/{ds}/{kind}", json={"language": "hi"}).status_code == 200
        assert i18n.misses() == {}
