"""
tests/test_bi_rigour.py — the BI engine had missed the rigour pass.

Found by opening the Business Intelligence page: the report opened with
"Root cause — low EmployeeNumber: 368 low performers (25.0%)" and stated
that low performers had "129.6% lower Years at Company", which is not a
figure that can exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.bi_engine import _is_performance_metric, run_bi


@pytest.fixture()
def workforce() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 1200
    dept = rng.choice(["Sales", "Research", "HR"], n, p=[0.45, 0.45, 0.10])
    tenure = rng.gamma(2.2, 3.0, n).round(1)
    return pd.DataFrame({
        "EmployeeNumber": range(1, n + 1),
        "Age": rng.integers(22, 60, n),
        "Department": dept,
        "MonthlyIncome": (6000 + tenure * 310
                          + rng.normal(0, 1500, n)).round(0),
        "YearsAtCompany": tenure,
        "JobSatisfaction": rng.integers(1, 5, n),
    })


class TestMetricSelection:
    def test_an_identifier_is_never_a_metric(self, workforce):
        """"Root cause — low EmployeeNumber" analyses a row number as
        underperformance."""
        report = run_bi(workforce)
        text = " ".join(report.key_insights) + report.executive_brief
        assert "EmployeeNumber" not in text, text
        assert not any(rc.target_col == "EmployeeNumber"
                       for rc in report.root_causes)

    def test_a_demographic_is_not_a_performance_metric(self):
        """Root-cause analysis asks "why is this low", which is only a
        question where low is worse."""
        assert not _is_performance_metric("Age")
        assert not _is_performance_metric("YearsAtCompany")
        assert not _is_performance_metric("EmployeeNumber")

    def test_a_real_metric_is(self):
        for col in ("MonthlyIncome", "Revenue", "conversion_rate",
                    "JobSatisfaction", "nps_score"):
            assert _is_performance_metric(col), col

    def test_root_cause_runs_on_a_metric(self, workforce):
        report = run_bi(workforce)
        for rc in report.root_causes:
            assert _is_performance_metric(rc.target_col), rc.target_col


class TestArithmetic:
    def test_a_shortfall_cannot_exceed_a_hundred_percent(self, workforce):
        """The gap was measured against the group that has less and then
        described as a reduction, so 3.44 against 7.90 read "129.6%
        lower"."""
        report = run_bi(workforce)
        for rc in report.root_causes:
            for driver in rc.drivers:
                gap = driver.get("diff_pct")
                if gap is not None and driver.get("direction") != "categorical":
                    assert gap <= 100, driver

    def test_a_negligible_driver_is_not_reported(self, workforce):
        """On 1,200 rows almost any difference clears p<0.05."""
        report = run_bi(workforce)
        for rc in report.root_causes:
            for driver in rc.drivers:
                if "effect" in driver:
                    assert driver["effect"] >= 0.2, driver


class TestWording:
    def test_no_p_value_prints_as_zero(self, workforce):
        report = run_bi(workforce)
        blob = " ".join(
            [report.executive_brief, *report.key_insights]
            + [d.get("detail", "") for rc in report.root_causes
               for d in rc.drivers])
        assert "p=0.0000" not in blob
        assert "p = 0.0000" not in blob

    def test_column_names_are_not_quoted_like_identifiers(self, workforce):
        report = run_bi(workforce)
        for insight in report.key_insights:
            assert "'" not in insight, insight

    def test_the_absence_of_a_finding_is_not_listed_as_one(self, workforce):
        """"Age is evenly distributed across Department" filled the
        insight list with things that are not news."""
        report = run_bi(workforce)
        assert not any("evenly distributed" in i for i in report.key_insights)

    def test_a_trivial_segment_spread_says_so(self, workforce):
        """A "healthiest" segment scoring 50 and a "needs most attention"
        scoring 48 is a ranking artefact, not a finding."""
        report = run_bi(workforce)
        health = [i for i in report.key_insights if "health" in i.lower()]
        for line in health:
            if "healthiest" in line.lower():
                assert "spread" in line
            else:
                assert "none stands out" in line
