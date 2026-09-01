"""
tests/test_governance.py — what the data is, and who it identifies.

The part a client's legal or security function asks for, and the part
almost nothing computes: removing the name column is what most teams
mean by anonymising, and it is not anonymising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines.governance import (K_ANONYMITY_FLOOR, assess_reidentification,
                                    build_dictionary, build_governance,
                                    classify_column)


@pytest.fixture()
def people() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 900
    return pd.DataFrame({
        "employee_id": range(n),
        "full_name": ["Person {}".format(i) for i in range(n)],
        "work_email": ["p{}@corp.example".format(i) for i in range(n)],
        "postcode": rng.choice(["SW1A 1AA", "EC1V 9NR", "M1 4BT"], n),
        "gender": rng.choice(["M", "F"], n),
        "age": rng.integers(22, 60, n),
        "department": rng.choice(["Sales", "Engineering"], n),
        "salary": rng.normal(60000, 12000, n).round(0),
        "health_condition": rng.choice(["none", "asthma", "diabetes"], n),
        "attrition": rng.choice([0, 1], n),
    })


class TestClassification:
    def test_a_direct_identifier_is_recognised(self, people):
        assert classify_column("full_name") == "direct"
        assert classify_column("work_email", people.work_email) == "direct"

    def test_special_category_data_is_recognised(self):
        assert classify_column("health_condition") == "special"
        assert classify_column("ethnic_group") == "special"
        assert classify_column("trade_union_member") == "special"

    def test_a_quasi_identifier_is_recognised(self):
        """Identifies nobody alone, most people in combination."""
        for col in ("postcode", "date_of_birth", "gender", "job_title"):
            assert classify_column(col) == "quasi-identifier", col

    def test_a_plain_measure_is_not_flagged(self, people):
        assert classify_column("salary", people.salary) == "none"

    def test_content_beats_the_column_name(self):
        """A column called "reference" full of email addresses is
        personal data whatever it is called."""
        emails = pd.Series(["a@b.com", "c@d.org", "e@f.net"] * 40)
        assert classify_column("reference", emails) == "direct"


class TestDictionary:
    def test_every_column_gets_a_row(self, people):
        assert len(build_dictionary(people)) == people.shape[1]

    def test_it_names_the_role_each_column_plays(self, people):
        by_name = {c.name: c for c in build_dictionary(people,
                                                       target="attrition")}
        assert by_name["employee_id"].role == "identifier"
        assert by_name["salary"].role == "measure"
        assert by_name["department"].role == "dimension"
        assert by_name["attrition"].role == "outcome"

    def test_a_sensitive_example_value_is_withheld(self, people):
        """The dictionary is itself a document that gets circulated."""
        by_name = {c.name: c for c in build_dictionary(people)}
        assert by_name["work_email"].example == "(withheld)"
        assert "@" not in by_name["work_email"].example
        assert by_name["salary"].example != "(withheld)"

    def test_completeness_is_measured_not_assumed(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0, None]})
        assert build_dictionary(df)[0].completeness_pct == 50.0


class TestReidentification:
    def test_it_finds_the_people_a_name_column_would_not_protect(self):
        """Every row unique on postcode + age + gender: deleting names
        anonymises none of them."""
        n = 400
        df = pd.DataFrame({
            "postcode": ["PC{}".format(i) for i in range(n)],
            "age": range(20, 20 + n),
            "gender": ["F"] * n,
        })
        risk = assess_reidentification(df)
        assert risk is not None
        assert risk.verdict == "High"
        assert risk.unique_pct > 50

    def test_a_well_grouped_file_is_low_risk(self):
        n = 600
        df = pd.DataFrame({
            "department": ["Sales"] * (n // 2) + ["Engineering"] * (n // 2),
            "gender": ["F", "M"] * (n // 2),
        })
        risk = assess_reidentification(df)
        assert risk.verdict == "Low"
        assert risk.k_min >= K_ANONYMITY_FLOOR

    def test_a_continuous_field_is_banded_before_counting(self):
        """Exact ages make everyone unique and say nothing about real
        risk; the question is whether a band identifies someone."""
        rng = np.random.default_rng(1)
        n = 800
        df = pd.DataFrame({
            "age": rng.normal(40, 9, n),          # continuous, all distinct
            "department": rng.choice(["A", "B"], n),
        })
        risk = assess_reidentification(df)
        assert risk is not None
        assert risk.unique_pct < 50, "exact values were not banded"

    def test_one_quasi_identifier_is_not_a_combination(self):
        df = pd.DataFrame({"gender": ["M", "F"] * 50})
        assert assess_reidentification(df) is None


class TestRecord:
    def test_it_states_the_obligation_not_the_citation(self, people):
        record = build_governance(people, retention_days=30)
        text = " ".join(record.obligations)
        assert "Article 9" in text
        assert "lawful basis" in text
        # And says what to do, not merely that a rule exists.
        assert "hash" in text or "Remove" in text

    def test_a_clean_dataset_says_it_can_be_shared(self):
        df = pd.DataFrame({"revenue": [1.0, 2.0, 3.0] * 40,
                           "units": [1, 2, 3] * 40})
        record = build_governance(df)
        assert any("can be shared" in o for o in record.obligations)

    def test_retention_is_stated_in_days(self, people):
        record = build_governance(people, retention_days=30)
        assert record.retention_days == 30
        assert "30 days" in record.retention_note

    def test_lineage_records_what_reached_the_report(self, people):
        record = build_governance(people)
        assert record.lineage
        assert any("900" in step for step in record.lineage)

    def test_counts_are_whole_numbers(self, people):
        """"2.00 are unique" — a tally formatted as a measure."""
        record = build_governance(people)
        if record.reidentification:
            assert ".0" not in record.reidentification.explanation.split(
                "%")[0]
