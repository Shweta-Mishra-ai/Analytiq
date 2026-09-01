"""
tests/test_report_polish.py — the defects that make a report look like a
debug dump rather than a deliverable.

Every case here was found by generating a report from a realistic HR
dataset and reading all 25 pages. None of them raised an error; each one
simply reached a client looking wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.engines import present
from app.engines.domains.general import (_is_obvious_segment_pair,
                                         _is_rating_scale)
from app.engines.domains.hr import _run_attrition
from app.engines.predictive import (compute_drivers, find_binary_target,
                                    find_top_cluster)


@pytest.fixture()
def attrition_df() -> pd.DataFrame:
    """HR data with one planted driver: overtime, recorded as Yes/No and
    converted to a boolean the way cleaning converts it."""
    rng = np.random.default_rng(11)
    n = 1400
    overtime = rng.choice([True, False], n, p=[0.28, 0.72])
    tenure = rng.gamma(2.2, 3.0, n).round(1)
    satisfaction = rng.integers(1, 5, n)
    logit = (-0.9 + 1.35 * overtime + 0.9 * (tenure < 2)
             - 0.42 * satisfaction)
    return pd.DataFrame({
        "EmployeeNumber": range(n),
        "Department": rng.choice(["Sales", "Research", "HR"], n),
        "JobRole": rng.choice(["Executive", "Scientist", "Technician",
                               "Manager"], n),
        "MonthlyIncome": rng.normal(6500, 1500, n).round(0),
        "YearsAtCompany": tenure,
        "OverTime": overtime,
        "JobSatisfaction": satisfaction,
        # 85% share one value — a column that cannot narrow anything.
        "PerformanceRating": rng.choice([3, 4], n, p=[0.85, 0.15]),
        "Attrition": rng.random(n) < 1 / (1 + np.exp(-logit)),
    })


# ── numbers and names a person can read ──────────────────

class TestPresentation:
    def test_a_measure_never_reaches_the_page_in_scientific_notation(self):
        # Median salaries were printing as "7.26e+03" in the findings.
        for value in (7260.0, 8024.30, 1_470_000, 0.00043, 12):
            assert "e+" not in present.num(value)
            assert "e-" not in present.num(value)

    def test_money_is_not_quoted_to_the_cent(self):
        assert present.num(8024.30) == "8,024"

    def test_column_names_become_words(self):
        assert present.label("MonthlyIncome") == "Monthly Income"
        assert present.label("years_at_company") == "Years at Company"
        assert present.label("OverTime") == "Overtime"

    def test_acronyms_survive_the_title_caser(self):
        assert present.label("mrr_usd") == "MRR USD"
        assert present.label("B2B_revenue") == "B2B Revenue"

    def test_a_cleaned_boolean_reads_as_the_value_the_client_typed(self):
        # Cleaning turns Yes/No into True/False. The report told people
        # their risk pocket was "OverTime = True".
        assert present.value(np.True_) == "Yes"
        assert present.value(np.False_) == "No"
        assert present.value("true") == "Yes"

    def test_quoting_is_stripped_from_values(self):
        assert present.value("'Sales'") == "Sales"

    def test_truncation_never_cuts_a_word_in_half(self):
        # "Northwind Manufacturing" became "Northwind Manufacturin" on the
        # cover of a client deliverable.
        out = present.truncate("Northwind Manufacturing", 22)
        assert out.endswith("…")
        assert "Manufacturin…" not in out

    def test_short_enough_text_is_left_alone(self):
        assert present.truncate("Sales", 22) == "Sales"


# ── findings that are actually findings ──────────────────

class TestFindingQuality:
    def test_a_rating_scale_has_levels_not_outliers(self):
        # IQR called the top 15% of a 3/4 performance rating a data
        # quality issue and advised capping or removing them.
        assert _is_rating_scale(pd.Series([3, 4] * 200))
        assert _is_rating_scale(pd.Series([1, 2, 3, 4] * 100))

    def test_a_measurement_still_gets_outlier_checks(self):
        assert not _is_rating_scale(pd.Series([6000.5, 7000.2] * 100))
        assert not _is_rating_scale(pd.Series(range(22, 60)))

    def test_pay_differing_by_job_role_is_not_a_finding(self):
        # The role IS the pay band. This was a HIGH severity headline.
        assert _is_obvious_segment_pair("JobRole", "MonthlyIncome")
        assert _is_obvious_segment_pair("JobLevel", "Salary")

    def test_pay_differing_by_department_still_is(self):
        assert not _is_obvious_segment_pair("Department", "MonthlyIncome")
        assert not _is_obvious_segment_pair("Region", "Revenue")


# ── drivers ──────────────────────────────────────────────

class TestDrivers:
    def test_a_boolean_driver_is_not_invisible(self, attrition_df):
        """select_dtypes("number") excludes bool and so does
        ["object", "string"], so a cleaned Yes/No column fell through
        both filters and never reached the drivers table — even when it
        was the strongest driver in the data."""
        result = _run_attrition(attrition_df)
        factors = [d["factor"] for d in result.top_drivers]
        assert "OverTime" in factors, factors

    def test_the_planted_driver_ranks_first(self, attrition_df):
        result = _run_attrition(attrition_df)
        assert result.top_drivers[0]["factor"] == "OverTime"

    def test_every_driver_carries_an_effect_size(self, attrition_df):
        for d in _run_attrition(attrition_df).top_drivers:
            assert 0 <= d["effect"] <= 1
            assert d["effect_label"] in ("small", "moderate", "large")

    def test_significant_but_negligible_is_not_reported(self, attrition_df):
        """At n=1,400 almost anything clears p<0.05. A driver that
        separates nobody is not a driver."""
        for d in _run_attrition(attrition_df).top_drivers:
            assert d["effect"] >= 0.147, d

    def test_a_categorical_driver_reports_its_group_sizes(self, attrition_df):
        cat = next(d for d in _run_attrition(attrition_df).top_drivers
                   if d["type"] == "categorical")
        assert cat["n_worst"] >= 20 and cat["n_best"] >= 20
        assert "n=" in cat["detail"]

    def test_driver_detail_speaks_the_clients_vocabulary(self, attrition_df):
        cat = next(d for d in _run_attrition(attrition_df).top_drivers
                   if d["type"] == "categorical")
        assert "True" not in cat["detail"], cat["detail"]
        assert "Yes" in cat["detail"] or "No" in cat["detail"]

    def test_the_model_and_the_tests_name_the_same_driver(self, attrition_df):
        """Importances used to be summed across a column's one-hot
        dummies, so a four-level job role collected four contributions
        against a binary flag's one. The report then headlined Job Role
        while the statistical tests on the same page ranked Overtime
        first — two answers to one question."""
        target = find_binary_target(attrition_df)
        model = compute_drivers(attrition_df, target)
        assert model is not None
        assert model.top_drivers[0][0] == "OverTime", model.top_drivers


class TestRiskCluster:
    def test_a_condition_that_narrows_nothing_is_dropped(self, attrition_df):
        """"Overtime = Yes AND Performance Rating = 3" reads as a precise
        two-factor pocket, but 85% of people are rated 3 — the second
        condition only makes the first look more targeted than it is."""
        cluster = find_top_cluster(attrition_df,
                                   find_binary_target(attrition_df))
        assert cluster is not None
        assert "Performance Rating" not in cluster.description, \
            cluster.description

    def test_the_cluster_reads_in_the_clients_own_values(self, attrition_df):
        cluster = find_top_cluster(attrition_df,
                                   find_binary_target(attrition_df))
        assert "= True" not in cluster.description
        assert "_" not in cluster.description

    def test_the_cluster_is_sharper_than_the_base_rate(self, attrition_df):
        cluster = find_top_cluster(attrition_df,
                                   find_binary_target(attrition_df))
        assert cluster.rate > cluster.base_rate * 1.4


# ── one figure, one source ───────────────────────────────

def test_replacement_cost_is_quoted_one_way_everywhere():
    """Two published ranges were in circulation — 50-200% and 50-150% —
    and the same page carried both for the same quantity."""
    import re
    from pathlib import Path
    from app.engines.industry_benchmarks import REPLACEMENT_COST_RANGE

    assert "50-200" in REPLACEMENT_COST_RANGE
    root = Path(__file__).resolve().parent.parent / "app" / "engines"
    stale = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue        # comments may recount the history
            if re.search(r"50[-–]150\s*%", line):
                stale.append(f"{path.name}: {stripped}")
    assert not stale, stale


# ── the document itself ──────────────────────────────────

class TestDocument:
    """Built from a real report, because these defects only exist once
    the flowables have been laid out on pages."""

    @staticmethod
    def _pdf(df, **over):
        from app.engines.pdf_builder import build_pdf
        config = {"title": "Workforce Review", "client_name": "Northwind "
                  "Manufacturing Group", "theme_name": "corporate",
                  "prepared_by": "S. Mishra"}
        config.update(over.pop("config", {}))
        return build_pdf(df=df, config=config, domain="hr", **over)

    @staticmethod
    def _pages(raw: bytes):
        import io
        import pypdf
        return [p.extract_text() for p in pypdf.PdfReader(io.BytesIO(raw)).pages]

    def test_no_section_heading_is_stranded_on_an_empty_page(self,
                                                             attrition_df):
        """The Findings section — the body of the report — rendered its
        heading, a promise of what was coming, and nothing else; the
        findings themselves started overleaf."""
        pages = self._pages(self._pdf(attrition_df))
        for i, text in enumerate(pages):
            body = [ln for ln in text.splitlines()
                    if ln.strip() and "Analytiq" not in ln
                    and "Page " not in ln and "Northwind" not in ln]
            assert len(body) > 3, \
                "page {} carries only {}".format(i + 1, body)

    def test_the_contents_locates_every_section(self, attrition_df):
        """A contents page that numbers the sections 1..n tells the reader
        the order of a document they are already holding."""
        first = self._pages(self._pdf(attrition_df))[1]
        assert "Contents" in first
        # Section rows end in a page number; at least most must resolve.
        numbers = [ln for ln in first.splitlines() if ln.strip().isdigit()]
        assert len(numbers) >= 8, first

    def test_confidential_is_not_claimed_unless_asked_for(self,
                                                          attrition_df):
        plain = "\n".join(self._pages(self._pdf(attrition_df)))
        assert "CONFIDENTIAL" not in plain.upper(), \
            "an unmarked report was stamped confidential"

    def test_confidential_is_honoured_when_asked_for(self, attrition_df):
        marked = "\n".join(self._pages(
            self._pdf(attrition_df, config={"confidential": True})))
        assert "CONFIDENTIAL" in marked.upper()

    def test_no_client_facing_page_shows_scientific_notation(self,
                                                             attrition_df):
        import re
        text = "\n".join(self._pages(self._pdf(attrition_df)))
        hits = re.findall(r"\d\.\d+e[+-]\d+", text)
        assert not hits, hits

    def test_exhibit_numbering_starts_at_one(self, attrition_df):
        """The contents needs two passes over the story to find its page
        numbers. Exhibit numbers are handed out during that pass, so
        without a reset the report opened at Exhibit 7."""
        text = "\n".join(self._pages(self._pdf(attrition_df)))
        if "Exhibit" in text:
            assert "Exhibit 1" in text


# ── a caption describes its own chart ────────────────────

class TestChartNarratives:
    """The narrator used to recover a chart's columns by matching the
    chart title against column names. Titles are prettified, so "JobRole"
    is not found in "Job Role", the lookup fell through to the first
    categorical column, and a chart of seven job roles was captioned with
    a confident paragraph about three departments."""

    @staticmethod
    def _charts(df):
        from app.engines.chart_exporter import generate_all_charts
        return generate_all_charts(df, "Corporate Light", max_charts=5)

    def test_every_chart_says_which_columns_it_used(self, attrition_df):
        for title, _img, spec in self._charts(attrition_df):
            assert spec.kind in ("bar", "hist", "trend", "correlation"), title
            if spec.kind == "bar":
                assert spec.metric in attrition_df.columns
                assert spec.dimension in attrition_df.columns

    def test_a_caption_names_its_own_dimension(self, attrition_df):
        from app.ai.report_narrator import generate_chart_narrative
        from app.engines.present import label

        for title, _img, spec in self._charts(attrition_df):
            if spec.kind != "bar":
                continue
            text = generate_chart_narrative(attrition_df, title, "", "hr",
                                            spec=spec)
            assert label(spec.dimension) in text, \
                "{!r} is captioned: {}".format(title, text)

    def test_a_caption_counts_the_groups_its_chart_shows(self, attrition_df):
        from app.ai.report_narrator import generate_chart_narrative

        for title, _img, spec in self._charts(attrition_df):
            if spec.kind != "bar":
                continue
            n = attrition_df[spec.dimension].nunique()
            text = generate_chart_narrative(attrition_df, title, "", "hr",
                                            spec=spec)
            assert "{} ".format(n) in text, \
                "{} has {} groups but reads: {}".format(title, n, text)

    def test_captions_carry_no_raw_floats(self, attrition_df):
        import re
        from app.ai.report_narrator import generate_chart_narrative

        for title, _img, spec in self._charts(attrition_df):
            text = generate_chart_narrative(attrition_df, title, "", "hr",
                                            spec=spec)
            # "8986.134" next to a chart axis reading "8,986".
            assert not re.search(r"\d{4,}\.\d", text), text
            assert "e+" not in text

    def test_a_narrow_spread_is_not_called_urgent(self):
        """Every gap used to be "requiring attention", however small."""
        from app.ai.report_narrator import _fb_bar
        narrow = _fb_bar({"ok": True, "metric_label": "Revenue",
                          "dimension_label": "Region", "n_groups": 4,
                          "top": "North", "top_val": 103.0,
                          "worst": "South", "worst_val": 100.0,
                          "gap_pct": 3.0, "above_avg": 2, "org_avg": 101.0})
        assert "narrow" in narrow
        assert "requiring attention" not in narrow

    def test_a_wide_spread_still_gets_a_recommendation(self):
        wide = _fb_bar_wide()
        assert "large enough to act on" in wide


def _fb_bar_wide() -> str:
    from app.ai.report_narrator import _fb_bar
    return _fb_bar({"ok": True, "metric_label": "Revenue",
                    "dimension_label": "Region", "n_groups": 4,
                    "top": "North", "top_val": 200.0, "worst": "South",
                    "worst_val": 100.0, "gap_pct": 50.0,
                    "above_avg": 1, "org_avg": 150.0})


def test_grammar_agrees_with_the_numbers():
    """"a 18% spread" and "1 of 3 groups sit" are the kind of slip that
    makes a paragraph read as generated."""
    assert present.article("18%") == "an"
    assert present.article("22%") == "a"
    assert present.article("8,024") == "an"
    assert present.plural(1, "sits", "sit") == "sits"
    assert present.plural(3, "sits", "sit") == "sit"


# ── the action plan ──────────────────────────────────────

class TestActionPlan:
    """The page a client runs a meeting from. It used to be two canned
    sentences — one of which described a flight-risk profile built on a
    promotion column the dataset did not contain, contradicting the
    analysis on the page before it."""

    @staticmethod
    def _plan(insights, actions):
        from reportlab.lib.units import mm
        from app.engines.pdf.data_sections import _recommendations
        from app.engines.pdf.theme import THEMES, _styles
        theme = THEMES["Corporate Light"]
        story: list = []
        _recommendations(story, _styles(theme), theme, actions, 170 * mm,
                         insights=insights)
        return story

    @staticmethod
    def _cells(story):
        from reportlab.platypus import Table
        for flowable in story:
            if isinstance(flowable, Table):
                return [[getattr(c, "text", "") for c in row]
                        for row in flowable._cellvalues]
        return []

    def test_each_action_cites_the_finding_it_came_from(self):
        """Matching an action to a finding on shared words cited an income
        finding beside an overtime action."""
        from app.engines.domains.base import build_insight
        insights = [
            build_insight(title="Attrition at 19.9%", problem="292 of 1,470 left",
                          cause="", evidence="", severity="critical",
                          category="attrition",
                          action="1. Take the Overtime finding to managers  2. x",
                          impact=""),
            build_insight(title="Pay varies by department",
                          problem="Median income runs from 7,469 to 8,737",
                          cause="", evidence="", severity="high",
                          category="segmentation",
                          action="1. Decide which end is desirable  2. y",
                          impact=""),
        ]
        rows = self._cells(self._plan(insights, []))[1:]
        overtime = next(r for r in rows if "Overtime" in r[1])
        assert "292 of 1,470" in overtime[2], overtime
        pay = next(r for r in rows if "desirable" in r[1])
        assert "7,469" in pay[2], pay

    def test_an_action_with_no_finding_cites_nothing(self):
        rows = self._cells(self._plan([], ["[CRITICAL] Do the thing"]))[1:]
        assert rows, "a standalone action should still get a row"
        assert rows[0][2].strip() in ("", "—") or "—" in rows[0][2]

    def test_owner_and_date_are_left_for_the_client(self):
        from app.engines.domains.base import build_insight
        insights = [build_insight(title="T", problem="P", cause="",
                                  evidence="", action="1. Do it", impact="",
                                  severity="critical", category="attrition")]
        header = self._cells(self._plan(insights, []))[0]
        assert "Owner" in header[3] and "By when" in header[4]
        assert self._cells(self._plan(insights, []))[1][3].strip() == ""

    def test_nothing_to_recommend_says_so(self):
        story = self._plan([], [])
        text = " ".join(getattr(f, "text", "") for f in story)
        assert "found nothing it could recommend" in text
