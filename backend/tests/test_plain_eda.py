"""The same result, said twice.

Analytiq has two readers: the director who needs to know what to do, and
the analyst who needs to check the working. Pitching at the midpoint
serves neither, so every explanatory field ships in both registers. The
rule these tests enforce is that the plain reading never replaces the
technical one and never claims more than it.
"""
import numpy as np
import pandas as pd
import pytest

from app.engines import plain_language as pl


# ══════════════════════════════════════════════════════════
#  THE VOCABULARY STAYS OUT OF THE PLAIN READING
# ══════════════════════════════════════════════════════════

JARGON = ("vif", "variance inflation", "multicollinear", "kurtosis",
          "kruskal", "shapiro", "anderson", "heteroscedast", "eta-squared",
          "p-value", "non-parametric", "log-transform", "stationar",
          "entropy", "quartile", "decile", "adf")


def _assert_plain(text: str):
    lowered = text.lower()
    found = [term for term in JARGON if term in lowered]
    assert not found, "plain wording still contains {}: {!r}".format(
        found, text[:160])


def test_the_vif_sentence_never_says_vif():
    """"1 feature(s) have high VIF (multicollinearity): revenue" is exact
    and tells a business reader nothing about why to care."""
    text = pl.vif_plain("revenue", 14.0, "High")
    _assert_plain(text)
    assert "repeats information" in text
    assert "93%" in text, "the figure is the reason to believe the sentence"


def test_the_skew_sentence_names_the_consequence():
    text = pl.skew_plain("revenue", 2.4, mean=523.0, median=404.0)
    _assert_plain(text)
    assert "average" in text and "middle value" in text


def test_the_normality_sentence_says_what_follows():
    _assert_plain(pl.normality_plain("revenue", False))
    _assert_plain(pl.normality_plain("units", True))


def test_the_kurtosis_sentence_never_says_kurtosis():
    _assert_plain(pl.kurtosis_plain("revenue", 3.2))
    _assert_plain(pl.kurtosis_plain("units", -1.4))


def test_the_entropy_sentence_talks_about_balance():
    concentrated = pl.entropy_plain("region", 0.4, 5, "North", 82.0)
    _assert_plain(concentrated)
    assert "North" in concentrated and "82%" in concentrated
    _assert_plain(pl.entropy_plain("region", 2.3, 5))


def test_the_trend_sentence_never_says_stationary():
    _assert_plain(pl.trend_plain("revenue", "upward", False))
    _assert_plain(pl.trend_plain("revenue", "flat", True))


def test_the_group_sentence_never_names_the_test():
    text = pl.group_difference_plain("revenue", "region", 4, True,
                                     "large", best="North", worst="East")
    _assert_plain(text)
    assert "North" in text and "East" in text


# ══════════════════════════════════════════════════════════
#  IT MUST NOT OVERCLAIM
# ══════════════════════════════════════════════════════════

def test_a_correlation_is_never_described_as_a_cause():
    text = pl.correlation_plain("revenue", "profit", 0.96, True, 40_000)
    assert "not proof that either one causes the other" in text
    _assert_plain(text)


def test_an_insignificant_correlation_says_so():
    text = pl.correlation_plain("revenue", "shoe_size", 0.02, False)
    assert "no dependable relationship" in text


def test_an_insignificant_group_difference_says_so():
    text = pl.group_difference_plain("revenue", "region", 3, False)
    assert "no bigger than ordinary variation" in text


def test_outliers_are_never_described_as_errors():
    text = pl.outliers_plain("revenue", 8.3, 120)
    assert "kept, not deleted" in text
    assert "as often a real event" in text


def test_the_direction_of_a_group_gap_is_not_assumed():
    """Which end is good depends on the metric, and the name may not say."""
    text = pl.group_difference_plain("ship_days", "region", 4, True, "large",
                                     best="North", worst="East")
    assert "depends on what" in text


# ══════════════════════════════════════════════════════════
#  BOTH READINGS TRAVEL TOGETHER
# ══════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def report():
    rng = np.random.default_rng(0)
    n = 3000
    revenue = rng.lognormal(6, 0.7, n).round(2)
    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "order_date": pd.date_range("2024-01-01", periods=n, freq="3h"),
        "region": rng.choice(["North", "South", "East"], n),
        "revenue": revenue,
        "profit": (revenue * 0.3 + rng.normal(0, 4, n)).round(2),
        "units": rng.integers(1, 40, n),
    })
    from app.engines.eda_engine import run_eda
    return run_eda(df)


def test_the_technical_findings_are_untouched(report):
    """Nothing here softens a computation. The statistical list is the
    one an analyst checks the work against and it keeps its wording."""
    joined = " ".join(report.key_findings).lower()
    assert report.key_findings
    assert any(term in joined for term in ("vif", "skew", "normal", "r=")), \
        "the technical findings lost their statistics"


def test_a_plain_reading_ships_beside_it(report):
    assert report.plain_findings, "no plain findings were produced"
    for finding in report.plain_findings:
        _assert_plain(finding)


def test_neither_list_is_a_subset_of_the_other(report):
    """They are two readings, not one truncated. If they were identical
    the toggle in the UI would be a lie."""
    assert set(report.plain_findings) != set(report.key_findings)


def test_every_univariate_result_carries_both(report):
    described = [r for r in report.univariate.values() if r.interpretation]
    assert described
    for result in described:
        assert result.plain, "{} has no plain reading".format(result.column)
        _assert_plain(result.plain)


def test_every_vif_row_carries_both(report):
    assert report.multicollinearity
    for row in report.multicollinearity:
        assert row.interpretation
        assert row.plain
        _assert_plain(row.plain)


def test_every_correlation_carries_both(report):
    for row in report.correlations:
        assert row.plain, "{} vs {} has no plain reading".format(
            row.col_a, row.col_b)
        _assert_plain(row.plain)


def test_identifier_columns_are_explained_not_just_dropped(report):
    """A reader who sent a column and does not see it in the analysis
    deserves to be told why."""
    if not report.identifier_cols:
        pytest.skip("no identifier columns in this fixture")
    joined = " ".join(report.plain_findings)
    assert "reference number" in joined


# ══════════════════════════════════════════════════════════
#  THE REPORT READS THE FIELDS IT THINKS IT IS READING
# ══════════════════════════════════════════════════════════

def test_the_report_reads_real_attribute_names():
    """`getattr(cs, "skew", None)` returned None on every column, because
    the field is `skewness`. Nothing raised — the plain sentence was
    simply absent from every report, silently.

    Any name the statistics section reaches for with a default must exist
    on the object it is reaching into.
    """
    import ast
    import dataclasses
    import inspect

    from app.engines.stats_engine import ColumnStats
    from app.engines.pdf import data_sections

    known = {f.name for f in dataclasses.fields(ColumnStats)}
    source = inspect.getsource(data_sections._stats_section)
    tree = ast.parse(source.lstrip())

    reached_for = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "cs"
                and isinstance(node.args[1], ast.Constant)):
            reached_for.add(node.args[1].value)

    assert reached_for, "no attribute reads found — has the section moved?"
    unknown = sorted(reached_for - known)
    assert not unknown, (
        "the statistics section reads {} off ColumnStats, which has no such "
        "field; the default hides it".format(unknown))


def test_a_constant_column_is_not_described_as_a_distribution(tmp_path):
    """"Employee Count: Non-normal | approximately symmetric | Outliers: 0"
    is three statements about a column that holds the number 1."""
    import io

    import pypdf

    from app.engines.pdf_builder import build_pdf
    from app.engines.stats_engine import analyze

    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "headcount": [1] * n,                       # constant
        "salary": rng.normal(50_000, 9_000, n).round(0),
        "team": rng.choice(["A", "B"], n),
    })
    pdf = build_pdf(
        df=df, stats_report=analyze(df),
        config={"title": "Constant Check", "client_name": "Test",
                "subtitle": "", "confidential": False, "theme_name": "",
                "logo_path": None, "prepared_by": "",
                "source_table": "src"},
        domain="general")
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    pages = [(p.extract_text() or "") for p in reader.pages]
    summary = next((t for t in pages if "Distribution Summary" in t), "")
    assert summary, "the distribution summary is missing"
    assert "Headcount" not in summary, (
        "a column of identical values was given a distribution")
    assert "Salary" in summary, "the real column was dropped too"
