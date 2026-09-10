"""
What the dashboard opens on.

The page used to choose its own tiles in the browser, from the order the
columns happen to sit in the file — the only information a browser has.
Measured against the server's column-role logic on thirteen domains,
that opened:

    finance     "budget by account"   a budget is ASSIGNED per account,
                                      so the bars restate the file's
                                      own construction
    education   "cohort by module"    an average of a year number
    healthcare  "age by department"   rather than length of stay
    ecommerce   "unit_price by …"     rather than revenue

and it never once showed the outcome — the returned/churned/attrition
flag the rest of the app leads with.

These tests cover the recommender that replaced it, and the three
defects found by rendering its output in a real browser.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from app.engines.chart_engine import (  # noqa: E402
    _is_chartable, _is_identifier, recommend_tiles,
)


@pytest.fixture()
def hr():
    r = np.random.default_rng(11)
    n = 2000
    overtime = r.choice(["Yes", "No"], n, p=[.3, .7])
    dept = r.choice(["Sales", "Support", "Engineering"], n)
    return pd.DataFrame({
        "EmployeeID": range(1, n + 1),
        "Department": dept,
        "OverTime": overtime,
        "MonthlyIncome": r.normal(6500, 1500, n).round(0),
        "YearsAtCompany": r.integers(0, 20, n),
        "Attrition": np.where(
            r.random(n) < np.where(overtime == "Yes", .38, .08), "Yes", "No"),
    })


@pytest.fixture()
def sales():
    r = np.random.default_rng(3)
    n = 2000
    return pd.DataFrame({
        "opportunity_id": range(1, n + 1),
        "sales_rep": r.choice(["Rep 01", "Rep 02", "Rep 03"], n),
        "region": r.choice(["North", "South"], n),
        "deal_size": r.lognormal(9.4, .7, n).round(0),
        "quota": 250000,                      # constant: an annual target
        "won": (r.random(n) < .25).astype(int),
    })


# ── the column that was thrown away ───────────────────────

def test_a_continuous_measure_is_not_an_identifier(sales):
    """`deal_size` is nearly all-distinct, as any money column is. The
    old rule was uniqueness alone, so the most important column in a
    sales file was classified as an identifier and removed from every
    chart and every correlation."""
    assert _is_identifier("deal_size", sales["deal_size"], len(sales)) is False


@pytest.mark.parametrize("values", [
    lambda r, n: pd.Series(range(1, n + 1)),              # 1..n
    lambda r, n: pd.Series(range(100001, 100001 + n)),    # offset block
    lambda r, n: pd.Series(r.permutation(np.arange(5000, 5000 + n))),
])
def test_an_assigned_key_is_still_an_identifier(values):
    r = np.random.default_rng(5)
    n = 2000
    assert _is_identifier("acct_no", values(r, n), n) is True


def test_a_measure_reaches_the_dashboard(sales):
    titles = " ".join(t["title"] for t in recommend_tiles(sales, "sales"))
    assert "Deal Size" in titles, titles


# ── metrics that cannot be charted ────────────────────────

def test_a_constant_column_is_never_the_headline(sales):
    """`quota` is 250,000 on every row: a flat bar chart and a
    single-spike histogram. The domain's preferred-metric list returned
    it early, before any usability check ran."""
    titles = " ".join(t["title"] for t in recommend_tiles(sales, "sales"))
    assert "Quota" not in titles, titles


def test_a_binary_flag_is_never_the_headline():
    """A logistics dashboard led with `on_time`, so "Distribution of On
    Time" was a two-bar histogram."""
    r = np.random.default_rng(7)
    n = 2000
    lane = r.choice(["West", "Midwest"], n)
    df = pd.DataFrame({
        "shipment_id": range(n),
        "lane": lane,
        "carrier": r.choice(["A", "B"], n),
        "freight_cost": r.gamma(4, 220, n).round(2),
        "on_time": (r.random(n) > np.where(lane == "West", .34, .09))
                   .astype(int),
    })
    tiles = recommend_tiles(df, "logistics")
    assert "Distribution of Freight Cost" in [t["title"] for t in tiles]
    assert not any(t["title"] == "Distribution of On Time" for t in tiles)


@pytest.mark.parametrize("series,ok", [
    (pd.Series([1.0, 2.0, 3.0] * 40), True),
    (pd.Series([0, 1] * 60), False),          # a flag
    (pd.Series([7] * 120), False),            # a constant
    (pd.Series([1.0, 2.0]), False),           # too few rows
])
def test_chartable_needs_real_variation(series, ok):
    df = pd.DataFrame({"x": series})
    assert _is_chartable("x", df) is ok


# ── the outcome the dashboard never showed ────────────────

def test_the_outcome_gets_a_tile(hr):
    titles = [t["title"] for t in recommend_tiles(hr, "hr")]
    assert any("Attrition rate" in t for t in titles), titles


def test_the_outcome_is_not_charted_against_itself(hr):
    """When the outcome is also the leading dimension, its rate by
    itself is 0% and 100%."""
    for tile in recommend_tiles(hr, "hr"):
        if "rate" in tile["title"]:
            assert tile["x"] != tile["y"]


def test_a_file_with_no_outcome_simply_has_no_rate_tile():
    r = np.random.default_rng(13)
    n = 800
    df = pd.DataFrame({
        "account": r.choice(["Payroll", "Cloud"], n),
        "revenue": r.normal(50000, 12000, n).round(0),
        "actual": r.normal(48000, 11000, n).round(0),
    })
    assert not any("rate" in t["title"]
                   for t in recommend_tiles(df, "finance"))


# ── how the tiles are aggregated ──────────────────────────

def test_aggregation_is_left_to_the_shared_rule(hr):
    """Hardcoding "sum" charted the TOTAL monthly income of each
    attrition group — 15M against 3M, which is a headcount chart wearing
    a salary label. `_agg_for_metric` already knows income is averaged;
    the tiles have to ask it rather than decide for themselves."""
    for tile in recommend_tiles(hr, "hr"):
        if tile["type"] in ("bar", "line") and "rate" not in tile["title"]:
            assert tile["agg"] == "auto", tile


def test_income_is_averaged_not_totalled():
    from app.engines.charts.style import _agg_for_metric
    assert _agg_for_metric("MonthlyIncome")[0] == "mean"


# ── nothing chartable at all ──────────────────────────────

def test_a_frame_with_no_measure_returns_no_tiles():
    df = pd.DataFrame({"note": ["a", "b", "c"] * 50,
                       "row_id": range(150)})
    assert recommend_tiles(df, "general") == []


def test_it_never_raises_on_an_odd_frame():
    for df in (pd.DataFrame(), pd.DataFrame({"a": [1]}),
               pd.DataFrame({"a": [None] * 50})):
        assert isinstance(recommend_tiles(df, "general"), list)


# ══════════════════════════════════════════════════════════
#  THE ENDPOINT, AND WHAT IT DOES WITH A YES/NO COLUMN
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def client_with_hr(client, hr):
    csv = hr.to_csv(index=False).encode()
    ds = client.post("/api/datasets/upload",
                     files={"file": ("hr.csv", csv, "text/csv")}
                     ).json()["meta"]["dataset_id"]
    return client, ds


def test_the_endpoint_returns_specs_the_builder_accepts(client_with_hr):
    client, ds = client_with_hr
    tiles = client.post(f"/api/charts/{ds}/recommend-tiles",
                        json={"filters": []}).json()["tiles"]
    assert tiles
    for tile in tiles:
        body = {"type": tile["type"], "x": tile.get("x"), "y": tile.get("y"),
                "agg": tile.get("agg", "auto"), "title": "", "filters": []}
        r = client.post(f"/api/charts/{ds}/build", json=body)
        assert r.status_code == 200, (tile, r.text)


def test_a_yes_no_column_charts_as_a_rate(client_with_hr):
    """Pandas cannot average a string, so "attrition rate by department"
    — the most useful tile a dashboard can carry — failed outright with
    "dtype 'str' does not support operation 'mean'"."""
    client, ds = client_with_hr
    r = client.post(f"/api/charts/{ds}/build", json={
        "type": "bar", "x": "Department", "y": "Attrition",
        "agg": "mean", "title": "", "filters": []})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "rate (%)" in str(body), "the axis must say it is a percentage"


def _numbers(axis):
    """Plotly serialises numeric arrays as base64 float64 ("bdata")."""
    import array
    import base64
    if isinstance(axis, dict) and "bdata" in axis:
        buf = array.array("d")
        buf.frombytes(base64.b64decode(axis["bdata"]))
        return list(buf)
    return [float(v) for v in (axis or [])]


def test_the_rate_is_a_percentage_not_a_fraction(client_with_hr):
    client, ds = client_with_hr
    fig = client.post(f"/api/charts/{ds}/build", json={
        "type": "bar", "x": "Department", "y": "Attrition",
        "agg": "mean", "title": "", "filters": []}).json()["figure"]
    values = [v for trace in fig["data"] for v in _numbers(trace.get("y"))]
    assert values
    assert all(0 <= v <= 100 for v in values), values
    assert max(values) > 1.5, "0.19 rather than 19% — reads as a fraction"


def test_text_that_is_not_binary_is_refused_with_a_reason(client_with_hr):
    client, ds = client_with_hr
    r = client.post(f"/api/charts/{ds}/build", json={
        "type": "bar", "x": "Attrition", "y": "Department",
        "agg": "mean", "title": "", "filters": []})
    assert r.status_code == 422
    assert "cannot be averaged" in r.json()["detail"]


def test_a_column_from_another_dataset_is_a_422_not_a_500(client_with_hr):
    """Switching datasets left the previous file's tiles in state, and
    they fired against the new id — plotly raised deep inside
    make_figure and the client got a 500 carrying no usable message."""
    client, ds = client_with_hr
    r = client.post(f"/api/charts/{ds}/build", json={
        "type": "histogram", "x": "not_a_real_column", "y": None,
        "agg": "auto", "title": "", "filters": []})
    assert r.status_code == 422
    assert "not in this dataset" in r.json()["detail"]


def test_summing_a_flag_counts_it_rather_than_adding_percentages(
        client_with_hr):
    """Averaging 0/100 gives a rate. Summing percentages does not —
    three affirmative rows would read as 300%."""
    client, ds = client_with_hr
    fig = client.post(f"/api/charts/{ds}/build", json={
        "type": "bar", "x": "Department", "y": "Attrition",
        "agg": "sum", "title": "", "filters": []}).json()["figure"]
    values = [v for trace in fig["data"] for v in _numbers(trace.get("y"))]
    assert values
    # Counts of people, not percentages, and every one a whole number.
    assert all(float(v).is_integer() for v in values), values
    assert max(values) > 100, "a count of 3,000 employees, not a rate"


def test_a_missing_value_is_not_counted_as_a_no(client):
    """Filling the gaps with False counted every blank row against the
    rate. A rate is over the records that recorded it."""
    r = np.random.default_rng(21)
    n = 600
    dept = np.array(["A"] * 300 + ["B"] * 300)
    churn = np.where(np.arange(n) % 2 == 0, "Yes", "No").astype(object)
    churn[:200] = None                        # every gap sits in dept A
    df = pd.DataFrame({"dept": dept, "churned": churn})
    ds = client.post("/api/datasets/upload",
                     files={"file": ("c.csv", df.to_csv(index=False).encode(),
                                     "text/csv")}).json()["meta"]["dataset_id"]
    fig = client.post(f"/api/charts/{ds}/build", json={
        "type": "bar", "x": "dept", "y": "churned",
        "agg": "mean", "title": "", "filters": []}).json()["figure"]
    values = [v for trace in fig["data"] for v in _numbers(trace.get("y"))]
    # Both groups alternate Yes/No among the rows that HAVE a value, so
    # both sit near 50%. Counting the 200 blanks as "No" would drag A
    # down to roughly 17%.
    assert values and min(values) > 35, values
