"""
Defects found by driving every page of the app in a browser with sales, HR
and deliberately messy files.
"""
from __future__ import annotations

import io
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from conftest import _make_hr_df  # noqa: E402


def _upload(client, df, name="data.csv"):
    r = client.post("/api/datasets/upload",
                    files={"file": (name, io.BytesIO(df.to_csv(index=False).encode()),
                                    "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["meta"]["dataset_id"]


def test_attrition_cost_is_a_computed_range_from_the_salary_column():
    from app.engines.domains.hr import _run_attrition as compute_attrition
    df = _make_hr_df()
    a = compute_attrition(df)
    leavers = df[df.attrition == "Yes"]
    med = leavers.salary.median()
    assert a.cost_low == 0.5 * med * len(leavers)
    assert a.cost_high == 2.0 * med * len(leavers)
    assert a.cost_estimate.startswith("At Gallup's estimate")


def test_attrition_cost_stays_a_sentence_without_salaries():
    from app.engines.domains.hr import _run_attrition as compute_attrition
    a = compute_attrition(_make_hr_df().drop(columns=["salary"]))
    assert a.cost_low is None and a.cost_estimate.startswith("Gallup estimates")


def test_survival_defaults_only_to_a_column_that_reads_as_a_duration(client):
    hr = client.get(f"/api/analytics/{_upload(client, _make_hr_df())}/survival/fields").json()
    assert hr["default_duration"] == "tenure_years" and hr["default_event"] == "attrition"
    assert hr["duration_columns"][0] == "tenure_years"
    rng = np.random.default_rng(1)
    shop = pd.DataFrame({"amount": rng.gamma(2, 100, 80), "returned": rng.choice(["Yes", "No"], 80)})
    f = client.get(f"/api/analytics/{_upload(client, shop)}/survival/fields").json()
    assert f["default_duration"] is None                 # money is not a duration


def test_rfm_finds_customer_columns_however_they_are_written():
    from app.engines.rfm_engine import detect_rfm_columns
    for name in ("Customer ID", "customer-id", "Customer", "Client"):
        df = pd.DataFrame({name: ["a", "b", "a", "b"],
                           "Order Date": pd.to_datetime(["2024-01-01"] * 4),
                           "Amount": [1.0, 2, 3, 4]})
        cols = detect_rfm_columns(df)
        assert cols and cols.customer_id == name and cols.monetary_col == "Amount", name


def _sales(n=1200, seed=2):
    rng = np.random.default_rng(seed)
    cat = rng.choice(["Electronics", "Furniture", "Office Supplies"], n)
    base = {"Electronics": 900, "Furniture": 600, "Office Supplies": 120}
    return pd.DataFrame({"category": cat, "region": rng.choice(list("NSEW"), n),
                         "sales": [rng.gamma(2, base[c] / 2) for c in cat],
                         "order_date": pd.date_range("2023-01-01", periods=n, freq="D")})


def test_bi_does_not_call_a_product_price_gap_outperformance():
    from app.engines.bi_engine import run_bi
    text = " ".join(run_bi(_sales()).key_insights)
    assert "outperforms" not in text and "best cohort" not in text
    assert "mostly reflects what each sells for" in text


def test_the_insight_card_for_product_groups_reads_as_price_mix():
    from app.engines.story_engine import generate_story
    titles = [getattr(i, "title", "") for i in generate_story(_sales()).top_insights]
    assert not any("Segments Differ Significantly" in t and "category" in t for t in titles)
    assert any("(price mix)" in t for t in titles), titles
