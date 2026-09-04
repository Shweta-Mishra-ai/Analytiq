"""The quick-read insight generator must only claim what the data shows.

Two of these tests pin defects that reached the API: a trend invented
from row order, and an employee ID used as a business dimension.
"""
import numpy as np
import pandas as pd

from app.engines.insight_engine import generate_insights


def _titles(df):
    return [i["title"] for i in generate_insights(df)]


def test_no_trend_is_claimed_without_a_date_column():
    """It reported "'revenue' has increased over time — +59.9%" on a file
    with no date column at all, by comparing the first half of the rows
    to the second half. Row order is not time."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "revenue": np.r_[rng.normal(100, 5, 500), rng.normal(160, 5, 500)],
        "team": ["a"] * 1000})
    assert not any("over time" in t for t in _titles(df))


def test_the_same_rows_shuffled_give_the_same_answer():
    """A finding that appears or vanishes when a file is sorted
    differently is a fact about the file, not about the business."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "revenue": np.r_[rng.normal(100, 5, 500), rng.normal(160, 5, 500)],
        "team": ["a"] * 1000})
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    assert _titles(df) == _titles(shuffled)


def test_a_real_trend_is_found_even_when_the_rows_are_out_of_order():
    """With a date column the generator sorts by it, so the trend is
    found regardless of how the file happens to be ordered."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "day": pd.date_range("2024-01-01", periods=1000, freq="D"),
        "revenue": np.r_[rng.normal(100, 5, 500), rng.normal(160, 5, 500)],
        "region": rng.choice(["North", "South"], 1000)})
    titles = _titles(df.sample(frac=1, random_state=2).reset_index(drop=True))
    assert any("has risen over time" in t for t in titles)


def test_an_identifier_is_never_used_as_a_business_dimension():
    """On an HR extract this produced "'RM0139' dominates Age — accounts
    for 0.1% of total": one employee, used as a segment."""
    rng = np.random.default_rng(2)
    n = 500
    df = pd.DataFrame({
        "EmpID": [f"RM{i:04d}" for i in range(n)],
        "Age": rng.integers(20, 60, n),
        "Department": rng.choice(["Sales", "R&D"], n)})
    assert not any("RM0" in t for t in _titles(df))


def test_an_identifier_is_never_used_as_a_measure():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({
        "employee_number": np.arange(1, n + 1),
        "salary": rng.normal(50_000, 8_000, n),
        "office": rng.choice(["London", "Leeds"], n)})
    joined = " ".join(i["title"] + i["body"] for i in generate_insights(df))
    assert "employee_number" not in joined


def test_a_difference_too_small_to_act_on_is_not_reported():
    """It reported "a gap of 0% across 3 'segment' groups" — true, and
    not a finding."""
    rng = np.random.default_rng(4)
    n = 2000
    df = pd.DataFrame({"segment": rng.choice(["A", "B", "C"], n),
                       "value": rng.normal(100, 5, n)})
    assert not any("highest average" in t for t in _titles(df))


def test_a_real_difference_between_groups_is_reported():
    rng = np.random.default_rng(5)
    n = 3000
    seg = rng.choice(["Enterprise", "Mid-Market", "SMB"], n, p=[.2, .3, .5])
    spend = np.where(seg == "Enterprise", rng.normal(900, 80, n),
                     np.where(seg == "Mid-Market", rng.normal(400, 60, n),
                              rng.normal(120, 30, n)))
    df = pd.DataFrame({"segment": seg, "spend": spend})
    found = [i for i in generate_insights(df)
             if "highest average" in i["title"]]
    assert found, "a 650% gap between segments went unreported"
    assert "Enterprise" in found[0]["title"]


def test_correlation_is_described_as_association_not_cause():
    rng = np.random.default_rng(6)
    n = 800
    a = rng.normal(0, 1, n)
    df = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(0, 0.2, n)})
    body = " ".join(i["body"] for i in generate_insights(df))
    assert "not cause" in body


def test_an_empty_result_is_allowed_rather_than_padded():
    """Nothing but the shape of the data is a legitimate answer for a
    file that holds no relationship worth reporting."""
    df = pd.DataFrame({"only": [1] * 50})
    out = generate_insights(df)
    assert len(out) == 1 and "rows and" in out[0]["title"]


# ══════════════════════════════════════════════════════════
#  THE ANDERSON-DARLING CALL MUST SURVIVE THE NEXT SCIPY
# ══════════════════════════════════════════════════════════

def test_the_normality_statistic_does_not_warn_on_every_column():
    """SciPy 1.17 warns on every `anderson` call that leaves `method`
    unset, and from 1.19 drops `critical_values` — the attribute this
    code read. Left alone it would have raised into a bare `except`, and
    the statistic would have silently stopped appearing."""
    import warnings

    from app.engines.eda.univariate import analyze_univariate

    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = analyze_univariate(pd.Series(rng.normal(size=2000), name="x"))
    assert result.anderson_stat is not None
    assert result.anderson_p is not None, (
        "the p-value is the number a reader can use; the critical value "
        "it replaced was written and never read")


# ══════════════════════════════════════════════════════════
#  WHAT COUNTS AS A PREDICTABLE OUTCOME
# ══════════════════════════════════════════════════════════

def _target(**cols):
    from app.engines.predictive import find_binary_target
    rng = np.random.default_rng(0)
    n = 400
    frame = {k: rng.choice(v, n) if isinstance(v, list) else v
             for k, v in cols.items()}
    frame["measure"] = rng.normal(size=n)
    return find_binary_target(pd.DataFrame(frame))


def test_outcome_columns_outside_the_hr_vocabulary_are_found():
    """The predictive module carried its own short list — attrition,
    left, churn, exited, resigned, terminated, is_fraud, default — so a
    sales file with a 0/1 `returned` column was told it had "no binary
    outcome column detected"."""
    for name in ("returned", "converted", "cancelled", "renewed",
                 "approved", "readmitted", "subscribed"):
        assert _target(**{name: [0, 1]}) == name, name


def test_the_hr_vocabulary_still_works():
    assert _target(Attrition=["Yes", "No"]) == "Attrition"
    assert _target(churn=[0, 1]) == "churn"


def test_a_word_is_matched_whole_not_as_a_substring():
    """`leftover_stock` is not a column about who left."""
    assert _target(leftover_stock=[0, 1]) is None


def test_a_column_that_is_not_binary_is_not_offered():
    assert _target(returned=[0, 1, 2, 3]) is None


def test_a_column_that_names_nothing_is_not_offered():
    assert _target(some_flag=[0, 1]) is None


# ══════════════════════════════════════════════════════════
#  "NOT TRAINED YET" IS AN ANSWER, NOT AN ERROR
# ══════════════════════════════════════════════════════════

def test_asking_for_an_untrained_model_is_not_an_error(client, hr_csv_bytes):
    """The ML page asks "is there a model for this dataset?" on every
    visit, before the user has trained anything. Answering 404 logged an
    error in every user's console on a page working exactly as designed.
    """
    ds = client.post("/api/datasets/upload",
                     files={"file": ("hr.csv", hr_csv_bytes, "text/csv")}
                     ).json()["meta"]["dataset_id"]

    resp = client.get(f"/api/ml/{ds}/report")
    assert resp.status_code == 200
    body = resp.json()
    assert body["report"] is None
    assert "trained" in body["reason"].lower()


def test_asking_for_a_prediction_without_a_model_still_fails(
        client, hr_csv_bytes):
    """`/what-if` is different: the caller asked for a prediction and
    there is no model to make it with. That request really did fail."""
    ds = client.post("/api/datasets/upload",
                     files={"file": ("hr.csv", hr_csv_bytes, "text/csv")}
                     ).json()["meta"]["dataset_id"]

    resp = client.post(f"/api/ml/{ds}/what-if",
                       json={"target": "attrition", "inputs": {}})
    assert resp.status_code == 404


def test_a_missing_dataset_is_still_a_404(client):
    """Softening "no model" must not soften "no dataset"."""
    assert client.get("/api/ml/nosuchdataset/report").status_code == 404
