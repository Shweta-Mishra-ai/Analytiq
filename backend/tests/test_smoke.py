"""
Every endpoint answers, on one ordinary dataset, in single-user open
mode. It is the cheapest test in the suite and the one that catches a
route that stopped existing.

It was a script that printed a tick per endpoint and exited 1 if any
failed — so the first failure told you nothing about the other forty.
As pytest, each endpoint fails on its own.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _csv() -> bytes:
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "employee_id": range(1, n + 1),
        "department": rng.choice(["Sales", "Engineering", "HR", "Marketing"], n),
        "salary": rng.normal(60000, 15000, n).round(0),
        "age": rng.integers(22, 60, n),
        "tenure_years": rng.integers(0, 20, n),
        "satisfaction": rng.uniform(1, 5, n).round(1),
        "attrition": rng.choice(["Yes", "No"], n, p=[0.2, 0.8]),
        "hire_date": pd.date_range("2015-01-01", periods=n, freq="3D"),
    })
    # Dirt on purpose: missing salaries and an impossible age. A smoke
    # test on spotless data does not exercise the paths that matter.
    df.loc[5:20, "salary"] = np.nan
    df.loc[3, "age"] = 250
    return df.to_csv(index=False).encode()


@pytest.fixture(scope="module")
def dataset(client):
    """One upload for the whole module, cleaned, then removed.

    Module-scoped because uploading and cleaning 500 rows for each of
    thirty endpoint checks is the whole runtime of this file.
    """
    res = client.post(
        "/api/datasets/upload",
        files={"file": ("hr_test.csv", io.BytesIO(_csv()), "text/csv")})
    assert res.status_code == 200, res.text
    ds = res.json()["meta"]["dataset_id"]

    assert client.post("/api/datasets/{}/clean".format(ds)).status_code == 200
    yield ds
    client.delete("/api/datasets/{}".format(ds))


def test_health_answers(client):
    assert client.get("/api/health").status_code == 200


def test_upload_and_clean_both_work(dataset):
    """The fixture does both; this names them so a failure in either
    reads as itself rather than as an error in thirty other tests."""
    assert dataset


@pytest.mark.parametrize("route", [
    "/api/datasets",
    "/api/datasets/{ds}/preview?rows=5",
    "/api/datasets/{ds}/profile",
    "/api/analytics/{ds}/stats",
    "/api/analytics/{ds}/eda",
    "/api/analytics/{ds}/bi",
    "/api/analytics/{ds}/story",
    "/api/analytics/{ds}/insights",
    "/api/analytics/{ds}/domain",
    "/api/charts/{ds}/fields",
    "/api/ml/{ds}/targets",
    "/api/reports/{ds}/csv",
    "/api/reports/{ds}/excel",
])
def test_every_read_route_answers(client, dataset, route):
    res = client.get(route.format(ds=dataset))
    assert res.status_code == 200, res.text[:300]


@pytest.mark.parametrize("body", [
    {"filters": []},
    {"filters": [{"column": "department", "op": "eq", "value": "Sales"}]},
])
def test_kpis_answer_filtered_and_unfiltered(client, dataset, body):
    res = client.post("/api/charts/{}/kpis".format(dataset), json=body)
    assert res.status_code == 200, res.text[:300]


def test_chart_recommendations_answer(client, dataset):
    res = client.post("/api/charts/{}/recommend".format(dataset),
                      json={"filters": []})
    assert res.status_code == 200, res.text[:300]


@pytest.mark.parametrize("spec", [
    {"type": "bar", "x": "department", "y": "salary", "agg": "mean"},
    {"type": "line", "x": "hire_date", "y": "salary", "agg": "mean"},
    {"type": "histogram", "x": "age"},
    {"type": "heatmap"},
    {"type": "pie", "x": "department", "y": "salary",
     "filters": [{"column": "age", "op": "between", "value": [30, 50]}]},
    {"type": "table", "x": "department", "y": "salary", "agg": "count"},
], ids=["bar", "line-on-a-date", "histogram", "heatmap", "pie-filtered",
        "table"])
def test_every_chart_type_builds(client, dataset, spec):
    res = client.post("/api/charts/{}/build".format(dataset), json=spec)
    assert res.status_code == 200, res.text[:300]


def test_a_model_trains_and_reports(client, dataset):
    assert client.post("/api/ml/{}/train".format(dataset),
                       json={"target": "attrition"}).status_code == 200
    assert client.get("/api/ml/{}/report".format(dataset)).status_code == 200


def test_the_pdf_report_builds_and_is_not_empty(client, dataset):
    res = client.post("/api/reports/{}/pdf".format(dataset),
                      json={"title": "Smoke Test Report",
                            "include_stats": True, "include_bi": True})
    assert res.status_code == 200, res.text[:300]
    assert res.content.startswith(b"%PDF"), "that is not a PDF"
    assert len(res.content) > 20_000, \
        "a report this small has pages with nothing on them"


# ══════════════════════════════════════════════════════════
#  The knowledge base, with no model configured
# ══════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def knowledge_base(client):
    res = client.post("/api/rag/kb", json={"name": "Test KB"})
    assert res.status_code == 200, res.text
    kb = res.json()["kb_id"]
    yield kb
    client.delete("/api/rag/kb/{}".format(kb))


def _pdf_bytes() -> bytes:
    from reportlab.pdfgen import canvas as rl_canvas
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf)
    c.drawString(100, 750, "Annual report: headcount reached 120 employees.")
    c.drawString(100, 730, "Attrition was 14 percent, above the 10% target.")
    c.save()
    return buf.getvalue()


@pytest.mark.parametrize("name,payload,mime", [
    ("notes.txt",
     b"Q3 revenue grew 18% to $4.2M. Churn rose to 6.1% in September. "
     b"The main driver was pricing changes in the Enterprise tier. "
     b"Recommendation: revert Enterprise pricing and add annual plans.",
     "text/plain"),
    ("sales.csv", None, "text/csv"),
    ("report.pdf", None, "application/pdf"),
])
def test_every_file_type_is_accepted(client, knowledge_base, name, payload,
                                     mime):
    body = payload
    if body is None:
        body = _csv() if name.endswith(".csv") else _pdf_bytes()

    res = client.post("/api/rag/kb/{}/files".format(knowledge_base),
                      files={"file": (name, io.BytesIO(body), mime)})
    assert res.status_code == 200, res.text[:300]


def test_the_knowledge_base_reports_what_is_in_it(client, knowledge_base):
    assert client.get("/api/rag/kb/{}".format(knowledge_base)).status_code == 200


def test_a_question_always_gets_an_answer(client, knowledge_base):
    """With a model it is written prose; without one it is the ranked
    passages that matched, which is a document search. The retrieval
    half runs locally and used to be thrown away with the generation
    half."""
    res = client.post("/api/rag/kb/{}/query".format(knowledge_base),
                      json={"question": "How did revenue do?"})
    assert res.status_code == 200, res.text[:300]


def test_a_report_refuses_rather_than_pretending(client, knowledge_base):
    """A report is a synthesis, not a lookup. There is no honest version
    of one without a model, so it declines instead of dressing up the
    passages it found."""
    has_model = bool(os.environ.get("GEMINI_API_KEY")
                     or os.environ.get("GROQ_API_KEY"))
    res = client.post("/api/rag/kb/{}/report".format(knowledge_base),
                      json={"title": "Test Report"})

    assert res.status_code == (200 if has_model else 503), res.text[:300]
