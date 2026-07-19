"""
End-to-end smoke test against the FastAPI app (no server needed).
Run:  python -m tests.smoke_test   (from backend/)
"""
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATA_DIR", "/tmp/dataforge-test-data")

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
FAILURES = []


def check(name, resp, expect=200):
    ok = resp.status_code == expect
    print(f"{'✅' if ok else '❌'} {name}: {resp.status_code}")
    if not ok:
        print("   ", resp.text[:300])
        FAILURES.append(name)
    return resp


def make_csv() -> bytes:
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
    # add some dirt
    df.loc[5:20, "salary"] = np.nan
    df.loc[3, "age"] = 250
    return df.to_csv(index=False).encode()


def main():
    r = check("health", client.get("/api/health"))

    r = check("upload", client.post(
        "/api/datasets/upload",
        files={"file": ("hr_test.csv", io.BytesIO(make_csv()), "text/csv")}))
    ds = r.json()["meta"]["dataset_id"]
    print(f"   dataset_id={ds}")

    check("list", client.get("/api/datasets"))
    check("preview", client.get(f"/api/datasets/{ds}/preview?rows=5"))
    check("profile", client.get(f"/api/datasets/{ds}/profile"))
    check("clean", client.post(f"/api/datasets/{ds}/clean"))

    check("stats", client.get(f"/api/analytics/{ds}/stats"))
    check("eda", client.get(f"/api/analytics/{ds}/eda"))
    check("bi", client.get(f"/api/analytics/{ds}/bi"))
    check("story", client.get(f"/api/analytics/{ds}/story"))
    check("insights", client.get(f"/api/analytics/{ds}/insights"))
    check("domain", client.get(f"/api/analytics/{ds}/domain"))

    check("fields", client.get(f"/api/charts/{ds}/fields"))
    check("kpis", client.post(f"/api/charts/{ds}/kpis", json={"filters": []}))
    check("kpis+filter", client.post(f"/api/charts/{ds}/kpis", json={
        "filters": [{"column": "department", "op": "eq", "value": "Sales"}]}))
    check("recommend", client.post(f"/api/charts/{ds}/recommend", json={"filters": []}))
    check("build bar", client.post(f"/api/charts/{ds}/build", json={
        "type": "bar", "x": "department", "y": "salary", "agg": "mean"}))
    check("build line(date)", client.post(f"/api/charts/{ds}/build", json={
        "type": "line", "x": "hire_date", "y": "salary", "agg": "mean"}))
    check("build hist", client.post(f"/api/charts/{ds}/build", json={
        "type": "histogram", "x": "age"}))
    check("build heatmap", client.post(f"/api/charts/{ds}/build", json={
        "type": "heatmap"}))
    check("build pie+filter", client.post(f"/api/charts/{ds}/build", json={
        "type": "pie", "x": "department", "y": "salary",
        "filters": [{"column": "age", "op": "between", "value": [30, 50]}]}))
    check("build table", client.post(f"/api/charts/{ds}/build", json={
        "type": "table", "x": "department", "y": "salary", "agg": "count"}))

    check("ml targets", client.get(f"/api/ml/{ds}/targets"))
    check("ml train", client.post(f"/api/ml/{ds}/train",
                                  json={"target": "attrition"}))
    check("ml report", client.get(f"/api/ml/{ds}/report"))

    check("export csv", client.get(f"/api/reports/{ds}/csv"))
    check("export excel", client.get(f"/api/reports/{ds}/excel"))
    r = check("pdf report", client.post(f"/api/reports/{ds}/pdf", json={
        "title": "Smoke Test Report", "include_stats": True, "include_bi": True}))
    if r.status_code == 200:
        print(f"   PDF size: {len(r.content)/1024:.0f} KB")

    # ── RAG (offline mode: local embedder, no LLM keys) ──
    r = check("rag create kb", client.post("/api/rag/kb",
                                           json={"name": "Test KB"}))
    kb = r.json()["kb_id"]
    check("rag upload txt", client.post(
        f"/api/rag/kb/{kb}/files",
        files={"file": ("notes.txt", io.BytesIO(
            b"Q3 revenue grew 18% to $4.2M. Churn rose to 6.1% in September. "
            b"The main driver was pricing changes in the Enterprise tier. "
            b"Recommendation: revert Enterprise pricing and add annual plans."),
            "text/plain")}))
    check("rag upload csv", client.post(
        f"/api/rag/kb/{kb}/files",
        files={"file": ("sales.csv", io.BytesIO(make_csv()), "text/csv")}))

    from reportlab.pdfgen import canvas as rl_canvas
    pdf_buf = io.BytesIO()
    c = rl_canvas.Canvas(pdf_buf)
    c.drawString(100, 750, "Annual report: headcount reached 120 employees.")
    c.drawString(100, 730, "Attrition was 14 percent, above the 10% target.")
    c.save()
    check("rag upload pdf", client.post(
        f"/api/rag/kb/{kb}/files",
        files={"file": ("report.pdf", io.BytesIO(pdf_buf.getvalue()),
                        "application/pdf")}))

    check("rag kb detail", client.get(f"/api/rag/kb/{kb}"))
    has_llm = bool(os.environ.get("GEMINI_API_KEY")
                   or os.environ.get("GROQ_API_KEY"))
    expect = 200 if has_llm else 503
    check(f"rag query (expect {expect})", client.post(
        f"/api/rag/kb/{kb}/query", json={"question": "How did revenue do?"}),
        expect=expect)
    check(f"rag report (expect {expect})", client.post(
        f"/api/rag/kb/{kb}/report", json={"title": "Test Report"}),
        expect=expect)
    check("rag delete kb", client.delete(f"/api/rag/kb/{kb}"))

    check("delete", client.delete(f"/api/datasets/{ds}"))

    print("\n" + ("💥 FAILURES: " + ", ".join(FAILURES) if FAILURES
                  else "🎉 ALL ENDPOINTS PASS"))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
