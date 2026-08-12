"""
Regression test for a real data-loss bug found in production use:
a mostly-numeric column with a few legitimate non-numeric values (e.g.
"Pending", "N/A") was silently having those values destroyed into NaN
by the >80%-numeric auto-conversion heuristic — both in the normal
upload path (_smart_dtype_inference) and in AI table extraction from
images/video (table_json_to_df). Fixed: a cell that fails to convert
now keeps its original value instead of becoming a blank.
Run:  python -m tests.data_integrity_test   (from backend/)
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATA_DIR", "/tmp/dataforge-data-integrity-test")
os.environ["APP_ADMIN_KEY"] = ""
os.environ["APP_PASSWORD"] = ""

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.services.table_extractor import table_json_to_df  # noqa: E402

client = TestClient(app)
FAILURES = []


def check(name, condition):
    print(f"{'✅' if condition else '❌'} {name}")
    if not condition:
        FAILURES.append(name)


def main():
    # ── 1. Normal CSV upload path (_smart_dtype_inference) ──
    csv_text = "Order,Discount\n" + "".join(
        f"{i+1},{(i+1)*10}\n" for i in range(9)) + "10,Pending\n"
    r = client.post("/api/datasets/upload",
                     files={"file": ("orders.csv", io.BytesIO(csv_text.encode()), "text/csv")})
    check("CSV upload succeeds", r.status_code == 200)
    ds_id = r.json()["meta"]["dataset_id"]
    records = client.get(f"/api/datasets/{ds_id}/preview").json()["records"]
    check("90%-numeric column keeps 'Pending' instead of nulling it",
          records[-1]["Discount"] == "Pending")
    check("the numeric values in that same column still converted properly",
          records[0]["Discount"] == 10.0)

    # ── 2. AI image/video table extraction (table_json_to_df) ──
    payload = {
        "found": True, "columns": ["Region", "Revenue"],
        "rows": [["R1", "12500"], ["R2", "8200"], ["R3", "9800"], ["R4", "1000"],
                 ["R5", "4300"], ["R6", "7700"], ["R7", "2200"], ["R8", "5600"],
                 ["R9", "3300"], ["R10", "Pending"]],
    }
    df, warnings = table_json_to_df(payload)
    check("extraction: 'Pending' survives instead of becoming NaN",
          df.loc[df["Region"] == "R10", "Revenue"].iloc[0] == "Pending")
    check("extraction: numeric rows still converted",
          df.loc[df["Region"] == "R1", "Revenue"].iloc[0] == 12500.0)
    check("extraction: warns about the preserved non-numeric value",
          any("kept as text" in w for w in warnings))

    # malformed / oversized rows: counted and surfaced, not silently vanished
    payload2 = {
        "found": True, "columns": ["A", "B"],
        "rows": [["x", "1"], ["y", "2"], ["z", "3", "extra-cell"], "not-a-row"],
    }
    df2, warnings2 = table_json_to_df(payload2)
    check("malformed row is dropped from the frame but not silently — 3 rows remain",
          len(df2) == 3)
    check("extraction warns about the skipped malformed row",
          any("unexpected format" in w for w in warnings2))
    check("extraction warns about the truncated oversized row",
          any("truncated" in w for w in warnings2))

    # accounting-negative format
    payload3 = {
        "found": True, "columns": ["Item", "Amount"],
        "rows": [["A", "1000"], ["B", "2000"], ["C", "3000"], ["D", "4000"],
                 ["E", "(500)"]],
    }
    df3, _ = table_json_to_df(payload3)
    check("accounting-style '(500)' parses as -500",
          df3.loc[df3["Item"] == "E", "Amount"].iloc[0] == -500)

    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("🎉 DATA INTEGRITY REGRESSION SUITE PASSES")


if __name__ == "__main__":
    main()
