"""
End-to-end test for multi-tenant client isolation (no server needed).
Run:  python -m tests.multi_tenant_test   (from backend/)

Unlike smoke_test.py (which runs in single-user open dev mode), this
sets APP_ADMIN_KEY before the app is imported, so auth is enforced —
then verifies two separate client accounts can never see, read, or
delete each other's datasets or RAG knowledge bases.
"""
import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATA_DIR", "/tmp/dataforge-mt-test-data")
os.environ["APP_ADMIN_KEY"] = "test-admin-key-do-not-use-in-prod"

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
ADMIN = os.environ["APP_ADMIN_KEY"]
FAILURES = []


def check(name, condition):
    print(f"{'✅' if condition else '❌'} {name}")
    if not condition:
        FAILURES.append(name)
    return condition


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN}"}


def login(username, password):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def make_csv(seed_col: str) -> bytes:
    return pd.DataFrame({seed_col: [1, 2, 3], "value": [10, 20, 30]}).to_csv(index=False).encode()


def main():
    check("health.auth_required is True with admin key set",
          client.get("/api/health").json()["auth_required"] is True)
    check("unauthenticated /api/datasets -> 401",
          client.get("/api/datasets").status_code == 401)

    r = client.post("/api/admin/users", json={"username": "tenant_a", "password": "password_a1"},
                     headers=admin_headers())
    check("admin creates tenant_a", r.status_code == 200)
    r = client.post("/api/admin/users", json={"username": "tenant_b", "password": "password_b1"},
                     headers=admin_headers())
    check("admin creates tenant_b", r.status_code == 200)
    check("non-admin key cannot create users",
          client.post("/api/admin/users", json={"username": "x", "password": "password_x1"},
                      headers={"Authorization": "Bearer wrong-key"}).status_code == 401)

    a_headers = login("tenant_a", "password_a1")
    b_headers = login("tenant_b", "password_b1")
    check("wrong password rejected",
          client.post("/api/auth/login", json={"username": "tenant_a", "password": "nope"}).status_code == 401)

    r = client.post("/api/datasets/upload",
                     files={"file": ("a.csv", io.BytesIO(make_csv("a_col")), "text/csv")},
                     headers=a_headers)
    a_ds = r.json()["meta"]["dataset_id"]
    check("tenant_a uploads a dataset", r.status_code == 200)

    r = client.post("/api/datasets/upload",
                     files={"file": ("b.csv", io.BytesIO(make_csv("b_col")), "text/csv")},
                     headers=b_headers)
    b_ds = r.json()["meta"]["dataset_id"]
    check("tenant_b uploads a dataset", r.status_code == 200)

    a_list = [d["dataset_id"] for d in client.get("/api/datasets", headers=a_headers).json()["datasets"]]
    check("tenant_a's list has only their own dataset", a_list == [a_ds])
    b_list = [d["dataset_id"] for d in client.get("/api/datasets", headers=b_headers).json()["datasets"]]
    check("tenant_b's list has only their own dataset", b_list == [b_ds])

    # cross-tenant access attempts must all 404 (not 403 — never confirm existence)
    for label, resp in [
        ("get meta", client.get(f"/api/datasets/{a_ds}", headers=b_headers)),
        ("preview", client.get(f"/api/datasets/{a_ds}/preview", headers=b_headers)),
        ("profile", client.get(f"/api/datasets/{a_ds}/profile", headers=b_headers)),
        ("eda", client.get(f"/api/analytics/{a_ds}/eda", headers=b_headers)),
        ("kpis", client.post(f"/api/charts/{a_ds}/kpis", json={"filters": []}, headers=b_headers)),
        ("chat", client.post(f"/api/chat/{a_ds}", json={"message": "hi"}, headers=b_headers)),
        ("csv export", client.get(f"/api/reports/{a_ds}/csv", headers=b_headers)),
        ("delete", client.delete(f"/api/datasets/{a_ds}", headers=b_headers)),
    ]:
        check(f"tenant_b -> tenant_a's dataset [{label}] => 404", resp.status_code == 404)

    check("tenant_a's dataset survived the attack attempts",
          client.get(f"/api/datasets/{a_ds}", headers=a_headers).status_code == 200)

    # RAG isolation
    r = client.post("/api/rag/kb", json={"name": "Tenant A KB"}, headers=a_headers)
    a_kb = r.json()["kb_id"]
    check("tenant_a creates a KB", r.status_code == 200)
    check("tenant_b cannot read tenant_a's KB",
          client.get(f"/api/rag/kb/{a_kb}", headers=b_headers).status_code == 404)
    check("tenant_b's KB list is empty",
          client.get("/api/rag/kb", headers=b_headers).json()["knowledge_bases"] == [])

    # regular client token can't reach admin routes
    check("tenant_a cannot list users (not admin)",
          client.get("/api/admin/users", headers=a_headers).status_code == 401)

    # offboarding cascades
    r = client.delete("/api/admin/users/tenant_a", headers=admin_headers())
    check("admin offboards tenant_a", r.status_code == 200)
    check("offboard removed tenant_a's dataset", r.json()["datasets_removed"] == [a_ds])
    check("offboard removed tenant_a's KB", r.json()["knowledge_bases_removed"] == [a_kb])
    check("tenant_a can no longer log in",
          client.post("/api/auth/login", json={"username": "tenant_a", "password": "password_a1"}).status_code == 401)
    check("tenant_a's old token is dead",
          client.get("/api/datasets", headers=a_headers).status_code == 401)

    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("🎉 MULTI-TENANT ISOLATION FULLY VERIFIED")


if __name__ == "__main__":
    main()
