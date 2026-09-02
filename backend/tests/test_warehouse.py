"""
Warehouse connector tests, run against a real SQLite database.

SQLite is a genuine SQLAlchemy dialect, so this exercises the whole path
— engine construction, inspection, statement execution, identifier
quoting, transaction rollback — rather than a mock that agrees with
whatever the code does. The parts that are specific to Postgres or
Snowflake are the connection string and the driver, and those are
covered by the availability report rather than by pretending to connect.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from app.services import warehouse as W


@pytest.fixture
def db():
    path = os.path.join(tempfile.mkdtemp(), "warehouse.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE employees "
                 "(id INTEGER, department TEXT, salary INTEGER)")
    conn.executemany("INSERT INTO employees VALUES (?, ?, ?)",
                     [(1, "Sales", 50000), (2, "Engineering", 90000),
                      (3, "Sales", 55000)])
    # A name that has to be quoted — the case the code would get wrong if
    # it interpolated identifiers by hand.
    conn.execute('CREATE TABLE "quarterly results" (period TEXT, revenue INTEGER)')
    conn.execute('INSERT INTO "quarterly results" VALUES ("Q1", 120)')
    conn.commit()
    conn.close()
    return f"sqlite:///{path}"


# ── availability ─────────────────────────────────────────

def test_backends_report_what_is_actually_installed():
    """A list that claims Snowflake support on a box with no Snowflake
    driver produces an ImportError traceback, which tells the user
    nothing."""
    rows = {b["key"]: b for b in W.backends()}
    assert rows["sqlite"]["available"] is True
    for key in ("postgresql", "mysql", "snowflake", "bigquery", "mssql"):
        row = rows[key]
        if not row["available"]:
            assert row["install"].startswith("pip install "), \
                f"{key} must name the package to install"


# ── credentials ──────────────────────────────────────────

def test_a_password_never_survives_redaction():
    url = "postgresql+psycopg://alice:hunter2@db.acme.com:5432/prod?sslmode=require"
    out = W.redact(url)
    assert "hunter2" not in out
    assert "alice" in out and "db.acme.com" in out and "prod" in out


def test_query_parameters_are_dropped_from_the_redacted_form():
    """Connection query strings carry tokens too."""
    out = W.redact("snowflake://u:p@acct/db/schema?private_key=SECRETVALUE")
    assert "SECRETVALUE" not in out
    assert "p@" not in out


def test_a_database_error_never_leaks_the_password():
    url = "postgresql+psycopg://alice:hunter2@db/prod"
    msg = W._clean_db_error(
        Exception("could not connect using postgresql://alice:hunter2@db/prod"),
        url)
    assert "hunter2" not in msg
    assert "***" in msg


def test_import_never_writes_a_credential_anywhere(monkeypatch):
    """The one place a leaked warehouse password would persist.

    Driven through a stubbed read rather than a live Postgres, because
    the thing under test is what the API does with the URL it was given
    — every path out of it (the dataset name, the audit trail, the
    response body) must carry the redacted form.
    """
    import tempfile as tmp
    import pandas as pd
    from fastapi.testclient import TestClient
    from app.services.dataset_store import DatasetStore
    import app.services.dataset_store as ds_mod
    import app.api.datasets as datasets_api

    secret_url = "postgresql+psycopg://alice:hunter2@db.acme.com:5432/prod"
    stub = W.QueryResult(
        df=pd.DataFrame({"department": ["Sales", "Engineering", "Sales"],
                         "salary": [50000, 90000, 55000]}),
        sql="SELECT * FROM employees",
        source=W.redact(secret_url))
    monkeypatch.setattr(W, "preview_table", lambda *a, **k: stub)
    monkeypatch.setattr(W, "run_query", lambda *a, **k: stub)

    store = DatasetStore(tmp.mkdtemp())
    monkeypatch.setattr(ds_mod, "store", store)
    monkeypatch.setattr(datasets_api, "store", store)

    from app.main import app
    client = TestClient(app)
    created = client.post("/api/datasets/warehouse/import", json={
        "url": secret_url, "table": "employees", "limit": 100})
    assert created.status_code == 200, created.text
    ds_id = created.json()["meta"]["dataset_id"]

    trail = client.get(f"/api/datasets/{ds_id}/integrity").text
    assert "hunter2" not in created.text
    assert "hunter2" not in trail
    # Provenance is still recorded — the host and database survive, so a
    # reader can tell where the data came from.
    assert "db.acme.com" in trail


def test_a_warehouse_dataset_records_where_it_came_from(monkeypatch):
    """Redaction must not go so far that provenance is lost."""
    label = W.redact("postgresql+psycopg://alice:hunter2@db.acme.com:5432/prod")
    assert "db.acme.com" in label and "prod" in label


# ── the read-only guard ──────────────────────────────────

@pytest.mark.parametrize("sql", [
    "DELETE FROM employees",
    "DROP TABLE employees",
    "UPDATE employees SET salary = 0",
    "INSERT INTO employees VALUES (4, 'x', 1)",
    "CREATE TABLE t (a INT)",
    "  \n  truncate table employees",
])
def test_writes_are_refused_before_they_reach_the_database(sql):
    with pytest.raises(W.WarehouseError, match="SELECT and WITH"):
        W.assert_read_only(sql)


def test_a_second_statement_is_refused_rather_than_split():
    """This is how a read-only check gets walked around."""
    with pytest.raises(W.WarehouseError, match="one statement"):
        W.assert_read_only("SELECT 1; DROP TABLE employees")


def test_a_trailing_semicolon_is_fine():
    W.assert_read_only("SELECT * FROM employees;")


def test_a_cte_is_allowed():
    W.assert_read_only("WITH t AS (SELECT 1) SELECT * FROM t")


def test_an_empty_query_is_refused():
    with pytest.raises(W.WarehouseError, match="No query"):
        W.assert_read_only("   ")


def test_a_write_that_somehow_reached_run_query_still_changes_nothing(db):
    """The rollback is the second guard: if a statement got past the
    check, it must still leave no trace."""
    with pytest.raises(W.WarehouseError):
        W.run_query(db, "DELETE FROM employees")
    assert len(W.run_query(db, "SELECT * FROM employees").df) == 3


# ── reading ──────────────────────────────────────────────

def test_connection_test_reports_the_dialect(db):
    result = W.test_connection(db)
    assert result["ok"] is True
    assert result["dialect"] == "sqlite"
    assert result["error"] == ""


def test_connection_test_reports_a_failure_without_raising():
    result = W.test_connection("sqlite:////nonexistent/dir/x.db")
    assert result["ok"] is False
    assert result["error"]


def test_an_unreadable_url_is_reported_not_raised():
    result = W.test_connection("not-a-url")
    assert result["ok"] is False
    assert result["error"]


def test_tables_are_listed(db):
    names = {t.name for t in W.list_tables(db)}
    assert "employees" in names
    assert "quarterly results" in names


def test_a_query_returns_a_frame(db):
    result = W.run_query(db, "SELECT department, SUM(salary) AS total "
                             "FROM employees GROUP BY department")
    assert set(result.df.columns) == {"department", "total"}
    assert len(result.df) == 2
    assert result.truncated is False


def test_a_table_with_a_space_in_its_name_is_quoted_correctly(db):
    """Interpolating an identifier raw is both a correctness bug and an
    injection route."""
    result = W.preview_table(db, "quarterly results")
    assert list(result.df.columns) == ["period", "revenue"]


def test_the_row_cap_is_enforced_and_reported(db):
    result = W.run_query(db, "SELECT * FROM employees", limit=2)
    assert len(result.df) == 2
    assert result.truncated is True
    assert any("more than 2 rows" in w for w in result.warnings)


def test_the_cap_cannot_be_raised_past_the_hard_maximum(db):
    result = W.run_query(db, "SELECT * FROM employees", limit=10**9)
    assert result.limit == W.MAX_ROWS


def test_an_empty_result_is_an_error_not_an_empty_dataset(db):
    """Silently creating a zero-row dataset sends the failure downstream
    to whichever engine trips over it first."""
    with pytest.raises(W.WarehouseError, match="no rows"):
        W.run_query(db, "SELECT * FROM employees WHERE salary < 0")


def test_a_bad_query_reports_the_databases_own_message(db):
    with pytest.raises(W.WarehouseError, match="no_such_table|no such table"):
        W.run_query(db, "SELECT * FROM no_such_table")


# ── through the API ──────────────────────────────────────

@pytest.fixture
def api(monkeypatch):
    from fastapi.testclient import TestClient
    from app.services.dataset_store import DatasetStore
    import app.services.dataset_store as ds_mod
    import app.api.datasets as datasets_api

    store = DatasetStore(tempfile.mkdtemp())
    monkeypatch.setattr(ds_mod, "store", store)
    monkeypatch.setattr(datasets_api, "store", store)
    from app.main import app
    return TestClient(app)


def test_backends_endpoint_lists_what_is_installable(api):
    body = api.get("/api/datasets/warehouse/backends").json()
    keys = {b["key"] for b in body["backends"]}
    assert {"postgresql", "mysql", "snowflake", "bigquery", "mssql",
            "sqlite"} == keys


def test_preview_keeps_nothing(api, db):
    before = api.get("/api/datasets").json()
    body = api.post("/api/datasets/warehouse/preview",
                    json={"url": db, "table": "employees"}).json()
    assert body["rows"] == 3
    assert api.get("/api/datasets").json() == before, \
        "a preview must not create a dataset"


def test_import_creates_a_dataset_the_rest_of_the_app_can_use(api, db):
    body = api.post("/api/datasets/warehouse/import", json={
        "url": db,
        "sql": "SELECT department, salary FROM employees",
    }).json()
    ds_id = body["meta"]["dataset_id"]
    assert body["meta"]["rows"] == 3
    # The proof it is a first-class dataset: the ordinary routes work.
    profile = api.get(f"/api/datasets/{ds_id}/profile")
    assert profile.status_code == 200


def test_the_audit_trail_records_the_query_the_data_came_from(api, db):
    body = api.post("/api/datasets/warehouse/import", json={
        "url": db, "table": "employees",
    }).json()
    trail = api.get(f"/api/datasets/{body['meta']['dataset_id']}/integrity").json()
    ingests = [e for e in trail["audit"] if e["event"] == "ingest"]
    assert any("SELECT" in str(e["detail"].get("sql", "")) for e in ingests), \
        "a warehouse pull must record the statement that produced it"


def test_a_write_through_the_api_is_refused(api, db):
    r = api.post("/api/datasets/warehouse/import",
                 json={"url": db, "sql": "DROP TABLE employees"})
    assert r.status_code == 422
    assert "SELECT" in r.json()["detail"]
