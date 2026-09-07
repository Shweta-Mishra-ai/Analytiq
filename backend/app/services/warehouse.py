"""
services/warehouse.py — reading a dataset out of a database instead of a
file upload.

Why this exists: every analysis in this app started with someone
exporting a CSV. That export is the weakest link in the whole chain. It
is a point-in-time copy with no lineage, it is usually stale by the time
it is analysed, it silently loses types (a date becomes a string, a
decimal becomes a float), and it means the platform can only ever see
what somebody remembered to extract. A client with their data in
Postgres or Snowflake was, until now, doing manual work to use this
product at all.

Design decisions worth stating, because each one is a refusal of an
easier option:

**Read-only, enforced twice.** Statements are checked against a
whitelist before they are sent, and the connection is opened in a
transaction that is always rolled back. An analytics tool has no
business writing to a client's warehouse, and "we only ever send
SELECTs" is not a guarantee — it is an intention.

**No credentials at rest.** A connection URL carries a password. This
module never writes one to disk, and never returns one: URLs are
redacted on the way out. A client's warehouse password living in our
storage is a liability with no upside, so the caller supplies it per
request. That is less convenient than saved connections, deliberately.

**Drivers stay optional.** SQLAlchemy is the only dependency added.
Each backend's driver (psycopg, pymysql, snowflake-connector-python,
the BigQuery dialect) is installed only if a deployment actually needs
it, and a missing one produces a message naming the exact package to
install rather than an ImportError traceback.

**Row caps are not a suggestion.** A query against a fact table can
return a hundred million rows and take the process down. Every read is
capped, and the cap is enforced while fetching rather than trusted to a
LIMIT the dialect may or may not honour inside a wrapper.

Ingested data goes through the same door as an upload — same store, same
integrity record, same audit trail — so a dataset that came from a
warehouse is exactly as traceable as one that came from a file, and the
audit trail records the query it came from.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from app.config import config

logger = logging.getLogger(__name__)

MAX_ROWS = 1_000_000
DEFAULT_ROWS = 200_000

# One entry per backend we claim to support. `driver` is the pip package
# a deployment installs to turn the row on; `prefix` is what a URL for it
# starts with. Claiming support for a backend nobody can install is worse
# than not listing it, so this table is also what the API reports as
# available — computed, not asserted.
BACKENDS = {
    "postgresql": {
        "label": "PostgreSQL",
        "prefix": "postgresql+psycopg://",
        "driver": "psycopg[binary]",
        "module": "psycopg",
        "example": "postgresql+psycopg://user:password@host:5432/database",
    },
    "mysql": {
        "label": "MySQL / MariaDB",
        "prefix": "mysql+pymysql://",
        "driver": "pymysql",
        "module": "pymysql",
        "example": "mysql+pymysql://user:password@host:3306/database",
    },
    "snowflake": {
        "label": "Snowflake",
        "prefix": "snowflake://",
        "driver": "snowflake-sqlalchemy",
        "module": "snowflake.sqlalchemy",
        "example": "snowflake://user:password@account/database/schema"
                   "?warehouse=COMPUTE_WH&role=ANALYST",
    },
    "bigquery": {
        "label": "Google BigQuery",
        "prefix": "bigquery://",
        "driver": "sqlalchemy-bigquery",
        "module": "sqlalchemy_bigquery",
        "example": "bigquery://project/dataset",
    },
    "mssql": {
        "label": "SQL Server",
        "prefix": "mssql+pyodbc://",
        "driver": "pyodbc",
        "module": "pyodbc",
        "example": "mssql+pyodbc://user:password@host/database"
                   "?driver=ODBC+Driver+18+for+SQL+Server",
    },
    "sqlite": {
        "label": "SQLite",
        "prefix": "sqlite:///",
        "driver": "",                  # in the standard library
        "module": "sqlite3",
        "example": "sqlite:////absolute/path/to/file.db",
    },
}

# Anything that is not one of these is refused before it reaches the
# database. A blacklist of dangerous words is the wrong shape here: it
# fails open on the one keyword nobody thought of.
_READ_ONLY_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
# A second statement is how a read-only check gets walked around, so a
# query carrying one is refused rather than split and partly run.
_STATEMENT_SPLIT = re.compile(r";\s*\S")

# The above two were the whole guard, and five things walked past them.
# Measured, not imagined:
#
#   WITH x AS (DELETE FROM users RETURNING *) SELECT * FROM x
#   WITH x AS (SELECT 1) DELETE FROM users
#       PostgreSQL data-modifying CTEs. Both start with "with", so the
#       start-anchor matched and the statement deleted rows.
#   SELECT * INTO new_table FROM users
#       PostgreSQL and SQL Server: a SELECT that creates a table.
#   SELECT * FROM users INTO OUTFILE '/tmp/pwned.txt'
#       MySQL writes a file on the database server. Not transactional,
#       so the rollback below does not undo it.
#   SELECT lo_import('/etc/passwd')
#       PostgreSQL large-object import: reads a server-side file.
#
# The rollback was described as the second of two guards. It is a
# genuine second guard against DML, and no guard at all against a file
# write or a DDL statement that auto-commits — so the statement check
# has to carry the weight it was assumed to share.
_FORBIDDEN_IN_READ = (
    (re.compile(r"\b(insert|update|delete|merge|truncate|drop|alter|create|"
                r"grant|revoke|call|do|copy|vacuum|reindex|attach|detach|"
                r"pragma|load_extension)\b", re.IGNORECASE),
     "It contains a statement that changes data or schema ({}). This "
     "tool only reads."),
    (re.compile(r"\binto\s+(outfile|dumpfile)\b", re.IGNORECASE),
     "It writes a file on the database server (INTO {})."),
    (re.compile(r"\bselect\b[\s\S]*?\binto\s+(?!outfile|dumpfile)"
                r"[a-z_\"\[]", re.IGNORECASE),
     "SELECT ... INTO creates a table. Use a plain SELECT."),
    (re.compile(r"\b(lo_import|lo_export|pg_read_file|pg_read_binary_file|"
                r"pg_ls_dir|pg_sleep|dblink|readfile|load_file|"
                r"sys_exec|sys_eval|xp_cmdshell|openrowset|opendatasource)\s*\(",
     re.IGNORECASE),
     "It calls {}, which reads server files, runs commands or holds "
     "the connection open."),
)


# Which URL schemes may be dialled at all. SQLAlchemy will happily build
# an engine for any dialect it can import, and `sqlite:////etc/passwd`
# made every file on the APPLICATION server readable by any
# authenticated client — other tenants' datasets, the user store with
# its credential hashes, the environment file with the API keys. The
# feature is "read from your warehouse", and a path on our own disk is
# not that.
_ALLOWED_SCHEMES = frozenset(
    spec["prefix"].split("://")[0] for key, spec in BACKENDS.items()
    if key != "sqlite"
)

# SQLite stays available for local and self-hosted use, but only inside
# a directory the deployment names, and only when it names one. Off by
# default: on a hosted install there is no such directory, and the
# answer to "can a client read /etc/passwd" must not depend on nobody
# trying.
_SQLITE_SCHEME = "sqlite"


def _sqlite_root() -> Optional[str]:
    import os
    root = (os.environ.get("WAREHOUSE_SQLITE_DIR") or "").strip()
    return os.path.realpath(root) if root else None


class WarehouseError(RuntimeError):
    """Every failure here is something the caller can act on, so the
    message is written for them rather than for a log."""


@dataclass
class TableRef:
    schema: str
    name: str
    rows: Optional[int] = None

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name


@dataclass
class QueryResult:
    df: pd.DataFrame
    sql: str
    source: str                        # redacted connection description
    truncated: bool = False
    limit: int = 0
    warnings: list = field(default_factory=list)


# ── availability ─────────────────────────────────────────

def sqlalchemy_available() -> bool:
    try:
        import sqlalchemy                            # noqa: F401
        return True
    except ImportError:
        return False


def _driver_installed(key: str) -> bool:
    module = BACKENDS[key]["module"]
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def backends() -> list[dict]:
    """What this deployment can actually connect to right now.

    Computed by importing, not declared: a list that claims Snowflake
    support on a box with no Snowflake driver produces a connection
    attempt that fails with an ImportError traceback, which tells the
    user nothing.
    """
    out = []
    for key, spec in BACKENDS.items():
        installed = _driver_installed(key)
        out.append({
            "key": key,
            "label": spec["label"],
            "available": installed and sqlalchemy_available(),
            "url_prefix": spec["prefix"],
            "example": spec["example"],
            "install": ("" if installed else
                        f"pip install {spec['driver']}" if spec["driver"]
                        else ""),
        })
    return out


# ── safety ───────────────────────────────────────────────

def redact(url: str) -> str:
    """A connection URL carries a password, and this string ends up in
    error messages, audit entries and the report's provenance line."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "database"
    if not parts.netloc:
        return url.split("?")[0]
    host = parts.hostname or ""
    user = parts.username or ""
    netloc = f"{user}:***@{host}" if user else host
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def assert_read_only(sql: str) -> None:
    """Refuse anything that is not a single SELECT or CTE.

    Whitelist rather than blacklist, and enforced before the statement
    is sent rather than relying on database permissions the client may
    not have set up. The connection is *also* rolled back — one guard
    that can be reasoned about is not the same as two.
    """
    if not sql or not sql.strip():
        raise WarehouseError("No query was given.")
    if not _READ_ONLY_START.match(sql):
        raise WarehouseError(
            "Only SELECT and WITH queries are allowed. This tool reads "
            "from your warehouse and never writes to it.")
    if _STATEMENT_SPLIT.search(sql):
        raise WarehouseError(
            "Send one statement at a time. A query containing a second "
            "statement after a semicolon is refused rather than split.")

    # Starting with SELECT or WITH is necessary and, on its own, not
    # close to sufficient — see the cases listed beside
    # _FORBIDDEN_IN_READ, every one of which walked past the check
    # above while starting with one of those two words.
    for pattern, message in _FORBIDDEN_IN_READ:
        found = pattern.search(sql)
        if found:
            offending = (found.group(1) if found.groups() else found.group(0))
            raise WarehouseError(
                "That query was refused. " + message.format(
                    str(offending).strip().upper()))


def assert_dialable(url: str) -> None:
    """Refuse a connection URL before an engine is built for it.

    Two separate risks, both of them the same shape — the SERVER makes a
    connection to a host the CLIENT names:

    * a scheme we never meant to support (``sqlite:////etc/passwd`` read
      arbitrary files on the application server), and
    * a host inside our own network (``127.0.0.1`` reaches services that
      are not exposed; ``169.254.169.254`` is the cloud metadata
      endpoint that hands out the instance's credentials).
    """
    import ipaddress
    import os

    try:
        parts = urlsplit(url.strip())
    except ValueError:
        raise WarehouseError("That connection URL could not be read.") from None

    scheme = (parts.scheme or "").lower()
    base = scheme.split("+")[0]
    if not scheme:
        raise WarehouseError(
            "A connection URL needs to start with the database type, "
            "for example postgresql+psycopg://")

    if base == _SQLITE_SCHEME:
        root = _sqlite_root()
        if not root:
            raise WarehouseError(
                "SQLite files are not available on this server. A path "
                "here is a path on the application's own disk rather "
                "than a database you own, so it is refused unless the "
                "deployment sets WAREHOUSE_SQLITE_DIR to a directory it "
                "is willing to expose.")
        # A SQLite URL is "sqlite://" + an empty netloc + the path, so
        # sqlite:///rel.db is relative and sqlite:////abs.db is
        # absolute. Stripping every leading slash turns the absolute
        # form into a relative one and lands the check on the wrong
        # file entirely.
        rest = url.split("://", 1)[1]
        raw = rest[1:] if rest.startswith("/") else rest
        target = os.path.realpath(raw)
        try:
            shared = os.path.commonpath([target, root])
        except ValueError:            # different drives / bad path
            shared = ""
        if shared != root:
            raise WarehouseError(
                "That SQLite file is outside the directory this server "
                "allows database files to be read from.")
        return

    if scheme not in _ALLOWED_SCHEMES:
        raise WarehouseError(
            "'{}' is not a database type this server connects to. "
            "Supported: {}.".format(
                scheme, ", ".join(sorted(_ALLOWED_SCHEMES))))

    host = (parts.hostname or "").strip()
    if not host:
        raise WarehouseError("That connection URL names no host.")

    # A name that resolves to a blocked address is the same attack with
    # a DNS record in front of it, so resolve before judging.
    addresses = []
    try:
        ip = ipaddress.ip_address(host)
        addresses = [ip]
    except ValueError:
        import socket
        try:
            for info in socket.getaddrinfo(host, None):
                try:
                    addresses.append(ipaddress.ip_address(info[4][0]))
                except ValueError:
                    continue
        except OSError:
            # Unresolvable is the database's problem to report, not a
            # reason to refuse here — the connection will fail with a
            # message about the host, which is what the user needs.
            logger.debug("could not resolve warehouse host %r", host,
                         exc_info=True)
            return

    allow_private = getattr(config, "warehouse_allow_private_hosts", True)
    for ip in addresses:
        if ip.is_loopback or ip.is_link_local or ip.is_multicast \
                or ip.is_reserved or ip.is_unspecified:
            raise WarehouseError(
                "That address ({}) is on the application server's own "
                "network rather than a database you own. Connections "
                "there are refused.".format(ip))
        if ip.is_private and not allow_private:
            raise WarehouseError(
                "That address ({}) is on a private network this server "
                "does not connect out to.".format(ip))


def _engine(url: str):
    assert_dialable(url)
    if not sqlalchemy_available():
        raise WarehouseError(
            "SQLAlchemy is not installed on this server, so database "
            "connections are unavailable. Install it with: "
            "pip install sqlalchemy")
    import sqlalchemy as sa
    try:
        return sa.create_engine(url, pool_pre_ping=True)
    except ModuleNotFoundError as e:
        raise WarehouseError(_missing_driver_message(url, e)) from None
    except Exception as e:                           # noqa: BLE001
        raise WarehouseError(f"Could not read that connection URL: {e}") from None


def _missing_driver_message(url: str, error: Exception) -> str:
    for spec in BACKENDS.values():
        if url.startswith(spec["prefix"].split("://")[0]) and spec["driver"]:
            return (f"The driver for {spec['label']} is not installed on "
                    f"this server. Install it with: pip install "
                    f"{spec['driver']}")
    return f"That database driver is not installed on this server: {error}"


# ── operations ───────────────────────────────────────────

def test_connection(url: str) -> dict:
    """Connect, ask the database what it is, disconnect.

    Returns rather than raises for the ordinary failures, because the
    whole point of a connection test is to report what went wrong in a
    form the person entering the URL can act on.
    """
    source = redact(url)
    try:
        engine = _engine(url)
    except WarehouseError as e:
        return {"ok": False, "source": source, "error": str(e)}

    import sqlalchemy as sa
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
            dialect = engine.dialect.name
            try:
                version = ".".join(
                    str(p) for p in (conn.dialect.server_version_info or ()))
            except Exception:                        # noqa: BLE001
                version = ""
        return {"ok": True, "source": source, "dialect": dialect,
                "version": version, "error": ""}
    except Exception as e:                           # noqa: BLE001
        return {"ok": False, "source": source, "error": _clean_db_error(e, url)}
    finally:
        engine.dispose()


def list_tables(url: str, schema: str = "") -> list[TableRef]:
    """What is in there, so a user is not asked to type a table name from
    memory."""
    engine = _engine(url)
    import sqlalchemy as sa
    try:
        inspector = sa.inspect(engine)
        schemas = [schema] if schema else _candidate_schemas(inspector)
        out = []
        for sch in schemas:
            try:
                names = inspector.get_table_names(schema=sch or None)
                views = inspector.get_view_names(schema=sch or None)
            except Exception:                        # noqa: BLE001
                logger.debug("could not list schema %r", sch, exc_info=True)
                continue
            for name in sorted(set(names) | set(views)):
                out.append(TableRef(schema=sch or "", name=name))
        return out
    except Exception as e:                           # noqa: BLE001
        raise WarehouseError(_clean_db_error(e, url)) from None
    finally:
        engine.dispose()


def _candidate_schemas(inspector) -> list[str]:
    """Every schema, minus the system ones nobody wants to scroll past."""
    system = {"information_schema", "pg_catalog", "pg_toast", "sys",
              "performance_schema", "mysql", "INFORMATION_SCHEMA"}
    try:
        found = [s for s in inspector.get_schema_names() if s not in system]
    except Exception:                                # noqa: BLE001
        return [""]
    return found or [""]


def run_query(url: str, sql: str, limit: int = DEFAULT_ROWS) -> QueryResult:
    """One guarded read.

    The cap is enforced while fetching rather than by wrapping the query
    in a LIMIT: dialects differ on whether a wrapper subquery is legal
    around an arbitrary statement, and a cap that some dialect quietly
    drops is not a cap. One extra row is fetched so that "there was
    more" can be reported rather than guessed at.
    """
    assert_read_only(sql)
    limit = max(1, min(int(limit or DEFAULT_ROWS), MAX_ROWS))
    engine = _engine(url)
    source = redact(url)
    warnings: list[str] = []

    import sqlalchemy as sa
    try:
        with engine.connect() as conn:
            # Always inside a transaction that is rolled back. Belt and
            # braces with assert_read_only: if a statement somehow got
            # through that check, it still leaves no trace.
            trans = conn.begin()
            try:
                result = conn.execute(sa.text(sql.rstrip().rstrip(";")))
                rows = result.fetchmany(limit + 1)
                columns = list(result.keys())
            finally:
                trans.rollback()
    except WarehouseError:
        raise
    except Exception as e:                           # noqa: BLE001
        raise WarehouseError(_clean_db_error(e, url)) from None
    finally:
        engine.dispose()

    truncated = len(rows) > limit
    if truncated:
        rows = rows[:limit]
        warnings.append(
            f"The query returned more than {limit:,} rows; the first "
            f"{limit:,} were loaded. Narrow the query with a WHERE clause "
            f"or a date range if the rest matter.")

    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        raise WarehouseError(
            "The query ran but returned no rows. There is nothing to "
            "analyse — check the filters in the WHERE clause.")
    return QueryResult(df=df, sql=sql.strip(), source=source,
                       truncated=truncated, limit=limit, warnings=warnings)


def preview_table(url: str, table: str, schema: str = "",
                  limit: int = 1000) -> QueryResult:
    """A first look at a table without asking the user to write SQL."""
    ident = _quote_identifier(url, table, schema)
    return run_query(url, f"SELECT * FROM {ident}", limit=limit)


def _quote_identifier(url: str, table: str, schema: str = "") -> str:
    """Quote through the dialect rather than by hand.

    A table name arrives from a client's warehouse and can contain
    anything the database allows — spaces, reserved words, mixed case.
    Interpolating it raw is both a correctness bug and an injection
    route; the dialect's own preparer is the one thing that knows the
    right quoting for that database.
    """
    engine = _engine(url)
    try:
        preparer = engine.dialect.identifier_preparer
        quoted = preparer.quote(table)
        if schema:
            quoted = f"{preparer.quote(schema)}.{quoted}"
        return quoted
    finally:
        engine.dispose()


def _clean_db_error(error: Exception, url: str) -> str:
    """Database errors are long, contain the driver's stack, and — worst
    — often echo the connection string back with the password in it."""
    text = " ".join(str(error).split())
    password = ""
    try:
        password = urlsplit(url).password or ""
    except ValueError:
        pass
    if password:
        text = text.replace(password, "***")
    if "(Background on this error" in text:
        text = text.split("(Background on this error")[0].strip()
    return text[:400] or error.__class__.__name__
