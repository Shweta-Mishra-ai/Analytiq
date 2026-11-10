"""
Analytiq's MCP server: the tools Claude calls, locally (stdio, in-process
here) and remotely (Streamable HTTP at /mcp, one token per user).

The remote tests run the real web app on a local port and talk to it with
the official MCP client, as Claude does.
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time

import numpy as np
import pandas as pd
import pytest

from mcp import Client

from app.mcp_server import mcp


def _text(result) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content)


def _sales_csv(path, n=2400, seed=11):
    rng = np.random.default_rng(seed)
    date = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    sales = rng.gamma(2, 250, n)
    disc = rng.choice([0, .1, .2, .3], n, p=[.5, .25, .15, .1])
    pd.DataFrame({"order_id": range(10001, 10001 + n), "order_date": date,
                  "region": rng.choice(["North", "South", "East", "West"], n),
                  "customer_id": rng.integers(1, 400, n), "sales": sales.round(2),
                  "discount": disc, "profit": (sales * (0.22 - 0.9 * disc)).round(2)}
                 ).to_csv(path, index=False)
    return path


def _run(coro):
    return asyncio.run(coro)


async def _local(calls):
    async with Client(mcp) as c:
        out = []
        for name, args in calls:
            out.append(_text(await c.call_tool(name, args)))
        return out


def _dataset_id(profile: str) -> str:
    return profile.split("`")[1]


# ── local (stdio) ────────────────────────────────────────────────────────

def test_the_tools_on_offer():
    async def names():
        async with Client(mcp) as c:
            return {t.name for t in (await c.list_tools()).tools}
    assert _run(names()) == {"list_datasets", "load_dataset", "business_summary",
                             "data_quality", "query", "find_drivers", "train_model",
                             "generate_report"}


def test_a_local_file_is_loaded_and_summarised_compactly(tmp_path):
    csv = _sales_csv(tmp_path / "sales.csv")
    profile, = _run(_local([("load_dataset", {"path": str(csv)})]))
    assert "2,400 rows × 7 columns" in profile
    assert "Identifiers (not analysed as measures): order_id, customer_id" in profile
    summary, = _run(_local([("business_summary", {"dataset_id": _dataset_id(profile)})]))
    assert "### Scorecard — 12 months to Dec 2024 vs 12 months to Dec 2023" in summary
    assert "Orders discounted over 20% lose money" in summary
    # The point of the server: the answer is a few hundred tokens, not the file.
    assert len(summary) < 4000 < csv.stat().st_size / 10
    assert "10001" not in summary


def test_queries_answer_with_the_chat_engines_honesty(tmp_path):
    csv = _sales_csv(tmp_path / "sales.csv")
    profile, = _run(_local([("load_dataset", {"path": str(csv)})]))
    ds = _dataset_id(profile)
    agg, corr, top, bad = _run(_local([
        ("query", {"dataset_id": ds, "operation": "aggregate", "column": "sales",
                   "group_by": "region", "agg": "mean"}),
        ("query", {"dataset_id": ds, "operation": "correlation", "column": "discount",
                   "other_column": "profit"}),
        ("query", {"dataset_id": ds, "operation": "top_n", "column": "profit", "n": 3}),
        ("query", {"dataset_id": ds, "operation": "drop_table"}),
    ]))
    assert "within what chance produces" in agg          # random regions
    assert "2,400 rows" in corr and "not proof" in corr
    assert top.count("\n|") == 5                         # header, rule, 3 rows
    assert bad.startswith("**Could not do that:** operation must be one of")


def test_a_local_report_is_written_where_asked(tmp_path):
    csv = _sales_csv(tmp_path / "sales.csv")
    profile, = _run(_local([("load_dataset", {"path": str(csv)})]))
    out = tmp_path / "out" / "review.pdf"
    msg, = _run(_local([("generate_report", {"dataset_id": _dataset_id(profile),
                                              "output_path": str(out), "language": "hi"})]))
    assert out.read_bytes()[:5] == b"%PDF-" and str(out) in msg


def test_input_mistakes_are_answered_not_raised(tmp_path):
    a, b, c, d = _run(_local([
        ("load_dataset", {}),
        ("load_dataset", {"path": str(tmp_path / "missing.csv")}),
        ("load_dataset", {"csv_text": "a,b\n" + "1,2\n" * 60_000}),
        ("business_summary", {"dataset_id": "nope"}),
    ]))
    assert "exactly one of path, url or csv_text" in a
    assert "No file at" in b
    assert "limited to 200 KB" in c
    assert "No dataset 'nope'" in d


def test_links_to_private_addresses_are_refused():
    from app.mcp_server import _public_host
    assert not _public_host("127.0.0.1")
    assert not _public_host("169.254.169.254")
    assert not _public_host("10.0.0.5")
    assert not _public_host("localhost")


def test_download_links_are_signed_and_expire(tmp_path, monkeypatch):
    from app import mcp_server
    monkeypatch.setattr(mcp_server, "report_dir", lambda: str(tmp_path))
    (tmp_path / "r.pdf").write_bytes(b"%PDF-1.4")
    exp = int(time.time()) + 60
    good = mcp_server._sign("r.pdf", exp)
    assert mcp_server.verify_download("r.pdf", exp, good) == str(tmp_path / "r.pdf")
    assert mcp_server.verify_download("r.pdf", exp, "0" * 32) is None
    assert mcp_server.verify_download("r.pdf", exp + 1, good) is None
    past = int(time.time()) - 1
    assert mcp_server.verify_download("r.pdf", past, mcp_server._sign("r.pdf", past)) is None
    assert mcp_server.verify_download("../r.pdf", exp, mcp_server._sign("../r.pdf", exp)) is None


# ── remote (Streamable HTTP) ─────────────────────────────────────────────

@pytest.fixture()
def server(monkeypatch):
    """The web app on a local port, with accounts switched on and two users."""
    import uvicorn
    from app.config import config
    from app.main import app
    from app.services.tokens import issue_token
    from app.services.user_store import user_store
    monkeypatch.setattr(config, "app_admin_key", "mcp-test-admin-key")
    for u in ("mcp_alice", "mcp_bob"):
        user_store.create(u, "correct-horse-7")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", {u: issue_token(u) for u in ("mcp_alice", "mcp_bob")}
    finally:
        srv.should_exit = True
        t.join(timeout=10)
        for u in ("mcp_alice", "mcp_bob"):
            user_store.delete(u)


async def _remote(base, token, name, args, extra_headers=None):
    import httpx2
    from mcp.client.streamable_http import streamable_http_client
    headers = dict(extra_headers or {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx2.AsyncClient(headers=headers, timeout=120) as h:
        async with Client(streamable_http_client(base + "/mcp/", http_client=h)) as c:
            return _text(await c.call_tool(name, args))


_CSV = "region,sales,profit\n" + "\n".join(
    f"{r},{100 + i},{10 + i % 7}" for i, r in zip(range(60), ["N", "S", "E", "W"] * 15))


def test_each_remote_user_sees_only_their_own_data(server):
    base, tok = server
    profile = _run(_remote(base, tok["mcp_alice"], "load_dataset",
                           {"csv_text": _CSV, "name": "alice_sales"}))
    ds = _dataset_id(profile)
    assert ds in _run(_remote(base, tok["mcp_alice"], "list_datasets", {}))
    assert ds not in _run(_remote(base, tok["mcp_bob"], "list_datasets", {}))
    assert "No dataset" in _run(_remote(base, tok["mcp_bob"], "business_summary",
                                        {"dataset_id": ds}))


def test_remote_callers_need_a_valid_token(server):
    base, _ = server
    for token in (None, "forged.token"):
        assert "Not signed in" in _run(_remote(base, token, "list_datasets", {}))


def test_remote_callers_cannot_read_server_files_or_internal_links(server):
    base, tok = server
    t = tok["mcp_alice"]
    assert "File paths work only" in _run(_remote(base, t, "load_dataset",
                                                  {"path": "/etc/passwd"}))
    assert "private or local" in _run(_remote(base, t, "load_dataset",
                                              {"url": base.replace("http", "https") + "/x"}))
    assert "Only https" in _run(_remote(base, t, "load_dataset",
                                        {"url": "http://example.com/a.csv"}))


def test_a_remote_report_comes_back_as_a_working_link(server):
    import httpx
    base, tok = server
    ds = _dataset_id(_run(_remote(base, tok["mcp_alice"], "load_dataset", {"csv_text": _CSV})))
    msg = _run(_remote(base, tok["mcp_alice"], "generate_report", {"dataset_id": ds}))
    link = msg.split("ready: ")[1].split(" (valid")[0].replace("https://", "http://")
    r = httpx.get(link)
    assert r.status_code == 200 and r.content[:5] == b"%PDF-"
    assert httpx.get(link.rsplit("sig=", 1)[0] + "sig=" + "0" * 32).status_code == 404


def test_open_mode_serves_this_machine_but_not_a_proxied_internet_caller():
    """Without accounts the server is single-user: local HTTP callers are
    served; a request forwarded by a proxy is not, even from 127.0.0.1."""
    import uvicorn
    from app.main import app
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.05)
    try:
        base = f"http://127.0.0.1:{port}"
        assert not _run(_remote(base, None, "list_datasets", {})).startswith("**Could not")
        proxied = _run(_remote(base, None, "list_datasets", {},
                               {"X-Forwarded-For": "203.0.113.9"}))
        assert "Not signed in" in proxied
    finally:
        srv.should_exit = True
        t.join(timeout=10)


def test_every_tool_is_annotated_for_the_connector_directory():
    """The directory requires a title and read-only / destructive hints on
    every tool; Claude uses them to decide when to ask before calling."""
    async def tools():
        async with Client(mcp) as c:
            return (await c.list_tools()).tools
    read_only = {"list_datasets", "business_summary", "data_quality", "query", "find_drivers"}
    for t in _run(tools()):
        assert t.annotations and t.annotations.title, t.name
        assert t.annotations.read_only_hint is (t.name in read_only), t.name
        assert t.annotations.destructive_hint is False, t.name
        assert len(t.name) <= 64


def test_a_report_never_overwrites_an_existing_file(tmp_path):
    csv = _sales_csv(tmp_path / "sales.csv", n=300)
    profile, = _run(_local([("load_dataset", {"path": str(csv)})]))
    keep = tmp_path / "review.pdf"
    keep.write_bytes(b"keep me")
    msg, = _run(_local([("generate_report", {"dataset_id": _dataset_id(profile),
                                              "output_path": str(keep)})]))
    assert keep.read_bytes() == b"keep me"
    saved = tmp_path / "review (1).pdf"
    assert saved.read_bytes()[:5] == b"%PDF-" and str(saved) in msg
