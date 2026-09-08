"""
The three ways this app fell over under ordinary use.

None of these needed an attacker. Each was reachable by a client doing
the thing the product invites them to do, and each took the service down
for everyone on it rather than for the person who triggered it.

**Memory.** Every upload endpoint began `await file.read()`, so the size
limit was checked after the whole body was already in RAM — and the
parse then multiplied it. Measured on a plain CSV, not an attack:

    payload      79 MB   (under the configured 200 MB limit)
    RSS before  203 MB
    RSS after 1,704 MB

Twenty-one times the file, from a file the product accepts. On the
512 MB and 1 GB containers this deploys to, that is the process gone.

**Concurrency.** A report is 3.0s of CPU on the request thread, with
nothing bounding how many run at once. Ten together put thirty
CPU-seconds ahead of the platform's health check, which times out, and
the container is restarted mid-report for every user on it.

**Rate.** One account looping the report endpoint costs what a hundred
clients cost using it normally.
"""
import threading

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.load_control import LIMITS, limiter


@pytest.fixture(autouse=True)
def _fresh_limiter():
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def dataset(client):
    csv = b"region,amount\n" + b"North,10\nSouth,20\nEast,30\nWest,40\n" * 60
    res = client.post("/api/datasets/upload",
                      files={"file": ("s.csv", csv, "text/csv")})
    assert res.status_code == 200
    return res.json()["meta"]["dataset_id"]


# ══════════════════════════════════════════════════════════
#  Memory
# ══════════════════════════════════════════════════════════

def test_an_oversized_upload_is_refused_before_it_is_held(client):
    """The cap is applied while reading, in chunks, so the peak is the
    limit plus one chunk rather than whatever was sent."""
    from app.config import config
    over = int((config.max_file_mb + 5) * 1024 * 1024)
    payload = b"a,b\n" + b"1,2\n" * (over // 4)

    res = client.post("/api/datasets/upload",
                      files={"file": ("huge.csv", payload, "text/csv")})
    assert res.status_code == 413
    body = res.json()["detail"]
    assert "MB" in body, "a size limit without a number is a support ticket"
    assert str(int(config.max_file_mb)) in body


def test_the_row_ceiling_is_applied_while_reading_not_after(client):
    """A 79 MB CSV was parsed in full — 1,704 MB of RSS — and only then
    rejected for having too many rows. Reading is what has to be
    bounded, not the verdict on it."""
    from app.engines.data_loader import ROW_CEILING

    csv = b"a,b\n" + b"1,2\n" * (ROW_CEILING + 5_000)
    res = client.post("/api/datasets/upload",
                      files={"file": ("many.csv", csv, "text/csv")})
    assert res.status_code == 422
    # The count it reports is the ceiling it stopped at, not the file's
    # real length — it never read that far.
    assert "{:,}".format(ROW_CEILING + 1) in res.json()["detail"]


def test_an_ordinary_file_is_unaffected(client):
    csv = b"a,b\n" + b"1,2\n" * 50_000
    res = client.post("/api/datasets/upload",
                      files={"file": ("ok.csv", csv, "text/csv")})
    assert res.status_code == 200
    assert res.json()["meta"]["rows"] == 50_000


# ══════════════════════════════════════════════════════════
#  Rate
# ══════════════════════════════════════════════════════════

def test_one_account_cannot_loop_an_expensive_endpoint(client, dataset):
    limit, _window = LIMITS["report"]
    codes = []
    for _ in range(limit + 2):
        res = client.post(f"/api/reports/{dataset}/pdf",
                          json={"title": "T", "client": "C"})
        codes.append(res.status_code)
        if res.status_code == 429:
            assert res.headers.get("Retry-After"), \
                "a 429 without Retry-After tells the client nothing"
            break
    assert 429 in codes
    assert codes.count(200) <= limit


def test_the_refusal_says_what_to_do(client, dataset):
    limit, _ = LIMITS["report"]
    for _ in range(limit + 2):
        res = client.post(f"/api/reports/{dataset}/pdf",
                          json={"title": "T", "client": "C"})
        if res.status_code == 429:
            detail = res.json()["detail"]
            assert "Wait about" in detail
            assert "this account" in detail
            return
    pytest.fail("never rate limited")


def test_one_operation_does_not_starve_another(client, dataset):
    """A client running reports must still be able to upload."""
    limit, _ = LIMITS["report"]
    for _ in range(limit + 2):
        client.post(f"/api/reports/{dataset}/pdf",
                    json={"title": "T", "client": "C"})

    res = client.post("/api/datasets/upload",
                      files={"file": ("later.csv", b"a,b\n1,2\n3,4\n",
                                      "text/csv")})
    assert res.status_code == 200


# ══════════════════════════════════════════════════════════
#  Concurrency
# ══════════════════════════════════════════════════════════

def test_concurrent_reports_queue_rather_than_all_run(client, dataset):
    """Six at once against two slots. They must all be served — the
    point is that only two are computing at any moment, so the process
    stays responsive to everything else."""
    from app.services.load_control import heavy

    results: list = []

    def build():
        results.append(client.post(f"/api/reports/{dataset}/pdf",
                                   json={"title": "T", "client": "C"}
                                   ).status_code)

    threads = [threading.Thread(target=build) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert heavy.slots < 4, "the gate must be narrower than the burst"
    assert all(code in (200, 429, 503) for code in results), results
    assert 200 in results


# ══════════════════════════════════════════════════════════
#  Being able to find one request again
# ══════════════════════════════════════════════════════════

def test_every_response_carries_an_id(client):
    res = client.get("/api/health")
    assert res.headers.get("X-Request-ID")


def test_an_id_is_echoed_across_a_proxy(client):
    res = client.get("/api/health", headers={"X-Request-ID": "trace-abc-123"})
    assert res.headers["X-Request-ID"] == "trace-abc-123"


@pytest.mark.parametrize("hostile", [
    "bad\ninjected: header",
    "x" * 300,
    "has spaces and ; punctuation",
])
def test_a_client_supplied_id_is_checked_before_it_is_echoed(client, hostile):
    """It lands in a log line and a response header, so it does not get
    to carry newlines or 8 KB of text."""
    res = client.get("/api/health", headers={"X-Request-ID": hostile})
    returned = res.headers["X-Request-ID"]
    assert returned != hostile
    assert "\n" not in returned and len(returned) <= 64


def test_a_failure_hands_the_client_something_to_quote(client, monkeypatch):
    """"It broke at about three o'clock" is not actionable. A 500 now
    carries the id that is also on every log line for that request."""
    from app.api import datasets as datasets_api

    def explode(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(datasets_api.store, "list_datasets", explode,
                        raising=False)
    res = client.get("/api/datasets")
    if res.status_code == 500:
        assert res.json().get("request_id")
        assert res.headers.get("X-Request-ID") == res.json()["request_id"]
