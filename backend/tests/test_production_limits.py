"""
What keeps one client from taking the service down for the rest, and what
a client can do with their own account: a queue for heavy jobs, storage
quotas, a request rate, error references, password change, admin tools.
"""
from __future__ import annotations

import io
import tarfile
import threading
import time
import uuid

import pandas as pd
import pytest
from fastapi import HTTPException

from app.config import config
from app.services import limits
from app.services.user_store import check_password, user_store

ADMIN = {"Authorization": "Bearer test-admin-key"}


@pytest.fixture
def tenant(client, monkeypatch):
    """A client account on a server with auth enforced: (headers, name)."""
    monkeypatch.setattr(config, "app_admin_key", "test-admin-key")
    name = "t" + uuid.uuid4().hex[:10]
    r = client.post("/api/admin/users", headers=ADMIN,
                    json={"username": name, "password": "first-pass-1"})
    assert r.status_code == 200, r.text
    tok = client.post("/api/auth/login",
                      json={"username": name, "password": "first-pass-1"}).json()["token"]
    yield {"Authorization": f"Bearer {tok}"}, name
    # Removed directly, not through the API: a test that leaves the request
    # rate exhausted must not leave its account behind, which would switch
    # every later test from open mode to "sign in first".
    from app.services.dataset_store import store
    for meta in store.list_meta(name):
        store.delete(name, meta.dataset_id)
    user_store.delete(name)


def _csv(rows=50) -> bytes:
    return pd.DataFrame({"a": range(rows), "b": ["x", "y"] * (rows // 2)}).to_csv(index=False).encode()


def _upload(client, headers, data=None):
    return client.post("/api/datasets/upload", headers=headers,
                       files={"file": ("d.csv", io.BytesIO(data or _csv()), "text/csv")})


# ── heavy jobs ───────────────────────────────────────────

def test_heavy_jobs_run_a_few_at_a_time_and_the_rest_are_told_busy():
    gate = limits.JobGate(slots=1, wait_sec=0.2)
    inside = threading.Event()
    release = threading.Event()

    def hold():
        with gate.job("report"):
            inside.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    inside.wait(5)
    with pytest.raises(HTTPException) as busy:
        with gate.job("report"):
            pass
    assert busy.value.status_code == 503 and "busy" in busy.value.detail
    release.set()
    t.join()
    with gate.job("report"):                    # free again once the first is done
        assert gate.stats()["running"] == 1
    assert gate.stats() == {"slots": 1, "running": 0, "waiting": 0}


def test_a_slot_is_given_back_when_the_job_fails():
    gate = limits.JobGate(slots=1, wait_sec=0.1)
    with pytest.raises(RuntimeError):
        with gate.job():
            raise RuntimeError("boom")
    with gate.job():
        pass


# ── storage ──────────────────────────────────────────────

def test_an_upload_past_the_dataset_count_is_refused_with_the_reason(client, tenant, monkeypatch):
    headers, _ = tenant
    monkeypatch.setattr(config, "max_datasets_per_owner", 2)
    assert _upload(client, headers).status_code == 200
    assert _upload(client, headers).status_code == 200
    r = _upload(client, headers)
    assert r.status_code == 413 and "Delete one" in r.json()["detail"]


def test_an_upload_past_the_storage_quota_is_refused(client, tenant, monkeypatch):
    headers, name = tenant
    assert _upload(client, headers).status_code == 200
    assert limits.owner_usage(name)["datasets"] == 1
    monkeypatch.setattr(config, "max_storage_mb_per_owner", 1)
    r = _upload(client, headers, _csv(200_000))
    assert r.status_code == 413 and "storage limit" in r.json()["detail"]


def test_uploads_pause_before_the_disk_is_full(client, tenant, monkeypatch):
    headers, _ = tenant
    monkeypatch.setattr(config, "min_free_disk_mb", 10**9)
    r = _upload(client, headers)
    assert r.status_code == 507 and "storage space" in r.json()["detail"]


def test_the_upload_limit_is_the_configured_one(client, tenant, monkeypatch):
    headers, _ = tenant
    monkeypatch.setattr(config, "max_file_mb", 1)
    r = _upload(client, headers, b"a,b\n" + b"1,2\n" * 400_000)
    assert r.status_code == 413


def test_a_dataset_says_when_it_will_be_deleted(client, tenant, monkeypatch):
    headers, _ = tenant
    monkeypatch.setattr(config, "data_ttl_days", 30)
    meta = _upload(client, headers).json()["meta"]
    assert abs(meta["expires_at"] - (meta["uploaded_at"] + 30 * 86400)) < 1
    listed = client.get("/api/datasets", headers=headers).json()["datasets"][0]
    assert listed["expires_at"] == meta["expires_at"]
    monkeypatch.setattr(config, "data_ttl_days", 0)
    assert client.get("/api/datasets", headers=headers).json()["datasets"][0]["expires_at"] is None


# ── request rate, errors, headers ────────────────────────

def test_a_caller_past_the_request_rate_is_slowed_not_served(client, tenant, monkeypatch):
    headers, _ = tenant
    monkeypatch.setattr(limits, "rate_limiter", limits.RateLimiter(5))
    codes = [client.get("/api/datasets", headers=headers).status_code for _ in range(8)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429] * 3
    assert client.get("/api/health").status_code == 200      # health checks are not counted


def test_the_request_rate_is_per_caller(client, tenant, monkeypatch):
    headers, _ = tenant
    monkeypatch.setattr(limits, "rate_limiter", limits.RateLimiter(3))
    for _ in range(5):
        client.get("/api/datasets", headers=headers)
    assert client.get("/api/admin/users", headers=ADMIN).status_code == 200


def test_an_unhandled_error_gets_a_reference_not_a_traceback(client, monkeypatch, caplog):
    from app.services import dataset_store
    monkeypatch.setattr(dataset_store.store, "list_meta",
                        lambda owner: (_ for _ in ()).throw(RuntimeError("secret internals")))
    r = client.get("/api/datasets")
    assert r.status_code == 500
    body = r.json()
    assert "secret internals" not in r.text
    assert body["reference"] in body["detail"] and r.headers["X-Request-ID"] == body["reference"]
    assert body["reference"] in caplog.text and "secret internals" in caplog.text


def test_every_answer_carries_the_security_headers(client):
    h = client.get("/api/health").headers
    assert h["X-Content-Type-Options"] == "nosniff" and h["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"]
    assert h["X-Request-ID"]


def test_other_websites_are_not_allowed_to_call_the_api_by_default(client):
    r = client.options("/api/datasets", headers={"Origin": "https://evil.example",
                                                 "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in r.headers


# ── passwords ────────────────────────────────────────────

@pytest.mark.parametrize("bad", ["password", "12345678", "aaaaaaaa", "short"])
def test_guessable_passwords_are_refused(bad):
    with pytest.raises(ValueError):
        check_password(bad, "someone")


def test_a_password_equal_to_the_username_is_refused():
    with pytest.raises(ValueError):
        check_password("acmecorp1", "AcmeCorp1")
    check_password("a-decent-pass-9", "acmecorp1")


def test_changing_a_password_signs_out_the_old_sessions(client, tenant):
    headers, name = tenant
    time.sleep(0.01)
    wrong = client.post("/api/auth/change-password", headers=headers,
                        json={"current_password": "nope-nope", "new_password": "second-pass-2"})
    assert wrong.status_code == 403
    weak = client.post("/api/auth/change-password", headers=headers,
                       json={"current_password": "first-pass-1", "new_password": "password"})
    assert weak.status_code == 422
    ok = client.post("/api/auth/change-password", headers=headers,
                     json={"current_password": "first-pass-1", "new_password": "second-pass-2"})
    assert ok.status_code == 200
    assert client.get("/api/datasets", headers=headers).status_code == 401      # old token
    new = {"Authorization": "Bearer " + ok.json()["token"]}
    assert client.get("/api/datasets", headers=new).status_code == 200
    assert client.post("/api/auth/login",
                       json={"username": name, "password": "first-pass-1"}).status_code == 401
    assert client.post("/api/auth/login",
                       json={"username": name, "password": "second-pass-2"}).status_code == 200


def test_the_account_page_shows_usage_and_limits(client, tenant):
    headers, name = tenant
    _upload(client, headers)
    acc = client.get("/api/account", headers=headers).json()
    assert acc["username"] == name and acc["has_password"]
    assert acc["usage"]["datasets"] == 1 and acc["usage"]["used_mb"] >= 0
    assert acc["max_file_mb"] == config.max_file_mb


# ── admin ────────────────────────────────────────────────

def test_admin_can_reset_a_lost_password(client, tenant):
    headers, name = tenant
    time.sleep(0.01)
    assert client.post(f"/api/admin/users/{name}/password", headers=headers,
                       json={"password": "third-pass-3"}).status_code == 401   # not a client's to call
    r = client.post(f"/api/admin/users/{name}/password", headers=ADMIN,
                    json={"password": "third-pass-3"})
    assert r.status_code == 200
    assert client.get("/api/datasets", headers=headers).status_code == 401
    assert client.post("/api/auth/login",
                       json={"username": name, "password": "third-pass-3"}).status_code == 200


def test_admin_sees_each_clients_usage_and_the_server_status(client, tenant):
    headers, name = tenant
    _upload(client, headers)
    users = client.get("/api/admin/users", headers=ADMIN).json()["users"]
    mine = next(u for u in users if u["username"] == name)
    assert mine["usage"]["datasets"] == 1
    status = client.get("/api/admin/status", headers=ADMIN).json()
    assert status["heavy_jobs"]["slots"] >= 1 and status["disk_free_mb"] > 0
    assert client.get("/api/admin/status", headers=headers).status_code == 401


def test_the_backup_holds_accounts_and_every_clients_data(client, tenant):
    headers, name = tenant
    ds = _upload(client, headers).json()["meta"]["dataset_id"]
    r = client.get("/api/admin/backup", headers=ADMIN)
    assert r.status_code == 200
    names = tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz").getnames()
    assert "data/users.pkl" in names
    assert f"data/datasets/{name}/{ds}/raw.pkl" in names
    assert client.get("/api/admin/backup", headers=headers).status_code == 401


def test_old_account_files_without_the_new_field_still_load():
    from app.services.user_store import User, _public
    u = User("old", b"s", b"h", 1.0)
    del u.__dict__["password_changed_at"]              # as unpickled from an older file
    assert _public(u).password_changed_at == 0.0


# ── the Claude connector key ─────────────────────────────

def test_the_account_page_hands_out_a_connector_key_once(client, tenant, monkeypatch):
    headers, name = tenant
    monkeypatch.setattr(config, "public_url", "https://analytiq.example.com/")
    c = client.get("/api/account", headers=headers).json()["connector"]
    assert c == {"url": "https://analytiq.example.com/mcp/", "has_key": False,
                 "key_created_at": None}
    made = client.post("/api/account/connector-key", headers=headers).json()
    assert made["key"].startswith("aqk-") and made["connector"]["has_key"]
    assert user_store.user_for_connector_key(made["key"]).username == name
    # Shown once: the account answer says a key exists, never the key.
    again = client.get("/api/account", headers=headers)
    assert again.json()["connector"]["has_key"] and made["key"] not in again.text
    assert client.delete("/api/account/connector-key", headers=headers).json() == {"removed": True}
    assert user_store.user_for_connector_key(made["key"]) is None
    assert client.delete("/api/account/connector-key", headers=headers).json() == {"removed": False}


def test_connector_keys_need_an_account_and_survive_nonsense(client):
    # Single-user mode has no accounts: nothing to key.
    assert client.get("/api/account").json()["connector"] is None
    assert client.post("/api/account/connector-key").status_code == 400
    for junk in ("", "aqk-", "aqk-!!!-x", "aqk-" + "A" * 500, "aqk-w6k-x", "é"):
        assert user_store.user_for_connector_key(junk) is None


# ── backups ──────────────────────────────────────────────

def test_snapshots_are_kept_on_disk_pruned_and_never_nest(client, tenant, monkeypatch):
    import os
    from app.services import backups
    headers, _ = tenant
    assert _upload(client, headers).status_code == 200
    for b in backups.list_backups():
        os.unlink(os.path.join(backups.backup_dir(), b["name"]))
    monkeypatch.setattr(config, "backup_keep", 2)
    stamps = iter(["20260101-000001", "20260101-000002", "20260101-000003"])
    monkeypatch.setattr(backups.time, "strftime", lambda _f: next(stamps))
    made = [client.post("/api/admin/backups", headers=ADMIN) for _ in range(3)]
    assert [m.status_code for m in made] == [200, 200, 200]
    listed = client.get("/api/admin/status", headers=ADMIN).json()["backups"]
    assert len(listed) == 2 and made[0].json()["name"] not in [b["name"] for b in listed]
    got = client.get(f"/api/admin/backups/{listed[0]['name']}", headers=ADMIN)
    names = tarfile.open(fileobj=io.BytesIO(got.content)).getnames()
    assert "data/users.pkl" in names
    assert not any(n.startswith("data/backups") for n in names)
    assert client.get("/api/admin/backups/..%2Fusers.pkl", headers=ADMIN).status_code == 404
    assert client.get(f"/api/admin/backups/{listed[0]['name']}", headers=headers).status_code == 401
    for b in backups.list_backups():
        os.unlink(os.path.join(backups.backup_dir(), b["name"]))


def test_a_snapshot_is_skipped_when_it_would_fill_the_disk(monkeypatch):
    import collections
    from app.services import backups
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(backups.shutil, "disk_usage", lambda _p: usage(10**9, 10**9, 10 * 2**20))
    before = backups.list_backups()
    assert backups.take_backup() is None
    assert backups.list_backups() == before


# ── knowing something broke ──────────────────────────────

def test_an_unexpected_error_is_listed_for_the_admin_and_alerted_once(client, monkeypatch):
    from app.services import alerts
    sent = []
    monkeypatch.setattr(config, "alert_webhook_url", "https://hooks.example.com/x")
    monkeypatch.setattr(alerts, "_post", lambda url, text: sent.append((url, text)))
    monkeypatch.setattr(alerts, "_last_sent", 0.0)
    monkeypatch.setattr(alerts, "_held", 0)
    monkeypatch.setattr(alerts.threading, "Thread", _Inline)
    from app.api import datasets as ds_api

    def boom(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(ds_api.store, "list_meta", boom)
    refs = [client.get("/api/datasets").json()["reference"] for _ in range(3)]
    listed = client.get("/api/admin/status").json()
    assert [e["reference"] for e in listed["recent_errors"][:3]] == refs[::-1]
    assert "disk on fire" in listed["recent_errors"][0]["error"]
    assert listed["alerts_configured"] is True
    # Three failures in a row are one message, not three.
    assert len(sent) == 1 and refs[0] in sent[0][1] and "disk on fire" in sent[0][1]
    monkeypatch.setattr(alerts, "_last_sent", 0.0)
    client.get("/api/datasets")
    assert len(sent) == 2 and "+2 more" in sent[1][1]


class _Inline:
    def __init__(self, target, args=(), daemon=None):
        self._run = lambda: target(*args)

    def start(self):
        self._run()


def test_alerts_go_only_to_https_and_never_raise(monkeypatch, caplog):
    from app.services import alerts
    alerts._post("http://plain.example.com/x", "hello")
    alerts._post("file:///etc/passwd", "hello")
    assert "alert could not be sent" in caplog.text
