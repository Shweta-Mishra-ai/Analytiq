"""
tests/test_security.py — the boundaries an unauthenticated or hostile
caller can push on: the static-file fallback, the bearer-token parser,
login, and account names that become directory names.
"""
from __future__ import annotations

import os

import pytest

from app import main


pytestmark = pytest.mark.skipif(
    not os.path.isdir(main._static), reason="no built frontend to serve")


@pytest.mark.parametrize("path", [
    "/%2Fetc%2Fpasswd",                       # absolute after decoding
    "/..%2Fapp%2Fconfig.py",                  # parent of the static root
    "/%2e%2e/app/config.py",
    "/..%2F..%2F..%2Fetc%2Fpasswd",
    "/assets%2F..%2F..%2Fapp%2Fmain.py",
])
def test_static_fallback_never_serves_files_outside_the_build(client, path):
    r = client.get(path)
    # The SPA shell or a 404 (the /assets mount refuses on its own) —
    # never the requested file.
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert "<!doctype html>" in r.text.lower()
    assert "root:" not in r.text
    assert "AppConfig" not in r.text and "FastAPI" not in r.text


def test_static_fallback_still_serves_real_build_files(client):
    r = client.get("/favicon.svg")
    assert r.status_code == 200
    assert "<svg" in r.text


def test_static_file_resolver_rejects_a_sibling_with_a_shared_prefix(tmp_path):
    # '/srv/static-private' starts with '/srv/static' as a string; a
    # startswith() check would let it through.
    root = os.path.realpath(main._static)
    sibling = root + "-private"
    assert main._static_file(os.path.relpath(sibling, root) + "/x") is None


# ── tokens and admin key ──────────────────────────────────

def test_non_ascii_bearer_tokens_are_rejected_not_crashed():
    from app.services.tokens import verify_token
    assert verify_token("abc.déf") is None
    assert verify_token("é.abc") is None
    assert verify_token("") is None
    assert verify_token("no-dot") is None


def test_non_ascii_admin_key_candidate_is_rejected_not_crashed(monkeypatch):
    from app.services import auth
    monkeypatch.setattr(auth.config, "app_admin_key", "real-key")
    assert auth.check_admin_key("é") is False
    assert auth.check_admin_key("real-key") is True


def test_token_round_trip_and_tamper():
    from app.services.tokens import issue_token, verify_token
    t = issue_token("tenant_x")
    assert verify_token(t) == "tenant_x"
    payload, sig = t.split(".")
    assert verify_token(payload + "." + sig[:-1] + ("0" if sig[-1] != "0" else "1")) is None
    assert verify_token(issue_token("tenant_x", ttl_days=-1)) is None


# ── accounts ──────────────────────────────────────────────

@pytest.fixture()
def users(tmp_path):
    from app.services.user_store import UserStore
    return UserStore(str(tmp_path / "users.pkl"))


@pytest.mark.parametrize("name", ["../evil", "a/b", "a\x00b", ".hidden", "", "x" * 65,
                                  "admin", "local", "sp ace"])
def test_unsafe_or_reserved_usernames_are_refused(users, name):
    with pytest.raises(ValueError):
        users.create(name, "password123")


def test_ordinary_usernames_are_accepted_and_case_folded(users):
    users.create("Tenant_A", "password123")
    assert users.exists("tenant_a") and users.exists("TENANT_A")
    assert users.verify("tenant_a", "password123")
    assert users.verify("tenant_a", "wrong-pass") is None
    assert users.verify("nobody", "password123") is None


def test_a_corrupt_account_store_refuses_to_load_instead_of_opening_the_app(tmp_path):
    # Loading as empty would mean open dev mode: no authentication at all.
    from app.services.user_store import UserStore, UserStoreCorrupt
    p = tmp_path / "users.pkl"
    p.write_bytes(b"\x80\x05 truncated")
    with pytest.raises(UserStoreCorrupt):
        UserStore(str(p))


def test_login_throttle_blocks_after_repeated_failures_and_resets_on_success():
    from app.services.auth import LoginThrottle
    t = LoginThrottle(max_failures=3, window_sec=60)
    for _ in range(3):
        assert t.retry_after("Bob") == 0
        t.record_failure("bob")
    assert t.retry_after("BOB") > 0
    t.record_success("bob")
    assert t.retry_after("bob") == 0


def test_login_endpoint_returns_429_once_throttled(client, monkeypatch):
    from app.services import auth
    monkeypatch.setattr(auth, "login_throttle", auth.LoginThrottle(max_failures=2))
    for _ in range(2):
        assert client.post("/api/auth/login",
                           json={"username": "ghost", "password": "x"}).status_code == 401
    r = client.post("/api/auth/login", json={"username": "ghost", "password": "x"})
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) > 0


# ── data wipe on offboarding ──────────────────────────────

def test_offboarding_removes_knowledge_bases_from_the_served_store(client, monkeypatch):
    # Deleting through a second RagStore left the KB in the shared one's
    # memory cache: still readable after a "full data wipe".
    from app.config import config
    from app.rag.vector_store import rag_store
    from app.services.user_store import user_store
    monkeypatch.setattr(config, "app_admin_key", "offboard-test-key")
    user_store.create("offboard_me", "password123")
    try:
        kb = rag_store.create("offboard_me", "contracts")
        assert rag_store.get("offboard_me", kb.kb_id) is not None
        r = client.delete("/api/admin/users/OFFBOARD_ME",
                          headers={"Authorization": "Bearer offboard-test-key"})
        assert r.status_code == 200, r.text
        assert kb.kb_id in r.json()["knowledge_bases_removed"]
        assert rag_store.get("offboard_me", kb.kb_id) is None
    finally:
        user_store.delete("offboard_me")


def test_ttl_sweep_removes_knowledge_bases_from_the_served_store():
    from app.rag.vector_store import rag_store
    from app.services.cleanup import sweep_expired
    kb = rag_store.create("local", "old")
    kb.created_at = 0
    kb.save()
    sweep_expired(ttl_days=1)
    assert rag_store.get("local", kb.kb_id) is None


# ── analysis cache ────────────────────────────────────────

def test_cache_is_invalidated_by_an_edit_beyond_the_first_rows(tmp_path):
    # The old key hashed only shape, dtypes and the first 100 rows.
    import numpy as np
    import pandas as pd
    from app.services.dataset_store import DatasetStore
    st = DatasetStore(str(tmp_path))
    df = pd.DataFrame({"v": np.arange(500, dtype=float)})
    m = st.create("local", df, "x.csv", 0.1)
    st.cache_set("local", m.dataset_id, "stats", "old-analysis")
    assert st.cache_get("local", m.dataset_id, "stats") == "old-analysis"
    edited = df.copy()
    edited.loc[400:, "v"] = 1e6
    st.update_active("local", m.dataset_id, edited)
    assert st.cache_get("local", m.dataset_id, "stats") is None


def test_cache_survives_a_restart_and_handles_slashes_in_keys(tmp_path):
    import pandas as pd
    from app.services.dataset_store import DatasetStore
    st = DatasetStore(str(tmp_path))
    m = st.create("local", pd.DataFrame({"a": [1, 2, 3]}), "x.csv", 0.1)
    st.cache_set("local", m.dataset_id, "ml_Revenue/Unit", {"r2": 0.5})
    st.cache_set("local", m.dataset_id, "ml_../../etc", {"r2": 0.1})
    fresh = DatasetStore(str(tmp_path))         # new process, empty memory
    assert fresh.cache_get("local", m.dataset_id, "ml_Revenue/Unit") == {"r2": 0.5}
    assert fresh.cache_get("local", m.dataset_id, "ml_../../etc") == {"r2": 0.1}
    assert fresh.cache_get("local", m.dataset_id, "ml_other") is None


def test_rfm_cache_distinguishes_price_and_quantity_columns(client, transactions_df):
    import io
    df = transactions_df.assign(qty=1, price_a=2.0, price_b=50.0)
    r = client.post("/api/datasets/upload", files={
        "file": ("tx.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")})
    ds = r.json()["meta"]["dataset_id"]
    base = f"/api/analytics/{ds}/rfm?customer_col=customer_id&date_col=order_date&quantity_col=qty"
    a = client.get(base + "&price_col=price_a")
    b = client.get(base + "&price_col=price_b")
    assert a.status_code == 200 and b.status_code == 200, (a.text, b.text)
    assert a.json() != b.json()
