"""
Two client accounts on one deployment must never see, read or delete
each other's work. This is the property a freelancer's own clients are
implicitly trusting, and the one whose failure is unrecoverable.

It was a script that set APP_ADMIN_KEY before importing the app and then
ran one long sequence, so a failure anywhere stopped everything after
it. The auth middleware reads the key per request, so the whole thing
runs in-process now — and every test builds its own tenants rather than
inheriting the previous test's, because the suite is run in random
order.
"""
from __future__ import annotations

import io
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ADMIN_KEY = "test-admin-key-do-not-use-in-prod"


def _csv(column: str) -> bytes:
    return pd.DataFrame({column: [1, 2, 3],
                         "value": [10, 20, 30]}).to_csv(index=False).encode()


class _Tenancy:
    """A deployment with auth switched on, and a way to make clients."""

    def __init__(self, client):
        self.client = client
        self.admin = {"Authorization": "Bearer " + ADMIN_KEY}
        self._made = []

    def tenant(self, label: str):
        """A fresh account, its auth header, and one uploaded dataset.

        The username carries a uuid because datasets are keyed by owner
        in a DATA_DIR shared across the session and across runs. A fixed
        name meant the previous run's datasets were still there, and the
        offboarding assertion below — exactly one dataset removed — would
        fail on the second `pytest` and pass on a clean CI runner.
        """
        name = "{}_{}".format(label, uuid.uuid4().hex[:10])
        password = "password_{}1".format(label)

        res = self.client.post("/api/admin/users",
                               json={"username": name, "password": password},
                               headers=self.admin)
        assert res.status_code == 200, res.text
        self._made.append(name)

        res = self.client.post("/api/auth/login",
                               json={"username": name, "password": password})
        assert res.status_code == 200, res.text
        headers = {"Authorization": "Bearer " + res.json()["token"]}

        res = self.client.post(
            "/api/datasets/upload",
            files={"file": (label + ".csv",
                            io.BytesIO(_csv(label + "_col")), "text/csv")},
            headers=headers)
        assert res.status_code == 200, res.text

        return name, password, headers, res.json()["meta"]["dataset_id"]

    def cleanup(self):
        for name in self._made:
            self.client.delete("/api/admin/users/" + name, headers=self.admin)


@pytest.fixture()
def tenancy(monkeypatch, tmp_path):
    from app.config import config
    from app.services.user_store import UserStore

    monkeypatch.setattr(config, "app_admin_key", ADMIN_KEY, raising=False)

    store = UserStore(str(tmp_path / "users.json"))
    # Every module that imported the singleton by name holds its own
    # reference. Miss the auth middleware's and its is_empty() still says
    # True, the app stays in single-user open mode, and none of this
    # tests anything.
    import app.api.accounts as accounts_api
    import app.api.billing as billing_api
    import app.main as main
    import app.services.auth as auth
    import app.services.user_store as us
    for module in (us, accounts_api, billing_api, auth, main):
        monkeypatch.setattr(module, "user_store", store, raising=False)

    app = main.app
    t = _Tenancy(TestClient(app))
    yield t
    t.cleanup()


@pytest.fixture()
def two_tenants(tenancy):
    a = tenancy.tenant("tenanta")
    b = tenancy.tenant("tenantb")
    return tenancy, a, b


# ══════════════════════════════════════════════════════════
#  Auth is actually on
# ══════════════════════════════════════════════════════════

def test_health_reports_that_auth_is_required(tenancy):
    assert tenancy.client.get("/api/health").json()["auth_required"] is True


def test_an_unauthenticated_request_is_refused(tenancy):
    assert tenancy.client.get("/api/datasets").status_code == 401


def test_a_wrong_admin_key_cannot_create_an_account(tenancy):
    res = tenancy.client.post(
        "/api/admin/users",
        json={"username": "x", "password": "password_x1"},
        headers={"Authorization": "Bearer wrong-key"})
    assert res.status_code == 401


def test_a_wrong_password_is_refused(tenancy):
    name, _password, _headers, _ds = tenancy.tenant("tenanta")
    res = tenancy.client.post("/api/auth/login",
                              json={"username": name, "password": "nope"})
    assert res.status_code == 401


# ══════════════════════════════════════════════════════════
#  Isolation
# ══════════════════════════════════════════════════════════

def test_each_tenant_lists_only_their_own_datasets(two_tenants):
    t, (_, _, a_headers, a_ds), (_, _, b_headers, b_ds) = two_tenants

    a_list = [d["dataset_id"]
              for d in t.client.get("/api/datasets", headers=a_headers).json()["datasets"]]
    b_list = [d["dataset_id"]
              for d in t.client.get("/api/datasets", headers=b_headers).json()["datasets"]]

    assert a_list == [a_ds]
    assert b_list == [b_ds]


@pytest.mark.parametrize("route", [
    "get-meta", "preview", "profile", "eda", "kpis", "chat", "csv", "delete",
])
def test_no_route_reaches_another_tenants_dataset(two_tenants, route):
    """404, never 403. A 403 confirms the dataset exists, which is
    itself a leak — it tells one client another client holds a
    particular id."""
    t, (_, _, _a_headers, a_ds), (_, _, b_headers, _b_ds) = two_tenants
    c = t.client

    attempt = {
        "get-meta": lambda: c.get("/api/datasets/{}".format(a_ds), headers=b_headers),
        "preview":  lambda: c.get("/api/datasets/{}/preview".format(a_ds), headers=b_headers),
        "profile":  lambda: c.get("/api/datasets/{}/profile".format(a_ds), headers=b_headers),
        "eda":      lambda: c.get("/api/analytics/{}/eda".format(a_ds), headers=b_headers),
        "kpis":     lambda: c.post("/api/charts/{}/kpis".format(a_ds),
                                   json={"filters": []}, headers=b_headers),
        "chat":     lambda: c.post("/api/chat/{}".format(a_ds),
                                   json={"message": "hi"}, headers=b_headers),
        "csv":      lambda: c.get("/api/reports/{}/csv".format(a_ds), headers=b_headers),
        "delete":   lambda: c.delete("/api/datasets/{}".format(a_ds), headers=b_headers),
    }[route]()

    assert attempt.status_code == 404, \
        "{} leaked tenant_a's dataset to tenant_b".format(route)


def test_the_dataset_survives_every_one_of_those_attempts(two_tenants):
    """Including the delete. A refused delete that deleted anyway would
    pass every assertion above."""
    t, (_, _, a_headers, a_ds), (_, _, b_headers, _) = two_tenants

    for call in (lambda: t.client.delete("/api/datasets/{}".format(a_ds),
                                         headers=b_headers),):
        call()

    assert t.client.get("/api/datasets/{}".format(a_ds),
                        headers=a_headers).status_code == 200


def test_a_knowledge_base_is_private_too(two_tenants):
    t, (_, _, a_headers, _), (_, _, b_headers, _) = two_tenants

    res = t.client.post("/api/rag/kb", json={"name": "Tenant A KB"},
                        headers=a_headers)
    assert res.status_code == 200
    a_kb = res.json()["kb_id"]

    assert t.client.get("/api/rag/kb/{}".format(a_kb),
                        headers=b_headers).status_code == 404
    assert t.client.get("/api/rag/kb",
                        headers=b_headers).json()["knowledge_bases"] == []


def test_a_client_token_does_not_reach_the_admin_routes(two_tenants):
    t, (_, _, a_headers, _), _ = two_tenants
    assert t.client.get("/api/admin/users", headers=a_headers).status_code == 401


# ══════════════════════════════════════════════════════════
#  Offboarding
# ══════════════════════════════════════════════════════════

def test_offboarding_takes_the_account_and_its_data_with_it(tenancy):
    """A client who leaves must leave nothing behind — that is the
    promise that makes holding their data acceptable at all."""
    name, password, headers, ds_id = tenancy.tenant("tenantc")
    kb_id = tenancy.client.post("/api/rag/kb", json={"name": "KB"},
                                headers=headers).json()["kb_id"]

    res = tenancy.client.delete("/api/admin/users/" + name,
                                headers=tenancy.admin)

    assert res.status_code == 200
    assert res.json()["datasets_removed"] == [ds_id]
    assert res.json()["knowledge_bases_removed"] == [kb_id]
    assert tenancy.client.post(
        "/api/auth/login",
        json={"username": name, "password": password}).status_code == 401
    assert tenancy.client.get("/api/datasets",
                              headers=headers).status_code == 401, \
        "the offboarded account's token still worked"


# ══════════════════════════════════════════════════════════
#  The fixture's own footing
# ══════════════════════════════════════════════════════════

def test_every_module_holding_a_user_store_is_redirected_by_the_fixture():
    """`from ... import user_store` binds a name per module, so a test
    that redirects the store has to redirect every one of them.

    app.main was missing from the list, and it is the module that serves
    /api/admin/users — so an account was created in the real store while
    the middleware checked the temporary one, and every authenticated
    request came back 401. Nothing about that pointed at the fixture.
    """
    import ast
    import os

    app_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")

    holders = set()
    for root, _dirs, files in os.walk(app_dir):
        if "__pycache__" in root:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
            # Module level only. An import inside a function re-reads
            # the attribute on every call, so redirecting the store's own
            # module is enough for those; a module-level one binds a name
            # once, at import, and has to be redirected by name.
            for node in tree.body:
                if (isinstance(node, ast.ImportFrom)
                        and node.module == "app.services.user_store"
                        and any(a.name == "user_store" for a in node.names)):
                    rel = os.path.relpath(path, os.path.dirname(app_dir))
                    holders.add(rel[:-3].replace(os.sep, "."))

    redirected = {"app.services.user_store", "app.api.accounts",
                  "app.api.billing", "app.services.auth", "app.main"}

    assert holders <= redirected, (
        "these modules import the user_store singleton but no test fixture "
        "redirects them: " + ", ".join(sorted(holders - redirected)))
