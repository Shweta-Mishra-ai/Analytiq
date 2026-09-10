"""
Becoming a customer, and what that customer is allowed to do.

Accounts could only be created by an operator holding the admin key, so
nobody could start using Analytiq on their own. Every other commercial
question was moot until that changed, and nothing metered usage either:
an account could upload any number of files of any size and train any
number of models on somebody else's compute.

Two rules shape the tests below, and both are about failing safely:

  * A self-hosted deployment is NOT metered. Capping the container its
    own operator pays for serves nobody, so signup being off means
    every account is unlimited whatever its stored plan says.
  * Billing CHANGES a plan, it never grants one. Stripe being absent,
    broken or slow must never take a customer's data away from them.
"""
import importlib
import json
import sys
import tempfile
import types

import pytest

from app.services import plans


# ── the tiers themselves ──────────────────────────────────

def test_every_public_plan_is_real():
    for key in plans.PUBLIC_PLANS:
        assert key in plans.PLANS


def test_a_retired_plan_does_not_lock_an_account_out():
    """A tier removed between a subscription starting and this deploy
    must degrade, not raise: the account still owns its data."""
    assert plans.get("a-tier-that-no-longer-exists").key == plans.FALLBACK_PLAN


def test_a_self_hosted_deployment_is_never_metered():
    assert plans.resolve("free", metered=False).key == plans.SELF_HOSTED_PLAN
    assert plans.resolve("free", metered=False).datasets is None


def test_zero_and_none_mean_different_things():
    """None is "no ceiling"; zero is "not included". Collapsing them
    would either give the free tier scheduled delivery or cap the
    unlimited tier at nothing."""
    assert plans.get("free").allows("scheduled_reports") is False
    assert plans.get("solo").allows("scheduled_reports") is True
    assert plans.get("unlimited").allows("scheduled_reports") is True
    assert plans.get("unlimited").scheduled_reports is None


def test_the_free_tier_is_enough_to_judge_the_product_on():
    """A trial that cannot hold a real file tells the user nothing."""
    free = plans.get("free")
    assert free.rows_per_dataset >= 10_000
    assert free.reports_per_month >= 3


def test_no_analysis_is_behind_a_paywall():
    """Plans cap volume, never correctness. Gating the statistics would
    make the free tier a demo that lies, which is the opposite of what
    this product sells."""
    for plan in plans.PLANS.values():
        for field in ("datasets", "rows_per_dataset", "reports_per_month"):
            value = plan.limit(field)
            assert value is None or value > 0, (plan.key, field)


# ── usage counting ────────────────────────────────────────

def test_usage_counts_per_account_and_period():
    from app.services.usage import UsageStore
    import os
    store = UsageStore(os.path.join(tempfile.mkdtemp(), "usage.json"))
    assert store.record("amy", "reports") == 1
    assert store.record("amy", "reports") == 2
    assert store.used("amy", "reports") == 2
    assert store.used("bob", "reports") == 0


def test_an_unknown_event_is_not_counted():
    from app.services.usage import UsageStore
    import os
    store = UsageStore(os.path.join(tempfile.mkdtemp(), "usage.json"))
    assert store.record("amy", "not-a-thing") == 0
    assert store.snapshot("amy") == {"reports": 0, "models": 0}


def test_an_unreadable_usage_file_does_not_stop_the_app():
    """Losing a counter resets somebody's month, which is generous.
    Refusing to start is not."""
    import os
    from app.services.usage import UsageStore
    path = os.path.join(tempfile.mkdtemp(), "usage.json")
    with open(path, "w") as f:
        f.write("{ this is not json")
    assert UsageStore(path).snapshot("amy") == {"reports": 0, "models": 0}


# ── the account file across a deploy ──────────────────────

def test_an_account_written_before_plans_existed_still_loads():
    """users.json on a running deployment predates every field added
    here. A KeyError would lock every existing client out."""
    import base64
    import os
    import time
    from app.services.user_store import UserStore

    path = os.path.join(tempfile.mkdtemp(), "users.json")
    with open(path, "w") as f:
        json.dump({"amy": {
            "username": "amy",
            "salt": base64.b64encode(b"s" * 16).decode(),
            "password_hash": base64.b64encode(b"h" * 32).decode(),
            "created_at": time.time(), "is_admin": False}}, f)

    user = UserStore(path).get("amy")
    assert user is not None
    assert user.plan == ""
    assert user.email == ""


def test_one_account_per_email_address():
    import os
    from app.services.user_store import UserStore
    store = UserStore(os.path.join(tempfile.mkdtemp(), "users.json"))
    store.create("amy", "password123", email="a@b.com")
    with pytest.raises(ValueError, match="email"):
        store.create("bob", "password123", email="A@B.com")


def test_changing_a_password_reissues_the_salt():
    """Reusing the salt tells anyone holding the previous file that the
    password changed but the salt did not."""
    import os
    from app.services.user_store import UserStore
    store = UserStore(os.path.join(tempfile.mkdtemp(), "users.json"))
    store.create("amy", "password123")
    before = store._users["amy"].salt
    store.set_password("amy", "adifferentpassword")
    assert store._users["amy"].salt != before
    assert store.verify("amy", "adifferentpassword")
    assert store.verify("amy", "password123") is None


# ══════════════════════════════════════════════════════════
#  THE ENDPOINTS, ON A METERED DEPLOYMENT
# ══════════════════════════════════════════════════════════

@pytest.fixture()
def metered(monkeypatch, tmp_path):
    """A deployment that has invited strangers: signup on, so metered."""
    from app.config import config
    from app.services import quota, usage
    from app.services.user_store import UserStore

    monkeypatch.setattr(config, "signup_enabled", True, raising=False)
    monkeypatch.setattr(config, "signup_plan", "free", raising=False)
    monkeypatch.setattr(usage, "usage_store",
                        usage.UsageStore(str(tmp_path / "usage.json")))
    monkeypatch.setattr(quota, "usage_store", usage.usage_store)

    store = UserStore(str(tmp_path / "users.json"))
    # Every module that imported the singleton by name holds its own
    # reference, so each has to be pointed at the temporary store. Miss
    # one — the auth middleware in particular — and its is_empty() still
    # answers True, the app stays in single-user open mode, and every
    # request arrives as "local" rather than as the account under test.
    import app.api.accounts as accounts_api
    import app.api.billing as billing_api
    import app.services.auth as auth
    import app.services.user_store as us
    for module in (us, accounts_api, billing_api, auth):
        monkeypatch.setattr(module, "user_store", store, raising=False)
    return store


@pytest.fixture()
def account_name(request):
    """A username unique to this test.

    conftest points DATA_DIR at one directory for the whole session, so
    the dataset store is shared and keyed by owner. A fixed name meant
    the datasets one test uploaded were still there for the next, and a
    row-ceiling test was refused for holding too many DATASETS instead —
    a green assertion away from hiding the thing it was written to check.
    """
    stem = "".join(c for c in request.node.name if c.isalnum())[-24:].lower()
    return "t{}".format(stem or "anon")


@pytest.fixture()
def clean_datasets(account_name):
    """Remove anything this account left behind on a previous run.

    conftest points DATA_DIR at one directory reused across the whole
    session AND across pytest invocations, so a per-test username is not
    enough on its own: the second run of the suite finds the first run's
    uploads still sitting there. A row-ceiling test then fails for
    holding too many DATASETS, which is a green assertion away from
    hiding the thing it was written to check.
    """
    from app.services.dataset_store import store

    def wipe():
        try:
            for meta in store.list_meta(account_name):
                store.delete(account_name, meta.dataset_id)
        except Exception:
            pass

    wipe()
    yield
    wipe()


@pytest.fixture()
def signed_up(client, metered, account_name, clean_datasets):
    r = client.post("/api/auth/signup", json={
        "username": account_name,
        "email": "{}@example.com".format(account_name),
        "password": "password123"})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


# ── signup ────────────────────────────────────────────────

def test_signup_is_refused_unless_the_deployment_asked_for_it(client,
                                                              monkeypatch):
    """The default has to be the safe one: a self-hosted instance that
    quietly accepted strangers is a hole its operator never opened."""
    from app.config import config
    monkeypatch.setattr(config, "signup_enabled", False, raising=False)
    r = client.post("/api/auth/signup", json={
        "username": "mallory", "email": "m@x.com", "password": "password123"})
    assert r.status_code == 403


def test_signup_returns_a_working_token(client, signed_up, account_name):
    me = client.get("/api/account", headers=signed_up)
    assert me.status_code == 200
    assert me.json()["username"] == account_name
    assert me.json()["plan"] == "free"


@pytest.mark.parametrize("payload,status", [
    ({"username": "Has Space", "email": "a@b.com", "password": "password123"}, 422),
    ({"username": "amy2", "email": "not-an-email", "password": "password123"}, 422),
    ({"username": "amy3", "email": "a@b.com", "password": "short"}, 422),
])
def test_a_malformed_signup_is_refused(client, metered, payload, status):
    assert client.post("/api/auth/signup", json=payload).status_code == status


def test_a_taken_email_does_not_confirm_the_address(client, signed_up,
                                                    account_name):
    """Answering "that address is registered" turns signup into a way to
    ask whether somebody is a customer."""
    taken = "{}@example.com".format(account_name)
    r = client.post("/api/auth/signup", json={
        "username": "someoneelse", "email": taken,
        "password": "password123"})
    assert r.status_code == 409
    assert taken not in r.json()["detail"]


def test_the_password_reset_path_does_not_pretend_to_work(client, metered):
    """Inventing a token nobody can receive is worse than saying the
    path is unfinished."""
    r = client.post("/api/auth/password-reset", json={"email": "a@b.com"})
    assert r.status_code == 200
    assert r.json()["sent"] is False


# ── ceilings ──────────────────────────────────────────────

def test_a_file_over_the_row_ceiling_is_refused_with_the_number(
        client, signed_up):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(3)
    n = 30_000
    csv = pd.DataFrame({"g": rng.choice(list("ABC"), n),
                        "v": rng.normal(0, 1, n)}).to_csv(index=False)
    r = client.post("/api/datasets/upload", headers=signed_up,
                    files={"file": ("big.csv", csv.encode(), "text/csv")})
    assert r.status_code == 402, r.text
    detail = r.json()["detail"]
    assert detail["limit"] == "rows_per_dataset"
    assert detail["allowed"] == plans.get("free").rows_per_dataset
    assert "25,000" in detail["error"]


def test_the_dataset_ceiling_counts_what_is_actually_held(client, signed_up):
    import pandas as pd
    csv = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(index=False)
    codes = [
        client.post("/api/datasets/upload", headers=signed_up,
                    files={"file": (f"s{i}.csv", csv.encode(), "text/csv")}
                    ).status_code
        for i in range(plans.get("free").datasets + 1)
    ]
    assert codes[:-1] == [200] * plans.get("free").datasets
    assert codes[-1] == 402


def test_a_refusal_is_402_not_403(client, signed_up):
    """The caller can act on this, and the action is a payment. 403 says
    "you may never", which is not what a plan ceiling means."""
    import pandas as pd
    csv = pd.DataFrame({"a": [1, 2, 3]}).to_csv(index=False)
    for i in range(plans.get("free").datasets):
        client.post("/api/datasets/upload", headers=signed_up,
                    files={"file": (f"x{i}.csv", csv.encode(), "text/csv")})
    r = client.post("/api/datasets/upload", headers=signed_up,
                    files={"file": ("over.csv", csv.encode(), "text/csv")})
    assert r.status_code == 402
    assert r.json()["detail"]["upgrade"]


def test_an_unmetered_deployment_has_no_ceilings(client, monkeypatch):
    """The person paying for the container is the person using it."""
    import numpy as np
    import pandas as pd
    from app.config import config
    monkeypatch.setattr(config, "signup_enabled", False, raising=False)
    rng = np.random.default_rng(5)
    n = 30_000
    csv = pd.DataFrame({"g": rng.choice(list("ABC"), n),
                        "v": rng.normal(0, 1, n)}).to_csv(index=False)
    r = client.post("/api/datasets/upload",
                    files={"file": ("big.csv", csv.encode(), "text/csv")})
    assert r.status_code == 200, r.text


# ── billing ───────────────────────────────────────────────

def test_the_price_list_is_public(client):
    r = client.get("/api/billing/plans")
    assert r.status_code == 200
    assert [p["key"] for p in r.json()["plans"]] == list(plans.PUBLIC_PLANS)


def test_checkout_says_so_when_payments_are_not_set_up(client, signed_up):
    r = client.post("/api/billing/checkout", headers=signed_up,
                    json={"plan": "solo"})
    assert r.status_code == 503
    assert "not set up" in r.json()["detail"]


def test_the_webhook_refuses_without_a_signing_secret(client):
    """An unsigned endpoint that moves accounts between plans lets
    anybody move any account onto any plan."""
    r = client.post("/api/billing/webhook", content=b"{}")
    assert r.status_code == 503


def test_the_webhook_is_reachable_without_a_token(client):
    """Stripe cannot hold a bearer token. Left out of the public paths
    the middleware answered 401 and no subscription would ever have been
    applied."""
    r = client.post("/api/billing/webhook", content=b"{}")
    assert r.status_code != 401


# ══════════════════════════════════════════════════════════
#  THE SLOT MUST COME BACK
# ══════════════════════════════════════════════════════════

def _admit_request():
    from starlette.requests import Request
    return Request({"type": "http", "headers": [], "method": "GET",
                    "path": "/x", "query_string": b""})


def _free_slots(gate) -> int:
    """How many slots the gate can still hand out.

    Read behaviourally rather than from a counter: the gate is a
    BoundedSemaphore with no public size, and asserting on a private
    attribute would pass while the thing it stands for was broken.
    """
    taken = []
    try:
        while gate._sem.acquire(blocking=False):
            taken.append(True)
        return len(taken)
    finally:
        for _ in taken:
            gate._sem.release()


def test_a_cancelled_request_still_returns_its_slot(monkeypatch):
    """A client that disconnects mid-report raises CancelledError, and
    that derives from BaseException — so `except Exception` around the
    yield looks equivalent to `finally` and is not. Narrowing it leaked
    the concurrency slot: enough disconnects and every caller gets 503
    until the process restarts, which is the exact failure this module
    exists to prevent.
    """
    import asyncio

    from app.services import load_control
    monkeypatch.setattr(load_control, "guard", lambda *a, **k: None)

    before = _free_slots(load_control.heavy)
    for _ in range(before + 2):          # more cancellations than slots
        gen = load_control.admit("report")(_admit_request())
        next(gen)
        with pytest.raises(asyncio.CancelledError):
            gen.throw(asyncio.CancelledError())
    assert _free_slots(load_control.heavy) == before, "the slot leaked"


def test_a_failed_handler_returns_its_slot_and_is_not_charged(monkeypatch):
    from app.services import load_control
    import app.services.quota as quota

    monkeypatch.setattr(load_control, "guard", lambda *a, **k: None)
    charged = []
    monkeypatch.setattr(quota, "record",
                        lambda owner, event, n=1: charged.append(event))
    monkeypatch.setattr(quota, "check_event", lambda *a, **k: None)

    before = _free_slots(load_control.heavy)
    gen = load_control.admit("report", "reports")(_admit_request())
    next(gen)
    with pytest.raises(RuntimeError):
        gen.throw(RuntimeError("the report failed to build"))
    assert _free_slots(load_control.heavy) == before, "the slot leaked"
    assert charged == [], "a report that failed was still charged for"


def test_a_successful_handler_is_charged_exactly_once(monkeypatch):
    from app.services import load_control
    import app.services.quota as quota

    monkeypatch.setattr(load_control, "guard", lambda *a, **k: None)
    charged = []
    monkeypatch.setattr(quota, "record",
                        lambda owner, event, n=1: charged.append(event))
    monkeypatch.setattr(quota, "check_event", lambda *a, **k: None)

    before = _free_slots(load_control.heavy)
    gen = load_control.admit("report", "reports")(_admit_request())
    next(gen)
    with pytest.raises(StopIteration):
        next(gen)
    assert _free_slots(load_control.heavy) == before
    assert charged == ["reports"]
