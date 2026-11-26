"""
Credits, sign-up and payments.

The provider APIs are replaced by a fake (`billing._http`), so these tests
check this app's side of each flow: what it charges, when it adds credits,
and that it refuses forged, mismatched or someone else's payments.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import time

import httpx
import pytest

from app.config import config
from app.services import billing
from app.services.credits import credit_store
from app.services.user_store import user_store

PASSWORD = "plenty-long-pw-9"


@pytest.fixture()
def world(client, tmp_path, monkeypatch):
    """A server with sign-in on, an empty credit store and two accounts."""
    from app import main
    monkeypatch.setattr(config, "app_admin_key", "billing-admin-key")
    monkeypatch.setattr(config, "free_daily_credits", 5)
    monkeypatch.setattr(config, "free_users", "")
    monkeypatch.setattr(config, "credit_costs", "")
    monkeypatch.setattr(config, "credit_packs", "")
    monkeypatch.setattr(main, "signup_limiter", main._SignupLimiter())
    from app.services.invoices import invoice_book
    credit_store.open(str(tmp_path / "billing"))
    billing.order_book.open(str(tmp_path / "billing"))
    billing.upi_queue.open(str(tmp_path / "billing"))
    invoice_book.open(str(tmp_path / "billing"))
    made = []

    def make(name):
        user_store.create(name, PASSWORD, email=f"{name}@example.com")
        made.append(name)
        tok = client.post("/api/auth/login", json={"username": name, "password": PASSWORD}).json()
        return {"Authorization": "Bearer " + tok["token"]}

    yield make
    for name in made + ["newbie", "newbie2"]:
        user_store.delete(name)
    credit_store.open()
    billing.order_book.open()
    billing.upi_queue.open()
    invoice_book.open()


ADMIN = {"Authorization": "Bearer billing-admin-key"}


def _upload(client, h, hr_csv_bytes):
    r = client.post("/api/datasets/upload", headers=h,
                    files={"file": ("hr.csv", io.BytesIO(hr_csv_bytes), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["meta"]["dataset_id"]


# ── credits ──────────────────────────────────────────────────

def test_free_credits_are_spent_then_the_action_is_refused_with_402(client, world, hr_csv_bytes):
    h = world("cred_a")
    ds = _upload(client, h, hr_csv_bytes)        # uploading is free
    assert client.get("/api/billing", headers=h).json()["balance"]["free_left"] == 5
    for _ in range(2):                            # 2 credits each
        assert client.post(f"/api/reports/{ds}/pdf", headers=h, json={}).status_code == 200
    bal = client.get("/api/billing", headers=h).json()["balance"]
    assert bal["free_left"] == 1 and bal["total"] == 1
    r = client.post(f"/api/reports/{ds}/pdf", headers=h, json={})
    assert r.status_code == 402 and "Billing" in r.json()["detail"]
    # Looking at the data stays free with no credits left.
    assert client.get(f"/api/analytics/{ds}/eda", headers=h).status_code == 200


def test_a_failed_action_gives_its_credits_back(client, world):
    h = world("cred_b")
    r = client.post("/api/reports/doesnotexist/pdf", headers=h, json={})
    assert r.status_code == 404
    bal = client.get("/api/billing", headers=h).json()["balance"]
    assert bal["free_left"] == 5, "charged for a report that was never made"
    hist = client.get("/api/billing", headers=h).json()["history"]
    assert [e["kind"] for e in hist[:2]] == ["refund", "spend"]


def test_free_credits_renew_each_day_and_bought_ones_stay(world, monkeypatch):
    world("cred_c")
    credit_store.grant("cred_c", 10, "test")
    for _ in range(5):
        credit_store.spend("cred_c", "chat")
    credit_store.spend("cred_c", "chat")          # first bought credit
    assert credit_store.balance("cred_c")["paid"] == 9
    from app.services import credits
    monkeypatch.setattr(credits, "_today", lambda: "2099-01-01")
    bal = credit_store.balance("cred_c")
    assert bal["free_left"] == 5 and bal["paid"] == 9


def test_unlimited_accounts_are_never_charged(client, world, hr_csv_bytes, monkeypatch):
    h = world("owner_me")
    ds = _upload(client, h, hr_csv_bytes)
    r = client.post("/api/admin/users/owner_me/unlimited", headers=ADMIN, json={"unlimited": True})
    assert r.status_code == 200 and r.json()["unlimited"]
    for _ in range(4):
        assert client.post(f"/api/reports/{ds}/pdf", headers=h, json={}).status_code == 200
    assert client.get("/api/billing", headers=h).json()["balance"]["free_left"] == 5
    # And by setting, without the admin screen.
    world("partner")
    monkeypatch.setattr(config, "free_users", "partner, someone")
    assert credit_store.is_unlimited("partner")


def test_admin_can_give_credits(client, world):
    world("cred_d")
    r = client.post("/api/admin/users/cred_d/credits", headers=ADMIN,
                    json={"amount": 25, "note": "UPI transfer checked by hand"})
    assert r.status_code == 200 and r.json()["paid"] == 25
    h = {"Authorization": "Bearer nope"}
    assert client.post("/api/admin/users/cred_d/credits", headers=h,
                       json={"amount": 25}).status_code == 401


def test_a_deleted_account_does_not_pass_its_credits_to_a_new_one_of_that_name(client, world):
    world("reused")
    credit_store.grant("reused", 100, "test")
    assert client.delete("/api/admin/users/reused", headers=ADMIN).status_code == 200
    user_store.create("reused", PASSWORD)
    assert credit_store.balance("reused")["paid"] == 0


def test_mcp_report_tool_is_charged(world, hr_csv_bytes, client):
    """The Claude connector must not be a free way round the prices."""
    h = world("mcp_user")
    ds = _upload(client, h, hr_csv_bytes)
    from app.services.credits import OutOfCredits, charging
    for _ in range(2):
        with charging("mcp_user", "pdf_report"):
            pass
    with pytest.raises(OutOfCredits):
        with charging("mcp_user", "pdf_report"):
            pass
    import inspect
    from app import mcp_server
    src = inspect.getsource(mcp_server.generate_report) + inspect.getsource(mcp_server.train_model)
    assert src.count("charging(owner") == 3
    assert ds


# ── sign-up ──────────────────────────────────────────────────

def test_signup_creates_an_account_with_free_credits(client, world):
    world("someone_existing")         # sign-in mode on
    r = client.post("/api/auth/signup", json={"username": "newbie", "password": PASSWORD,
                                              "email": "n@example.com"})
    assert r.status_code == 200, r.text
    h = {"Authorization": "Bearer " + r.json()["token"]}
    assert client.get("/api/account", headers=h).json()["credits"]["free_left"] == 5
    assert user_store.get("newbie").email == "n@example.com"
    dup = client.post("/api/auth/signup", json={"username": "newbie", "password": PASSWORD})
    assert dup.status_code == 422


def test_signup_is_rate_limited_per_address(client, world):
    world("someone_existing2")
    codes = [client.post("/api/auth/signup", json={"username": f"x{i}a", "password": "x"}).status_code
             for i in range(5)]
    assert all(c == 422 for c in codes)       # bad passwords, but counted
    r = client.post("/api/auth/signup", json={"username": "newbie2", "password": PASSWORD})
    assert r.status_code == 429


def test_signup_can_be_switched_off(client, world, monkeypatch):
    monkeypatch.setattr(config, "signup_enabled", False)
    r = client.post("/api/auth/signup", json={"username": "newbie", "password": PASSWORD})
    assert r.status_code == 403


# ── a fake payment provider ─────────────────────────────────

class FakeProviders:
    def __init__(self):
        self.payments = {}      # razorpay payment id -> entity
        self.sessions = {}      # stripe session id -> object
        self.paypal = {}        # paypal order id -> status
        self.calls = []

    def __call__(self, method, url, **kw):
        self.calls.append((method, url))
        req = httpx.Request(method, url)

        def out(data, code=200):
            return httpx.Response(code, json=data, request=req)
        if url.endswith("/v1/orders") and "razorpay" in url:
            return out({"id": "order_TEST1", "amount": kw["json"]["amount"]})
        if "razorpay.com/v1/payments/" in url and url.endswith("/capture"):
            pid = url.split("/")[-2]
            self.payments[pid]["status"] = "captured"
            return out(self.payments[pid])
        if "razorpay.com/v1/payments/" in url:
            return out(self.payments[url.rsplit("/", 1)[1]])
        if "razorpay.com/v1/orders/" in url and url.endswith("/payments"):
            oid = url.split("/")[-2]
            return out({"items": [p for p in self.payments.values() if p["order_id"] == oid]})
        if url.endswith("/v1/checkout/sessions"):
            return out({"id": "cs_TEST1", "url": "https://checkout.stripe.test/cs_TEST1"})
        if "/v1/checkout/sessions/" in url:
            return out(self.sessions[url.rsplit("/", 1)[1]])
        if url.endswith("/v1/oauth2/token"):
            return out({"access_token": "tok"})
        if url.endswith("/v2/checkout/orders"):
            return out({"id": "PP1", "links": [{"rel": "payer-action",
                                                "href": "https://paypal.test/approve"}]})
        if "/v2/checkout/orders/" in url and url.endswith("/capture"):
            oid = url.split("/")[-2]
            state = self.paypal.get(oid, "APPROVED")
            if state == "CAPTURED":
                return httpx.Response(422, text='{"name":"ORDER_ALREADY_CAPTURED"}', request=req)
            if state == "CREATED":
                return httpx.Response(422, text='{"name":"ORDER_NOT_APPROVED"}', request=req)
            self.paypal[oid] = "CAPTURED"
            return out(self._pp(oid))
        if "/v2/checkout/orders/" in url:
            return out(self._pp(url.rsplit("/", 1)[1]))
        raise AssertionError("unexpected call " + url)

    def _pp(self, oid, value="5.00"):
        return {"status": "COMPLETED", "purchase_units": [{"payments": {"captures": [
            {"id": "CAP1", "status": "COMPLETED",
             "amount": {"value": value, "currency_code": "USD"}}]}}]}


@pytest.fixture()
def fake(monkeypatch):
    f = FakeProviders()
    monkeypatch.setattr(billing, "_http", f)
    monkeypatch.setattr(config, "razorpay_key_id", "rzp_test_x")
    monkeypatch.setattr(config, "razorpay_key_secret", "rzp-secret")
    monkeypatch.setattr(config, "razorpay_webhook_secret", "rzp-hook")
    monkeypatch.setattr(config, "stripe_secret_key", "sk_test_x")
    monkeypatch.setattr(config, "stripe_webhook_secret", "whsec_x")
    monkeypatch.setattr(config, "stripe_currency", "usd")
    monkeypatch.setattr(config, "paypal_client_id", "pp-id")
    monkeypatch.setattr(config, "paypal_client_secret", "pp-secret")
    return f


def _rzp_sig(order_id, payment_id, secret="rzp-secret"):
    return hmac.new(secret.encode(), f"{order_id}|{payment_id}".encode(),
                    hashlib.sha256).hexdigest()


def test_plans_are_public_and_show_which_providers_are_on(client, world, fake):
    world("plans_user")
    r = client.get("/api/billing/plans")
    assert r.status_code == 200
    body = r.json()
    assert [p["credits"] for p in body["packs"]] == [50, 150, 600]
    assert {k: body["providers"][k] for k in ("razorpay", "stripe", "paypal")} == \
        {"razorpay": True, "stripe": True, "paypal": True}


def test_razorpay_upi_payment_adds_credits_once(client, world, fake):
    h = world("rzp_user")
    r = client.post("/api/billing/checkout", headers=h, json={"provider": "razorpay", "pack": "starter"})
    assert r.status_code == 200, r.text
    assert r.json()["amount"] == 19900 and r.json()["currency"] == "INR"
    fake.payments["pay_1"] = {"id": "pay_1", "order_id": "order_TEST1", "status": "authorized",
                              "amount": 19900, "currency": "INR", "method": "upi"}
    body = {"order_id": "order_TEST1", "payment_id": "pay_1",
            "signature": _rzp_sig("order_TEST1", "pay_1")}
    r = client.post("/api/billing/razorpay/verify", headers=h, json=body)
    assert r.status_code == 200 and r.json()["credited"] == 50
    assert fake.payments["pay_1"]["status"] == "captured", "authorised payment was not captured"
    # The same confirmation again, and the webhook for it, add nothing.
    assert client.post("/api/billing/razorpay/verify", headers=h, json=body).json()["credited"] == 0
    hook = json.dumps({"event": "order.paid", "payload": {"payment": {"entity": fake.payments["pay_1"]}}}).encode()
    sig = hmac.new(b"rzp-hook", hook, hashlib.sha256).hexdigest()
    r = client.post("/api/billing/webhook/razorpay", content=hook,
                    headers={"X-Razorpay-Signature": sig})
    assert r.status_code == 200 and r.json()["credited"] is False
    assert credit_store.balance("rzp_user")["paid"] == 50


def test_razorpay_forged_signature_or_someone_elses_order_adds_nothing(client, world, fake):
    h = world("rzp_owner")
    other = world("rzp_thief")
    client.post("/api/billing/checkout", headers=h, json={"provider": "razorpay", "pack": "pro"})
    fake.payments["pay_2"] = {"id": "pay_2", "order_id": "order_TEST1", "status": "captured",
                              "amount": 49900, "currency": "INR"}
    forged = {"order_id": "order_TEST1", "payment_id": "pay_2", "signature": "0" * 64}
    assert client.post("/api/billing/razorpay/verify", headers=h, json=forged).status_code == 400
    good = dict(forged, signature=_rzp_sig("order_TEST1", "pay_2"))
    assert client.post("/api/billing/razorpay/verify", headers=other, json=good).status_code == 404
    assert credit_store.balance("rzp_thief")["paid"] == 0
    bad_hook = client.post("/api/billing/webhook/razorpay", content=b"{}",
                           headers={"X-Razorpay-Signature": "x"})
    assert bad_hook.status_code == 400


def test_a_payment_for_less_than_the_pack_price_adds_nothing(client, world, fake):
    h = world("rzp_cheap")
    client.post("/api/billing/checkout", headers=h, json={"provider": "razorpay", "pack": "business"})
    fake.payments["pay_3"] = {"id": "pay_3", "order_id": "order_TEST1", "status": "captured",
                              "amount": 100, "currency": "INR"}
    r = client.post("/api/billing/razorpay/verify", headers=h, json={
        "order_id": "order_TEST1", "payment_id": "pay_3", "signature": _rzp_sig("order_TEST1", "pay_3")})
    assert r.status_code == 409
    assert credit_store.balance("rzp_cheap")["paid"] == 0


def test_stripe_checkout_credits_only_a_paid_session(client, world, fake):
    h = world("stripe_user")
    r = client.post("/api/billing/checkout", headers=h, json={"provider": "stripe", "pack": "starter"})
    assert r.json()["url"].startswith("https://checkout.stripe.test")
    fake.sessions["cs_TEST1"] = {"id": "cs_TEST1", "payment_status": "unpaid",
                                 "amount_total": 500, "currency": "usd"}
    assert client.post("/api/billing/stripe/confirm", headers=h,
                       json={"session_id": "cs_TEST1"}).status_code == 402
    fake.sessions["cs_TEST1"]["payment_status"] = "paid"
    r = client.post("/api/billing/stripe/confirm", headers=h, json={"session_id": "cs_TEST1"})
    assert r.json()["credited"] == 50
    assert credit_store.balance("stripe_user")["paid"] == 50


def test_stripe_webhook_checks_the_signature_and_its_age(client, world, fake):
    h = world("stripe_hook")
    client.post("/api/billing/checkout", headers=h, json={"provider": "stripe", "pack": "pro"})
    event = {"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_TEST1", "payment_status": "paid", "amount_total": 1200, "currency": "usd"}}}
    body = json.dumps(event).encode()
    t = str(int(time.time()))
    sig = hmac.new(b"whsec_x", t.encode() + b"." + body, hashlib.sha256).hexdigest()
    bad = client.post("/api/billing/webhook/stripe", content=body,
                      headers={"Stripe-Signature": f"t={t},v1={'0' * 64}"})
    assert bad.status_code == 400
    old_t = str(int(time.time()) - 3600)
    old_sig = hmac.new(b"whsec_x", old_t.encode() + b"." + body, hashlib.sha256).hexdigest()
    replay = client.post("/api/billing/webhook/stripe", content=body,
                         headers={"Stripe-Signature": f"t={old_t},v1={old_sig}"})
    assert replay.status_code == 400
    ok = client.post("/api/billing/webhook/stripe", content=body,
                     headers={"Stripe-Signature": f"t={t},v1={sig}"})
    assert ok.status_code == 200 and ok.json()["credited"] is True
    assert credit_store.balance("stripe_hook")["paid"] == 150


def test_paypal_is_captured_by_the_server_and_credited_once(client, world, fake):
    h = world("pp_user")
    r = client.post("/api/billing/checkout", headers=h, json={"provider": "paypal", "pack": "starter"})
    assert r.json()["url"] == "https://paypal.test/approve"
    fake.paypal["PP1"] = "CREATED"
    assert client.post("/api/billing/paypal/confirm", headers=h,
                       json={"order_id": "PP1"}).status_code == 402
    fake.paypal["PP1"] = "APPROVED"
    assert client.post("/api/billing/paypal/confirm", headers=h,
                       json={"order_id": "PP1"}).json()["credited"] == 50
    again = client.post("/api/billing/paypal/confirm", headers=h, json={"order_id": "PP1"})
    assert again.status_code == 200 and again.json()["credited"] == 0
    assert credit_store.balance("pp_user")["paid"] == 50


def test_a_buyer_who_closed_the_tab_is_credited_on_the_next_visit(client, world, fake):
    h = world("tab_closer")
    client.post("/api/billing/checkout", headers=h, json={"provider": "razorpay", "pack": "starter"})
    fake.payments["pay_9"] = {"id": "pay_9", "order_id": "order_TEST1", "status": "captured",
                              "amount": 19900, "currency": "INR"}
    r = client.get("/api/billing", headers=h).json()
    assert r["recovered"] == 50 and r["balance"]["paid"] == 50
    assert client.get("/api/billing", headers=h).json()["recovered"] == 0


def test_a_provider_without_keys_is_refused(client, world, monkeypatch):
    h = world("nokeys")
    monkeypatch.setattr(config, "stripe_secret_key", "")
    r = client.post("/api/billing/checkout", headers=h, json={"provider": "stripe", "pack": "starter"})
    assert r.status_code == 400
    r = client.post("/api/billing/checkout", headers=h, json={"provider": "stripe", "pack": "nope"})
    assert r.status_code in (400, 404)


# ── invoices ─────────────────────────────────────────────────

def _buy_with_razorpay(client, h, fake, pay_id="pay_inv", pack="starter", amount=19900):
    client.post("/api/billing/checkout", headers=h, json={"provider": "razorpay", "pack": pack})
    fake.payments[pay_id] = {"id": pay_id, "order_id": "order_TEST1", "status": "captured",
                             "amount": amount, "currency": "INR"}
    return client.post("/api/billing/razorpay/verify", headers=h, json={
        "order_id": "order_TEST1", "payment_id": pay_id,
        "signature": _rzp_sig("order_TEST1", pay_id)})


def test_a_purchase_gets_one_numbered_invoice_with_gst_shown(client, world, fake, monkeypatch):
    import pypdf
    monkeypatch.setattr(config, "gstin", "27ABCDE1234F1Z5")
    monkeypatch.setattr(config, "gst_rate", 18.0)
    h = world("inv_user")
    assert _buy_with_razorpay(client, h, fake).status_code == 200
    # Confirmed twice (return + webhook): still one invoice.
    _buy_with_razorpay(client, h, fake)
    invs = client.get("/api/billing", headers=h).json()["invoices"]
    assert len(invs) == 1 and invs[0]["tax_invoice"]
    from app.services.invoices import invoice_book
    inv = invoice_book.get(invs[0]["number"])
    assert inv["taxable"] + inv["gst"] == 19900 and inv["taxable"] == 16864
    pdf = client.get("/api/billing/invoices/" + invs[0]["number"], headers=h)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    text = "".join(pg.extract_text() for pg in pypdf.PdfReader(io.BytesIO(pdf.content)).pages)
    assert "Tax Invoice" in text and "27ABCDE1234F1Z5" in text and "199.00" in text
    other = world("inv_other")
    assert client.get("/api/billing/invoices/" + invs[0]["number"], headers=other).status_code == 404


def test_invoice_numbers_run_in_order_and_without_gst_it_is_a_receipt(client, world, fake, monkeypatch):
    monkeypatch.setattr(config, "gstin", "")
    from app.services.invoices import invoice_book
    a = invoice_book.issue(user="x", email="", description="d", credits=1, amount=100,
                           currency="INR", provider="razorpay", ref="r1")
    b = invoice_book.issue(user="x", email="", description="d", credits=1, amount=100,
                           currency="INR", provider="razorpay", ref="r2")
    assert int(b["number"][-5:]) == int(a["number"][-5:]) + 1
    assert not a["tax_invoice"] and a["gst"] == 0


# ── manual UPI ───────────────────────────────────────────────

def test_upi_transfer_is_credited_only_when_an_admin_approves_it(client, world, monkeypatch):
    monkeypatch.setattr(config, "upi_id", "shop@okaxis")
    h = world("upi_buyer")
    plans = client.get("/api/billing/plans").json()
    assert plans["providers"]["upi"] and plans["upi"]["links"]["starter"].startswith("upi://pay?pa=shop%40okaxis")
    bad = client.post("/api/billing/upi", headers=h, json={"pack": "starter", "utr": "12345"})
    assert bad.status_code == 422
    r = client.post("/api/billing/upi", headers=h, json={"pack": "starter", "utr": "4123 5678 9012"})
    assert r.status_code == 200 and r.json()["status"] == "pending"
    assert credit_store.balance("upi_buyer")["paid"] == 0
    dup = client.post("/api/billing/upi", headers=world("upi_other"),
                      json={"pack": "starter", "utr": "412356789012"})
    assert dup.status_code == 409
    pending = client.get("/api/admin/upi-payments", headers=ADMIN).json()["pending"]
    assert len(pending) == 1
    assert client.post(f"/api/admin/upi-payments/{pending[0]['id']}",
                       headers=h, json={"approve": True}).status_code == 401
    ok = client.post(f"/api/admin/upi-payments/{pending[0]['id']}", headers=ADMIN,
                     json={"approve": True})
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    assert credit_store.balance("upi_buyer")["paid"] == 50
    assert client.get("/api/billing", headers=h).json()["invoices"]
    again = client.post(f"/api/admin/upi-payments/{pending[0]['id']}", headers=ADMIN,
                        json={"approve": True})
    assert again.status_code == 409 and credit_store.balance("upi_buyer")["paid"] == 50


def test_a_rejected_upi_report_adds_nothing(client, world, monkeypatch):
    monkeypatch.setattr(config, "upi_id", "shop@okaxis")
    h = world("upi_fake")
    item = client.post("/api/billing/upi", headers=h, json={"pack": "pro", "utr": "999988887777"}).json()
    r = client.post(f"/api/admin/upi-payments/{item['id']}", headers=ADMIN,
                    json={"approve": False, "note": "not found in bank statement"})
    assert r.json()["status"] == "rejected"
    assert credit_store.balance("upi_fake")["paid"] == 0
