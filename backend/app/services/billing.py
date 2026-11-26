"""
services/billing.py — buying credits with Razorpay, Stripe or PayPal.

The rule every path follows: the price comes from the server's own pack
list, never from the browser, and credits are added only after the server
itself has confirmed with the provider that the money was taken — for the
order it created, for this account, for the full amount. A confirmation can
arrive more than once (the buyer's return and the provider's webhook); the
credit ledger adds each payment once (credits.grant with a ref).

  Razorpay  cards, UPI, netbanking, wallets (INR). The checkout runs in the
            page; the server checks Razorpay's signature on the result,
            captures the payment if it was only authorised, and checks the
            amount. The order.paid webhook is a second path.
  Stripe    a hosted Checkout page. On return the server fetches the
            session from Stripe and checks it is paid. The
            checkout.session.completed webhook (signature checked) is a
            second path.
  PayPal    an approval page. On return the server captures the order
            itself; until it does, no money moves, so a buyer who never
            returns is not charged.

Opening the billing page also re-checks this account's recent unpaid
orders with the provider (`reconcile`), so a buyer who paid and closed the
tab before returning still gets their credits.

Provider calls go through `_http`, which the tests replace.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from typing import Optional

import httpx
from fastapi import HTTPException

from app.config import config
from app.services.credits import credit_store

logger = logging.getLogger(__name__)

PENDING_RECHECK_SEC = 2 * 86400

DEFAULT_PACKS = [
    {"id": "starter", "name": "Starter", "credits": 50, "inr": 19900, "usd": 500},
    {"id": "pro", "name": "Pro", "credits": 150, "inr": 49900, "usd": 1200},
    {"id": "business", "name": "Business", "credits": 600, "inr": 149900, "usd": 3500},
]


def packs() -> list[dict]:
    if config.credit_packs:
        try:
            out = json.loads(config.credit_packs)
            assert isinstance(out, list) and out
            for p in out:
                assert {"id", "credits", "inr", "usd"} <= set(p)
                p.setdefault("name", str(p["id"]).title())
                p["credits"], p["inr"], p["usd"] = int(p["credits"]), int(p["inr"]), int(p["usd"])
            return out
        except Exception:
            logger.error("CREDIT_PACKS is not a valid list of packs; using the defaults")
    return [dict(p) for p in DEFAULT_PACKS]


def pack(pack_id: str) -> dict:
    for p in packs():
        if p["id"] == pack_id:
            return p
    raise HTTPException(404, "No such credit pack.")


def providers() -> dict[str, bool]:
    return {
        "upi": bool(config.upi_id),
        "razorpay": bool(config.razorpay_key_id and config.razorpay_key_secret),
        "stripe": bool(config.stripe_secret_key),
        "paypal": bool(config.paypal_client_id and config.paypal_client_secret),
    }


def price_for(provider: str, p: dict) -> tuple[int, str]:
    """(amount in the smallest unit, ISO currency) the provider charges."""
    if provider == "razorpay":
        return p["inr"], "INR"
    if provider == "stripe":
        cur = (config.stripe_currency or "usd").lower()
        return (p["inr"], "INR") if cur == "inr" else (p["usd"], "USD")
    return p["usd"], "USD"


# ── orders the server created ──────────────────────────────

class OrderBook:
    def __init__(self, root: Optional[str] = None):
        self._lock = threading.RLock()
        self.open(root)

    def open(self, root: Optional[str] = None) -> None:
        self.path = os.path.join(root or os.path.join(config.data_dir, "billing"), "orders.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._orders: dict = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self._orders = json.load(f)

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._orders, f)
        os.replace(tmp, self.path)

    def add(self, key: str, **fields):
        with self._lock:
            self._orders[key] = dict(fields, created=time.time(), status="pending")
            self._save()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            o = self._orders.get(key)
            return dict(o) if o else None

    def mark(self, key: str, status: str, **extra):
        with self._lock:
            if key in self._orders:
                self._orders[key].update(status=status, **extra)
                self._save()

    def pending_for(self, username: str) -> list[tuple[str, dict]]:
        now = time.time()
        with self._lock:
            return [(k, dict(o)) for k, o in self._orders.items()
                    if o["user"] == username and o["status"] == "pending"
                    and now - o["created"] < PENDING_RECHECK_SEC]


order_book = OrderBook()


def _fulfil(key: str, order: dict, amount: int, currency: str, provider_ref: str) -> bool:
    """Credit a confirmed payment. The amount and currency the provider
    reports must be what the server asked for."""
    if int(amount) != int(order["amount"]) or currency.upper() != order["currency"].upper():
        logger.error("payment %s amount mismatch: got %s %s, expected %s %s", key, amount,
                     currency, order["amount"], order["currency"])
        order_book.mark(key, "mismatch")
        raise HTTPException(409, "The payment does not match the order. Nothing was "
                                 "credited; please contact support with reference "
                                 f"{provider_ref}.")
    added = credit_store.grant(order["user"], order["credits"],
                               "{} pack via {}".format(order["pack"], order["provider"]),
                               ref=key)
    order_book.mark(key, "paid", provider_ref=provider_ref)
    if added:
        logger.info("credited %s credits to %s for %s", order["credits"], order["user"], key)
        _invoice(order["user"], order["pack"], order["credits"], order["amount"],
                 order["currency"], order["provider"], key)
    return added


def _invoice(user, pack_id, credits, amount, currency, provider, ref):
    """Issue the purchase's invoice. A failure here is logged, never undoes
    the credits: the payment is real either way."""
    try:
        from app.services.invoices import issue_and_send
        name = next((p["name"] for p in packs() if p["id"] == pack_id), str(pack_id))
        issue_and_send(user=user, description="{} credits — {} pack".format(credits, name),
                       credits=credits, amount=amount, currency=currency,
                       provider=provider, ref=ref)
    except Exception:
        logger.exception("invoice for %s could not be issued", ref)


# ── HTTP ────────────────────────────────────────────────────

def _http(method: str, url: str, **kw) -> httpx.Response:
    try:
        return httpx.request(method, url, timeout=20, **kw)
    except httpx.HTTPError as e:
        logger.warning("payment provider unreachable: %s %s: %s", method, url, e)
        raise HTTPException(502, "The payment provider could not be reached. "
                                 "Please try again in a minute.")


def _ok(r: httpx.Response, what: str) -> dict:
    if r.status_code >= 400:
        logger.error("%s failed: %s %s", what, r.status_code, r.text[:500])
        raise HTTPException(502, f"The payment provider refused the request ({what}).")
    return r.json()


def _require(provider: str):
    if not providers().get(provider):
        raise HTTPException(400, f"{provider.title()} payments are not switched on.")


# ── checkout ────────────────────────────────────────────────

def start_checkout(provider: str, pack_id: str, username: str, email: str,
                   base_url: str) -> dict:
    _require(provider)
    p = pack(pack_id)
    amount, currency = price_for(provider, p)
    common = dict(user=username, pack=p["id"], credits=p["credits"], provider=provider,
                  amount=amount, currency=currency)
    label = "{} credits ({})".format(p["credits"], p["name"])
    if provider == "razorpay":
        r = _ok(_http("POST", "https://api.razorpay.com/v1/orders",
                      auth=(config.razorpay_key_id, config.razorpay_key_secret),
                      json={"amount": amount, "currency": currency,
                            "receipt": "aq_" + secrets.token_hex(8),
                            "notes": {"user": username, "pack": p["id"]}}),
                "create Razorpay order")
        order_book.add("razorpay:" + r["id"], **common)
        return {"provider": "razorpay", "key_id": config.razorpay_key_id,
                "order_id": r["id"], "amount": amount, "currency": currency,
                "name": config.app_name, "description": label,
                "prefill": {"email": email} if email else {}}
    if provider == "stripe":
        form = {
            "mode": "payment",
            "line_items[0][quantity]": "1",
            "line_items[0][price_data][currency]": currency.lower(),
            "line_items[0][price_data][unit_amount]": str(amount),
            "line_items[0][price_data][product_data][name]": f"{config.app_name} — {label}",
            "success_url": base_url + "/billing?provider=stripe&session_id={CHECKOUT_SESSION_ID}",
            "cancel_url": base_url + "/billing?cancelled=1",
            "client_reference_id": username,
            "metadata[pack]": p["id"],
        }
        if email:
            form["customer_email"] = email
        r = _ok(_http("POST", "https://api.stripe.com/v1/checkout/sessions",
                      auth=(config.stripe_secret_key, ""), data=form),
                "create Stripe checkout")
        order_book.add("stripe:" + r["id"], **common)
        return {"provider": "stripe", "url": r["url"]}
    if provider == "paypal":
        body = {"intent": "CAPTURE",
                "purchase_units": [{"reference_id": p["id"], "custom_id": username,
                                    "description": label,
                                    "amount": {"currency_code": currency,
                                               "value": "{:.2f}".format(amount / 100)}}],
                "payment_source": {"paypal": {"experience_context": {
                    "brand_name": config.app_name, "user_action": "PAY_NOW",
                    "shipping_preference": "NO_SHIPPING",
                    "return_url": base_url + "/billing?provider=paypal",
                    "cancel_url": base_url + "/billing?cancelled=1"}}}}
        r = _ok(_http("POST", _paypal_base() + "/v2/checkout/orders", json=body,
                      headers=_paypal_headers()), "create PayPal order")
        link = next((l["href"] for l in r.get("links", [])
                     if l.get("rel") in ("payer-action", "approve")), None)
        if not link:
            raise HTTPException(502, "PayPal did not return an approval link.")
        order_book.add("paypal:" + r["id"], **common)
        return {"provider": "paypal", "url": link}
    raise HTTPException(400, "Unknown payment provider.")


def _own(key: str, username: str) -> dict:
    order = order_book.get(key)
    if order is None or order["user"] != username:
        raise HTTPException(404, "No such order for this account.")
    return order


# ── Razorpay ────────────────────────────────────────────────

def _rzp_auth():
    return (config.razorpay_key_id, config.razorpay_key_secret)


def razorpay_verify(username: str, order_id: str, payment_id: str, signature: str) -> dict:
    _require("razorpay")
    key = "razorpay:" + order_id
    order = _own(key, username)
    expected = hmac.new(config.razorpay_key_secret.encode(),
                        f"{order_id}|{payment_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(signature)):
        raise HTTPException(400, "The payment could not be verified.")
    pay = _ok(_http("GET", f"https://api.razorpay.com/v1/payments/{payment_id}",
                    auth=_rzp_auth()), "fetch Razorpay payment")
    return _settle_razorpay_payment(key, order, pay)


def _settle_razorpay_payment(key: str, order: dict, pay: dict) -> dict:
    if pay.get("order_id") != key.split(":", 1)[1]:
        raise HTTPException(400, "That payment belongs to a different order.")
    if pay.get("status") == "authorized":
        pay = _ok(_http("POST", f"https://api.razorpay.com/v1/payments/{pay['id']}/capture",
                        auth=_rzp_auth(),
                        json={"amount": order["amount"], "currency": order["currency"]}),
                  "capture Razorpay payment")
    if pay.get("status") != "captured":
        raise HTTPException(402, "The payment has not gone through yet.")
    added = _fulfil(key, order, pay["amount"], pay["currency"], pay["id"])
    return {"credited": order["credits"] if added else 0, "already": not added}


def razorpay_webhook(body: bytes, signature: str) -> dict:
    if not config.razorpay_webhook_secret:
        raise HTTPException(400, "Razorpay webhooks are not configured.")
    expected = hmac.new(config.razorpay_webhook_secret.encode(), body,
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature or ""):
        raise HTTPException(400, "Bad signature.")
    event = json.loads(body)
    if event.get("event") not in ("order.paid", "payment.captured"):
        return {"ignored": event.get("event")}
    pay = event.get("payload", {}).get("payment", {}).get("entity", {})
    key = "razorpay:" + str(pay.get("order_id"))
    order = order_book.get(key)
    if order is None:
        return {"ignored": "unknown order"}
    if pay.get("status") != "captured":
        return {"ignored": "not captured"}
    added = _fulfil(key, order, pay["amount"], pay["currency"], pay["id"])
    return {"credited": added}


# ── Stripe ──────────────────────────────────────────────────

def stripe_confirm(username: str, session_id: str) -> dict:
    _require("stripe")
    key = "stripe:" + session_id
    order = _own(key, username)
    s = _ok(_http("GET", f"https://api.stripe.com/v1/checkout/sessions/{session_id}",
                  auth=(config.stripe_secret_key, "")), "fetch Stripe session")
    if s.get("payment_status") != "paid":
        raise HTTPException(402, "The payment has not gone through yet.")
    added = _fulfil(key, order, s["amount_total"], s["currency"], s.get("payment_intent") or s["id"])
    return {"credited": order["credits"] if added else 0, "already": not added}


def stripe_webhook(body: bytes, header: str, now: Optional[float] = None) -> dict:
    if not config.stripe_webhook_secret:
        raise HTTPException(400, "Stripe webhooks are not configured.")
    parts = dict(p.split("=", 1) for p in (header or "").split(",") if "=" in p)
    sigs = [p.split("=", 1)[1] for p in (header or "").split(",") if p.startswith("v1=")]
    t = parts.get("t", "")
    if not t.isdigit() or abs((now or time.time()) - int(t)) > 300:
        raise HTTPException(400, "Bad signature.")
    expected = hmac.new(config.stripe_webhook_secret.encode(), t.encode() + b"." + body,
                        hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, s) for s in sigs):
        raise HTTPException(400, "Bad signature.")
    event = json.loads(body)
    if event.get("type") not in ("checkout.session.completed",
                                 "checkout.session.async_payment_succeeded"):
        return {"ignored": event.get("type")}
    s = event["data"]["object"]
    key = "stripe:" + s["id"]
    order = order_book.get(key)
    if order is None or s.get("payment_status") != "paid":
        return {"ignored": "unknown or unpaid"}
    added = _fulfil(key, order, s["amount_total"], s["currency"], s.get("payment_intent") or s["id"])
    return {"credited": added}


# ── PayPal ──────────────────────────────────────────────────

def _paypal_base() -> str:
    return ("https://api-m.paypal.com" if config.paypal_mode.lower() == "live"
            else "https://api-m.sandbox.paypal.com")


def _paypal_headers() -> dict:
    r = _ok(_http("POST", _paypal_base() + "/v1/oauth2/token",
                  auth=(config.paypal_client_id, config.paypal_client_secret),
                  data={"grant_type": "client_credentials"}), "PayPal sign-in")
    return {"Authorization": "Bearer " + r["access_token"],
            "Content-Type": "application/json"}


def paypal_confirm(username: str, order_id: str) -> dict:
    _require("paypal")
    key = "paypal:" + order_id
    order = _own(key, username)
    headers = _paypal_headers()
    r = _http("POST", f"{_paypal_base()}/v2/checkout/orders/{order_id}/capture",
              headers=dict(headers, **{"PayPal-Request-Id": key}))
    if r.status_code == 422 and "ORDER_ALREADY_CAPTURED" in r.text:
        r = _http("GET", f"{_paypal_base()}/v2/checkout/orders/{order_id}", headers=headers)
    if r.status_code == 422:
        raise HTTPException(402, "The payment was not approved on PayPal.")
    data = _ok(r, "capture PayPal order")
    if data.get("status") != "COMPLETED":
        raise HTTPException(402, "The payment has not gone through yet.")
    cap = data["purchase_units"][0]["payments"]["captures"][0]
    if cap.get("status") != "COMPLETED":
        raise HTTPException(402, "PayPal is still processing this payment; your credits "
                                 "will be added when it completes.")
    value = int(round(float(cap["amount"]["value"]) * 100))
    added = _fulfil(key, order, value, cap["amount"]["currency_code"], cap["id"])
    return {"credited": order["credits"] if added else 0, "already": not added}


# ── catching up ─────────────────────────────────────────────

def reconcile(username: str) -> int:
    """Credit this account's recent orders that were paid but never
    confirmed (the buyer closed the tab). PayPal orders are not captured
    here: an order the buyer did not come back to approve is not taken."""
    credited = 0
    for key, order in order_book.pending_for(username):
        provider, ref = key.split(":", 1)
        try:
            if provider == "stripe" and providers()["stripe"]:
                credited += stripe_confirm(username, ref)["credited"]
            elif provider == "razorpay" and providers()["razorpay"]:
                r = _http("GET", f"https://api.razorpay.com/v1/orders/{ref}/payments",
                          auth=_rzp_auth())
                if r.status_code < 400:
                    for pay in r.json().get("items", []):
                        if pay.get("status") in ("captured", "authorized"):
                            credited += _settle_razorpay_payment(key, order, pay)["credited"]
                            break
        except HTTPException:
            continue
    return credited


# ── direct UPI transfers, approved by hand ──────────────────────────────

class UpiQueue:
    """Payments made straight to the business's UPI ID. The app cannot see
    the bank account, so the buyer gives the transaction reference (UTR)
    and an admin approves it after finding the money in the account."""

    MAX_PENDING = 3

    def __init__(self, root: Optional[str] = None):
        self._lock = threading.RLock()
        self.open(root)

    def open(self, root: Optional[str] = None) -> None:
        self.path = os.path.join(root or os.path.join(config.data_dir, "billing"),
                                 "upi_requests.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with self._lock:
            self._items: list = []
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as f:
                    self._items = json.load(f)

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._items, f)
        os.replace(tmp, self.path)

    def submit(self, username: str, pack_id: str, utr: str) -> dict:
        if not config.upi_id:
            raise HTTPException(400, "UPI transfers are not switched on.")
        utr = "".join(utr.split())
        if not (utr.isdigit() and len(utr) == 12):
            raise HTTPException(422, "The UPI reference (UTR) is the 12-digit number in "
                                     "your UPI app's payment details.")
        p = pack(pack_id)
        with self._lock:
            if any(i["utr"] == utr for i in self._items if i["status"] != "rejected"):
                raise HTTPException(409, "That UPI reference has already been submitted.")
            pending = [i for i in self._items if i["user"] == username and i["status"] == "pending"]
            if len(pending) >= self.MAX_PENDING:
                raise HTTPException(429, "You have payments waiting to be checked; please "
                                         "wait for them before sending another.")
            item = {"id": secrets.token_hex(6), "user": username, "pack": p["id"],
                    "credits": p["credits"], "amount": p["inr"], "utr": utr,
                    "created": time.time(), "status": "pending", "note": ""}
            self._items.append(item)
            self._save()
        logger.info("UPI payment reported by %s: %s credits, UTR ending %s",
                    username, p["credits"], utr[-4:])
        return dict(item)

    def for_user(self, username: str) -> list:
        with self._lock:
            return [dict(i) for i in self._items if i["user"] == username][-10:][::-1]

    def pending(self) -> list:
        with self._lock:
            return [dict(i) for i in self._items if i["status"] == "pending"]

    def decide(self, item_id: str, approve: bool, note: str = "") -> dict:
        with self._lock:
            item = next((i for i in self._items if i["id"] == item_id), None)
            if item is None:
                raise HTTPException(404, "No such UPI payment.")
            if item["status"] != "pending":
                raise HTTPException(409, "This payment was already {}.".format(item["status"]))
            item["status"] = "approved" if approve else "rejected"
            item["note"] = note[:200]
            item["decided"] = time.time()
            self._save()
        if approve:
            ref = "upi:" + item["utr"]
            if credit_store.grant(item["user"], item["credits"],
                                  "{} pack via UPI transfer".format(item["pack"]), ref=ref):
                _invoice(item["user"], item["pack"], item["credits"], item["amount"],
                         "INR", "upi", ref)
        return dict(item)


upi_queue = UpiQueue()


def upi_link(p: dict) -> str:
    """A upi:// link that opens any UPI app with the payee and amount
    filled in (on a phone)."""
    from urllib.parse import quote
    return "upi://pay?pa={}&pn={}&am={:.2f}&cu=INR&tn={}".format(
        quote(config.upi_id), quote(config.upi_payee_name or config.business_name
                                    or config.app_name),
        p["inr"] / 100, quote("{} credits".format(p["credits"])))
