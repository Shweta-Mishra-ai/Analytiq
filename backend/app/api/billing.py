"""
api/billing.py — credits balance, buying credits, and the providers'
webhooks. The money logic is in services/billing.py and credits.py.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Request, UploadFile
from pydantic import BaseModel

from app.config import config
from app.services import billing
from app.services.auth import current_owner
from app.services.credits import ACTIONS, costs, credit_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/billing", tags=["billing"])


def _public_plans() -> dict:
    return {
        "packs": [{"id": p["id"], "name": p["name"], "credits": p["credits"],
                   "inr": p["inr"], "usd": p["usd"]} for p in billing.packs()],
        "costs": [{"action": k, "label": ACTIONS[k], "credits": v} for k, v in costs().items()],
        "free_daily": config.free_daily_credits,
        "providers": billing.providers(),
        "stripe_currency": (config.stripe_currency or "usd").upper(),
        "signup_enabled": config.signup_enabled,
        "upi": ({"id": config.upi_id,
                 "payee": config.upi_payee_name or config.business_name or config.app_name,
                 "links": {p["id"]: billing.upi_link(p) for p in billing.packs()}}
                if config.upi_id else None),
    }


@router.get("/plans")
def plans():
    """Prices and what things cost. Public: the pricing page shows it."""
    return _public_plans()


def _admin(request: Request) -> bool:
    return bool(getattr(request.state, "is_admin", False))


@router.get("")
def overview(request: Request, owner: str = Depends(current_owner)):
    """Balance and recent activity. Also credits any payment of this
    account's that was completed but never confirmed."""
    caught_up = billing.reconcile(owner)
    from app.services.invoices import invoice_book
    return {"balance": credit_store.balance(owner, _admin(request)),
            "history": credit_store.history(owner, 30),
            "invoices": [{"number": i["number"], "issued_at": i["issued_at"],
                          "amount": i["amount"], "currency": i["currency"],
                          "credits": i["credits"], "tax_invoice": i["tax_invoice"]}
                         for i in invoice_book.for_user(owner)],
            "upi_requests": billing.upi_queue.for_user(owner),
            "recovered": caught_up, **_public_plans()}


@router.get("/invoices/{number}")
def invoice_pdf(number: str, request: Request, owner: str = Depends(current_owner)):
    from fastapi import HTTPException
    from fastapi.responses import Response
    from app.services.invoices import invoice_book, render_pdf
    inv = invoice_book.get(number)
    if inv is None or (inv["user"] != owner and not _admin(request)):
        raise HTTPException(404, "No such invoice.")
    return Response(render_pdf(inv), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{number}.pdf"'})


class UpiReport(BaseModel):
    pack: str
    utr: str


@router.post("/upi")
def report_upi_payment(req: UpiReport, owner: str = Depends(current_owner)):
    """The buyer has paid the UPI ID directly; an admin will check it."""
    from fastapi import HTTPException
    from app.services.user_store import user_store
    if user_store.get(owner) is None:
        raise HTTPException(400, "Buying credits needs a signed-in account.")
    return billing.upi_queue.submit(owner, req.pack, req.utr)


class CheckoutRequest(BaseModel):
    provider: str
    pack: str


def _base_url(request: Request) -> str:
    if config.public_url:
        return config.public_url.rstrip("/")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return f"{proto}://{host}"


@router.post("/checkout")
def checkout(req: CheckoutRequest, request: Request, owner: str = Depends(current_owner)):
    from fastapi import HTTPException
    from app.services.user_store import user_store
    account = user_store.get(owner)
    if account is None:
        raise HTTPException(400, "Buying credits needs a signed-in account.")
    return billing.start_checkout(req.provider, req.pack, owner, account.email,
                                  _base_url(request))


class RazorpayResult(BaseModel):
    order_id: str
    payment_id: str
    signature: str


@router.post("/razorpay/verify")
def razorpay_verify(req: RazorpayResult, owner: str = Depends(current_owner)):
    return billing.razorpay_verify(owner, req.order_id, req.payment_id, req.signature)


class StripeResult(BaseModel):
    session_id: str


@router.post("/stripe/confirm")
def stripe_confirm(req: StripeResult, owner: str = Depends(current_owner)):
    return billing.stripe_confirm(owner, req.session_id)


class PaypalResult(BaseModel):
    order_id: str


@router.post("/paypal/confirm")
def paypal_confirm(req: PaypalResult, owner: str = Depends(current_owner)):
    return billing.paypal_confirm(owner, req.order_id)


@router.post("/webhook/razorpay")
async def razorpay_webhook(request: Request):
    body = await request.body()
    return billing.razorpay_webhook(body, request.headers.get("x-razorpay-signature", ""))


@router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    body = await request.body()
    return billing.stripe_webhook(body, request.headers.get("stripe-signature", ""))


# ── admin ────────────────────────────────────────────────────

admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


class GrantRequest(BaseModel):
    amount: int
    note: str = ""


@admin_router.post("/users/{username}/credits")
def grant_credits(username: str, req: GrantRequest):
    """Give credits by hand: a payment taken outside the app (bank
    transfer, a UPI transfer you checked yourself), a refund, a goodwill
    gesture."""
    from fastapi import HTTPException
    from app.services.user_store import user_store
    username = username.strip().lower()
    if not user_store.exists(username):
        raise HTTPException(404, "No such user")
    if not 0 < req.amount <= 1_000_000:
        raise HTTPException(422, "Amount must be between 1 and 1,000,000 credits.")
    credit_store.grant(username, req.amount, req.note or "given by the administrator")
    logger.info("admin gave %s credits to %s", req.amount, username)
    return credit_store.balance(username)


@admin_router.get("/upi-payments")
def upi_payments():
    """UPI transfers waiting for you to find them in the bank account."""
    return {"pending": billing.upi_queue.pending()}


class UpiDecision(BaseModel):
    approve: bool
    note: str = ""


@admin_router.post("/upi-payments/statement")
async def match_upi_statement(file: UploadFile = File(...)):
    """Approve the UPI payments found in a bank or UPI-app statement
    (CSV or Excel): same UTR and same amount."""
    from fastapi import HTTPException
    from starlette.concurrency import run_in_threadpool
    data = await file.read()
    if len(data) > 10 * 2**20:
        raise HTTPException(413, "A statement larger than 10 MB is not expected.")

    def _rows():
        from app.api.datasets import _UploadShim
        from app.engines.data_loader import load_file
        res = load_file(_UploadShim(file.filename or "statement.csv", data))
        if not res.success:
            raise HTTPException(422, "The statement could not be read: {}".format(res.error))
        df = res.df.astype(str)
        # The header row too: a statement's first line is sometimes a payment.
        return [list(df.columns)] + df.values.tolist()

    rows = await run_in_threadpool(_rows)
    out = await run_in_threadpool(billing.upi_queue.match_statement, rows)
    logger.info("UPI statement matched: %d approved, %d wrong amount",
                len(out["approved"]), len(out["wrong_amount"]))
    return out


# Before "/upi-payments/{item_id}", which would take "statement" as an id.


@admin_router.post("/upi-payments/{item_id}")
def decide_upi_payment(item_id: str, req: UpiDecision):
    item = billing.upi_queue.decide(item_id, req.approve, req.note)
    logger.info("UPI payment %s %s", item_id, item["status"])
    return item


class UnlimitedRequest(BaseModel):
    unlimited: bool


@admin_router.post("/users/{username}/unlimited")
def set_unlimited(username: str, req: UnlimitedRequest):
    """Never charge this account (your own, a partner's, a demo)."""
    from fastapi import HTTPException
    from app.services.user_store import user_store
    username = username.strip().lower()
    if not user_store.exists(username):
        raise HTTPException(404, "No such user")
    credit_store.set_unlimited(username, req.unlimited)
    return credit_store.balance(username)
