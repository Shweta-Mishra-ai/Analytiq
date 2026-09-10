"""
api/billing.py — changing a plan, and surviving the payment processor.

One rule shapes this module: **billing changes a plan, it never grants
one.** The plan lives on the account, in this app's own store. Stripe is
told about upgrades and tells us about renewals and cancellations, but
if Stripe is unreachable — misconfigured, down, or simply not set up on
this deployment — every account keeps exactly the plan it already had
and the product keeps working. A customer locked out of their own data
because a webhook was late is a worse outcome than a month of service
somebody did not pay for.

So the module degrades in three steps rather than failing:

  * no Stripe keys        → the plan catalogue is still readable, and
                            checkout says plainly that it is unavailable
  * keys but no library   → same, with the reason logged for the operator
  * everything present    → real checkout sessions and a signed webhook

The webhook verifies its signature and refuses to act without one. An
unsigned endpoint that moves accounts between plans is an endpoint that
lets anyone move any account onto any plan.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.config import config
from app.services import plans
from app.services.auth import current_owner
from app.services.load_control import admit
from app.services.user_store import user_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/billing", tags=["billing"])

# Stripe events that change what an account is entitled to. Anything
# else is acknowledged and ignored — an endpoint that acts on every
# event it receives is one deploy away from acting on a new one nobody
# reviewed.
SUBSCRIPTION_ACTIVE = {"checkout.session.completed",
                       "customer.subscription.created",
                       "customer.subscription.updated"}
SUBSCRIPTION_ENDED = {"customer.subscription.deleted"}

# What an account falls back to when a subscription ends. Not "no
# access": the data is theirs, and the free tier is the honest place for
# it to sit until they decide.
LAPSED_PLAN = "free"


class CheckoutRequest(BaseModel):
    plan: str


def _stripe():
    """The Stripe SDK, or None with the reason logged."""
    if not config.stripe_configured:
        return None
    try:
        import stripe
    except ImportError:
        logger.warning("STRIPE_SECRET_KEY is set but the stripe package is "
                       "not installed — checkout is unavailable")
        return None
    stripe.api_key = config.stripe_secret_key
    return stripe


@router.get("/plans")
def catalogue():
    """The tiers, and whether this deployment can actually sell them.

    Public: it is what a signed-out visitor is deciding on.
    """
    return {
        "plans": plans.public_catalogue(),
        "checkout_available": bool(_stripe()),
        "metered": bool(config.is_metered),
        "currency": "INR",
    }


@router.get("/subscription")
def subscription(owner: str = Depends(current_owner)):
    """What this account is on, and where it came from."""
    from app.services.quota import allowance

    user = user_store.get(owner)
    quota = allowance(owner)
    return {
        "plan": quota.plan,
        "label": quota.label,
        "since": user.plan_since if user else 0.0,
        "billed": bool(user and user.billing_customer_id),
        "checkout_available": bool(_stripe()),
        "limits": quota.limits,
        "used": quota.used,
        "remaining": quota.remaining,
    }


@router.post("/checkout", dependencies=[Depends(admit("billing"))])
def checkout(req: CheckoutRequest, request: Request,
             owner: str = Depends(current_owner)):
    """A Stripe Checkout URL for moving this account onto a paid plan."""
    stripe = _stripe()
    if stripe is None:
        raise HTTPException(
            503, "Payments are not set up on this deployment. Ask whoever "
                 "runs it to move your account.")

    target = str(req.plan or "").strip().lower()
    if target not in plans.PUBLIC_PLANS or target == plans.FALLBACK_PLAN:
        raise HTTPException(422, "That is not a plan you can subscribe to.")
    price_id = config.stripe_price_map.get(target)
    if not price_id:
        raise HTTPException(
            503, "The {} plan has no price configured on this deployment."
                 .format(target))

    user = user_store.get(owner)
    return_url = config.billing_return_url or str(
        request.base_url).rstrip("/")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url="{}/account?upgraded={}".format(return_url, target),
            cancel_url="{}/account".format(return_url),
            customer_email=(user.email or None) if user else None,
            # The account this is for, echoed back on the webhook. Email
            # is not enough: a customer may pay with a different address
            # than the one they signed up with.
            client_reference_id=owner,
            metadata={"analytiq_owner": owner, "analytiq_plan": target},
            subscription_data={
                "metadata": {"analytiq_owner": owner,
                             "analytiq_plan": target},
            },
        )
    except Exception as e:
        logger.warning("stripe checkout failed for %r", owner, exc_info=True)
        raise HTTPException(
            502, "The payment processor could not start a checkout: {}"
                 .format(e)) from None

    return {"url": session.url, "plan": target}


@router.post("/webhook")
async def webhook(request: Request):
    """Stripe telling us a subscription started, renewed or ended.

    Signature-verified, and refuses outright without a secret: an
    unsigned endpoint that moves accounts between plans lets anybody
    move any account onto any plan.
    """
    stripe = _stripe()
    if stripe is None or not config.stripe_webhook_secret:
        raise HTTPException(503, "Billing webhooks are not configured.")

    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(
            payload, signature, config.stripe_webhook_secret)
    except Exception:
        logger.warning("rejected a webhook with a bad signature")
        raise HTTPException(400, "Bad signature") from None

    kind = str(event.get("type") or "")
    obj = (event.get("data") or {}).get("object") or {}

    if kind in SUBSCRIPTION_ACTIVE:
        _apply_active(obj)
    elif kind in SUBSCRIPTION_ENDED:
        _apply_ended(obj)
    else:
        logger.info("ignoring stripe event %s", kind)

    # 200 for everything we understood, including events we chose to
    # ignore. A non-2xx makes Stripe retry, and retrying an event this
    # app will never act on is noise for both sides.
    return {"received": True}


def _owner_from(obj: dict) -> str:
    """Which account a Stripe object is about."""
    metadata = obj.get("metadata") or {}
    owner = (obj.get("client_reference_id")
             or metadata.get("analytiq_owner") or "")
    if owner:
        return str(owner).strip().lower()
    # Fall back to the customer id, which is how a renewal months later
    # arrives — it carries no metadata from the original checkout.
    found = user_store.by_customer_id(str(obj.get("customer") or ""))
    return found.username if found else ""


def _apply_active(obj: dict) -> None:
    owner = _owner_from(obj)
    if not owner:
        logger.warning("stripe event names no account this app knows")
        return
    metadata = obj.get("metadata") or {}
    target = str(metadata.get("analytiq_plan") or "").strip().lower()
    if target not in plans.PLANS:
        logger.warning("stripe event for %r names unknown plan %r — leaving "
                       "the account as it is", owner, target)
        return
    user_store.set_plan(
        owner, target,
        customer_id=str(obj.get("customer") or ""),
        subscription_id=str(obj.get("subscription") or obj.get("id") or ""))
    logger.info("account %s moved to %s by stripe", owner, target)


def _apply_ended(obj: dict) -> None:
    owner = _owner_from(obj)
    if not owner:
        return
    user_store.set_plan(owner, LAPSED_PLAN,
                        customer_id=str(obj.get("customer") or ""),
                        subscription_id="")
    logger.info("account %s lapsed to %s", owner, LAPSED_PLAN)
