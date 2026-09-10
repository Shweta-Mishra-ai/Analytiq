"""
services/plans.py — what an account is allowed to do.

Analytiq had no notion of a plan. Accounts existed only because an
operator created them with an admin key, so there was nothing to meter:
every account could upload any number of files of any size and train any
number of models, on compute somebody else pays for. That is fine for a
tool one person runs for their own clients and impossible for a product
strangers can sign up to.

A plan is a set of ceilings, not a set of features. Nothing here hides
an engine behind a paywall — the analysis a free account gets is the
same analysis, on less data. Gating correctness would make the free tier
a demo that lies, which is the opposite of what this product sells.

The caps are deliberately generous at the top and honest at the bottom:
the free tier has to be enough to judge the product on real work, or the
trial tells the user nothing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# The plan an account has when nothing says otherwise. Existing accounts
# were created before plans existed and must not suddenly lose access,
# so the store maps a missing plan to this and the value is chosen to be
# the unlimited one for self-hosted deployments (see `resolve`).
FALLBACK_PLAN = "free"

# What an operator running this for their own clients gets. Self-hosted
# deployments are not metered: the person paying for the container is
# the person using it, and a ceiling there protects nobody.
SELF_HOSTED_PLAN = "unlimited"


@dataclass(frozen=True)
class Plan:
    """One tier, and every ceiling that applies to it.

    A ceiling of None means "no limit". Zero means "not allowed", which
    is different, and the distinction matters: a free account has
    `scheduled_reports = 0` because scheduling is a delivery promise the
    free tier does not make, while `datasets = None` on unlimited means
    genuinely uncapped.
    """
    key: str
    label: str
    price_inr_month: int

    # Storage-shaped ceilings, checked when data arrives.
    datasets: Optional[int] = None
    rows_per_dataset: Optional[int] = None
    max_upload_mb: Optional[int] = None

    # Work-shaped ceilings, counted over a rolling month.
    reports_per_month: Optional[int] = None
    models_per_month: Optional[int] = None

    # Delivery-shaped ceilings.
    scheduled_reports: Optional[int] = None
    share_links: Optional[int] = None

    # Whether the deliverable carries Analytiq's own name. The one thing
    # a freelancer actually pays to remove.
    branding: bool = True

    notes: str = ""

    def limit(self, name: str) -> Optional[int]:
        return getattr(self, name, None)

    def allows(self, name: str) -> bool:
        """False only for a ceiling of exactly zero."""
        return self.limit(name) != 0


PLANS: Dict[str, Plan] = {
    "free": Plan(
        key="free", label="Free", price_inr_month=0,
        datasets=3, rows_per_dataset=25_000, max_upload_mb=10,
        reports_per_month=5, models_per_month=10,
        scheduled_reports=0, share_links=2,
        branding=True,
        notes="Enough to judge the product on real work. Reports carry "
              "the Analytiq mark.",
    ),
    "solo": Plan(
        key="solo", label="Solo", price_inr_month=2400,
        datasets=100, rows_per_dataset=2_000_000, max_upload_mb=200,
        reports_per_month=200, models_per_month=300,
        scheduled_reports=10, share_links=200,
        branding=False,
        notes="One analyst, unlimited clients. Your name on the report, "
              "not ours.",
    ),
    "practice": Plan(
        key="practice", label="Practice", price_inr_month=8000,
        datasets=1000, rows_per_dataset=10_000_000, max_upload_mb=500,
        reports_per_month=2000, models_per_month=3000,
        scheduled_reports=100, share_links=2000,
        branding=False,
        notes="A team of analysts, scheduled delivery, and client "
              "workspaces.",
    ),
    "unlimited": Plan(
        key="unlimited", label="Unlimited", price_inr_month=0,
        datasets=None, rows_per_dataset=None, max_upload_mb=None,
        reports_per_month=None, models_per_month=None,
        scheduled_reports=None, share_links=None,
        branding=False,
        notes="Self-hosted, or an account an operator runs directly. Not "
              "metered — the person paying for the container is the "
              "person using it.",
    ),
}

# The order a pricing page lists them in.
PUBLIC_PLANS: Tuple[str, ...] = ("free", "solo", "practice")


def get(key: str) -> Plan:
    """The named plan, falling back rather than raising.

    A plan key that no longer exists — a tier retired between a
    subscription starting and this deploy — must not lock the account
    out of its own data.
    """
    resolved = str(key or "").strip().lower()
    plan = PLANS.get(resolved)
    if plan is None:
        logger.warning("unknown plan %r — falling back to %s",
                       key, FALLBACK_PLAN)
        return PLANS[FALLBACK_PLAN]
    return plan


def resolve(plan_key: str, *, metered: bool) -> Plan:
    """The plan actually in force for an account.

    `metered` is the deployment's own answer to "is this a product
    strangers pay for, or a tool I run for myself". Where it is False —
    a self-hosted instance with signup switched off — every account is
    unlimited whatever its stored plan says, because metering an
    operator's own container serves no one.
    """
    if not metered:
        return PLANS[SELF_HOSTED_PLAN]
    return get(plan_key)


def public_catalogue() -> list:
    """The tiers a pricing page shows, cheapest first."""
    return [
        {
            "key": PLANS[k].key,
            "label": PLANS[k].label,
            "price_inr_month": PLANS[k].price_inr_month,
            "datasets": PLANS[k].datasets,
            "rows_per_dataset": PLANS[k].rows_per_dataset,
            "max_upload_mb": PLANS[k].max_upload_mb,
            "reports_per_month": PLANS[k].reports_per_month,
            "scheduled_reports": PLANS[k].scheduled_reports,
            "branding": PLANS[k].branding,
            "notes": PLANS[k].notes,
        }
        for k in PUBLIC_PLANS
    ]
