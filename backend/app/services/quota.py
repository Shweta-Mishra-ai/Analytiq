"""
services/quota.py — the one place a ceiling is checked.

plans.py says what a tier allows, usage.py counts what an account has
done, and the dataset store knows what it holds. This joins the three
and answers one question: may this account do this, now.

It is a separate module so that answer has exactly one implementation.
Scattering `if plan.datasets and len(datasets) >= plan.datasets` through
the routes is how two endpoints end up disagreeing about whether the
third dataset is the last allowed one or the first refused one.

Every refusal names the ceiling, the current usage and the plan, because
"quota exceeded" tells a paying customer nothing about what to do next.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.config import config
from app.services import plans
from app.services.usage import usage_store

logger = logging.getLogger(__name__)


class QuotaExceeded(Exception):
    """A ceiling was reached. Carries what a client needs to act."""

    def __init__(self, message: str, *, limit_name: str, limit: int,
                 used: int, plan: str):
        super().__init__(message)
        self.limit_name = limit_name
        self.limit = limit
        self.used = used
        self.plan = plan

    def as_detail(self) -> dict:
        return {
            "error": str(self),
            "limit": self.limit_name,
            "allowed": self.limit,
            "used": self.used,
            "plan": self.plan,
            "upgrade": "/api/billing/plans",
        }


@dataclass(frozen=True)
class Allowance:
    """What an account may still do, for the UI to show before it acts."""
    plan: str
    label: str
    branding: bool
    limits: dict
    used: dict
    remaining: dict


def plan_for(owner: str) -> plans.Plan:
    """The plan in force for this account."""
    from app.services.user_store import user_store

    metered = config.is_metered
    if not metered:
        return plans.resolve("", metered=False)
    user = user_store.get(owner)
    return plans.resolve(user.plan if user else plans.FALLBACK_PLAN,
                         metered=True)


def _dataset_count(owner: str) -> int:
    from app.services.dataset_store import store
    try:
        return len(store.list_meta(owner))
    except Exception:
        # Counting must never be the reason an upload fails. A miscount
        # low is a free extra dataset; a raised exception here is an
        # account that cannot use the product at all.
        logger.warning("could not count datasets for %r", owner,
                       exc_info=True)
        return 0


def check_new_dataset(owner: str, *, rows: int = 0, size_mb: float = 0.0
                      ) -> None:
    """Raise QuotaExceeded if this upload would cross a ceiling."""
    plan = plan_for(owner)

    cap = plan.datasets
    if cap is not None:
        held = _dataset_count(owner)
        if held >= cap:
            raise QuotaExceeded(
                "The {} plan holds {} datasets and you have {}. Delete one, "
                "or move to a plan with more room.".format(
                    plan.label, cap, held),
                limit_name="datasets", limit=cap, used=held, plan=plan.key)

    cap = plan.rows_per_dataset
    if cap is not None and rows > cap:
        raise QuotaExceeded(
            "That file has {:,} rows and the {} plan analyses up to {:,} in "
            "one dataset.".format(rows, plan.label, cap),
            limit_name="rows_per_dataset", limit=cap, used=int(rows),
            plan=plan.key)

    cap = plan.max_upload_mb
    if cap is not None and size_mb > cap:
        raise QuotaExceeded(
            "That file is {:.1f} MB and the {} plan accepts up to {} MB."
            .format(size_mb, plan.label, cap),
            limit_name="max_upload_mb", limit=cap, used=int(size_mb),
            plan=plan.key)


def check_event(owner: str, event: str) -> None:
    """Raise QuotaExceeded if another `event` would cross the month's cap."""
    plan = plan_for(owner)
    cap = plan.limit("{}_per_month".format(event))
    if cap is None:
        return
    used = usage_store.used(owner, event)
    if used >= cap:
        raise QuotaExceeded(
            "The {} plan includes {} {} a month and you have used {}. The "
            "count resets at the start of next month.".format(
                plan.label, cap, event, used),
            limit_name="{}_per_month".format(event), limit=cap, used=used,
            plan=plan.key)


def check_feature(owner: str, feature: str) -> None:
    """Raise QuotaExceeded for a capability this plan does not include."""
    plan = plan_for(owner)
    if plan.allows(feature):
        return
    raise QuotaExceeded(
        "{} is not part of the {} plan.".format(
            feature.replace("_", " ").capitalize(), plan.label),
        limit_name=feature, limit=0, used=0, plan=plan.key)


def record(owner: str, event: str, n: int = 1) -> None:
    """Count work that has actually completed.

    Called after the work succeeds, never before: charging for a report
    that then failed to build is the kind of thing customers remember.
    """
    try:
        usage_store.record(owner, event, n)
    except Exception:
        logger.warning("could not record %s for %r", event, owner,
                       exc_info=True)


def allowance(owner: str) -> Allowance:
    """Everything the UI needs to show what is left."""
    plan = plan_for(owner)
    used = dict(usage_store.snapshot(owner))
    used["datasets"] = _dataset_count(owner)

    limits = {
        "datasets": plan.datasets,
        "rows_per_dataset": plan.rows_per_dataset,
        "max_upload_mb": plan.max_upload_mb,
        "reports_per_month": plan.reports_per_month,
        "models_per_month": plan.models_per_month,
        "scheduled_reports": plan.scheduled_reports,
        "share_links": plan.share_links,
    }
    remaining = {}
    for name, cap in limits.items():
        if cap is None:
            remaining[name] = None
            continue
        spent = used.get(name.replace("_per_month", ""), used.get(name, 0))
        remaining[name] = max(0, cap - int(spent))

    return Allowance(plan=plan.key, label=plan.label, branding=plan.branding,
                     limits=limits, used=used, remaining=remaining)
