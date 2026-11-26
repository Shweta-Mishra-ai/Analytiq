"""
services/credits.py — what each account may spend, and the record of it.

Every account gets a free allowance each day (FREE_DAILY_CREDITS) and can
buy more (services/billing.py). Heavy actions — a PDF report, training
models, a chat answer — cost credits (CREDIT_COSTS); looking at data,
dashboards and charts never does, so a new user sees what the product does
before being asked to pay.

Free credits are spent first and do not carry over; bought credits never
expire. Unlimited accounts (admins, the open single-user mode, usernames in
FREE_USERS, and any account an admin marks unlimited) are never charged.

A charge is taken when the action starts and given back if it fails, so a
client is never billed for an error (see `charge`). Every change is written
to an append-only ledger, and a payment is credited at most once whatever
number of times its confirmation arrives (`grant(..., ref=...)`).

Single process, like the rest of the app's state: one lock, one JSON file
written atomically.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from starlette.requests import Request

from app.config import config

logger = logging.getLogger(__name__)

# What each charged action is called in CREDIT_COSTS and in the ledger.
ACTIONS = {
    "pdf_report": "PDF report",
    "health_report": "Data health PDF",
    "ml_train": "Model training",
    "chat": "AI chat answer",
    "kb_report": "Knowledge-base report",
}


def costs() -> dict[str, int]:
    out = {"pdf_report": 2, "health_report": 2, "ml_train": 1, "chat": 1, "kb_report": 1}
    try:
        extra = json.loads(config.credit_costs) if config.credit_costs else {}
        out.update({k: max(0, int(v)) for k, v in extra.items() if k in ACTIONS})
    except (ValueError, TypeError, AttributeError):
        logger.error("CREDIT_COSTS is not valid JSON; using the defaults")
    return out


def _today() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(config.credits_timezone)).strftime("%Y-%m-%d")
    except Exception:
        return datetime.utcnow().strftime("%Y-%m-%d")


class OutOfCredits(HTTPException):
    def __init__(self, need: int, have: int):
        super().__init__(
            402, "This needs {} credit{} and you have {}. Your free credits renew "
                 "tomorrow, or buy more on the Billing page.".format(
                     need, "" if need == 1 else "s", have))


class CreditStore:
    def __init__(self, root: Optional[str] = None):
        self._lock = threading.RLock()
        self.open(root)

    def open(self, root: Optional[str] = None) -> None:
        """(Re)read the store from `root` (tests point it at a temp dir)."""
        self.root = root or os.path.join(config.data_dir, "billing")
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, "credits.json")
        self.ledger_path = os.path.join(self.root, "ledger.jsonl")
        with self._lock:
            self._data = self._load()

    # ── storage ─────────────────────────────────────────
    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {"accounts": {}, "refs": []}
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("accounts", {})
        data.setdefault("refs", [])
        return data

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f)
        os.replace(tmp, self.path)

    def _log(self, username: str, change: int, kind: str, note: str, ref: str = "") -> None:
        entry = {"t": time.time(), "user": username, "change": change, "kind": kind,
                 "note": note, "ref": ref}
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _acct(self, username: str) -> dict:
        a = self._data["accounts"].setdefault(
            username, {"paid": 0, "free_day": "", "free_used": 0, "unlimited": False})
        if a.get("free_day") != _today():
            a["free_day"], a["free_used"] = _today(), 0
        return a

    # ── reading ─────────────────────────────────────────
    def is_unlimited(self, username: str, is_admin: bool = False) -> bool:
        from app.services.auth import LOCAL_OWNER
        if is_admin or username in ("admin", LOCAL_OWNER):
            return True
        from app.services.user_store import user_store
        account = user_store.get(username)
        if account is not None and account.is_admin:
            return True
        free = {u.strip().lower() for u in config.free_users.split(",") if u.strip()}
        if username in free:
            return True
        with self._lock:
            return bool(self._data["accounts"].get(username, {}).get("unlimited"))

    @staticmethod
    def free_allowance(username: str) -> int:
        """Free credits a day for this account: none until a signed-up
        account confirms its email (see services/emails.py), which is what
        keeps one person from collecting them on many accounts."""
        from app.services.user_store import user_store
        account = user_store.get(username)
        if account is not None and not account.email_verified:
            return 0
        return config.free_daily_credits

    def balance(self, username: str, is_admin: bool = False) -> dict:
        unlimited = self.is_unlimited(username, is_admin)
        allowance = self.free_allowance(username)
        with self._lock:
            a = self._acct(username)
            free_left = max(0, allowance - a["free_used"])
            return {"unlimited": unlimited, "free_left": free_left,
                    "free_daily": allowance, "needs_verification":
                        allowance == 0 and config.free_daily_credits > 0,
                    "paid": int(a["paid"]),
                    "total": free_left + int(a["paid"]),
                    "account_unlimited": bool(a.get("unlimited"))}

    def history(self, username: str, limit: int = 50) -> list[dict]:
        if not os.path.exists(self.ledger_path):
            return []
        out = []
        with open(self.ledger_path, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("user") == username:
                    out.append(e)
        return out[-limit:][::-1]

    # ── changing ────────────────────────────────────────
    def spend(self, username: str, action: str, is_admin: bool = False) -> dict:
        """Take the cost of `action`. Returns what was taken, for `refund`."""
        cost = costs().get(action, 0)
        if cost <= 0 or self.is_unlimited(username, is_admin):
            return {"free": 0, "paid": 0}
        allowance = self.free_allowance(username)
        with self._lock:
            a = self._acct(username)
            free_left = max(0, allowance - a["free_used"])
            if free_left + a["paid"] < cost:
                if allowance == 0 and config.free_daily_credits > 0 and not a["paid"]:
                    raise HTTPException(
                        402, "Confirm your email address to start using your free daily "
                             "credits: open the link we sent you (or send it again from "
                             "the banner at the top of the page). You can also buy "
                             "credits on the Billing page.")
                raise OutOfCredits(cost, free_left + int(a["paid"]))
            from_free = min(free_left, cost)
            from_paid = cost - from_free
            a["free_used"] += from_free
            a["paid"] -= from_paid
            self._save()
            self._log(username, -cost, "spend", ACTIONS.get(action, action))
            return {"free": from_free, "paid": from_paid, "day": a["free_day"]}

    def refund(self, username: str, taken: dict, action: str) -> None:
        if not (taken.get("free") or taken.get("paid")):
            return
        with self._lock:
            a = self._acct(username)
            # Free credits come back only on the day they were taken from.
            if taken.get("day") == a["free_day"]:
                a["free_used"] = max(0, a["free_used"] - taken["free"])
            a["paid"] += taken["paid"]
            self._save()
            self._log(username, taken["free"] + taken["paid"], "refund",
                      "{} did not complete".format(ACTIONS.get(action, action)))

    def grant(self, username: str, amount: int, note: str, ref: str = "") -> bool:
        """Add bought or given credits. With `ref` (a payment id), only once:
        returns False if that ref was already credited."""
        if amount <= 0:
            raise ValueError("amount must be positive")
        with self._lock:
            if ref and ref in self._data["refs"]:
                return False
            self._acct(username)["paid"] += int(amount)
            if ref:
                self._data["refs"].append(ref)
            self._save()
            self._log(username, int(amount), "purchase" if ref else "grant", note, ref)
            return True

    def set_unlimited(self, username: str, on: bool) -> None:
        with self._lock:
            self._acct(username)["unlimited"] = bool(on)
            self._save()
            self._log(username, 0, "unlimited", "on" if on else "off")

    def forget(self, username: str) -> None:
        """An account was deleted. Its balance must not pass to a new
        account created later under the same name; the ledger stays."""
        with self._lock:
            if self._data["accounts"].pop(username, None) is not None:
                self._save()


credit_store = CreditStore()


@contextmanager
def charging(username: str, action: str, is_admin: bool = False):
    """`with charging(owner, "pdf_report"):` around work that costs credits.
    The charge is taken before the work and given back if it raises."""
    taken = credit_store.spend(username, action, is_admin)
    try:
        yield
    except BaseException:
        credit_store.refund(username, taken, action)
        raise


def charge(action: str):
    """FastAPI dependency: charge the signed-in account for `action`, and
    refund it if the route fails."""
    def dep(request: Request):
        from app.services.auth import LOCAL_OWNER
        user = getattr(request.state, "username", LOCAL_OWNER)
        admin = bool(getattr(request.state, "is_admin", False))
        with charging(user, action, admin):
            yield
    return dep
