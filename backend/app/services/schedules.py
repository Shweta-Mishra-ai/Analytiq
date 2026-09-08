"""
services/schedules.py — a report that arrives without anyone asking.

The difference between a tool someone opens and a service they rely on
is whether it shows up on its own. Everything needed for that now
exists — a run that survives the request, a result with an identity, a
link that reaches someone without an account — so a schedule is a small
thing on top rather than a feature of its own.

Deliberately modest:

**Cadences, not cron.** Daily, weekly and monthly. A cron expression is
more expressive and nobody outside engineering can read one, and the
person setting this up is the analyst who wants Monday numbers on
Monday.

**The tick is idempotent.** A schedule records the period it last ran
for, not the last time it ran, so a restart, a slow tick, or two ticks
in the same minute produce one report — not none and not three.

**Missed windows are not backfilled.** A server that was down for a week
does not wake up and send seven reports. It sends the current one, which
is what anybody wanted.

**Delivery degrades honestly.** With SMTP configured, the link is
emailed. Without it, the report is still built and still waiting on the
Reports page, and the schedule says so rather than silently doing
nothing — a schedule that quietly produces no output is worse than no
schedule.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.config import config

logger = logging.getLogger(__name__)

CADENCES = ("daily", "weekly", "monthly")


@dataclass
class Schedule:
    schedule_id: str
    owner: str
    dataset_id: str
    kind: str = "health-report"
    cadence: str = "weekly"
    hour: int = 7                    # local hour to run at
    recipients: List[str] = field(default_factory=list)
    enabled: bool = True
    created_at: float = 0.0
    # The PERIOD last produced ("2026-W37"), not the timestamp of the
    # last run: this is what makes a tick idempotent across restarts.
    last_period: str = ""
    last_run_at: float = 0.0
    last_status: str = ""
    last_artifact_id: str = ""
    params: Dict = field(default_factory=dict)


def period_key(cadence: str, when: Optional[datetime] = None) -> str:
    """The window a moment belongs to. Two moments in one window share
    a key, which is the whole mechanism against double-sending."""
    now = when or datetime.now()
    if cadence == "daily":
        return now.strftime("%Y-%m-%d")
    if cadence == "monthly":
        return now.strftime("%Y-%m")
    iso = now.isocalendar()
    return "{}-W{:02d}".format(iso[0], iso[1])


class ScheduleStore:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.join(config.data_dir, "schedules")
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, owner: str, schedule_id: str) -> str:
        from app.services.artifacts import _safe
        directory = os.path.join(self.base_dir, _safe(owner))
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, _safe(schedule_id) + ".json")

    def save(self, s: Schedule) -> None:
        from app.services.artifacts import _atomic_write
        with self._lock:
            _atomic_write(self._path(s.owner, s.schedule_id),
                          json.dumps(asdict(s)).encode("utf-8"))

    def get(self, owner: str, schedule_id: str) -> Optional[Schedule]:
        path = self._path(owner, schedule_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return Schedule(**json.load(fh))
        except Exception:
            logger.warning("unreadable schedule at %s", path, exc_info=True)
            return None

    def list(self, owner: str) -> List[Schedule]:
        from app.services.artifacts import _safe
        directory = os.path.join(self.base_dir, _safe(owner))
        if not os.path.isdir(directory):
            return []
        out = []
        for name in os.listdir(directory):
            if name.endswith(".json"):
                s = self.get(owner, name[:-5])
                if s:
                    out.append(s)
        return sorted(out, key=lambda s: s.created_at, reverse=True)

    def all_owners(self) -> List[str]:
        if not os.path.isdir(self.base_dir):
            return []
        return [d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))]

    def delete(self, owner: str, schedule_id: str) -> bool:
        path = self._path(owner, schedule_id)
        with self._lock:
            if os.path.exists(path):
                os.remove(path)
                return True
        return False


store = ScheduleStore()


def create(owner: str, dataset_id: str, kind: str = "health-report",
           cadence: str = "weekly", hour: int = 7,
           recipients: Optional[List[str]] = None,
           params: Optional[Dict] = None) -> Schedule:
    if cadence not in CADENCES:
        raise ValueError("Cadence must be one of: {}.".format(
            ", ".join(CADENCES)))
    s = Schedule(schedule_id=uuid.uuid4().hex[:12], owner=owner,
                 dataset_id=dataset_id, kind=kind, cadence=cadence,
                 hour=max(0, min(int(hour), 23)),
                 recipients=list(recipients or []),
                 created_at=time.time(), params=dict(params or {}))
    store.save(s)
    logger.info("schedule %s created (%s %s) for %s",
                s.schedule_id, cadence, kind, owner)
    return s


def due(s: Schedule, now: Optional[datetime] = None) -> bool:
    """Whether this schedule owes a report for the current window."""
    if not s.enabled:
        return False
    now = now or datetime.now()
    if now.hour < s.hour:
        return False
    return s.last_period != period_key(s.cadence, now)


def next_run(s: Schedule, now: Optional[datetime] = None) -> datetime:
    """When it will next produce something — shown in the UI, because a
    schedule whose next run you cannot see is one you do not trust."""
    now = now or datetime.now()
    candidate = now.replace(hour=s.hour, minute=0, second=0, microsecond=0)
    if due(s, now):
        return now
    if candidate <= now:
        candidate += timedelta(days=1)
    if s.cadence == "weekly":
        while candidate.weekday() != 0:          # Monday
            candidate += timedelta(days=1)
    elif s.cadence == "monthly":
        while candidate.day != 1:
            candidate += timedelta(days=1)
    return candidate


def run_due(now: Optional[datetime] = None,
            only_owner: str = "") -> int:
    """Submit a job for every schedule that owes one. Returns how many.

    The period is claimed BEFORE the job is submitted. A tick that
    crashes between claiming and submitting loses one report; the other
    order sends two, and a client who receives the same report twice
    stops trusting the ones that arrive.

    `only_owner` narrows the sweep to one account. The timed loop runs
    every account, because that is its job; a tick driven from the API
    by a signed-in client runs only theirs, so one tenant pressing a
    button cannot consume another tenant's window or spend their CPU.
    """
    from app.services.jobs import submit

    now = now or datetime.now()
    started = 0
    owners = [only_owner] if only_owner else store.all_owners()
    for owner in owners:
        for s in store.list(owner):
            if not due(s, now):
                continue
            claimed = period_key(s.cadence, now)
            s.last_period = claimed
            s.last_run_at = time.time()
            s.last_status = "queued"
            store.save(s)
            try:
                job = submit(s.owner, s.kind, s.dataset_id,
                             {**s.params, "_schedule_id": s.schedule_id})
                s.last_status = "queued:{}".format(job.job_id)
                started += 1
            except Exception as exc:                       # noqa: BLE001
                # The window is claimed before submitting so a crash
                # mid-tick cannot double-send. A submit that RAISED is
                # a different thing: nothing was queued, so holding the
                # claim would mean this schedule silently skips the
                # whole week over a transient failure. Give it back.
                s.last_period = ""
                s.last_status = "failed: {}".format(exc)[:200]
                logger.error("schedule %s could not start: %s",
                             s.schedule_id, exc, exc_info=True)
            store.save(s)
    if started:
        logger.info("%d scheduled report(s) started", started)
    return started


def deliver(schedule_id: str, owner: str, artifact_id: str) -> str:
    """Hand the finished report to whoever the schedule names.

    Returns what happened, in words, so the schedule can display it.
    Email needs SMTP; without it the report is still built and still
    waiting, and the schedule says exactly that rather than implying
    something was sent.
    """
    from app.services import sharing

    s = store.get(owner, schedule_id)
    if s is None:
        return "ready"

    token, expires = sharing.mint(owner, artifact_id,
                                  ttl_days=_link_days(s.cadence))
    sharing.record_share(owner, artifact_id, expires,
                         note="scheduled {}".format(s.cadence))
    s.last_artifact_id = artifact_id

    if not s.recipients:
        s.last_status = "ready"
        store.save(s)
        return "ready"

    sent = _send_email(s, token)
    s.last_status = sent
    store.save(s)
    return sent


def _link_days(cadence: str) -> int:
    """A link outlives its cadence but not by much — a weekly report
    should not still be openable a quarter later."""
    return {"daily": 3, "weekly": 10, "monthly": 40}.get(cadence, 10)


def _send_email(s: Schedule, token: str) -> str:
    host = (os.environ.get("SMTP_HOST") or "").strip()
    if not host:
        return ("ready (no email configured — set SMTP_HOST to have this "
                "delivered instead of waiting here)")

    import smtplib
    from email.message import EmailMessage

    base = (os.environ.get("PUBLIC_BASE_URL") or "").strip().rstrip("/")
    link = "{}/api/shared/{}".format(base, token) if base else \
           "the Reports page"
    msg = EmailMessage()
    msg["Subject"] = "Your {} Analytiq report".format(s.cadence)
    msg["From"] = os.environ.get("SMTP_FROM", "reports@analytiq.local")
    msg["To"] = ", ".join(s.recipients)
    msg.set_content(
        "Your {} report is ready.\n\n{}\n\n"
        "The link expires in {} days. It opens the report only — it is "
        "not a sign-in.\n".format(s.cadence, link, _link_days(s.cadence)))

    try:
        port = int(os.environ.get("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            if os.environ.get("SMTP_STARTTLS", "1") == "1":
                smtp.starttls()
            user = os.environ.get("SMTP_USER")
            if user:
                smtp.login(user, os.environ.get("SMTP_PASSWORD", ""))
            smtp.send_message(msg)
        return "sent to {}".format(", ".join(s.recipients)[:80])
    except Exception as exc:                                # noqa: BLE001
        # The report exists either way; only the delivery failed, and
        # saying which is the difference between a fixable problem and
        # a mystery.
        logger.error("schedule %s built its report but could not email it: %s",
                     s.schedule_id, exc, exc_info=True)
        return "built, but the email failed: {}".format(exc)[:200]
