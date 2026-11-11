"""
services/alerts.py — what went wrong recently, and telling someone.

Every unexpected error is kept in a short in-memory list the admin screen
shows, so the owner sees a failure without reading server logs. With
ALERT_WEBHOOK_URL set, it is also posted to that address — at most once
every five minutes, with a count of what was held back, so a failing page
hit a hundred times is one message, not a hundred.
"""
from __future__ import annotations

import collections
import json
import logging
import threading
import time
import urllib.request

from app.config import config

logger = logging.getLogger(__name__)

_recent: collections.deque = collections.deque(maxlen=50)
_lock = threading.Lock()
_last_sent = 0.0
_held = 0
MIN_GAP_SEC = 300


def recent_errors() -> list:
    with _lock:
        return list(reversed(_recent))


def _post(url: str, text: str) -> None:
    try:
        if not url.lower().startswith("https://"):
            raise ValueError("ALERT_WEBHOOK_URL must start with https://")
        req = urllib.request.Request(
            url, data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        # The address is checked to be https just above.
        urllib.request.urlopen(req, timeout=10).close()  # nosec B310
    except Exception as e:
        logger.warning("alert could not be sent: %s", e)


def record_error(reference: str, method: str, path: str, user: str, error: BaseException) -> None:
    """Note an unexpected error and, if configured, send an alert. Never
    raises and never waits on the network: this runs while a client is
    waiting for their answer."""
    global _last_sent, _held
    entry = {"reference": reference, "at": time.time(), "method": method, "path": path,
             "user": user, "error": "{}: {}".format(type(error).__name__, str(error)[:200])}
    with _lock:
        _recent.append(entry)
        url = config.alert_webhook_url
        if not url:
            return
        now = time.time()
        if now - _last_sent < MIN_GAP_SEC:
            _held += 1
            return
        extra, _held, _last_sent = _held, 0, now
    text = "{} error: {} {} (user {}) - {} - reference {}".format(
        config.app_name, method, path, user, entry["error"], reference)
    if extra:
        text += " (+{} more since the last alert)".format(extra)
    threading.Thread(target=_post, args=(url, text), daemon=True).start()
