"""
services/request_context.py — being able to find one request again.

"It failed at about three o'clock" is the most common bug report a
hosted product gets, and there was no way to act on it. Log lines
carried no identifier, so a failure could not be tied to the request
that caused it, to the account that made it, or to the other lines
emitted while it ran. With several clients on one deployment the log is
interleaved and none of it is attributable.

Each request now carries an id. It goes into every log line emitted
while that request is being served, into the response headers, and into
the body of any 500 — so a client can quote the id from their screen and
it can be found in the log immediately, with the account and the route
beside it.

The id is generated here rather than trusted from the client: an
X-Request-ID a caller supplies is echoed for tracing across a proxy, but
only after it is checked for length and character set, because it lands
in log lines and a header.
"""
from __future__ import annotations

import contextvars
import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

HEADER = "X-Request-ID"

# A supplied id is echoed, not trusted: it reaches a log line and a
# response header, and neither should carry newlines or 8 KB of text.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-")
_account: contextvars.ContextVar[str] = contextvars.ContextVar(
    "account", default="-")


def current_request_id() -> str:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Puts the id on every record, so the format string can use it.

    A filter rather than an adapter because it applies to logs from
    every module — including libraries — without any of them knowing
    this exists.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        record.account = _account.get()
        return True


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Tags the request, times it, and logs one line when it finishes."""

    async def dispatch(self, request: Request, call_next):
        supplied = request.headers.get(HEADER, "")
        rid = supplied if _SAFE_ID.match(supplied) else uuid.uuid4().hex[:16]
        token = _request_id.set(rid)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # The one place an unhandled error becomes something a
            # client can quote back. Without the id, "500 Internal
            # Server Error" is the whole of what anyone has to go on.
            elapsed = (time.perf_counter() - started) * 1000
            logger.exception("request failed %s %s in %.0fms",
                             request.method, request.url.path, elapsed)
            _request_id.reset(token)
            return JSONResponse(
                {"detail": "The server hit an unexpected error handling that "
                           "request. Quote this id when reporting it.",
                 "request_id": rid},
                status_code=500, headers={HEADER: rid})

        elapsed = (time.perf_counter() - started) * 1000
        response.headers[HEADER] = rid

        # One line per request, at a level that matches what happened:
        # a 500 is not an INFO event, and a health check is not a
        # WARNING one.
        level = (logging.ERROR if response.status_code >= 500 else
                 logging.WARNING if response.status_code >= 400 else
                 logging.INFO)
        if request.url.path.startswith("/api"):
            logger.log(level, "%s %s -> %d in %.0fms",
                       request.method, request.url.path,
                       response.status_code, elapsed)
        _request_id.reset(token)
        return response


def set_account(name: str) -> None:
    """Called once the request is authenticated, so the log line and
    every line under it can say who it was for."""
    _account.set(name or "-")


def install(app) -> None:
    """Add the middleware and put the id into the log format."""
    app.add_middleware(RequestContextMiddleware)

    root = logging.getLogger()
    filt = RequestIdFilter()
    for handler in root.handlers:
        handler.addFilter(filt)
        try:
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(request_id)s %(account)s] "
                "%(name)s: %(message)s"))
        except Exception:
            logger.debug("could not set the request-id log format",
                         exc_info=True)
