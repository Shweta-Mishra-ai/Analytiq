"""
services/upload_limits.py — reading an upload without handing the
process to whoever sent it.

Every upload endpoint began `data = await file.read()`, which pulls the
whole body into memory and only then checks how big it was. The size
limit was real and it was enforced too late to matter.

Measured, on an ordinary CSV rather than an attack:

    payload      79 MB   (well under the configured 200 MB limit)
    RSS before  203 MB
    RSS after 1,704 MB

A twenty-one-fold amplification, from a file the product is supposed to
accept. On the 512 MB and 1 GB containers this deploys to, that is the
process gone — not for the client who uploaded it, for everyone using
the service at that moment. It needs no malice; a client with a big
export finds it on their first day.

Two separate costs, so two separate caps:

  * the RAW BYTES, capped here while reading, in chunks, so the limit
    is enforced before the memory is committed rather than after; and
  * the PARSED FRAME, which is where the other twenty-fold went —
    pandas holds a 79 MB CSV as something far larger, and no byte cap
    can see that coming. That one is bounded by the row and column
    limits the loader already applies, which is why they stay.

A refusal here is a 413 with the actual number in it, because "file too
large" without a size is a support ticket.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# How much is read at a time. Large enough that a 200 MB file is not a
# million round trips, small enough that overshooting the cap costs at
# most this much.
CHUNK_BYTES = 1024 * 1024


class UploadTooLarge(Exception):
    """Raised with a message written for the person who sent the file."""

    def __init__(self, limit_mb: float, seen_mb: float):
        self.limit_mb = limit_mb
        self.seen_mb = seen_mb
        super().__init__(
            "That file is larger than this server accepts: it is over "
            "{:,.0f} MB and the limit is {:,.0f} MB. Filter or sample the "
            "export before uploading, or split it into parts — the "
            "analysis works the same on a representative sample."
            .format(seen_mb, limit_mb))


async def read_capped(file, limit_mb: float) -> bytes:
    """Read an UploadFile, refusing past `limit_mb` without holding it.

    The refusal happens at the chunk that crosses the line, so the peak
    memory is the limit plus one chunk rather than whatever was sent.
    """
    limit_bytes = int(limit_mb * 1024 * 1024)
    chunks: list = []
    total = 0

    while True:
        chunk = await file.read(CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            # Drop what was read before raising: on a large upload this
            # is the difference between releasing the memory now and
            # holding it until the exception has finished propagating.
            chunks.clear()
            logger.warning("upload refused at %.0f MB (limit %.0f MB)",
                           total / 1024 / 1024, limit_mb)
            raise UploadTooLarge(limit_mb, total / 1024 / 1024)
        chunks.append(chunk)

    data = b"".join(chunks)
    chunks.clear()
    return data
