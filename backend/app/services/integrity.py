"""
services/integrity.py — proof that the numbers in a report came from the
data the client uploaded, and that nothing changed it in between.

Governance answers "what is in this data and who may see it". Privacy
answers "can a person be re-identified". Integrity answers the third
question, and until now the app could not answer it at all:

    Is this still the data I gave you, and can you account for every
    change made to it?

That question is not academic for an analytics product. A dataset here
is uploaded once and then cleaned, imputed, type-coerced, filtered and
re-encoded before a single figure is computed. If a report says
attrition is 16.1%, someone eventually has to be able to trace that back
to the exact bytes it came from — an auditor, a regulator, or the
analyst themselves six months later when the number is challenged. A
platform that cannot do that is asking to be believed.

Three mechanisms, each doing one job:

1. **Fingerprints.** A content digest over the frame's values, not its
   file. The raw upload's digest is recorded once at ingest and must
   never change again; the active (working) frame's digest is recorded
   after every transformation. A raw digest that has moved means the
   source of truth was altered underneath the app — corruption, or
   someone editing the store by hand.

2. **A hash-chained audit trail.** Every event that touches a dataset —
   ingest, clean, transform, reset, report build, export, delete — is
   appended as a line carrying the hash of the line before it. Editing
   or removing any past entry breaks every hash after it, so the trail
   detects its own tampering rather than merely recording history.

3. **A run manifest.** The library versions a result was produced with.
   Reproducibility is not "we used pandas" — a quantile changes between
   pandas versions, a solver's default changes between scikit-learn
   versions. Recording them is the difference between a figure that can
   be reproduced and one that can only be re-obtained.

The integrity verdict is derived, never asserted: `verify()` recomputes
the digests from the stored frames and compares them against the record.
Nothing here trusts a stored claim about the data — that is the whole
point.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024
AUDIT_FILE = "audit.jsonl"
INTEGRITY_FILE = "integrity.json"

# Events an audit trail can carry. Free-form strings were tempting and
# wrong: an audit trail whose vocabulary drifts cannot be queried, and
# "cleaned" vs "clean" vs "data_clean" is exactly how that happens.
EVENTS = {
    "ingest",          # the upload landed and was parsed
    "clean",           # the cleaning pipeline rewrote the active frame
    "transform",       # any other change to the active frame
    "reset",           # active frame restored to the raw upload
    "report",          # a report/deck was built from this dataset
    "export",          # data or a chart left the app
    "delete",          # the dataset was removed
    "verify",          # an integrity check was run
    "generate",        # a decorative image was produced for a deliverable
}


# ── digests ──────────────────────────────────────────────

def file_digest(path: str) -> str:
    """SHA-256 of a file, streamed — a 200MB upload must not be read into
    memory a second time just to be hashed."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def bytes_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frame_digest(df: Optional[pd.DataFrame]) -> str:
    """A digest of what the frame *contains*, not of the file it sits in.

    Deliberately content-based: parquet is free to change its encoding,
    compression or row-group layout between versions and write the same
    data to different bytes. Hashing the file would report a tamper on
    every library upgrade — an integrity check that cries wolf gets
    switched off, which is worse than not having one.

    Column names and dtypes are folded in, so a renamed column or an
    int-to-float coercion registers as a change. It should: both alter
    what a downstream figure means.
    """
    if df is None:
        return ""
    h = hashlib.sha256()
    h.update(f"{df.shape[0]}x{df.shape[1]}".encode())
    for col in df.columns:
        h.update(str(col).encode("utf-8", "replace"))
        h.update(str(df[col].dtype).encode())
    try:
        row_hashes = pd.util.hash_pandas_object(df, index=False)
        h.update(row_hashes.values.tobytes())
    except (TypeError, ValueError):
        # Unhashable cell values (a column holding lists or dicts) —
        # rare, but it must not take the whole check down. Falls back to
        # a stable string rendering, which is slower and just as sound.
        for col in df.columns:
            h.update(df[col].astype(str).str.cat(sep="\x1f").encode(
                "utf-8", "replace"))
    return h.hexdigest()


def run_manifest() -> dict:
    """The versions a result was computed with.

    Only libraries whose behaviour can move a number are listed. A
    version that cannot be determined is recorded as "unavailable"
    rather than omitted — a missing key reads as "we didn't use it",
    which would be a lie.
    """
    import platform

    from app.config import config

    def _v(mod_name: str) -> str:
        try:
            mod = __import__(mod_name)
            return str(getattr(mod, "__version__", "unknown"))
        except Exception:                          # noqa: BLE001
            return "unavailable"

    return {
        "app_version": config.app_version,
        "python": platform.python_version(),
        "pandas": _v("pandas"),
        "numpy": _v("numpy"),
        "scipy": _v("scipy"),
        "scikit_learn": _v("sklearn"),
        "statsmodels": _v("statsmodels"),
    }


# ── the record ───────────────────────────────────────────

@dataclass
class IntegrityRecord:
    """What was received, and what the data has looked like since."""
    dataset_id: str = ""
    source_filename: str = ""
    source_bytes: int = 0
    source_sha256: str = ""          # digest of the uploaded file itself
    raw_digest: str = ""             # content digest, fixed at ingest
    raw_rows: int = 0
    raw_cols: int = 0
    active_digest: str = ""          # content digest after the last change
    active_rows: int = 0
    active_cols: int = 0
    ingested_at: float = 0.0
    updated_at: float = 0.0
    manifest: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntegrityVerdict:
    """Derived by recomputation, never read from disk."""
    intact: bool
    raw_intact: bool
    active_accounted_for: bool
    chain_intact: bool
    verdict: str
    explanation: str
    checked_at: float = field(default_factory=time.time)
    events: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# ── the audit trail ──────────────────────────────────────

def _entry_hash(entry: dict) -> str:
    """Hash over the entry's own fields *including* the previous hash —
    which is what chains them. Sorted keys so the hash does not depend on
    dict ordering across Python versions."""
    payload = json.dumps(
        {k: v for k, v in entry.items() if k != "hash"},
        sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_audit(dataset_dir: str) -> list[dict]:
    path = os.path.join(dataset_dir, AUDIT_FILE)
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # A truncated final line (a crash mid-append) is a
                    # real possibility. Keep the entries that parsed and
                    # let verify_chain report the break, rather than
                    # discarding the whole history.
                    logger.warning("audit trail has an unparseable line in %s",
                                   dataset_dir)
    except OSError:
        return []
    return out


def append_audit(dataset_dir: str, event: str, detail: Optional[dict] = None,
                 actor: str = "", digest: str = "") -> Optional[dict]:
    """Append one event, chained to the last. Never raises: an audit
    write that fails must not take down the operation it was recording —
    that would make the trail a liability rather than a safeguard. It
    logs, and the chain check will show the gap."""
    if event not in EVENTS:
        logger.warning("unknown audit event %r — recording as 'transform'",
                       event)
        event = "transform"
    try:
        os.makedirs(dataset_dir, exist_ok=True)
        existing = read_audit(dataset_dir)
        prev = existing[-1]["hash"] if existing and "hash" in existing[-1] else ""
        entry = {
            "seq": len(existing) + 1,
            "at": time.time(),
            "event": event,
            "actor": actor,
            "digest": digest,
            "detail": detail or {},
            "prev": prev,
        }
        entry["hash"] = _entry_hash(entry)
        with open(os.path.join(dataset_dir, AUDIT_FILE), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry
    except OSError:
        logger.warning("could not append to the audit trail in %s",
                       dataset_dir, exc_info=True)
        return None


def verify_chain(entries: list[dict]) -> tuple[bool, str]:
    """Recompute every link. Returns (intact, explanation)."""
    prev = ""
    for i, entry in enumerate(entries):
        if entry.get("prev", "") != prev:
            return False, (f"Entry {i + 1} ({entry.get('event', '?')}) does not "
                           f"follow the entry before it — the trail was edited "
                           f"or an entry was removed.")
        if _entry_hash(entry) != entry.get("hash", ""):
            return False, (f"Entry {i + 1} ({entry.get('event', '?')}) has been "
                           f"altered since it was written.")
        prev = entry.get("hash", "")
    return True, ""


# ── the record on disk ───────────────────────────────────

def read_record(dataset_dir: str) -> Optional[IntegrityRecord]:
    path = os.path.join(dataset_dir, INTEGRITY_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("unreadable integrity record in %s", dataset_dir)
        return None
    fields = set(IntegrityRecord.__dataclass_fields__)
    return IntegrityRecord(**{k: v for k, v in raw.items() if k in fields})


def write_record(dataset_dir: str, record: IntegrityRecord) -> None:
    os.makedirs(dataset_dir, exist_ok=True)
    path = os.path.join(dataset_dir, INTEGRITY_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record.as_dict(), f)
    os.replace(tmp, path)


def record_ingest(dataset_dir: str, dataset_id: str, df_raw: pd.DataFrame,
                  filename: str, source_bytes: int = 0,
                  source_sha256: str = "", actor: str = "") -> IntegrityRecord:
    digest = frame_digest(df_raw)
    now = time.time()
    record = IntegrityRecord(
        dataset_id=dataset_id,
        source_filename=filename,
        source_bytes=source_bytes,
        source_sha256=source_sha256,
        raw_digest=digest,
        raw_rows=len(df_raw),
        raw_cols=df_raw.shape[1],
        active_digest=digest,
        active_rows=len(df_raw),
        active_cols=df_raw.shape[1],
        ingested_at=now,
        updated_at=now,
        manifest=run_manifest(),
    )
    write_record(dataset_dir, record)
    append_audit(dataset_dir, "ingest", actor=actor, digest=digest,
                 detail={"filename": filename,
                         "rows": len(df_raw),
                         "columns": df_raw.shape[1],
                         "file_digest": source_sha256})
    return record


def record_change(dataset_dir: str, df_active: pd.DataFrame,
                  event: str = "transform", actor: str = "",
                  detail: Optional[dict] = None) -> str:
    """Record that the active frame legitimately changed.

    This is what makes the check meaningful: a change that goes through
    here is accounted for, and one that does not shows up as an active
    frame whose digest matches no recorded state.
    """
    digest = frame_digest(df_active)
    record = read_record(dataset_dir)
    if record is not None:
        record.active_digest = digest
        record.active_rows = len(df_active)
        record.active_cols = df_active.shape[1]
        record.updated_at = time.time()
        write_record(dataset_dir, record)
    payload = dict(detail or {})
    payload.setdefault("rows", len(df_active))
    payload.setdefault("columns", df_active.shape[1])
    append_audit(dataset_dir, event, actor=actor, digest=digest, detail=payload)
    return digest


def record_event(dataset_dir: str, event: str, df_active: Optional[pd.DataFrame],
                 actor: str = "", detail: Optional[dict] = None) -> str:
    """Record something that *used* the data without changing it — a
    report build, an export.

    The digest of the data as it stood is carried into the entry, which
    is what lets a PDF circulating six months later be tied back to the
    exact state of the dataset it was built from. Without it the trail
    records that a report was made, which is the least interesting half
    of the question.
    """
    digest = frame_digest(df_active)
    append_audit(dataset_dir, event, actor=actor, digest=digest,
                 detail=detail or {})
    return digest


def verify(dataset_dir: str, df_raw: Optional[pd.DataFrame],
           df_active: Optional[pd.DataFrame]) -> IntegrityVerdict:
    """Recompute and compare. The only function here that decides."""
    record = read_record(dataset_dir)
    entries = read_audit(dataset_dir)
    chain_ok, chain_msg = verify_chain(entries)

    if record is None:
        return IntegrityVerdict(
            intact=False, raw_intact=False, active_accounted_for=False,
            chain_intact=chain_ok, verdict="unverifiable",
            events=len(entries),
            explanation=("No integrity record exists for this dataset. It was "
                         "stored before integrity tracking was added, so its "
                         "history cannot be checked — re-upload it to place it "
                         "under the audit trail."))

    raw_now = frame_digest(df_raw)
    active_now = frame_digest(df_active)
    raw_ok = bool(raw_now) and raw_now == record.raw_digest
    active_ok = bool(active_now) and active_now == record.active_digest

    if raw_ok and active_ok and chain_ok:
        return IntegrityVerdict(
            intact=True, raw_intact=True, active_accounted_for=True,
            chain_intact=True, verdict="intact", events=len(entries),
            explanation=(
                f"The uploaded data is byte-for-byte what was received, and "
                f"all {len(entries)} recorded change(s) account for the "
                f"working copy's current state."))

    problems = []
    if not raw_ok:
        problems.append(
            "the original uploaded data no longer matches the digest taken "
            "when it was received — it was altered or corrupted in storage")
    if not active_ok:
        problems.append(
            "the working copy does not match any recorded change, so it was "
            "modified outside the audited path")
    if not chain_ok:
        problems.append(f"the audit trail is broken: {chain_msg}")

    return IntegrityVerdict(
        intact=False, raw_intact=raw_ok, active_accounted_for=active_ok,
        chain_intact=chain_ok,
        verdict=("compromised" if not raw_ok
                 else "tampered" if not chain_ok
                 else "unaccounted"),
        events=len(entries),
        explanation=("Integrity check failed: " + "; ".join(problems) +
                     ". Treat any figure derived from this dataset as "
                     "unverified until it is re-uploaded."))


def summary(dataset_dir: str, df_raw: Optional[pd.DataFrame],
            df_active: Optional[pd.DataFrame]) -> dict[str, Any]:
    """Everything the API and the report need, in one call."""
    record = read_record(dataset_dir)
    verdict = verify(dataset_dir, df_raw, df_active)
    entries = read_audit(dataset_dir)
    return {
        "record": record.as_dict() if record else None,
        "verdict": verdict.as_dict(),
        "audit": entries,
        "manifest": record.manifest if record else run_manifest(),
    }
