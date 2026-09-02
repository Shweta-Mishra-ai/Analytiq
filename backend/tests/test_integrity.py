"""
Tests for the data integrity layer.

The property under test is not "does it compute a hash" — it is: can a
change to the data be made without the app noticing? Most of these tests
therefore make a change behind the store's back and assert that the check
catches it.
"""
from __future__ import annotations

import io
import json
import os
import tempfile

import pandas as pd
import pytest

from app.services import integrity as I
from app.services.dataset_store import DatasetStore


@pytest.fixture
def frame():
    return pd.DataFrame({
        "employee": ["A", "B", "C"],
        "salary": [50000, 61000, 72000],
        "left": [True, False, False],
    })


@pytest.fixture
def workspace():
    return tempfile.mkdtemp()


# ── digests ──────────────────────────────────────────────

def test_digest_is_stable_across_identical_frames(frame):
    assert I.frame_digest(frame) == I.frame_digest(frame.copy())


def test_digest_changes_when_a_value_changes(frame):
    other = frame.copy()
    other.loc[0, "salary"] = 50001
    assert I.frame_digest(other) != I.frame_digest(frame)


def test_digest_changes_when_a_column_is_renamed(frame):
    """A rename alters what a downstream figure means, so it must
    register as a change."""
    assert I.frame_digest(frame.rename(columns={"salary": "pay"})) \
        != I.frame_digest(frame)


def test_digest_changes_when_a_dtype_changes(frame):
    other = frame.copy()
    other["salary"] = other["salary"].astype(float)
    assert I.frame_digest(other) != I.frame_digest(frame)


def test_digest_survives_a_parquet_round_trip(frame, workspace):
    """Content-based, not file-based: parquet is free to change its
    encoding between versions and write the same data to different
    bytes. A check that cries wolf on a library upgrade gets switched
    off, which is worse than not having one."""
    from app.services.frame_io import read_frame, write_frame
    path = os.path.join(workspace, "f.parquet")
    write_frame(path, frame)
    assert I.frame_digest(read_frame(path)) == I.frame_digest(frame)


def test_digest_handles_unhashable_cell_values():
    """A column of lists must not take the whole check down."""
    df = pd.DataFrame({"tags": [["a", "b"], ["c"], []], "n": [1, 2, 3]})
    first = I.frame_digest(df)
    assert first
    assert first == I.frame_digest(df.copy())


def test_bytes_digest_matches_hashlib():
    import hashlib
    payload = b"name,value\nA,1\n"
    assert I.bytes_digest(payload) == hashlib.sha256(payload).hexdigest()


def test_file_digest_matches_bytes_digest(workspace):
    path = os.path.join(workspace, "src.csv")
    payload = b"a,b\n1,2\n"
    with open(path, "wb") as f:
        f.write(payload)
    assert I.file_digest(path) == I.bytes_digest(payload)


def test_file_digest_of_a_missing_file_is_empty(workspace):
    assert I.file_digest(os.path.join(workspace, "nope.csv")) == ""


# ── the audit chain ──────────────────────────────────────

def test_ingest_starts_the_trail(workspace, frame):
    I.record_ingest(workspace, "ds", frame, "hr.csv", 120, "abc123")
    entries = I.read_audit(workspace)
    assert [e["event"] for e in entries] == ["ingest"]
    assert entries[0]["detail"]["filename"] == "hr.csv"
    assert entries[0]["prev"] == ""


def test_each_entry_links_to_the_one_before_it(workspace, frame):
    I.record_ingest(workspace, "ds", frame, "hr.csv")
    I.record_change(workspace, frame, "clean")
    entries = I.read_audit(workspace)
    assert entries[1]["prev"] == entries[0]["hash"]
    assert I.verify_chain(entries)[0]


def test_editing_a_past_entry_breaks_the_chain(workspace, frame):
    I.record_ingest(workspace, "ds", frame, "hr.csv")
    I.record_change(workspace, frame, "clean")
    path = os.path.join(workspace, I.AUDIT_FILE)
    lines = open(path).read().strip().split("\n")
    entry = json.loads(lines[0])
    entry["detail"]["filename"] = "something-else.csv"
    lines[0] = json.dumps(entry)
    open(path, "w").write("\n".join(lines) + "\n")

    ok, why = I.verify_chain(I.read_audit(workspace))
    assert not ok
    assert "altered" in why


def test_removing_an_entry_breaks_the_chain(workspace, frame):
    I.record_ingest(workspace, "ds", frame, "hr.csv")
    I.record_change(workspace, frame, "clean")
    I.record_change(workspace, frame, "transform")
    path = os.path.join(workspace, I.AUDIT_FILE)
    lines = open(path).read().strip().split("\n")
    open(path, "w").write(lines[0] + "\n" + lines[2] + "\n")

    ok, why = I.verify_chain(I.read_audit(workspace))
    assert not ok
    assert "removed" in why or "follow" in why


def test_an_unknown_event_is_recorded_not_dropped(workspace, frame):
    I.record_ingest(workspace, "ds", frame, "hr.csv")
    I.append_audit(workspace, "sorcery", {"x": 1})
    assert [e["event"] for e in I.read_audit(workspace)][-1] == "transform"


def test_an_audit_write_failure_never_raises(workspace, frame, monkeypatch):
    """The trail is a safeguard; if it could take down the operation it
    was recording, it would be a liability."""
    monkeypatch.setattr(I, "read_audit",
                        lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        I.read_audit(workspace)          # the stub really does raise
    monkeypatch.setattr("builtins.open",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert I.append_audit(workspace, "clean") is None


# ── verdicts ─────────────────────────────────────────────

def test_untouched_data_verifies(workspace, frame):
    I.record_ingest(workspace, "ds", frame, "hr.csv")
    verdict = I.verify(workspace, frame, frame)
    assert verdict.intact
    assert verdict.verdict == "intact"


def test_a_recorded_change_still_verifies(workspace, frame):
    I.record_ingest(workspace, "ds", frame, "hr.csv")
    cleaned = frame.assign(salary=frame["salary"] + 1)
    I.record_change(workspace, cleaned, "clean")
    assert I.verify(workspace, frame, cleaned).intact


def test_an_unrecorded_change_is_caught(workspace, frame):
    """The whole point: a change that did not go through the audited
    path shows up as a working copy matching no recorded state."""
    I.record_ingest(workspace, "ds", frame, "hr.csv")
    tampered = frame.assign(salary=[1, 2, 3])
    verdict = I.verify(workspace, frame, tampered)
    assert not verdict.intact
    assert verdict.verdict == "unaccounted"
    assert not verdict.active_accounted_for
    assert verdict.raw_intact


def test_altering_the_original_upload_is_caught(workspace, frame):
    I.record_ingest(workspace, "ds", frame, "hr.csv")
    corrupted = frame.assign(employee=["X", "Y", "Z"])
    verdict = I.verify(workspace, corrupted, frame)
    assert verdict.verdict == "compromised"
    assert not verdict.raw_intact
    assert "no longer matches" in verdict.explanation


def test_a_broken_trail_over_intact_data_is_its_own_verdict(workspace, frame):
    """"The data changed" and "the record of changes was edited" are
    different failures with different responses."""
    I.record_ingest(workspace, "ds", frame, "hr.csv")
    path = os.path.join(workspace, I.AUDIT_FILE)
    entry = json.loads(open(path).read().strip())
    entry["actor"] = "someone-else"
    open(path, "w").write(json.dumps(entry) + "\n")

    verdict = I.verify(workspace, frame, frame)
    assert verdict.verdict == "tampered"
    assert verdict.raw_intact
    assert not verdict.chain_intact


def test_a_dataset_with_no_record_is_unverifiable_not_intact(workspace, frame):
    """Absence of evidence must not read as a pass."""
    verdict = I.verify(workspace, frame, frame)
    assert not verdict.intact
    assert verdict.verdict == "unverifiable"


def test_the_manifest_records_versions_that_move_results():
    manifest = I.run_manifest()
    for key in ("pandas", "numpy", "scipy", "scikit_learn", "python"):
        assert manifest[key], f"{key} missing from the run manifest"


# ── through the store ────────────────────────────────────

@pytest.fixture
def store(workspace):
    return DatasetStore(workspace)


def test_upload_is_recorded_with_the_source_digest(store, frame):
    meta = store.create("u", frame, "hr.csv", 0.1,
                        source_bytes=99, source_sha256="deadbeef")
    result = store.integrity("u", meta.dataset_id)
    assert result["record"]["source_sha256"] == "deadbeef"
    assert result["record"]["source_bytes"] == 99
    assert result["verdict"]["verdict"] == "intact"


def test_cleaning_through_the_store_stays_accounted_for(store, frame):
    meta = store.create("u", frame, "hr.csv", 0.1)
    store.update_active("u", meta.dataset_id, frame.assign(salary=[1, 2, 3]),
                        event="clean", detail={"mode": "non-destructive"})
    result = store.integrity("u", meta.dataset_id)
    assert result["verdict"]["intact"]
    assert [e["event"] for e in result["audit"]] == ["ingest", "clean"]


def test_a_write_behind_the_stores_back_is_caught(store, frame):
    """Simulates the real failure: someone editing the stored parquet
    directly, or a bug writing to it outside update_active."""
    meta = store.create("u", frame, "hr.csv", 0.1)
    from app.services.frame_io import write_frame
    path = os.path.join(store.dataset_dir("u", meta.dataset_id), "active.parquet")
    write_frame(path, frame.assign(salary=[9, 9, 9]))
    store._mem.clear()                   # force a read from disk

    result = store.integrity("u", meta.dataset_id)
    assert not result["verdict"]["intact"]
    assert result["verdict"]["verdict"] == "unaccounted"


def test_reset_is_recorded_as_its_own_event(store, frame):
    meta = store.create("u", frame, "hr.csv", 0.1)
    store.update_active("u", meta.dataset_id, frame.head(1), event="clean")
    store.reset_active("u", meta.dataset_id)
    result = store.integrity("u", meta.dataset_id)
    assert [e["event"] for e in result["audit"]] == ["ingest", "clean", "reset"]
    assert result["verdict"]["intact"]


def test_using_the_data_is_recorded_without_changing_the_digest(store, frame):
    meta = store.create("u", frame, "hr.csv", 0.1)
    before = store.integrity("u", meta.dataset_id)["record"]["active_digest"]
    store.record_event("u", meta.dataset_id, "report", {"format": "pdf"})
    after = store.integrity("u", meta.dataset_id)
    assert after["record"]["active_digest"] == before
    assert after["audit"][-1]["event"] == "report"
    assert after["audit"][-1]["digest"] == before, \
        "a report entry must name the data state it was built from"


def test_integrity_of_an_unknown_dataset_is_none(store):
    assert store.integrity("u", "nope") is None


def test_one_owner_cannot_see_anothers_trail(store, frame):
    meta = store.create("owner-a", frame, "hr.csv", 0.1)
    assert store.integrity("owner-b", meta.dataset_id) is None


# ── through the API ──────────────────────────────────────

def test_the_endpoint_reports_a_verifiable_digest(monkeypatch, workspace):
    import hashlib
    from fastapi.testclient import TestClient
    from app.config import config
    monkeypatch.setattr(config, "data_dir", workspace)
    from app.services import dataset_store as ds_mod
    monkeypatch.setattr(ds_mod, "store", DatasetStore(workspace))
    import app.api.datasets as datasets_api
    monkeypatch.setattr(datasets_api, "store", ds_mod.store)
    import app.api.reports as reports_api
    monkeypatch.setattr(reports_api, "store", ds_mod.store)

    from app.main import app
    client = TestClient(app)
    payload = b"dept,salary\nSales,100\nEng,200\n"
    up = client.post("/api/datasets/upload",
                     files={"file": ("t.csv", io.BytesIO(payload), "text/csv")})
    assert up.status_code == 200
    ds_id = up.json()["meta"]["dataset_id"]

    body = client.get(f"/api/datasets/{ds_id}/integrity").json()
    assert body["verdict"]["verdict"] == "intact"
    # The claim the report makes to the reader: run sha256sum on the
    # original file and it matches.
    assert body["record"]["source_sha256"] == hashlib.sha256(payload).hexdigest()

    client.get(f"/api/reports/{ds_id}/csv")
    after = client.get(f"/api/datasets/{ds_id}/integrity").json()
    assert after["audit"][-1]["event"] == "export"
