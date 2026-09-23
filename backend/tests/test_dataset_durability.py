"""
A dataset has to outlive the container that received it.

Datasets lived on the container's local disk and nowhere else. Render
happens to mount a volume; railway.json declares none at all, so every
deploy wiped every upload. And on either platform a second instance has
a second disk, so a dataset was present on one request and missing on
the next — the same account, seconds apart.

The tests below simulate exactly that: write through one store, then
build a *different* store on an empty directory sharing only the blob
backing, which is what a redeployed or second container is.
"""
from __future__ import annotations

import os
import shutil

import pandas as pd
import pytest

from app.services.blobstore import LocalBlobStore
from app.services.dataset_store import DatasetStore


@pytest.fixture()
def df():
    return pd.DataFrame({
        "region": ["East", "West", "North", "South"] * 25,
        "revenue": [100 + i for i in range(100)],
        "won": [1, 0, 0, 0] * 25,
    })


@pytest.fixture()
def durable(tmp_path):
    """One blob store, standing in for the bucket both containers share."""
    return LocalBlobStore(str(tmp_path / "bucket"))


@pytest.fixture()
def first_container(tmp_path, durable):
    return DatasetStore(base_dir=str(tmp_path / "disk-a"), blobs=durable)


def _restarted(tmp_path, durable, name="disk-b"):
    """A container that has never seen this data: empty disk, same
    bucket. A redeploy, or simply the other instance."""
    return DatasetStore(base_dir=str(tmp_path / name), blobs=durable)


# ══════════════════════════════════════════════════════════
#  Surviving the restart
# ══════════════════════════════════════════════════════════

def test_a_dataset_is_listed_by_a_container_that_never_saw_it(
        tmp_path, durable, first_container, df):
    """The symptom the client reports: upload, come back, nothing there."""
    meta = first_container.create("amy", df, "sales.csv", 0.1)

    fresh = _restarted(tmp_path, durable)
    listed = [m.dataset_id for m in fresh.list_meta("amy")]

    assert meta.dataset_id in listed, \
        "a restarted container told the account it had no datasets"


def test_the_frame_itself_comes_back(tmp_path, durable, first_container, df):
    meta = first_container.create("amy", df, "sales.csv", 0.1)

    fresh = _restarted(tmp_path, durable)
    back = fresh.get_df("amy", meta.dataset_id)

    assert back is not None
    assert len(back) == len(df)
    assert list(back.columns) == list(df.columns)
    pd.testing.assert_frame_equal(back.reset_index(drop=True),
                                  df.reset_index(drop=True),
                                  check_dtype=False)


def test_the_metadata_comes_back(tmp_path, durable, first_container, df):
    meta = first_container.create("amy", df, "sales.csv", 0.1)

    fresh = _restarted(tmp_path, durable)
    back = fresh.get_meta("amy", meta.dataset_id)

    assert back is not None
    assert back.filename == "sales.csv"
    assert back.rows == len(df)


def test_a_cleaned_dataset_keeps_its_cleaned_version(
        tmp_path, durable, first_container, df):
    """update_active writes a second frame. Restoring only the raw one
    would silently undo the client's cleaning."""
    meta = first_container.create("amy", df, "sales.csv", 0.1)
    cleaned = df.drop(columns=["won"])
    first_container.update_active("amy", meta.dataset_id, cleaned,
                                  event="cleaned")

    fresh = _restarted(tmp_path, durable)
    active = fresh.get_df("amy", meta.dataset_id)
    raw = fresh.get_raw_df("amy", meta.dataset_id)

    assert "won" not in active.columns, "the cleaning was lost"
    assert "won" in raw.columns, "the original was lost"


def test_the_audit_trail_travels_with_the_data(
        tmp_path, durable, first_container, df):
    """A restored dataset that cannot say what was done to it defeats
    the point of having an integrity record at all."""
    meta = first_container.create("amy", df, "sales.csv", 0.1)
    first_container.update_active("amy", meta.dataset_id,
                                  df.drop(columns=["won"]), event="cleaned")

    fresh = _restarted(tmp_path, durable)
    summary = fresh.integrity("amy", meta.dataset_id)

    assert summary is not None


# ══════════════════════════════════════════════════════════
#  Not resurrecting what was deleted
# ══════════════════════════════════════════════════════════

def test_a_deleted_dataset_stays_deleted(
        tmp_path, durable, first_container, df):
    """Deleting only the local copy would let the next cold container
    restore it — the client deletes their data and it comes back."""
    meta = first_container.create("amy", df, "sales.csv", 0.1)
    assert first_container.delete("amy", meta.dataset_id) is True

    fresh = _restarted(tmp_path, durable)

    assert fresh.get_meta("amy", meta.dataset_id) is None
    assert meta.dataset_id not in [m.dataset_id for m in fresh.list_meta("amy")]


# ══════════════════════════════════════════════════════════
#  Isolation still holds across the durable store
# ══════════════════════════════════════════════════════════

def test_one_account_cannot_see_another_through_the_durable_store(
        tmp_path, durable, first_container, df):
    """The whole isolation story would be worth nothing if the restore
    path ignored the owner."""
    amy = first_container.create("amy", df, "amy.csv", 0.1)
    bob = first_container.create("bob", df, "bob.csv", 0.1)

    fresh = _restarted(tmp_path, durable)

    assert [m.dataset_id for m in fresh.list_meta("amy")] == [amy.dataset_id]
    assert [m.dataset_id for m in fresh.list_meta("bob")] == [bob.dataset_id]
    assert fresh.get_df("bob", amy.dataset_id) is None


# ══════════════════════════════════════════════════════════
#  Without a bucket, nothing changes
# ══════════════════════════════════════════════════════════

def test_a_local_only_deployment_behaves_exactly_as_before(tmp_path, df):
    """No S3 configured is the common case — a laptop, a test, a single
    container whose operator accepts the trade. It must not pay for any
    of this."""
    store = DatasetStore(base_dir=str(tmp_path / "solo"))

    assert store.blobs is None
    meta = store.create("amy", df, "sales.csv", 0.1)
    assert store.get_df("amy", meta.dataset_id) is not None
    assert [m.dataset_id for m in store.list_meta("amy")] == [meta.dataset_id]


def test_the_analysis_caches_are_not_shipped_to_the_bucket(
        tmp_path, durable, first_container, df):
    """A fitted model regenerates from the frame in seconds. Copying it
    to object storage on every analysis would cost more than recomputing
    it ever does."""
    meta = first_container.create("amy", df, "sales.csv", 0.1)
    first_container.cache_set("amy", meta.dataset_id, "stats", {"n": 100})

    keys = durable.list_prefix("amy/{}".format(meta.dataset_id))

    assert keys, "the dataset itself should be backed up"
    assert not [k for k in keys if "cache_" in k], \
        "an analysis cache was shipped to durable storage"


def test_a_backup_failure_does_not_lose_the_upload(tmp_path, df, monkeypatch):
    """A dataset the client can still use beats no dataset at all."""
    class _Broken(LocalBlobStore):
        def put(self, key, data):
            raise OSError("bucket unreachable")

    store = DatasetStore(base_dir=str(tmp_path / "disk"),
                         blobs=_Broken(str(tmp_path / "bucket")))
    meta = store.create("amy", df, "sales.csv", 0.1)

    assert store.get_df("amy", meta.dataset_id) is not None
