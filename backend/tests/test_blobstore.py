"""
Where a finished report actually lives.

Artifacts were written to DATA_DIR, which on a container platform is
scratch space: it survives a request, not a deploy. Share links,
scheduled delivery and background jobs all exist so that a report
outlives the request that built it — and a disk that does not outlive
the process quietly undoes all three. A report built on Monday and sent
as a link is a 404 after Tuesday's restart.

The artifact store no longer knows where bytes are. These tests hold
both backends to the same contract, because the only way that
indirection is worth anything is if the S3 path behaves exactly as the
local one does.
"""
import datetime
import sys
import time
import types

import pytest

from app.services.artifacts import ArtifactStore
from app.services.blobstore import LocalBlobStore


# ── a stand-in for the S3 client surface used here ────────

class _FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = (Body, time.time())

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _missing()
        body, _ = self.objects[(Bucket, Key)]
        return {"Body": types.SimpleNamespace(read=lambda: body)}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _missing()
        _, when = self.objects[(Bucket, Key)]
        return {"LastModified": datetime.datetime.fromtimestamp(
            when, datetime.timezone.utc)}

    def list_objects_v2(self, Bucket, Prefix="", **kwargs):
        keys = sorted(k for (b, k) in self.objects
                      if b == Bucket and k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            self.objects.pop((Bucket, obj["Key"]), None)


def _missing():
    error = Exception("no such key")
    error.response = {"Error": {"Code": "NoSuchKey"}}
    return error


@pytest.fixture()
def s3(monkeypatch):
    fake = _FakeS3()
    stub = types.ModuleType("boto3")
    stub.client = lambda *a, **k: fake
    monkeypatch.setitem(sys.modules, "boto3", stub)
    from app.services.blobstore import S3BlobStore
    return S3BlobStore("bucket", prefix="analytiq/artifacts"), fake


@pytest.fixture(params=["local", "s3"])
def blobs(request, tmp_path, s3):
    """Both backends, so every test below runs against each."""
    return LocalBlobStore(str(tmp_path)) if request.param == "local" else s3[0]


# ── the contract both backends must meet ──────────────────

def test_bytes_come_back_unchanged(blobs):
    blobs.put("a/b/blob", b"\x00\x01binary\xff")
    assert blobs.get("a/b/blob") == b"\x00\x01binary\xff"


def test_a_missing_key_is_none_not_an_error(blobs):
    assert blobs.get("nothing/here") is None
    assert blobs.exists("nothing/here") is False


def test_overwriting_replaces(blobs):
    blobs.put("k", b"first")
    blobs.put("k", b"second")
    assert blobs.get("k") == b"second"


def test_listing_is_scoped_to_the_prefix(blobs):
    blobs.put("amy/1/blob", b"x")
    blobs.put("amy/2/blob", b"y")
    blobs.put("bob/1/blob", b"z")
    assert blobs.list_prefix("amy") == ["amy/1/blob", "amy/2/blob"]
    assert blobs.list_prefix("bob") == ["bob/1/blob"]


def test_deleting_a_prefix_takes_everything_under_it(blobs):
    blobs.put("amy/1/blob", b"x")
    blobs.put("amy/1/meta.json", b"{}")
    blobs.put("amy/2/blob", b"y")
    assert blobs.delete_prefix("amy/1") is True
    assert blobs.list_prefix("amy") == ["amy/2/blob"]


def test_deleting_nothing_reports_nothing(blobs):
    assert blobs.delete_prefix("never/existed") is False


def test_a_key_cannot_climb_out_of_the_store(blobs):
    """Owner names and ids come from tokens and uuid4, so this is
    defence against a future caller rather than a current hole."""
    blobs.put("../escape", b"nope")
    assert blobs.get("../escape") == b"nope"
    assert all(".." not in k for k in blobs.list_prefix(""))


def test_modification_time_is_available(blobs):
    blobs.put("k", b"x")
    when = blobs.modified_at("k")
    assert when is not None
    assert abs(time.time() - when) < 60
    assert blobs.modified_at("absent") is None


# ── the artifact store, on both backends ──────────────────

@pytest.fixture()
def store(blobs):
    return ArtifactStore(blobs=blobs)


def test_a_report_round_trips(store):
    art = store.put("amy", "ds1", "report", "pdf", b"%PDF-x", "r.pdf",
                    title="Q3")
    assert store.get("amy", art.artifact_id).title == "Q3"
    assert store.read("amy", art.artifact_id) == b"%PDF-x"


def test_one_account_cannot_read_another_s(store):
    art = store.put("amy", "ds1", "report", "pdf", b"secret", "r.pdf")
    assert store.get("bob", art.artifact_id) is None
    assert store.read("bob", art.artifact_id) is None
    assert store.list("bob") == []


def test_listing_filters_by_dataset(store):
    store.put("amy", "ds1", "report", "pdf", b"a", "a.pdf")
    store.put("amy", "ds2", "deck", "pptx", b"b", "b.pptx")
    assert [a.kind for a in store.list("amy", "ds1")] == ["report"]
    assert len(store.list("amy")) == 2


def test_an_expired_report_is_not_served(store):
    art = store.put("amy", "ds1", "report", "pdf", b"old", "o.pdf",
                    retention_days=-1)
    assert store.get("amy", art.artifact_id) is None
    assert store.read("amy", art.artifact_id) is None


def test_the_sweep_removes_only_what_expired(store):
    keep = store.put("amy", "ds1", "report", "pdf", b"new", "n.pdf")
    store.put("amy", "ds1", "report", "pdf", b"old", "o.pdf",
              retention_days=-1)
    assert store.sweep() == 1
    assert [a.artifact_id for a in store.list("amy")] == [keep.artifact_id]


def test_the_sweep_spares_an_artifact_still_being_written(store):
    """put() writes the blob, then the metadata. A sweep landing between
    the two sees metadata it cannot read, and without a grace window
    would delete the report somebody is at that moment waiting for."""
    art = store.put("amy", "ds1", "report", "pdf", b"x", "r.pdf")
    store.blobs.put(store._meta_key("amy", art.artifact_id), b"not json")
    assert store.sweep() == 0
    assert store._recent("amy", art.artifact_id) is True


def test_unreadable_metadata_is_swept_once_it_is_old(store, monkeypatch):
    art = store.put("amy", "ds1", "report", "pdf", b"x", "r.pdf")
    store.blobs.put(store._meta_key("amy", art.artifact_id), b"not json")
    monkeypatch.setattr(store, "_recent", lambda *a: False)
    assert store.sweep() == 1


def test_deleting_removes_both_the_bytes_and_the_record(store):
    art = store.put("amy", "ds1", "report", "pdf", b"x", "r.pdf")
    assert store.delete("amy", art.artifact_id) is True
    assert store.get("amy", art.artifact_id) is None
    assert store.blobs.list_prefix("amy") == []


# ── how the deployment is configured ──────────────────────

def test_local_storage_does_not_claim_to_be_durable(tmp_path):
    assert LocalBlobStore(str(tmp_path)).durable is False


def test_object_storage_is_durable(s3):
    assert s3[0].durable is True


def test_a_bucket_without_boto3_fails_loudly(monkeypatch):
    """Falling back to local disk here would leave the operator
    believing reports were durable when they were not."""
    monkeypatch.setitem(sys.modules, "boto3", None)
    from app.services.blobstore import S3BlobStore
    with pytest.raises(RuntimeError, match="boto3"):
        S3BlobStore("bucket")


def test_no_bucket_means_local_disk(monkeypatch, tmp_path):
    from app.config import config
    from app.services import blobstore
    monkeypatch.setattr(config, "s3_bucket", "", raising=False)
    monkeypatch.setattr(config, "data_dir", str(tmp_path), raising=False)
    assert isinstance(blobstore.build_store("artifacts"), LocalBlobStore)


def test_the_bucket_layout_keeps_accounts_apart(s3):
    blobs, fake = s3
    store = ArtifactStore(blobs=blobs)
    store.put("amy", "ds1", "report", "pdf", b"a", "a.pdf")
    store.put("bob", "ds1", "report", "pdf", b"b", "b.pdf")
    keys = sorted(k for (_b, k) in fake.objects)
    assert all(k.startswith("analytiq/artifacts/") for k in keys)
    assert any("/amy/" in k for k in keys)
    assert any("/bob/" in k for k in keys)


# ══════════════════════════════════════════════════════════
#  WHAT ELSE DEPENDS ON THIS MODULE
# ══════════════════════════════════════════════════════════
# Moving artifacts onto the blob store deleted two private helpers that
# three other modules imported by name. Nothing in the type system says
# so, and the artifact tests above all passed while background reports,
# schedules and share links were completely broken. These pin the
# dependency down.

def test_the_shared_atomic_write_is_importable():
    """jobs, schedules and sharing all keep small JSON files on local
    disk and all imported this from artifacts.py."""
    from app.services.blobstore import atomic_write
    assert callable(atomic_write)


def test_atomic_write_replaces_rather_than_truncates(tmp_path):
    from app.services.blobstore import atomic_write
    path = str(tmp_path / "f.json")
    atomic_write(path, b'{"a": 1}')
    atomic_write(path, b'{"b": 2}')
    with open(path, "rb") as fh:
        assert fh.read() == b'{"b": 2}'


def test_atomic_write_creates_the_directory(tmp_path):
    from app.services.blobstore import atomic_write
    path = str(tmp_path / "deep" / "nested" / "f.json")
    atomic_write(path, b"{}")
    assert open(path, "rb").read() == b"{}"


@pytest.mark.parametrize("module", [
    "app.services.jobs", "app.services.schedules", "app.services.sharing",
])
def test_every_dependent_module_still_imports(module):
    import importlib
    assert importlib.import_module(module) is not None


def test_sharing_updates_metadata_through_the_store(store):
    """Sharing used to write the artifact store's own metadata path
    itself, so renaming that path broke share links silently. The store
    owns where its metadata lives."""
    art = store.put("amy", "ds1", "report", "pdf", b"x", "r.pdf")
    art.meta["share_views"] = 3
    art.meta["shares_revoked_at"] = 1234.5
    store.update(art)

    reloaded = store.get("amy", art.artifact_id)
    assert reloaded.meta["share_views"] == 3
    assert reloaded.meta["shares_revoked_at"] == 1234.5


def test_updating_metadata_does_not_disturb_the_bytes(store):
    art = store.put("amy", "ds1", "report", "pdf", b"%PDF-keep", "r.pdf")
    art.title = "renamed"
    store.update(art)
    assert store.read("amy", art.artifact_id) == b"%PDF-keep"
    assert store.get("amy", art.artifact_id).title == "renamed"
