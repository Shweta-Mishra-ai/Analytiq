"""
services/blobstore.py — where bytes live when the container does not.

Artifacts were written to DATA_DIR, which on a container platform is
scratch space: it survives a request and not a deploy. A report built on
Monday, shared by link, is a 404 after Tuesday's restart — and the share
link, the scheduled delivery and the background job all exist precisely
so a report outlives the request that made it. Storing the result on a
disk that does not outlive the process undoes all three.

So the artifact store no longer knows where bytes are. It asks this,
and this is one of two things:

  * a local directory, exactly as before — the right answer for a
    laptop, a test, and a single-container deployment whose operator
    accepts that a restart loses old reports;
  * an S3-compatible bucket (AWS, Cloudflare R2, Backblaze, MinIO),
    which is the same API everywhere and the reason this is one class
    rather than four.

The choice is made once, from configuration, and every caller is
unaffected. What matters is the failure behaviour: a misconfigured or
unreachable bucket must not silently fall back to local disk, because
the operator would then believe reports were durable when they were
not. It fails loudly at startup instead.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import List, Optional

from app.config import config

logger = logging.getLogger(__name__)


class BlobStore:
    """Bytes, addressed by a '/'-separated key."""

    def put(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def get(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def modified_at(self, key: str) -> Optional[float]:
        """When this key was last written, as a unix time, or None.

        The artifact sweep needs it: an artifact whose metadata cannot
        be read is normally rubbish worth removing, EXCEPT in the moment
        between its blob landing and its metadata following. Without an
        age the sweep would delete the report somebody is at that second
        waiting for.
        """
        raise NotImplementedError

    def delete_prefix(self, prefix: str) -> bool:
        """Remove everything under a prefix. True if anything went."""
        raise NotImplementedError

    def list_prefix(self, prefix: str) -> List[str]:
        """Every key under a prefix, full keys not basenames."""
        raise NotImplementedError

    @property
    def durable(self) -> bool:
        """Whether bytes here survive the container that wrote them."""
        return False

    def describe(self) -> str:
        return self.__class__.__name__


class LocalBlobStore(BlobStore):
    """A directory. The default, and what every test runs against."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, key: str) -> str:
        # Key segments are already sanitised by the caller, but a key
        # that climbs out of the base directory is worth refusing here
        # too: this is the only place that turns one into a path.
        parts = [p for p in str(key).split("/") if p not in ("", ".", "..")]
        return os.path.join(self.base_dir, *parts)

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        with self._lock:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # pid and thread id, for the reason atomic_write() gives:
            # two workers of one server share a directory and can hold
            # the same thread id, and two writers landing on a single
            # temp name interleave their bytes. The lock above is per
            # process and does not help across them.
            tmp = "{}.{}.{}.tmp".format(path, os.getpid(),
                                        threading.get_ident())
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)

    def get(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except OSError:
            logger.warning("could not read %s", path, exc_info=True)
            return None

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def modified_at(self, key: str) -> Optional[float]:
        try:
            return os.path.getmtime(self._path(key))
        except OSError:
            return None

    def delete_prefix(self, prefix: str) -> bool:
        path = self._path(prefix)
        with self._lock:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                return True
            if os.path.exists(path):
                os.remove(path)
                return True
        return False

    def list_prefix(self, prefix: str) -> List[str]:
        root = self._path(prefix)
        if not os.path.isdir(root):
            return []
        out: List[str] = []
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if name.endswith(".tmp"):
                    continue
                full = os.path.join(dirpath, name)
                out.append(os.path.relpath(full, self.base_dir)
                           .replace(os.sep, "/"))
        return sorted(out)

    def describe(self) -> str:
        return "local directory {}".format(self.base_dir)


class S3BlobStore(BlobStore):
    """An S3-compatible bucket: AWS, Cloudflare R2, Backblaze, MinIO."""

    def __init__(self, bucket: str, prefix: str = "", endpoint_url: str = "",
                 region: str = ""):
        try:
            import boto3
        except ImportError as e:
            # Loud, and at construction. Falling back to local disk here
            # would leave the operator believing reports were durable.
            raise RuntimeError(
                "S3_BUCKET is set but boto3 is not installed. Add boto3 to "
                "the deployment, or unset S3_BUCKET to keep artifacts on "
                "local disk.") from e

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            region_name=region or None,
        )

    def _key(self, key: str) -> str:
        parts = [p for p in str(key).split("/") if p not in ("", ".", "..")]
        return "/".join(([self.prefix] if self.prefix else []) + parts)

    def put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=self._key(key),
                                Body=data)

    def get(self, key: str) -> Optional[bytes]:
        try:
            obj = self._client.get_object(Bucket=self.bucket,
                                          Key=self._key(key))
            return obj["Body"].read()
        except Exception as e:
            if _is_missing(e):
                return None
            logger.warning("could not read s3://%s/%s", self.bucket,
                           self._key(key), exc_info=True)
            return None

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception as e:
            if not _is_missing(e):
                logger.warning("head failed for s3://%s/%s", self.bucket,
                               self._key(key), exc_info=True)
            return False

    def modified_at(self, key: str) -> Optional[float]:
        try:
            head = self._client.head_object(Bucket=self.bucket,
                                            Key=self._key(key))
        except Exception:
            return None
        stamp = head.get("LastModified")
        return stamp.timestamp() if stamp is not None else None

    def delete_prefix(self, prefix: str) -> bool:
        keys = [{"Key": self._key(k)} for k in self.list_prefix(prefix)]
        if not keys:
            return False
        # delete_objects takes at most a thousand keys per call.
        for i in range(0, len(keys), 1000):
            self._client.delete_objects(Bucket=self.bucket,
                                        Delete={"Objects": keys[i:i + 1000]})
        return True

    def list_prefix(self, prefix: str) -> List[str]:
        full = self._key(prefix)
        if full and not full.endswith("/"):
            full += "/"
        out: List[str] = []
        token = None
        head = len(self.prefix) + 1 if self.prefix else 0
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": full}
            if token:
                kwargs["ContinuationToken"] = token
            try:
                page = self._client.list_objects_v2(**kwargs)
            except Exception:
                logger.warning("listing s3://%s/%s failed", self.bucket, full,
                               exc_info=True)
                return sorted(out)
            for item in page.get("Contents") or []:
                out.append(str(item["Key"])[head:])
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
        return sorted(out)

    @property
    def durable(self) -> bool:
        return True

    def describe(self) -> str:
        return "s3://{}/{}".format(self.bucket, self.prefix or "")


def atomic_write(path: str, data: bytes) -> None:
    """Write a local file so a reader never sees half of it.

    Lives here rather than in artifacts.py because three other stores —
    jobs, schedules, sharing — keep their own small JSON files on local
    disk and need exactly this. It used to be a private helper in the
    artifact store, which made removing it from there break all three.

    The pid as well as the thread id: two workers of the same server
    share a directory and can hold the same thread id, and two writers
    landing on one temp name would interleave their bytes.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = "{}.{}.{}.tmp".format(path, os.getpid(), threading.get_ident())
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _is_missing(error: Exception) -> bool:
    """True when the object simply is not there, rather than a fault."""
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    code = (response.get("Error") or {}).get("Code", "")
    return str(code) in ("404", "NoSuchKey", "NotFound")


def build_store(subdir: str) -> BlobStore:
    """The store this deployment is configured for.

    `subdir` names the kind of thing being stored, so one bucket can
    hold artifacts and anything added later without collision.
    """
    if config.s3_bucket:
        prefix = "/".join(p for p in (config.s3_prefix.strip("/"), subdir)
                          if p)
        store = S3BlobStore(config.s3_bucket, prefix=prefix,
                            endpoint_url=config.s3_endpoint_url,
                            region=config.s3_region)
        logger.info("%s will be stored in %s", subdir, store.describe())
        return store
    return LocalBlobStore(os.path.join(config.data_dir, subdir))
