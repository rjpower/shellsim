"""Durable build blobs and journals using Rigging's native object-store CAS.

Rigging owns credentials, endpoint selection and filesystem routing. Import it
only when a remote operation runs so local builds do not depend on Iris packages.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from urllib.parse import urlsplit

DEFAULT_PREFIX = "s3://marin-us-east-02a/marin/shellsim/buildomatic/v1"
MAX_METADATA_BYTES = 32 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_JOURNAL_HEADER = b"buildomatic-journal-v1:"


def validate_prefix(prefix: str) -> str:
    """Reject unsupported storage and ambiguous object paths before any I/O."""
    parsed = urlsplit(prefix)
    if parsed.scheme not in ("s3", "gs") or not parsed.netloc:
        raise ValueError("remote build storage requires an s3:// or gs:// bucket")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("storage prefixes cannot contain credentials, queries or fragments")
    if any(part in (".", "..") for part in parsed.path.split("/")):
        raise ValueError("storage prefixes cannot traverse directories")
    return prefix.rstrip("/")


def _conditional(path: str):
    from rigging.filesystem.conditional_object import conditional_object

    if path.startswith("s3://"):
        from rigging.filesystem.cluster_config import StoreType, data_buckets
        from rigging.filesystem.s3_compat import configure_coreweave_s3

        spec = data_buckets().get(urlsplit(path).netloc)
        if spec is not None and spec.store == StoreType.COREWEAVE:
            configure_coreweave_s3()
    return conditional_object(path)


class RemoteStore:
    """Implement the core Store protocol with opaque backend-native revisions.

    A failed conditional write is a concurrency conflict, not permission to
    overwrite. Network and authorization failures propagate to the caller.
    """

    def __init__(self, prefix: str = DEFAULT_PREFIX, *, journal_prefix: str | None = None, local_cache=None):
        self.prefix = validate_prefix(prefix)
        self.journal_prefix = validate_prefix(journal_prefix) if journal_prefix is not None else self.prefix
        self._local_cache = local_cache

    def _journal_path(self, key: str) -> str:
        if not key or key.startswith("/") or any(part in ("", ".", "..") for part in key.split("/")):
            raise ValueError("journal keys must be relative nonempty object paths")
        if "\\" in key or "?" in key or "#" in key:
            raise ValueError("invalid journal object path")
        return f"{self.journal_prefix}/journals/{key}"

    def put_blob(self, data: bytes) -> str:
        """Publish immutable SHA256 bytes; validate an existing concurrent winner."""
        from rigging.filesystem.conditional_object import ConditionalWriteError

        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("blob exceeds the supported storage bound")
        digest = hashlib.sha256(data).hexdigest()
        obj = _conditional(f"{self.prefix}/blobs/{digest}")
        try:
            obj.write(data, expected_version=None)
        except ConditionalWriteError:
            if self._get_remote_blob(digest) != data:
                raise ValueError("content-addressed blob is missing or corrupt") from None
        if self._local_cache is not None:
            self._local_cache.put_blob(data)
        return digest

    def get_blob(self, digest: str) -> bytes:
        """Reuse verified worker-local chunks; corruption never becomes a miss."""
        if _DIGEST.fullmatch(digest) is None:
            raise ValueError("invalid SHA256 digest")
        if self._local_cache is not None:
            try:
                return self._local_cache.get_blob(digest)
            except FileNotFoundError:
                pass
        data = self._get_remote_blob(digest)
        if self._local_cache is not None:
            self._local_cache.put_blob(data)
        return data

    def _get_remote_blob(self, digest: str) -> bytes:
        from rigging.filesystem.buckets import filesystem_for

        fs, path = filesystem_for(f"{self.prefix}/blobs/{digest}")
        with fs.open(path, "rb") as handle:
            data = handle.read(MAX_METADATA_BYTES + 1)
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("blob exceeds the supported storage bound")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("content-addressed blob is corrupt")
        return data

    def read_journal(self, key: str):
        """Return core VersionedBytes without interpreting the native revision."""
        from ports.buildomatic import VersionedBytes

        found = _conditional(self._journal_path(key)).read()
        if found is None:
            return None
        header, separator, data = found.data.partition(b"\n")
        if not separator or not header.startswith(_JOURNAL_HEADER) or len(data) > MAX_METADATA_BYTES:
            raise ValueError("invalid remote journal envelope")
        return VersionedBytes(data, found.version)

    def write_journal(self, key: str, data: bytes, expected_version: str | None) -> str:
        """Translate only native CAS conflicts to the core concurrency exception."""
        from rigging.filesystem.conditional_object import ConditionalWriteError as NativeConflict

        from ports.buildomatic import ConditionalWriteError

        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("journal exceeds the supported storage bound")
        # S3 ETags depend on bytes. A fresh write nonce prevents ABA when a
        # journal returns to identical content, while preserving native CAS.
        envelope = _JOURNAL_HEADER + uuid.uuid4().hex.encode() + b"\n" + data
        try:
            return _conditional(self._journal_path(key)).write(envelope, expected_version=expected_version)
        except NativeConflict as error:
            raise ConditionalWriteError("remote journal changed") from error
