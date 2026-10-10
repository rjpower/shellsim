"""Durable build blobs and journals using Rigging's native object-store CAS.

Rigging owns credentials, endpoint selection and filesystem routing. Import it
only when a remote operation runs so local builds do not depend on Iris packages.
"""

from __future__ import annotations

import functools
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_PREFIX = "s3://marin-us-east-02a/marin/shellsim/buildomatic/v1"
MAX_METADATA_BYTES = 32 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_JOURNAL_HEADER = b"buildomatic-journal-v1:"


@dataclass(frozen=True)
class BlobAccessRequest:
    """Authorize only one bounded immutable cache object, never a journal key."""

    operation: Literal["get", "put"]
    digest: str
    size: int | None = None

    def __post_init__(self):
        if (
            self.operation not in ("get", "put")
            or not isinstance(self.digest, str)
            or _DIGEST.fullmatch(self.digest) is None
        ):
            raise ValueError("invalid blob signing request")
        if self.operation == "get":
            if self.size is not None:
                raise ValueError("GET size is determined by the bounded response")
        elif type(self.size) is not int or not 0 <= self.size <= MAX_METADATA_BYTES:
            raise ValueError("PUT blob size exceeds its bound")


@dataclass(frozen=True)
class SignedBlobAccess:
    """Ephemeral transfer authority; its URL and headers must never be logged."""

    request: BlobAccessRequest
    url: str = field(repr=False)
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)


@functools.cache
def _signing_client(prefix: str):
    import botocore.config
    import botocore.session
    from rigging.filesystem.cluster_config import StoreType, data_buckets
    from rigging.filesystem.s3_compat import s3_credentials

    parsed = urlsplit(prefix)
    spec = data_buckets().get(parsed.netloc)
    if parsed.scheme != "s3" or spec is None or spec.store != StoreType.COREWEAVE:
        raise ValueError("signed transfers require a configured CoreWeave S3 bucket")
    credentials = s3_credentials(spec.store)
    if credentials is None:
        raise PermissionError("worker has no configured object-store credentials")
    # LOTA is cluster-local. Sign against the public origin itself: changing a
    # signed URL's host afterwards invalidates its V4 signature.
    return botocore.session.get_session().create_client(
        "s3",
        endpoint_url="https://cwobject.com",
        region_name=spec.signing_region,
        aws_access_key_id=credentials[0],
        aws_secret_access_key=credentials[1],
        config=botocore.config.Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def sign_blob(prefix: str, request: BlobAccessRequest) -> SignedBlobAccess:
    """Mint a two-minute transfer using service-side Rigging credentials only."""
    prefix = validate_prefix(prefix)
    request = BlobAccessRequest(request.operation, request.digest, request.size)
    parsed = urlsplit(prefix)
    params = {"Bucket": parsed.netloc, "Key": (parsed.path.strip("/") + "/blobs/" + request.digest).lstrip("/")}
    headers = ()
    if request.operation == "put":
        # CoreWeave rejects presigned SHA256 checksum headers with
        # SignatureDoesNotMatch. Sign length and conditional creation; clients
        # verify the SHA256 on every read, including a concurrent winner.
        params.update(IfNoneMatch="*", ContentLength=request.size)
        headers = (("If-None-Match", "*"), ("Content-Length", str(request.size)))
    url = _signing_client(prefix).generate_presigned_url(
        f"{request.operation}_object", Params=params, ExpiresIn=120, HttpMethod=request.operation.upper()
    )
    return SignedBlobAccess(request, url, headers)


class SignedBlobStore:
    """External BlobStore using scoped signing RPCs and direct bounded HTTPS.

    No cloud credentials or transfer URLs are retained. CAS journals belong to
    the coordinator and are explicitly unavailable through this client store.
    """

    def __init__(self, signer: Callable[[BlobAccessRequest], SignedBlobAccess]):
        self._signer = signer

    def _access(self, request: BlobAccessRequest) -> SignedBlobAccess:
        access = self._signer(request)
        parsed = urlsplit(access.url)
        if (
            access.request != request
            or parsed.scheme != "https"
            or not (parsed.hostname or "").endswith(".cwobject.com")
            or parsed.username
            or parsed.password
        ):
            raise ValueError("invalid signed blob authority")
        return access

    def put_blob(self, data: bytes) -> str:
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("blob exceeds the supported storage bound")
        digest = hashlib.sha256(data).hexdigest()
        # Cache eviction can remove a previously uploaded blob. Verify remote
        # bytes on every publication, uploading only a confirmed missing key.
        try:
            existing = self.get_blob(digest)
        except FileNotFoundError:
            pass
        else:
            if existing != data:
                raise ValueError("content-addressed blob is corrupt")
            return digest
        access = self._access(BlobAccessRequest("put", digest, len(data)))
        try:
            with urlopen(Request(access.url, data=data, headers=dict(access.headers), method="PUT"), timeout=30):
                pass
        except HTTPError as error:
            status = error.code
            error.close()
            if status not in (409, 412):
                raise OSError(f"signed blob PUT rejected: HTTP {status}") from None
            if self.get_blob(digest) != data:
                raise ValueError("concurrent blob winner is corrupt") from None
        except (URLError, OSError):
            raise OSError("signed blob PUT transport failed") from None
        return digest

    def get_blob(self, digest: str) -> bytes:
        access = self._access(BlobAccessRequest("get", digest))
        try:
            with urlopen(Request(access.url, headers=dict(access.headers), method="GET"), timeout=30) as response:
                size = response.headers.get("Content-Length")
                if size is not None and not 0 <= int(size) <= MAX_METADATA_BYTES:
                    raise ValueError("blob exceeds the supported storage bound")
                data = response.read(MAX_METADATA_BYTES + 1)
        except HTTPError as error:
            status = error.code
            error.close()
            if status == 404:
                raise FileNotFoundError(digest) from None
            raise OSError(f"signed blob GET rejected: HTTP {status}") from None
        except (URLError, OSError):
            raise OSError("signed blob GET transport failed") from None
        if len(data) > MAX_METADATA_BYTES or (size is not None and len(data) != int(size)):
            raise ValueError("blob has invalid stored size")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("content-addressed blob is corrupt")
        return data

    def read_journal(self, key: str):
        raise NotImplementedError("external blob clients cannot read coordinator journals")

    def write_journal(self, key: str, data: bytes, expected_version: str | None):
        raise NotImplementedError("external blob clients cannot write coordinator journals")


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
    Rigging's native conditional read returns the entire journal and offers no
    bounded read parameter. Journal objects therefore require an operator-owned
    namespace with writes restricted to these bounded metadata writers; the
    post-read bound detects corruption but cannot bound that native allocation.
    Blob reads use a bounded stream and do not share this trust assumption.
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
        """Verify remote bytes before uploading; recheck a concurrent winner.

        A worker-local hit cannot prove an object still exists remotely after
        cache eviction, so publication always reads the remote namespace.
        """
        from rigging.filesystem.conditional_object import ConditionalWriteError

        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("blob exceeds the supported storage bound")
        digest = hashlib.sha256(data).hexdigest()
        try:
            existing = self._get_remote_blob(digest)
        except FileNotFoundError:
            obj = _conditional(f"{self.prefix}/blobs/{digest}")
            try:
                obj.write(data, expected_version=None)
            except ConditionalWriteError:
                existing = self._get_remote_blob(digest)
            else:
                existing = data
        if existing != data:
            raise ValueError("content-addressed blob is missing or corrupt")
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
