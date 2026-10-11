"""POSIX local immutable blob storage and durable journal CAS.

Each journal has an advisory lock and opaque UUID revision. Readers see atomic
replacements; successful writers fsync the file and directory before returning.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from .contracts import MAX_METADATA_BYTES, ConditionalWriteError, VersionedBytes, encode
from .contracts import digest as check_digest


def atomic_write(path: Path, data: bytes) -> None:
    """Replace one durable record without exposing a partial JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".write-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def read_bounded(path: Path, limit: int = MAX_METADATA_BYTES) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("stored object exceeds byte bound")
    return data


class LocalStore:
    """A store shared by processes on one trusted POSIX filesystem."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        for directory in ("blobs", "journals", "locks"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)

    def put_blob(self, data: bytes) -> str:
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("blob exceeds byte bound")
        key = hashlib.sha256(data).hexdigest()
        path = self.root / "blobs" / key
        if path.exists():
            self.get_blob(key)
            return key
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".blob-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                os.chmod(temporary, 0o444)
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    self.get_blob(key)
                sync_directory(path.parent)
            finally:
                temporary.unlink(missing_ok=True)
        return key

    def get_blob(self, digest: str) -> bytes:
        key = check_digest(digest)
        data = read_bounded(self.root / "blobs" / key, MAX_METADATA_BYTES)
        if hashlib.sha256(data).hexdigest() != key:
            raise ValueError("blob SHA256 mismatch")
        return data

    def _key(self, key: str) -> str:
        if not isinstance(key, str) or not 1 <= len(key.encode()) <= 512:
            raise ValueError("invalid journal key")
        return hashlib.sha256(key.encode()).hexdigest()

    def read_journal(self, key: str) -> VersionedBytes | None:
        path = self.root / "journals" / self._key(key)
        try:
            envelope = json.loads(read_bounded(path, 2 * MAX_METADATA_BYTES))
        except FileNotFoundError:
            return None
        data = base64.b64decode(envelope["data"], validate=True)
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("journal exceeds byte bound")
        return VersionedBytes(data, envelope["version"])

    def write_journal(self, key: str, data: bytes, expected_version: str | None) -> str:
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("journal exceeds byte bound")
        hashed = self._key(key)
        with locked(self.root / "locks" / hashed):
            current = self.read_journal(key)
            if (current.version if current else None) != expected_version:
                raise ConditionalWriteError("journal revision changed")
            version = uuid.uuid4().hex
            atomic_write(
                self.root / "journals" / hashed,
                encode(
                    {
                        "version": version,
                        "data": base64.b64encode(data).decode("ascii"),
                    }
                ),
            )
            return version
