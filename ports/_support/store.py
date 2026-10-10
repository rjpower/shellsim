"""Fetch pinned sources and publish verified graph results atomically.

These are trusted host build operations. Failed builds retain their work directory
for diagnosis and never become cache hits. Every reuse verifies the stored bytes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

_MAX_SOURCE = 512 * 1024**2
_MAX_EXPANDED = 2 * 1024**3
_MAX_FILES = 100_000
_MAX_RECEIPT = 32 * 1024**2
_HASH = re.compile(r"[a-f0-9]{64}\Z")


def identity(value: object) -> str:
    """Hash JSON build inputs independently of dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path, *, limit: int = _MAX_EXPANDED) -> str:
    """Hash a regular file with a bound that also covers concurrent file growth."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            size += len(chunk)
            if size > limit:
                raise ValueError(f"file exceeds build bounds: {path}")
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value: object) -> str:
    """Admit a canonical relative file name without traversal or aliases."""
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("invalid build path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError("build path must be canonical and relative")
    return value


def fetch(source: Mapping[str, Any], cache: Path, *, offline: bool = False) -> Path:
    """Return a verified pinned source, fetching HTTPS only on a cache miss."""
    expected = source["sha256"]
    if not isinstance(expected, str) or not _HASH.fullmatch(expected):
        raise ValueError("source needs a SHA256 pin")
    url = source["url"]
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("source URL must be HTTPS without credentials or fragments")
    name = relative_path(source.get("filename") or PurePosixPath(parsed.path).name)
    if "/" in name:
        raise ValueError("source filename must be a basename")
    directory = cache / expected
    destination = directory / name
    if destination.exists() or destination.is_symlink():
        if file_hash(destination, limit=_MAX_SOURCE) != expected:
            raise ValueError(f"cached source hash mismatch: {destination}")
        return destination
    if offline:
        raise ValueError(f"source is missing from the offline cache: {name}")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".download-", delete=False) as output:
        temporary = Path(output.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                    raise ValueError("source redirected outside HTTPS")
                size = 0
                while chunk := response.read(1024**2):
                    size += len(chunk)
                    if size > _MAX_SOURCE:
                        raise ValueError("download exceeds source bounds")
                    output.write(chunk)
            output.flush()
            if file_hash(temporary, limit=_MAX_SOURCE) != expected:
                raise ValueError(f"downloaded source hash mismatch: {name}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def extract(archive: Path, destination: Path, *, subdirectory: str) -> Path:
    """Expand bounded release sources; links and special files fail explicitly.

    Extraction happens in private build work. Requiring regular members avoids
    archive-order-dependent traversal and host file references.
    """
    subdirectory = relative_path(subdirectory)
    if destination.exists() or destination.is_symlink():
        raise ValueError("source extraction destination already exists")
    destination.mkdir(parents=True)
    seen: set[str] = set()
    total = 0

    def member(name: str, size: int, directory: bool) -> Path:
        nonlocal total
        name = name.removesuffix("/")
        # Release tarballs often spell members with a harmless leading './'.
        while name.startswith("./"):
            name = name[2:]
        name = relative_path(name)
        if name in seen or len(seen) >= _MAX_FILES or size < 0:
            raise ValueError("duplicate or excessive archive members")
        seen.add(name)
        total += size
        if total > _MAX_EXPANDED:
            raise ValueError("source archive exceeds expansion bounds")
        path = destination / name
        if directory:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                mode = item.external_attr >> 16
                if mode & 0o170000 not in {0, 0o100000, 0o040000}:
                    raise ValueError("source archive contains a link or special file")
                path = member(item.filename, item.file_size, item.is_dir())
                if not item.is_dir():
                    with source.open(item) as stream, path.open("xb") as output:
                        shutil.copyfileobj(stream, output, 1024**2)
                    path.chmod(0o755 if mode & 0o111 else 0o644)
                    timestamp = datetime(*item.date_time, tzinfo=timezone.utc).timestamp()
                    os.utime(path, (timestamp, timestamp), follow_symlinks=False)
    else:
        with tarfile.open(archive, "r|*") as source:
            for item in source:
                if not item.isfile() and not item.isdir():
                    raise ValueError("source archive contains a link or special file")
                path = member(item.name, item.size, item.isdir())
                if item.isfile():
                    with source.extractfile(item) as stream, path.open("xb") as output:
                        shutil.copyfileobj(stream, output, 1024**2)
                    path.chmod(0o755 if item.mode & 0o111 else 0o644)
                    # Autotools release sources rely on generated files retaining
                    # their order relative to configure.ac and included macros.
                    os.utime(path, (item.mtime, item.mtime), follow_symlinks=False)
    root = destination / subdirectory
    if not root.is_dir():
        raise ValueError("declared source subdirectory is missing")
    return root


def _files(root: Path) -> dict[str, dict[str, object]]:
    files = {}
    total = 0
    for directory, directories, names in os.walk(root, followlinks=False):
        for name in [*directories, *names]:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"build result contains a symlink: {path}")
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative == "build-receipt.json":
                continue
            total += path.stat().st_size
            if len(files) >= _MAX_FILES or total > _MAX_EXPANDED:
                raise ValueError("build result exceeds storage bounds")
            files[relative] = {"sha256": file_hash(path), "executable": bool(path.stat().st_mode & 0o111)}
    if not files:
        raise ValueError("build produced no files")
    return files


def verify(result: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Admit an exact cache entry, including added files and executable bits."""
    receipt_path = result / "build-receipt.json"
    if result.is_symlink() or receipt_path.is_symlink():
        raise ValueError("build cache contains a linked result or receipt")
    with receipt_path.open("rb") as stream:
        data = stream.read(_MAX_RECEIPT + 1)
    if len(data) > _MAX_RECEIPT:
        raise ValueError("build receipt exceeds its size bound")
    receipt = json.loads(data)
    if receipt["schema_version"] != 1 or receipt["inputs"] != inputs or receipt["key"] != identity(inputs):
        raise ValueError("cached build inputs differ")
    if receipt["files"] != _files(result):
        raise ValueError("cached build files differ")
    return receipt


@dataclass(frozen=True)
class Build:
    """A verified hit or a private workspace whose result will be published."""

    key: str
    cached: bool
    work: Path
    result: Path


@contextmanager
def build_slot(store: Path, inputs: Mapping[str, Any]) -> Iterator[Build]:
    """Serialize builds per input key and publish only after successful sealing.

    The caller writes all deliverables into result. Logs and intermediates belong
    in work and remain available after failure. Exceptions are never cache hits.
    """
    key = identity(inputs)
    store = store.resolve()
    for name in ("locks", "work", "results"):
        (store / name).mkdir(parents=True, exist_ok=True)
    destination = store / "results" / key
    with (store / "locks" / key).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            verify(destination, inputs)
            yield Build(key, True, destination, destination)
            return
        work = Path(tempfile.mkdtemp(prefix=key[:12] + "-", dir=store / "work"))
        result = work / "result"
        result.mkdir()
        yield Build(key, False, work, result)
        receipt = {"schema_version": 1, "key": key, "inputs": dict(inputs), "files": _files(result)}
        (result / "build-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        verify(result, inputs)
        os.rename(result, destination)
