"""Chunked immutable trees with bounded transport and checked extraction.

Manifests describe directories, regular files and internal relative symlinks.
No archive extraction machinery, device nodes or host-absolute links are used.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import shutil
import stat
from collections import deque
from pathlib import Path, PurePosixPath

from .contracts import CHUNK_BYTES, MAX_METADATA_BYTES, ResourceLimits, Store, TreeBundle, digest, encode

_DEFAULT_LIMITS = ResourceLimits()


def _path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("invalid tree path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or value == "." or str(parsed) != value:
        raise ValueError("tree path must be canonical and relative")
    if len(value.encode()) > 4096:
        raise ValueError("tree path exceeds bound")
    return value


def _link(path: str, target: str) -> None:
    if not isinstance(target, str) or not target or "\\" in target or "\0" in target or len(target.encode()) > 4096:
        raise ValueError("invalid symlink target")
    if target.startswith("/"):
        raise ValueError("absolute symlink target")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
    if resolved == ".." or resolved.startswith("../"):
        raise ValueError("symlink escapes tree")


def _verify_links(entries: list[dict]) -> None:
    links = {entry["path"]: entry["target"] for entry in entries if entry["kind"] == "symlink"}
    for entry in entries:
        if entry["kind"] == "symlink":
            pending = deque(PurePosixPath(entry["path"]).parts)
            resolved = []
            followed = 0
            while pending:
                part = pending.popleft()
                if part == ".":
                    continue
                if part == "..":
                    if not resolved:
                        raise ValueError("symlink escapes tree")
                    resolved.pop()
                    continue
                candidate = "/".join((*resolved, part))
                if candidate in links:
                    followed += 1
                    if followed > 40:
                        raise ValueError("symlink cycle or excessive link depth")
                    pending.extendleft(reversed(PurePosixPath(links[candidate]).parts))
                else:
                    resolved.append(part)


def _blob(store: Store, key: str, limit: int) -> bytes:
    digest(key)
    data = store.get_blob(key)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != key:
        raise ValueError("invalid or oversized tree blob")
    return data


def _put(store: Store, data: bytes) -> str:
    key = store.put_blob(data)
    if key != hashlib.sha256(data).hexdigest():
        raise ValueError("store returned an incorrect blob digest")
    return key


def capture_tree(root: Path, store: Store, *, limits: ResourceLimits = _DEFAULT_LIMITS) -> TreeBundle:
    """Seal a quiescent tree, reading regular files in at most 4 MiB chunks.

    The caller owns the tree and must stop writers before capture. Empty trees,
    empty directories and dangling internal symlinks are valid results.
    """
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("tree root must be a directory")
    entries = []
    directories = [root]
    total = metadata = 0
    while directories:
        directory = directories.pop()
        with os.scandir(directory) as children:
            for child in children:
                if len(entries) >= limits.max_files:
                    raise ValueError("tree exceeds entry bound")
                path = Path(child.path)
                relative = _path(path.relative_to(root).as_posix())
                info = child.stat(follow_symlinks=False)
                entry = {"path": relative, "mode": stat.S_IMODE(info.st_mode) & 0o777}
                if stat.S_ISLNK(info.st_mode):
                    target = os.readlink(path)
                    _link(relative, target)
                    entry.update(kind="symlink", target=target)
                elif stat.S_ISDIR(info.st_mode):
                    entry.update(kind="directory")
                    directories.append(path)
                elif stat.S_ISREG(info.st_mode):
                    chunks = []
                    size = 0
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, "rb") as stream:
                        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                            raise ValueError("tree changed during capture")
                        while data := stream.read(min(CHUNK_BYTES, limits.output_bytes - total + 1)):
                            size += len(data)
                            total += len(data)
                            if total > limits.output_bytes:
                                raise ValueError("tree exceeds expanded byte bound")
                            chunks.append(_put(store, data))
                            if metadata + len(chunks) * 68 > MAX_METADATA_BYTES:
                                raise ValueError("tree exceeds manifest bound")
                    entry.update(kind="file", size=size, chunks=chunks)
                else:
                    raise ValueError("unsupported tree entry")
                metadata += len(encode(entry)) + 1
                if metadata > MAX_METADATA_BYTES - 128:
                    raise ValueError("tree exceeds manifest bound")
                entries.append(entry)
    _verify_links(entries)
    manifest = encode({"version": 1, "entries": sorted(entries, key=lambda e: e["path"])})
    if len(manifest) > MAX_METADATA_BYTES:
        raise ValueError("tree exceeds manifest bound")
    return TreeBundle(_put(store, manifest))


def extract_tree(
    bundle: TreeBundle, store: Store, destination: Path, *, limits: ResourceLimits = _DEFAULT_LIMITS
) -> None:
    """Verify and expand into a new private directory, never through symlinks.

    All names, parents and byte counts are checked before writing. Symlinks are
    installed last. Failed extraction removes only the new destination tree.
    """
    manifest = json.loads(_blob(store, bundle.digest, MAX_METADATA_BYTES))
    if manifest["version"] != 1 or not isinstance(manifest["entries"], list):
        raise ValueError("unsupported tree manifest")
    entries = manifest["entries"]
    if len(entries) > limits.max_files:
        raise ValueError("tree exceeds entry bound")
    paths = {}
    total = 0
    for entry in entries:
        path = _path(entry["path"])
        if path in paths:
            raise ValueError("duplicate tree path")
        paths[path] = entry
        mode = entry["mode"]
        if type(mode) is not int or not 0 <= mode <= 0o777:
            raise ValueError("invalid tree mode")
        if entry["kind"] == "file":
            size, chunks = entry["size"], entry["chunks"]
            if type(size) is not int or size < 0 or not isinstance(chunks, list):
                raise ValueError("invalid file size or chunks")
            total += size
            if total > limits.output_bytes or len(chunks) != (size + CHUNK_BYTES - 1) // CHUNK_BYTES:
                raise ValueError("tree exceeds expanded byte bound or has invalid chunks")
            for key in chunks:
                digest(key)
        elif entry["kind"] == "symlink":
            _link(path, entry["target"])
        elif entry["kind"] != "directory":
            raise ValueError("unsupported tree entry")
    for path in paths:
        for parent in PurePosixPath(path).parents:
            if str(parent) != "." and (str(parent) not in paths or paths[str(parent)]["kind"] != "directory"):
                raise ValueError("tree entry has missing or linked parent")
    destination = Path(destination).absolute()
    destination.mkdir(parents=False, exist_ok=False)
    try:
        directories = sorted((e for e in entries if e["kind"] == "directory"), key=lambda e: e["path"].count("/"))
        for entry in directories:
            (destination / entry["path"]).mkdir()
        for entry in entries:
            path = destination / entry["path"]
            if entry["kind"] == "file":
                remaining = entry["size"]
                with path.open("xb") as stream:
                    for key in entry["chunks"]:
                        data = _blob(store, key, CHUNK_BYTES)
                        if len(data) != min(remaining, CHUNK_BYTES):
                            raise ValueError("file chunk has incorrect length")
                        stream.write(data)
                        remaining -= len(data)
                path.chmod(entry["mode"])
        for entry in entries:
            if entry["kind"] == "symlink":
                (destination / entry["path"]).symlink_to(entry["target"])
        _verify_links(entries)
        for entry in reversed(directories):
            (destination / entry["path"]).chmod(entry["mode"])
    except BaseException:
        shutil.rmtree(destination)
        raise
