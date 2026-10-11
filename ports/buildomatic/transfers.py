"""Bounded bundle transport; extraction and build admission remain core operations.

Only fixed worker loops are submitted. Each loop owns its remote-store context,
and a failed operation stops new work and joins all contexts before returning.
Upload manifests follow all verified ordinary chunks, including cross-bundle
deduplication. Downloads only warm immutable local blobs for checked extraction.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, ContextManager, Protocol, Sequence

from ports.buildomatic import LocalStore, ResourceLimits, TreeBundle
from ports.buildomatic.contracts import CHUNK_BYTES, MAX_METADATA_BYTES, digest

MAX_PARALLEL = 8
# The core admits 512 actions with at most 512 input mounts each, plus one
# output bundle per action. Per-manifest bounds then bound dedup bookkeeping;
# no payloads or per-object futures enter the plan.
_MAX_BUNDLES = 512 * (512 + 1)
DEFAULT_LIMITS = ResourceLimits()


class Blobs(Protocol):
    """Immutable store reads are bounded to core MAX_METADATA_BYTES and verified."""

    def get_blob(self, digest: str) -> bytes: ...
    def put_blob(self, data: bytes) -> str: ...


BlobFactory = Callable[[], ContextManager[Blobs]]


@dataclass(frozen=True)
class TransferCounts:
    """Unique bundles/objects processed; bytes supplied to or fetched from remote.

    Upload bytes include payloads supplied to put_blob even for existing remote
    objects. Download local hits contribute objects but zero bytes. These counts
    exclude signing RPCs and HTTP overhead and are not wire-traffic accounting.
    """

    bundles: int = 0
    objects: int = 0
    bytes: int = 0


@dataclass(frozen=True)
class _Blob:
    digest: str
    size: int


@dataclass(frozen=True)
class _Plan:
    chunks: tuple[_Blob, ...]
    manifests: tuple[_Blob, ...]


def _arguments(bundles: Sequence[TreeBundle], max_parallel: int) -> tuple[str, ...]:
    if type(max_parallel) is not int or not 1 <= max_parallel <= MAX_PARALLEL:
        raise ValueError("transfer parallelism must be in 1..8")
    if len(bundles) > _MAX_BUNDLES:
        raise ValueError("bundle transfer exceeds manifest count bound")
    return tuple(dict.fromkeys(digest(bundle.digest) for bundle in bundles))


def _manifest(data: bytes, key: str) -> dict:
    if len(data) > MAX_METADATA_BYTES or hashlib.sha256(data).hexdigest() != key:
        raise ValueError("invalid or oversized bundle manifest")
    value = json.loads(data)
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("unsupported bundle manifest")
    if not isinstance(value.get("entries"), list):
        raise ValueError("invalid bundle entries")
    return value


def _plan(read: Callable[[str], bytes], bundle_keys: tuple[str, ...], limits: ResourceLimits) -> _Plan:
    chunks: dict[str, _Blob] = {}
    manifests: dict[str, _Blob] = {}
    references: dict[str, set[str]] = {}
    manifest_keys = frozenset(bundle_keys)
    for key in bundle_keys:
        data = read(key)
        entries = _manifest(data, key)["entries"]
        if len(entries) > limits.max_files:
            raise ValueError("bundle exceeds entry bound")
        manifests[key] = _Blob(key, len(data))
        needed = references[key] = set()
        total = 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("invalid bundle entry")
            if entry.get("kind") != "file":
                if entry.get("kind") not in ("directory", "symlink") or entry.get("chunks"):
                    raise ValueError("unsupported bundle entry")
                continue
            size, chunk_keys = entry.get("size"), entry.get("chunks")
            if type(size) is not int or size < 0 or not isinstance(chunk_keys, list):
                raise ValueError("invalid file size or chunks")
            total += size
            if total > limits.output_bytes or len(chunk_keys) != (size + CHUNK_BYTES - 1) // CHUNK_BYTES:
                raise ValueError("bundle exceeds byte bound or has invalid chunk count")
            for index, key in enumerate(chunk_keys):
                item = _Blob(digest(key), min(CHUNK_BYTES, size - index * CHUNK_BYTES))
                if key in chunks and chunks[key] != item:
                    raise ValueError("shared chunk has inconsistent lengths")
                chunks[key] = item
                if key in manifest_keys:
                    needed.add(key)

    for key in chunks.keys() & manifests.keys():
        if chunks[key] != manifests[key]:
            raise ValueError("chunk and manifest have inconsistent lengths")
        del chunks[key]
    # A regular file can contain another bundle's manifest bytes. Publish that
    # manifest first instead of uploading its digest twice or exposing a parent
    # manifest before its referenced bytes exist.
    waiting = {key: len(references[key]) for key in manifests}
    dependents: dict[str, list[str]] = {}
    for parent, children in references.items():
        for child in children:
            dependents.setdefault(child, []).append(parent)
    ready = deque(key for key in manifests if waiting[key] == 0)
    ordered = []
    while ready:
        key = ready.popleft()
        ordered.append(manifests[key])
        for parent in dependents.get(key, ()):
            waiting[parent] -= 1
            if waiting[parent] == 0:
                ready.append(parent)
    if len(ordered) != len(manifests):
        raise ValueError("cyclic manifest blob references")
    return _Plan(tuple(chunks.values()), tuple(ordered))


def _verified(store: Blobs, item: _Blob) -> bytes:
    data = store.get_blob(item.digest)
    if len(data) != item.size or hashlib.sha256(data).hexdigest() != item.digest:
        raise ValueError("blob digest or exact length differs")
    return data


def _put(store: Blobs, item: _Blob, data: bytes) -> None:
    if store.put_blob(data) != item.digest:
        raise ValueError("store returned an incorrect blob digest")


def _parallel(
    jobs: tuple[_Blob, ...], factory: BlobFactory, transfer: Callable[[Blobs, _Blob], int], max_parallel: int
) -> int:
    cursor = iter(jobs)
    lock, stop = threading.Lock(), threading.Event()
    transferred = 0

    def run():
        nonlocal transferred
        try:
            with factory() as remote:
                try:
                    while True:
                        with lock:
                            if stop.is_set():
                                return
                            item = next(cursor, None)
                        if item is None:
                            return
                        count = transfer(remote, item)
                        with lock:
                            transferred += count
                except BaseException:
                    # Stop peers before a potentially slow context close.
                    stop.set()
                    raise
        except BaseException:
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="buildomatic-transfer") as pool:
        try:
            futures = [pool.submit(run) for _ in range(min(max_parallel, len(jobs)))]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            raise
    return transferred


def upload_bundles(
    source: LocalStore,
    bundles: Sequence[TreeBundle],
    *,
    factory: BlobFactory,
    max_parallel: int = MAX_PARALLEL,
    limits: ResourceLimits = DEFAULT_LIMITS,
) -> TransferCounts:
    """Publish bounded unique chunks, then manifests, without submitting a build.

    Bundle count follows the core's 512-action/512-input protocol bounds. Each
    manifest is at most 32 MiB; ResourceLimits apply separately to each distinct
    tree. Increase them explicitly for an SDK. Missing/corrupt blobs and
    uncertain remote writes propagate.
    """
    keys = _arguments(bundles, max_parallel)
    plan = _plan(source.get_blob, keys, limits)

    def put(remote: Blobs, item: _Blob) -> int:
        data = _verified(source, item)
        _put(remote, item, data)
        return len(data)

    transferred = _parallel(plan.chunks, factory, put, max_parallel)
    if plan.manifests:
        with factory() as remote:
            for item in plan.manifests:
                transferred += put(remote, item)
    return TransferCounts(len(keys), len(plan.chunks) + len(plan.manifests), transferred)


def download_bundles(
    destination: LocalStore,
    bundles: Sequence[TreeBundle],
    *,
    factory: BlobFactory,
    max_parallel: int = MAX_PARALLEL,
    limits: ResourceLimits = DEFAULT_LIMITS,
) -> TransferCounts:
    """Warm verified local blobs; normal core extraction still checks the tree.

    Existing local corruption is an error, never a cache miss. Remote reads and
    local writes must preserve exact manifest/chunk digests and chunk lengths.
    ResourceLimits apply to each distinct bundle, without an aggregate DAG
    byte/file limit. Path, mode and symlink checks remain extraction's job.
    """
    keys = _arguments(bundles, max_parallel)
    if not keys:
        return TransferCounts()
    transferred = 0
    with factory() as remote:

        def read(key: str) -> bytes:
            nonlocal transferred
            try:
                return destination.get_blob(key)
            except FileNotFoundError:
                data = remote.get_blob(key)
                _manifest(data, key)
                _put(destination, _Blob(key, len(data)), data)
                transferred += len(data)
                return data

        plan = _plan(read, keys, limits)

    def get(remote: Blobs, item: _Blob) -> int:
        try:
            _verified(destination, item)
        except FileNotFoundError:
            data = _verified(remote, item)
            _put(destination, item, data)
            return len(data)
        return 0

    transferred += _parallel(plan.chunks, factory, get, max_parallel)
    return TransferCounts(len(keys), len(plan.chunks) + len(plan.manifests), transferred)
