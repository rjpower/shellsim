"""Canonical artifact records and disposable worker installations.

Artifacts are compact archives transferred in parallel. Store reads supply
transfer integrity; installation does not hash or inventory their contents.
"""

from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from ports.buildomatic.contracts import CHUNK_BYTES, Store, encode
from ports.buildomatic.store import atomic_write


def relative_path(key: str) -> str:
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts or not key or str(path) != key:
        raise ValueError("artifact keys must be relative paths")
    return key


def resolve(store: Store, key: str) -> dict | None:
    """Read only the small catalog record, without fetching product bytes."""
    found = store.read_journal("artifacts/" + relative_path(key))
    return None if found is None else json.loads(found.data)


def register(store: Store, key: str, value: dict) -> None:
    """Make an artifact discoverable after its upload succeeds."""
    path = "artifacts/" + relative_path(key)
    current = store.read_journal(path)
    store.write_journal(path, encode(value), None if current is None else current.version)


def upload(store: Store, root: Path, scratch: Path, *, workers: int = 8) -> dict:
    """Pack a product once and upload bounded chunks concurrently."""
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="archive-", dir=scratch) as temporary:
        archive = Path(temporary) / "product.tar.gz"
        compressor = "pigz -1 -p 8" if shutil.which("pigz") else "gzip -1"
        subprocess.run(["tar", "-I", compressor, "-cf", str(archive), "-C", str(root), "."], check=True)
        chunks = []
        with archive.open("rb") as source, ThreadPoolExecutor(max_workers=workers) as pool:
            while batch := [chunk for _ in range(workers) if (chunk := source.read(CHUNK_BYTES))]:
                chunks.extend(pool.map(store.put_blob, batch))
        return {"id": uuid.uuid4().hex, "chunks": chunks, "bytes": archive.stat().st_size}


def install(store: Store, artifact: dict, cache: Path, *, workers: int = 8) -> Path:
    """Fetch once per worker, using a lock and rename to share installations."""
    key = relative_path(artifact["id"])
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    with (cache / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_dir():
            return destination
        with tempfile.TemporaryDirectory(prefix=key + "-", dir=cache) as temporary:
            staging = Path(temporary)
            archive = staging / "product.tar.gz"
            with archive.open("wb") as output, ThreadPoolExecutor(max_workers=workers) as pool:
                chunks = artifact["chunks"]
                for offset in range(0, len(chunks), workers):
                    for data in pool.map(store.get_blob, chunks[offset : offset + workers]):
                        output.write(data)
            product = staging / "product"
            product.mkdir()
            with tarfile.open(archive) as source:
                source.extractall(product, filter="data")
            product.rename(destination)
    return destination


def write_reference(path: Path, value: dict) -> None:
    """Save a small result reference; callers choose whether to fetch its bytes."""
    atomic_write(path, encode(value))
