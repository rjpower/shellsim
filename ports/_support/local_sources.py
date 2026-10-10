"""Stage explicitly pinned in-tree platform sources for ordinary build adapters.

Source paths are relative to the ports tree. Only declared regular files enter
the build directory; recipes cannot use this mechanism to read ambient host files.
"""

import hashlib
from pathlib import Path
from typing import Mapping

from ports._support.store import identity, relative_path

MAX_FILES = 256
MAX_BYTES = 16 * 1024**2


def local_source_files(root: Path, source: Mapping) -> dict[str, bytes]:
    """Verify the complete local source manifest and return destination bytes.

    Each entry contains a ports-relative ``path``, source-relative
    ``destination`` and SHA256. The source SHA256 pins the canonical entry list.
    """
    entries = source.get("files")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_FILES:
        raise ValueError("local source requires a bounded file list")
    if source.get("sha256") != identity(entries):
        raise ValueError("local source manifest hash differs")
    if "url" in source:
        raise ValueError("source cannot declare both local files and a URL")
    root = root.resolve(strict=True)
    result = {}
    total = 0
    for item in entries:
        name = relative_path(item["path"])
        destination = relative_path(item["destination"])
        path = root / name
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("local source escapes ports tree")
        if path.is_symlink() or any((root / parent).is_symlink() for parent in Path(name).parents):
            raise ValueError("local source contains a symlink")
        if not path.is_file() or destination in result:
            raise ValueError("local source is not a regular file or repeats a destination")
        with path.open("rb") as stream:
            data = stream.read(MAX_BYTES - total + 1)
        total += len(data)
        if total > MAX_BYTES:
            raise ValueError("local sources exceed their byte limit")
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"local source hash differs: {name}")
        result[destination] = data
    names = set(result)
    if any(str(parent) in names for name in names for parent in Path(name).parents):
        raise ValueError("local source file/directory conflict")
    return result


def stage_local_sources(root: Path, source: Mapping, destination: Path) -> Path:
    """Copy verified source bytes into a fresh isolated adapter source tree."""
    files = local_source_files(root, source)
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return destination
