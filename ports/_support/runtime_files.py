"""Verify immutable CPython rootfs inventories before copying runtime files."""

import shutil
from pathlib import Path

from ports.native.dependencies import file_hash

MAX_FILES = 10_000
MAX_BYTES = 128 * 1024 * 1024


def _verified_rootfs(bundle: Path, manifest: dict) -> None:
    rootfs = bundle / "rootfs"
    if rootfs.is_symlink() or not rootfs.is_dir():
        raise ValueError("input CPython rootfs must be a real directory")
    files = manifest["files"]
    if not isinstance(files, dict) or len(files) > MAX_FILES:
        raise ValueError("input CPython rootfs has too many files")
    actual = {}
    total = 0
    for path in rootfs.rglob("*"):
        if path.is_symlink():
            raise ValueError("input CPython rootfs contains a symbolic link")
        if path.is_file():
            total += path.stat().st_size
            if total > MAX_BYTES or len(actual) >= MAX_FILES:
                raise ValueError("input CPython rootfs exceeds its size limit")
            actual["/" + path.relative_to(rootfs).as_posix()] = file_hash(path)
        elif not path.is_dir():
            raise ValueError("input CPython rootfs contains a special file")
    if actual != files:
        raise ValueError("input CPython rootfs differs from its manifest")


def _copy(rootfs: Path, destination: str, source: Path, expected_hash: str) -> None:
    if file_hash(source) != expected_hash:
        raise ValueError(f"native artifact changed: {source}")
    target = rootfs / destination.lstrip("/")
    if target.exists():
        raise ValueError(f"input CPython already contains {destination}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if file_hash(target) != expected_hash:
        raise ValueError(f"native artifact changed while copying: {source}")
