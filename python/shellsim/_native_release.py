"""Fetch and verify a native package asset from a trusted release descriptor.

Native delivery shares the Python release transport and bounds, but does not
select a host uv resolver or construct a CPython runtime.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ._cpython_release import (
    _MAX_FILES,
    _MAX_UNPACKED,
    _digest,
    _extract,
    _fetch,
    _native_asset,
    _read_descriptor,
)
from .native_packages import verify_release_catalog
from .pypi import PackageInstallError


def _descriptor(path: Path) -> dict[str, Any]:
    data = _read_descriptor(path)
    if data.get("schema_version") != 1 or data.get("target") != "wasm32-wasip1":
        raise PackageInstallError("unsupported native release descriptor")
    _native_asset(data)
    return data


def _validate_entry(entry: Path, native: dict[str, Any]) -> None:
    if entry.is_symlink() or not entry.is_dir():
        raise PackageInstallError("native release cache is not a real directory")
    actual: set[str] = set()
    visited = total = 0
    for path in entry.rglob("*"):
        visited += 1
        if visited > _MAX_FILES * 2 or path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise PackageInstallError("native release cache contains too many, linked, or special paths")
        if path.is_file():
            total += path.stat().st_size
            if total > _MAX_UNPACKED or len(actual) >= _MAX_FILES:
                raise PackageInstallError("native release cache exceeds its size limit")
            actual.add(path.relative_to(entry).as_posix())
    catalog = entry / "native/catalog.json"
    if _digest(catalog, 1024 * 1024) != native["catalog_sha256"]:
        raise PackageInstallError("cached native catalog differs from the release descriptor")
    try:
        expected = {"native/" + name for name in verify_release_catalog(catalog)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PackageInstallError("native release catalog or artifacts are invalid") from error
    if actual != expected:
        raise PackageInstallError("native release cache contains missing or undeclared files")


def materialize_native(descriptor_path: Path, *, cache_dir: Path | None, offline: bool) -> Path:
    """Return a verified native catalog; offline mode only uses a valid cache hit."""
    descriptor = _descriptor(descriptor_path)
    native = descriptor["native"]
    root = cache_dir if cache_dir is not None else Path.home() / ".cache/shellsim/native-releases"
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    entry = root / ("native-" + native["archive"]["sha256"])
    if entry.exists() or entry.is_symlink():
        try:
            _validate_entry(entry, native)
        except (OSError, ValueError, PackageInstallError) as error:
            raise PackageInstallError(f"cached native release is corrupt: {entry}") from error
        return entry / "native/catalog.json"
    if offline:
        raise PackageInstallError("verified native release is not cached for offline use")
    with tempfile.TemporaryDirectory(prefix=".shellsim-native-", dir=root) as temporary:
        work = Path(temporary)
        archive = work / "native.zip"
        stage = work / "stage"
        stage.mkdir()
        _fetch(native["archive"], descriptor_path, archive)
        _extract(archive, stage, allowed_roots=frozenset({"native"}))
        _validate_entry(stage, native)
        try:
            stage.rename(entry)
        except OSError:
            if not entry.exists():
                raise
            _validate_entry(entry, native)
    return entry / "native/catalog.json"
