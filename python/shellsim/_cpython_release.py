"""Fetch and verify a trusted CPython/WASI release without entering the guest."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from ._cpython_universe import Universe, _inspect_wheel, _verify_file, _verify_wasm
from .pypi import PackageInstallError

_MAX_DESCRIPTOR = 1024 * 1024
_MAX_ARCHIVE = 256 * 1024 * 1024
_MAX_RESOLVER = 256 * 1024 * 1024
_MAX_UNPACKED = 384 * 1024 * 1024
_MAX_FILES = 12_000
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_LOCAL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")


class _ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, fp: Any, code: int, message: str, headers: Any, url: str) -> Any:
        origin = urllib.parse.urlsplit(request.full_url)
        target = urllib.parse.urlsplit(url)
        allowed = {origin.hostname}
        if origin.hostname == "github.com":
            allowed.update({"release-assets.githubusercontent.com", "objects.githubusercontent.com"})
        if target.scheme != "https" or target.hostname not in allowed or target.username or target.password:
            raise PackageInstallError("release asset redirected outside its HTTPS host")
        return super().redirect_request(request, fp, code, message, headers, url)


def _digest(path: Path, limit: int) -> str:
    try:
        valid = not path.is_symlink() and path.is_file() and path.stat().st_size <= limit
    except OSError:
        valid = False
    if not valid:
        raise PackageInstallError(f"release asset is missing, linked, or oversized: {path.name}")
    result = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            total += len(chunk)
            if total > limit:
                raise PackageInstallError(f"release asset exceeds its size limit: {path.name}")
            result.update(chunk)
    return result.hexdigest()


def _asset(value: Any, limit: int) -> dict[str, Any]:
    if not isinstance(value, dict) or not {"url", "sha256", "size"} <= value.keys():
        raise PackageInstallError("invalid release asset descriptor")
    url, digest, size = value["url"], value["sha256"], value["size"]
    if not isinstance(url, str) or len(url) > 4096 or not url:
        raise PackageInstallError("invalid release asset URL")
    if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
        raise PackageInstallError("invalid release asset SHA-256")
    if type(size) is not int or not 0 < size <= limit:
        raise PackageInstallError("invalid release asset size")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme:
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise PackageInstallError("release assets require HTTPS")
    elif _LOCAL_NAME.fullmatch(url) is None or url in {".", ".."}:
        raise PackageInstallError("local release assets must be sibling files")
    return value


def _read_descriptor(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_DESCRIPTOR:
        raise PackageInstallError("trusted release descriptor is missing, linked, or oversized")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as error:
        raise PackageInstallError("trusted release descriptor is invalid JSON") from error
    if not isinstance(data, dict):
        raise PackageInstallError("trusted release descriptor must be an object")
    return data


def _native_asset(data: dict[str, Any]) -> dict[str, Any]:
    native = data.get("native")
    if not isinstance(native, dict):
        raise PackageInstallError("release has no native package catalog")
    digest = native.get("catalog_sha256")
    if not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
        raise PackageInstallError("invalid native release catalog SHA-256")
    _asset(native.get("archive"), _MAX_ARCHIVE)
    return native


def _descriptor(path: Path) -> dict[str, Any]:
    data = _read_descriptor(path)
    if (
        data.get("schema_version") != 1
        or data.get("target") != "wasm32-wasip1"
        or data.get("python_version") != "3.13.7"
        or not isinstance(data.get("abi"), str)
        or not 0 < len(data["abi"]) <= 128
        or any(
            not isinstance(data.get(name), str) or _HASH.fullmatch(data[name]) is None
            for name in ("runtime_manifest_sha256", "catalog_sha256")
        )
    ):
        raise PackageInstallError("unsupported CPython release descriptor")
    _asset(data.get("archive"), _MAX_ARCHIVE)
    resolvers = data.get("resolvers")
    if not isinstance(resolvers, dict) or not resolvers or len(resolvers) > 16:
        raise PackageInstallError("invalid release resolver map")
    for resolver in resolvers.values():
        if not isinstance(resolver, dict):
            raise PackageInstallError("invalid release resolver")
        _asset(resolver, _MAX_RESOLVER)
        version = resolver.get("min_glibc")
        if not isinstance(version, str) or re.fullmatch(r"[0-9]+\.[0-9]+", version) is None:
            raise PackageInstallError("invalid release resolver libc requirement")
    if "native" in data:
        _native_asset(data)
    return data


def _resolver(descriptor: dict[str, Any]) -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    if system != "Linux" or machine not in {"x86_64", "AMD64"}:
        raise PackageInstallError(f"no patched WASI resolver for host {system}/{machine}")
    candidate = descriptor["resolvers"].get("linux-x86_64-glibc")
    if candidate is None:
        raise PackageInstallError("release has no patched WASI resolver for Linux x86-64")
    libc, version = platform.libc_ver()
    required = tuple(map(int, candidate["min_glibc"].split(".")))
    if libc != "glibc" or not version or tuple(map(int, version.split(".")[:2])) < required:
        raise PackageInstallError(f"patched WASI resolver requires glibc {candidate['min_glibc']} or newer")
    return candidate


def _fetch(asset: dict[str, Any], descriptor: Path, output: Path) -> None:
    url = asset["url"]
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme:
        opener = urllib.request.build_opener(_ReleaseRedirect())
        try:
            source = opener.open(url, timeout=30)
        except OSError as error:
            raise PackageInstallError(f"could not fetch release asset: {error}") from error
    else:
        local = descriptor.parent / url
        if local.is_symlink() or not local.is_file():
            raise PackageInstallError(f"local release asset is missing or linked: {url}")
        source = local.open("rb")
    digest = hashlib.sha256()
    size = 0
    try:
        with source, output.open("wb") as target:
            while chunk := source.read(64 * 1024):
                size += len(chunk)
                if size > asset["size"]:
                    raise PackageInstallError("release asset exceeds its declared size")
                digest.update(chunk)
                target.write(chunk)
    except OSError as error:
        raise PackageInstallError(f"could not read release asset: {error}") from error
    if size != asset["size"] or digest.hexdigest() != asset["sha256"]:
        raise PackageInstallError("release asset size or SHA-256 mismatch")


def _member(name: str, allowed_roots: frozenset[str]) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\0" in name
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts[0] not in allowed_roots
        or len(name.encode("utf-8")) > 4096
        or len(path.parts) > 32
    ):
        raise PackageInstallError("release archive contains an unsafe path")
    return path


def _extract(
    archive: Path, destination: Path, *, allowed_roots: frozenset[str] = frozenset({"runtime", "universe"})
) -> None:
    try:
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            if len(members) > _MAX_FILES or sum(item.file_size for item in members) > _MAX_UNPACKED:
                raise PackageInstallError("release archive exceeds its file or unpacked size limit")
            seen: set[PurePosixPath] = set()
            for item in members:
                path = _member(item.filename, allowed_roots)
                mode = item.external_attr >> 16
                if path in seen or item.is_dir() or (stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
                    raise PackageInstallError("release archive contains duplicate or non-regular members")
                seen.add(path)
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output, 64 * 1024)
                if target.stat().st_size != item.file_size:
                    raise PackageInstallError("release archive member size mismatch")
                target.chmod(0o755 if mode & 0o111 else 0o644)
    except PackageInstallError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise PackageInstallError("release archive is invalid or unreadable") from error


def _validate_entry(entry: Path, descriptor: dict[str, Any], resolver: dict[str, Any]) -> None:
    from .cpython import CPythonRuntime

    if entry.is_symlink() or not entry.is_dir():
        raise PackageInstallError("release cache is not a real directory")
    runtime_dir = entry / "runtime"
    universe_dir = entry / "universe"
    uv = entry / "uv"
    actual = set()
    visited = 0
    total = 0
    for path in entry.rglob("*"):
        visited += 1
        if visited > _MAX_FILES * 2 or path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise PackageInstallError("release cache contains too many, linked, or special paths")
        if path.is_file():
            total += path.stat().st_size
            if total > _MAX_UNPACKED + _MAX_RESOLVER:
                raise PackageInstallError("release cache exceeds its size limit")
            actual.add(path.relative_to(entry).as_posix())
    if _digest(runtime_dir / "manifest.json", _MAX_DESCRIPTOR) != descriptor["runtime_manifest_sha256"]:
        raise PackageInstallError("cached runtime manifest differs from the release descriptor")
    if _digest(universe_dir / "catalog.json", _MAX_DESCRIPTOR) != descriptor["catalog_sha256"]:
        raise PackageInstallError("cached catalog differs from the release descriptor")
    if _digest(uv, _MAX_RESOLVER) != resolver["sha256"] or uv.stat().st_size != resolver["size"]:
        raise PackageInstallError("cached patched uv differs from the release descriptor")
    if not uv.stat().st_mode & 0o111:
        raise PackageInstallError("cached patched uv is not executable")
    try:
        runtime = CPythonRuntime(runtime_dir, universe=universe_dir, uv=uv)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PackageInstallError("release runtime manifest or rootfs is invalid") from error
    if runtime.manifest.get("dynamic_abi") != descriptor["abi"] or runtime.version != descriptor["python_version"]:
        raise PackageInstallError("release runtime ABI or Python version differs from its descriptor")
    try:
        universe = Universe(universe_dir, abi=descriptor["abi"], python_version=runtime.version)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PackageInstallError("release catalog is invalid or incompatible") from error
    if universe.default_index != "https://pypi.org/simple":
        raise PackageInstallError("downloadable release requires catalogued pure wheels")
    expected = {"runtime/manifest.json", "universe/catalog.json", "uv"}
    expected.update("runtime/rootfs/" + name.lstrip("/") for name in runtime.manifest["files"])
    needed_providers: set[str] = set()
    for (name, version), package in universe.packages.items():
        path = package["path"]
        _verify_file(path, package["sha256"], 128 * 1024 * 1024)
        needed_providers.update(
            _inspect_wheel(path, name=name, version=version, abi=universe.abi, curated=True).dependencies
        )
        expected.add(path.relative_to(entry).as_posix())
    for provider in universe.providers.values():
        path = provider["path"]
        _verify_wasm(_verify_file(path, provider["sha256"], 16 * 1024 * 1024), universe.abi)
        expected.add(path.relative_to(entry).as_posix())
    universe.provider_closure(needed_providers | set(universe.providers))
    if actual != expected:
        raise PackageInstallError("release cache contains missing or undeclared files")


def materialize(descriptor_path: Path, *, cache_dir: Path | None, offline: bool) -> tuple[Path, Path, Path]:
    """Return immutable local runtime, catalog and resolver paths from one release."""
    descriptor = _descriptor(descriptor_path)
    resolver = _resolver(descriptor)
    root = cache_dir if cache_dir is not None else Path.home() / ".cache/shellsim/cpython-releases"
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    entry = root / (descriptor["archive"]["sha256"] + "-" + resolver["sha256"])
    if entry.exists() or entry.is_symlink():
        try:
            _validate_entry(entry, descriptor, resolver)
        except (OSError, ValueError, PackageInstallError) as error:
            raise PackageInstallError(f"cached CPython release is corrupt: {entry}") from error
        return entry / "runtime", entry / "universe", entry / "uv"
    if offline:
        raise PackageInstallError("verified CPython release is not cached for offline use")
    with tempfile.TemporaryDirectory(prefix=".shellsim-cpython-", dir=root) as temporary:
        work = Path(temporary)
        archive = work / "cohort.zip"
        stage = work / "stage"
        stage.mkdir()
        _fetch(descriptor["archive"], descriptor_path, archive)
        _fetch(resolver, descriptor_path, stage / "uv")
        (stage / "uv").chmod(0o755)
        _extract(archive, stage)
        _validate_entry(stage, descriptor, resolver)
        try:
            stage.rename(entry)
        except OSError:
            if not entry.exists():
                raise
            _validate_entry(entry, descriptor, resolver)
    return entry / "runtime", entry / "universe", entry / "uv"
