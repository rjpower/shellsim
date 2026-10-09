"""Resolve a pinned CPython/WASI universe and stage its verified wheel closure.

The catalog and patched uv binary are explicit host inputs. Resolution and wheel
unpacking happen before one atomic VFS mount; guest code never sees host paths.
"""

from __future__ import annotations

import configparser
import email
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, Sequence

from ._api import Environment, SimulationError
from .pypi import PackageInstallError

_MAX_CATALOG_BYTES = 1024 * 1024
_MAX_WHEEL_BYTES = 128 * 1024 * 1024
_MAX_TOTAL_BYTES = 128 * 1024 * 1024
_MAX_FILES = 10_000
_MAX_PACKAGES = 256
_MAX_PROVIDERS = 64
_MAX_NATIVE_BYTES = 16 * 1024 * 1024
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PROVIDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*\.so(?:\.[0-9]+)*\Z")
_SCRIPT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
_PYTHON_INTERPRETER = re.compile(r"python(?:[0-9]+(?:\.[0-9]+)?)?\Z")
_NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib", ".a", ".wasm")


class _Runtime(Protocol):
    manifest: dict[str, Any]
    universe: Path | None
    uv: Path | None
    version: str
    venv: str | None
    site_packages: str


@dataclass
class _WheelInspection:
    dependencies: set[str]
    artifacts: dict[str, str]
    members: dict[str, tuple[bool, str | None]]
    file_count: int
    uncompressed_bytes: int


def _parse_pylock(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError as error:
            raise PackageInstallError("Python 3.9/3.10 hosts need shellsim[pypi] for TOML parsing") from error
    try:
        return tomllib.loads(_read_bounded(path, _MAX_CATALOG_BYTES).decode())
    except (UnicodeError, ValueError) as error:
        raise PackageInstallError("uv returned an invalid WASI wheel lock") from error


def _name(value: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise PackageInstallError("invalid distribution name in universe")
    return re.sub(r"[-_.]+", "-", value).lower()


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise PackageInstallError("invalid relative path in universe")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise PackageInstallError("universe path traversal is not allowed")
    return path


def _inside(root: Path, value: str) -> Path:
    relative = _relative(value)
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise PackageInstallError("universe path contains a symbolic link")
    if not path.resolve().is_relative_to(root.resolve()):
        raise PackageInstallError("universe path escapes its directory")
    return path


def _sha(value: Any) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise PackageInstallError("invalid SHA-256 in universe")
    return value


def _read_bounded(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise PackageInstallError(f"missing, linked, or oversized artifact: {path.name}")
    return path.read_bytes()


def _verify_file(path: Path, expected: str, limit: int) -> bytes:
    data = _read_bounded(path, limit)
    if hashlib.sha256(data).hexdigest() != _sha(expected):
        raise PackageInstallError(f"artifact SHA-256 mismatch: {path.name}")
    return data


def _uleb(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 35, 7):
        if offset >= len(data):
            break
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
    raise PackageInstallError("invalid core Wasm section length")


def _verify_wasm(data: bytes, abi: str) -> None:
    if len(data) > _MAX_NATIVE_BYTES or not data.startswith(b"\0asm\x01\0\0\0"):
        raise PackageInstallError("native artifact is not a bounded core Wasm module")
    offset = 8
    markers = []
    while offset < len(data):
        section = data[offset]
        size, offset = _uleb(data, offset + 1)
        end = offset + size
        if end > len(data):
            raise PackageInstallError("invalid core Wasm section size")
        if section == 0:
            length, start = _uleb(data, offset)
            name_end = start + length
            if name_end > end:
                raise PackageInstallError("invalid core Wasm custom section")
            if data[start:name_end] == b"shellsim.abi":
                markers.append(data[name_end:end])
        offset = end
    if markers != [abi.encode()]:
        raise PackageInstallError("native artifact ABI does not match the CPython bundle")


def _dependencies(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > _MAX_PROVIDERS:
        raise PackageInstallError("invalid native dependency list")
    for name in value:
        if not isinstance(name, str) or _PROVIDER.fullmatch(name) is None or ".." in name:
            raise PackageInstallError("invalid native dependency name")
    if len(value) != len(set(value)):
        raise PackageInstallError("duplicate native dependency")
    return value


class Universe:
    """One verified local catalog for the bundle's exact dynamic ABI."""

    def __init__(self, root: Path, *, abi: str, python_version: str) -> None:
        self.root = root.resolve()
        data = _read_bounded(self.root / "catalog.json", _MAX_CATALOG_BYTES)
        catalog = json.loads(data)
        if not isinstance(catalog, dict):
            raise PackageInstallError("invalid universe catalog")
        if (
            catalog.get("schema_version") != 1
            or catalog.get("abi") != abi
            or catalog.get("target") != "wasm32-wasip1"
            or catalog.get("python_version") != python_version
        ):
            raise PackageInstallError("universe ABI, target, or Python version differs from the CPython bundle")
        self.abi = abi
        self.packages: dict[tuple[str, str], dict[str, Any]] = {}
        entries = catalog.get("packages")
        if not isinstance(entries, list) or len(entries) > _MAX_PACKAGES:
            raise PackageInstallError("invalid universe package catalog")
        for entry in entries:
            if not isinstance(entry, dict) or not {"name", "version", "wheel", "sha256"} <= entry.keys():
                raise PackageInstallError("invalid curated package entry")
            name = _name(entry["name"])
            version = entry["version"]
            if not isinstance(version, str) or not version or len(version) > 128:
                raise PackageInstallError("invalid curated package version")
            key = (name, version)
            if key in self.packages:
                raise PackageInstallError("duplicate curated package version")
            path = _inside(self.root, entry["wheel"])
            if path.suffix != ".whl" or not path.is_file() or path.stat().st_size > _MAX_WHEEL_BYTES:
                raise PackageInstallError("curated artifact must be a bounded wheel")
            self.packages[key] = {"path": path, "sha256": _sha(entry["sha256"])}
        self.providers: dict[str, dict[str, Any]] = {}
        providers = catalog.get("native_providers", [])
        if not isinstance(providers, list) or len(providers) > _MAX_PROVIDERS:
            raise PackageInstallError("invalid native provider catalog")
        for entry in providers:
            if not isinstance(entry, dict) or not {"name", "path", "sha256", "native_dependencies"} <= entry.keys():
                raise PackageInstallError("invalid native provider entry")
            name = entry["name"]
            if not isinstance(name, str) or _PROVIDER.fullmatch(name) is None or ".." in name:
                raise PackageInstallError("invalid native provider name")
            if name in self.providers or entry.get("destination") != f"/lib/{name}":
                raise PackageInstallError("duplicate provider or invalid /lib destination")
            path = _inside(self.root, entry["path"])
            if not path.is_file() or path.stat().st_size > _MAX_NATIVE_BYTES:
                raise PackageInstallError("native provider must be a bounded file")
            self.providers[name] = {
                "path": path,
                "sha256": _sha(entry["sha256"]),
                "native_dependencies": _dependencies(entry["native_dependencies"]),
            }
        pure_index = catalog.get("pure_index")
        self.default_index = (
            _inside(self.root, pure_index).as_uri() if pure_index is not None else "https://pypi.org/simple"
        )
        if pure_index is not None and not _inside(self.root, pure_index).is_dir():
            raise PackageInstallError("local pure index must be a directory")

    def index(self, root: Path) -> Path:
        """Generate a Simple index from catalogued wheel entries."""
        index = root / "curated-index"
        index.mkdir()
        for (name, _), entry in self.packages.items():
            directory = index / name
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "index.html").open("a") as stream:
                stream.write(f'<a href="{entry["path"].as_uri()}#sha256={entry["sha256"]}">{entry["path"].name}</a>\n')
        return index

    def provider_closure(self, names: set[str]) -> dict[str, Path]:
        resolved: dict[str, Path] = {}
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in resolved:
                return
            if name in visiting:
                raise PackageInstallError("native provider dependency cycle")
            provider = self.providers.get(name)
            if provider is None:
                raise PackageInstallError(f"missing native provider: {name}")
            visiting.add(name)
            for dependency in provider["native_dependencies"]:
                visit(dependency)
            visiting.remove(name)
            _verify_wasm(_verify_file(provider["path"], provider["sha256"], _MAX_NATIVE_BYTES), self.abi)
            resolved[name] = provider["path"]

        for name in sorted(names):
            visit(name)
        return resolved


def _run_uv(uv: Path, args: list[str], *, cwd: Path) -> None:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("UV_", "PIP_")) and name not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    environment["UV_CACHE_DIR"] = str(cwd / "uv-cache")
    try:
        completed = subprocess.run(
            [str(uv), "--no-config", *args],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except FileNotFoundError as error:
        raise PackageInstallError("patched uv executable is missing") from error
    except subprocess.TimeoutExpired as error:
        raise PackageInstallError("WASI package resolution timed out") from error
    if completed.returncode:
        raise PackageInstallError("WASI package resolution failed: " + completed.stderr.strip()[-8000:])


def _wheel_path(url: str, universe: Universe) -> Path | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "file" or parsed.netloc:
        return None
    path = Path(urllib.parse.unquote(parsed.path)).resolve()
    if not path.is_relative_to(universe.root):
        raise PackageInstallError("resolved wheel file escapes the universe")
    return path


class _PyPIOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, fp: Any, code: int, message: str, headers: Any, newurl: str) -> Any:
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
            raise PackageInstallError("PyPI wheel redirected outside files.pythonhosted.org")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _download(url: str, destination: Path, universe: Universe) -> bytes:
    local = _wheel_path(url, universe)
    if local is not None:
        data = _read_bounded(local, _MAX_WHEEL_BYTES)
    else:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
            raise PackageInstallError("pure wheel URL is outside the trusted PyPI file host")
        opener = urllib.request.build_opener(_PyPIOnlyRedirect())
        try:
            with opener.open(url, timeout=30) as response:
                data = response.read(_MAX_WHEEL_BYTES + 1)
        except OSError as error:
            raise PackageInstallError(f"could not fetch PyPI wheel: {error}") from error
        if len(data) > _MAX_WHEEL_BYTES:
            raise PackageInstallError("resolved wheel exceeds 128 MiB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return data


def _native_path(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(_NATIVE_SUFFIXES) or ".so." in lower


def _inspect_wheel(path: Path, *, name: str, version: str, abi: str, curated: bool) -> _WheelInspection:
    dependencies: set[str] = set()
    artifacts: dict[str, str] = {}
    file_members: dict[str, tuple[bool, str | None]] = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        uncompressed_bytes = sum(member.file_size for member in members)
        if len(members) > _MAX_FILES or uncompressed_bytes > _MAX_TOTAL_BYTES:
            raise PackageInstallError("resolved wheel exceeds file or uncompressed size limit")
        seen: set[PurePosixPath] = set()
        for member in members:
            relative = _relative(member.filename.rstrip("/"))
            mode = member.external_attr >> 16
            if relative in seen or stat.S_ISLNK(mode):
                raise PackageInstallError("resolved wheel contains a duplicate or linked path")
            if any(part.endswith(".data") for part in relative.parts):
                raise PackageInstallError("wheel data relocation is not supported")
            seen.add(relative)
            for parent in relative.parents:
                if parent != PurePosixPath("."):
                    prior = file_members.get(parent.as_posix())
                    if prior is not None and not prior[0]:
                        raise PackageInstallError("wheel file conflicts with a directory")
                    file_members[parent.as_posix()] = (True, None)
            key = relative.as_posix()
            if member.is_dir():
                prior = file_members.get(key)
                if prior is not None and not prior[0]:
                    raise PackageInstallError("wheel directory conflicts with a file")
                file_members[key] = (True, None)
            else:
                if key in file_members:
                    raise PackageInstallError("wheel file conflicts with a directory")
                file_members[key] = (False, hashlib.sha256(archive.read(member)).hexdigest())
        dist_infos = [part for part in {path.parts[0] for path in seen} if part.endswith(".dist-info")]
        if len(dist_infos) != 1:
            raise PackageInstallError("resolved wheel has invalid distribution metadata")
        dist_info = dist_infos[0]
        for filename in (f"{dist_info}/METADATA", f"{dist_info}/WHEEL"):
            if filename not in archive.namelist() or archive.getinfo(filename).file_size > 65536:
                raise PackageInstallError("resolved wheel has missing or oversized metadata")
        metadata = email.message_from_bytes(archive.read(f"{dist_info}/METADATA"))
        if _name(metadata["Name"]) != name or metadata["Version"] != version:
            raise PackageInstallError("resolved wheel identity differs from uv's resolution")
        wheel_text = archive.read(f"{dist_info}/WHEEL").decode()
        wheel_lines = {line.strip() for line in wheel_text.splitlines()}
        pure = "Root-Is-Purelib: true" in wheel_lines and any(
            line.startswith("Tag: ")
            and line[5:].split("-")[-2:] == ["none", "any"]
            and ("py3" in line[5:].split("-")[0].split(".") or "py313" in line[5:].split("-")[0].split("."))
            for line in wheel_lines
        )
        # Curated provenance controls which source/hash is authoritative. Wheel
        # metadata controls kind; approved source-only releases stay pure wheels.
        if pure:
            if any(_native_path(member.filename) for member in members if not member.is_dir()):
                raise PackageInstallError("pure wheel contains native files")
        elif curated:
            if "Tag: cp313-cp313-wasm32_wasip1" not in wheel_lines:
                raise PackageInstallError("curated wheel has an incompatible WASI tag")
            native_path = f"{dist_info}/shellsim-native.json"
            if native_path not in {member.filename for member in members}:
                raise PackageInstallError("curated native wheel lacks shellsim-native.json")
            info = archive.getinfo(native_path)
            if info.file_size > _MAX_CATALOG_BYTES:
                raise PackageInstallError("native wheel manifest exceeds 1 MiB")
            native = json.loads(archive.read(info))
            if (
                native.get("schema_version") != 1
                or _name(native.get("name")) != name
                or native.get("version") != version
                or native.get("abi") != abi
                or not isinstance(native.get("recipe"), dict)
            ):
                raise PackageInstallError("native wheel manifest identity or ABI mismatch")
            entries = native.get("artifacts")
            if not isinstance(entries, list) or len(entries) > _MAX_FILES:
                raise PackageInstallError("invalid native artifact list")
            for entry in entries:
                relative = _relative(entry["path"]).as_posix()
                if relative in artifacts or not _native_path(relative):
                    raise PackageInstallError("duplicate or invalid native artifact")
                data = archive.read(relative)
                if hashlib.sha256(data).hexdigest() != _sha(entry["sha256"]):
                    raise PackageInstallError(f"native artifact SHA-256 mismatch: {relative}")
                _verify_wasm(data, abi)
                artifacts[relative] = entry["sha256"]
                dependencies.update(_dependencies(entry["native_dependencies"]))
            present = {member.filename for member in members if not member.is_dir() and _native_path(member.filename)}
            if present != set(artifacts):
                raise PackageInstallError("native wheel has undeclared native files")
        else:
            raise PackageInstallError("non-curated wheel is not pure Python")
    return _WheelInspection(dependencies, artifacts, file_members, len(members), uncompressed_bytes)


def _add_wheel_to_set(
    inspection: _WheelInspection,
    members: dict[str, tuple[bool, str | None]],
    file_count: int,
    uncompressed_bytes: int,
) -> tuple[int, int]:
    file_count += inspection.file_count
    uncompressed_bytes += inspection.uncompressed_bytes
    if file_count > _MAX_FILES or uncompressed_bytes > _MAX_TOTAL_BYTES:
        raise PackageInstallError("resolved wheel set exceeds file or uncompressed size limit")
    for path, member in inspection.members.items():
        prior = members.get(path)
        if prior is not None and prior != member:
            raise PackageInstallError(f"resolved wheel member path conflicts across wheels: {path}")
        members[path] = member
    return file_count, uncompressed_bytes


def _stage_provider(universe: Universe, name: str, source: Path, destination: Path) -> None:
    provider = universe.providers[name]
    data = _verify_file(source, provider["sha256"], _MAX_NATIVE_BYTES)
    _verify_wasm(data, universe.abi)
    destination.write_bytes(data)


def _stage_console_scripts(site: Path, staging: Path, venv: str) -> None:
    scripts = site / "bin"
    if not scripts.exists():
        return
    declared: set[str] = set()
    for metadata in site.glob("*.dist-info/entry_points.txt"):
        if metadata.stat().st_size > 65536:
            raise PackageInstallError("installed entry point metadata exceeds 64 KiB")
        config = configparser.ConfigParser(interpolation=None)
        config.optionxform = str
        try:
            config.read_string(metadata.read_text(encoding="utf-8"))
        except (configparser.Error, UnicodeError) as error:
            raise PackageInstallError("installed entry point metadata is invalid") from error
        if config.has_section("console_scripts"):
            declared.update(config.options("console_scripts"))
    destination = staging / venv.lstrip("/") / "bin"
    destination.mkdir(parents=True, exist_ok=True)
    for script in scripts.iterdir():
        if (
            not script.is_file()
            or script.is_symlink()
            or _SCRIPT_NAME.fullmatch(script.name) is None
            or ".." in script.name
            or script.name in {"python", "python3", "python3.13", "activate"}
            or script.name not in declared
        ):
            raise PackageInstallError("installed script is not a supported console entry point")
        source = script.read_bytes()
        first, separator, body = source.partition(b"\n")
        try:
            interpreter = first[2:].decode("utf-8").split()[0]
        except (UnicodeError, IndexError) as error:
            raise PackageInstallError("installed console script has no Python shebang") from error
        if (
            not separator
            or not first.startswith(b"#!/")
            or _PYTHON_INTERPRETER.fullmatch(Path(interpreter).name) is None
        ):
            raise PackageInstallError("installed console script has no Python shebang")
        target = destination / script.name
        if target.exists():
            raise PackageInstallError("installed console script conflicts with the CPython venv")
        target.write_bytes(b"#!" + (venv + "/bin/python").encode() + b"\n" + body)
        target.chmod(script.stat().st_mode & 0o777)
        script.unlink()
    scripts.rmdir()


def _lock_versions(lock: dict[str, Any]) -> set[tuple[str, str]]:
    packages = lock.get("packages")
    if not isinstance(packages, list) or len(packages) > _MAX_PACKAGES:
        raise PackageInstallError("uv returned an oversized WASI wheel lock")
    versions: set[tuple[str, str]] = set()
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("version"), str):
            raise PackageInstallError("invalid package in uv's WASI wheel lock")
        pair = (_name(package.get("name")), package["version"])
        if pair in versions:
            raise PackageInstallError("duplicate package in uv's WASI wheel lock")
        versions.add(pair)
    return versions


def _curated_cohorts(
    universe: Universe, requirements: Sequence[str], locked_versions: set[tuple[str, str]] | None
) -> str:
    requested = {
        _name(match.group())
        for requirement in requirements
        if (match := re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement))
    }
    requested.update(name for name, _ in locked_versions or ())
    cohorts = []
    for name in sorted(requested):
        versions = sorted(version for package, version in universe.packages if package == name)
        if versions:
            cohorts.append(f"{name}: {', '.join(versions[:8])}")
    return "; curated WASI versions available: " + "; ".join(cohorts[:8]) if cohorts else ""


def install(
    runtime: _Runtime,
    environment: Environment,
    requirements: Sequence[str],
    *,
    locked_versions: set[tuple[str, str]] | None = None,
) -> None:
    """Resolve, verify, and atomically stage a dynamic package set."""
    if runtime.universe is None or runtime.uv is None or runtime.venv is None:
        raise PackageInstallError("dynamic CPython installs need a local universe and patched uv")
    universe = Universe(runtime.universe, abi=runtime.manifest["dynamic_abi"], python_version=runtime.version)
    with tempfile.TemporaryDirectory(prefix="shellsim-cpython-universe-") as temporary:
        work = Path(temporary)
        index = universe.index(work)
        requirements_file = work / "requirements.in"
        requirements_file.write_text("\n".join(requirements) + "\n")
        pylock = work / "pylock.toml"
        common = [
            "--python",
            sys.executable,
            "--no-python-downloads",
            "--python-platform",
            "wasm32-wasip1",
            "--python-version",
            runtime.version,
            "--only-binary",
            ":all:",
            "--no-cache",
            "--index-strategy",
            "first-index",
        ]
        compile_args = [
            "pip",
            "compile",
            "--format",
            "pylock.toml",
            "--index",
            index.as_uri(),
            "--default-index",
            universe.default_index,
            *common,
        ]
        active_versions = None
        try:
            if locked_versions is not None:
                active_lock = work / "pylock.active.toml"
                _run_uv(
                    runtime.uv,
                    [*compile_args, "--no-deps", "--output-file", str(active_lock), str(requirements_file)],
                    cwd=work,
                )
                active_versions = _lock_versions(_parse_pylock(active_lock))
                if not active_versions <= locked_versions:
                    raise PackageInstallError("uv.lock exports a package version absent from its package table")
            _run_uv(
                runtime.uv,
                [*compile_args, "--output-file", str(pylock), str(requirements_file)],
                cwd=work,
            )
        except PackageInstallError as error:
            cohorts = _curated_cohorts(universe, requirements, locked_versions)
            if cohorts:
                raise PackageInstallError(str(error) + cohorts) from error
            raise
        lock = _parse_pylock(pylock)
        packages = lock.get("packages")
        if not isinstance(packages, list) or not packages or len(packages) > _MAX_PACKAGES:
            raise PackageInstallError("uv returned an empty or oversized WASI wheel lock")
        if active_versions is not None and _lock_versions(lock) != active_versions:
            raise PackageInstallError(
                "uv.lock dependency closure differs for WASI; regenerate the lock for this target"
            )
        wheel_dir = work / "wheels"
        selected: list[tuple[str, str, Path, str]] = []
        native_artifacts: dict[str, str] = {}
        needed: set[str] = set()
        total_bytes = 0
        total_uncompressed = 0
        total_files = 0
        all_members: dict[str, tuple[bool, str | None]] = {}
        for number, package in enumerate(packages):
            if not isinstance(package, dict) or not {"name", "version", "wheels"} <= package.keys():
                raise PackageInstallError("invalid package entry in uv's wheel lock")
            name = _name(package["name"])
            version = package["version"]
            if not isinstance(version, str) or len(version) > 128:
                raise PackageInstallError("invalid package version in uv's wheel lock")
            wheels = package.get("wheels")
            if not isinstance(wheels, list) or len(wheels) != 1 or package.get("sdist"):
                raise PackageInstallError(f"{name}: expected exactly one resolved wheel")
            wheel = wheels[0]
            if not isinstance(wheel, dict) or not isinstance(wheel.get("url"), str):
                raise PackageInstallError("invalid wheel entry in uv's wheel lock")
            url = wheel["url"]
            digest = _sha(wheel.get("hashes", {}).get("sha256"))
            filename = PurePosixPath(urllib.parse.urlparse(url).path).name
            if not filename.endswith(".whl") or len(filename) > 255:
                raise PackageInstallError("resolved artifact is not a wheel")
            curated = universe.packages.get((name, version))
            if curated:
                source = _wheel_path(url, universe)
                if source != curated["path"] or digest != curated["sha256"]:
                    raise PackageInstallError("curated wheel URL or hash differs from the catalog")
            elif any(key[0] == name for key in universe.packages):
                raise PackageInstallError(f"curated distribution {name} has no compatible catalog version")
            target = wheel_dir / str(number) / filename
            data = _download(url, target, universe)
            total_bytes += len(data)
            if total_bytes > _MAX_TOTAL_BYTES or hashlib.sha256(data).hexdigest() != digest:
                raise PackageInstallError("resolved wheel exceeds total size limit or has a corrupt hash")
            inspection = _inspect_wheel(target, name=name, version=version, abi=universe.abi, curated=bool(curated))
            total_files, total_uncompressed = _add_wheel_to_set(
                inspection, all_members, total_files, total_uncompressed
            )
            needed.update(inspection.dependencies)
            for path, artifact_hash in inspection.artifacts.items():
                if path in native_artifacts and native_artifacts[path] != artifact_hash:
                    raise PackageInstallError("native artifact path conflicts across wheels")
                native_artifacts[path] = artifact_hash
            selected.append((name, version, target, digest))
        providers = universe.provider_closure(needed)
        local_lock = work / "pylock.local.toml"
        lines = ['lock-version = "1.0"', 'created-by = "shellsim"', 'requires-python = ">=3.13.7"']
        for name, version, path, digest in selected:
            lines.extend(
                [
                    "",
                    "[[packages]]",
                    f"name = {json.dumps(name)}",
                    f"version = {json.dumps(version)}",
                    f"wheels = [{{ url = {json.dumps(path.as_uri())}, hashes = {{ sha256 = {json.dumps(digest)} }} }}]",
                ]
            )
        local_lock.write_text("\n".join(lines) + "\n")
        staging = work / "root"
        site = staging / runtime.site_packages.lstrip("/")
        site.mkdir(parents=True)
        _run_uv(
            runtime.uv,
            ["pip", "sync", str(local_lock), "--target", str(site), "--require-hashes", *common],
            cwd=work,
        )
        _stage_console_scripts(site, staging, runtime.venv)
        (site / ".lock").unlink(missing_ok=True)
        if any(path.is_symlink() for path in staging.rglob("*")):
            raise PackageInstallError("staged package set contains links")
        staged_artifacts = {
            path.relative_to(site).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in site.rglob("*")
            if path.is_file() and _native_path(path.name)
        }
        if staged_artifacts != native_artifacts:
            raise PackageInstallError("installed native files differ from verified wheel manifests")
        lib = staging / "lib"
        lib.mkdir()
        for name, source in providers.items():
            _stage_provider(universe, name, source, lib / name)
        files = [path for path in staging.rglob("*") if path.is_file()]
        if len(files) > _MAX_FILES or sum(path.stat().st_size for path in files) > _MAX_TOTAL_BYTES:
            raise PackageInstallError("staged package set exceeds VFS import limits")
        for path in files:
            destination = "/" + path.relative_to(staging).as_posix()
            try:
                existing = environment.read_file(destination)
            except SimulationError as error:
                if "No such file" not in str(error):
                    raise
            else:
                if existing != path.read_bytes():
                    raise PackageInstallError(f"package file would overwrite an existing VFS file: {destination}")
        environment._native.mount_package_tree(str(staging))
