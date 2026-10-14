"""Install a curated WASI package closure from independently cached delivery assets.

Downloads and extraction are trusted host setup. Guests receive a private VFS;
cache reuse does not hash artifacts or invoke a host package manager.
"""

from __future__ import annotations

import configparser
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Sequence
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from ._cpython_release import _extract
from ._cpython_universe import _inside, _inspect_wheel, _relative, _stage_console_scripts, _verify_wasm
from ._startup import logger as startup_logger
from ._startup import phase
from .pypi import PackageInstallError, _mount_package_tree

if TYPE_CHECKING:
    from ._api import Environment
    from .cpython import CPythonRuntime

_MAX_CATALOG = 4 * 1024**2
_MAX_ASSET = 512 * 1024**2
_MAX_UNPACKED = 1024**3
_MAX_FILES = 100_000


def _read_url(location: str):
    parsed = urlsplit(location)
    if parsed.scheme in {"http", "https", "file"} and not parsed.username and not parsed.password:
        return urllib.request.urlopen(location, timeout=60)
    if not parsed.scheme:
        return Path(location).open("rb")
    raise PackageInstallError("catalog assets need HTTP(S), file URLs or local paths")


class Catalog:
    """One version per distribution, with explicit CPython/WASI marker values."""

    def __init__(
        self,
        location: str | Path,
        *,
        cache_dir: Path | None = None,
        offline: bool = False,
        trusted_precompiled: bool = False,
    ):
        self.trusted_precompiled = trusted_precompiled
        self.cache = (
            cache_dir or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "shellsim/ports"
        ).resolve()
        # An arbitrary catalog must not seed bytes that a later trusted catalog
        # deserializes. Both policies need sibling roots, never overlapping roots.
        self.cache /= "trusted-precompiled" if trusted_precompiled else "portable"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        location = str(location)
        snapshot = self.cache / "catalogs" / (hashlib.sha256(location.encode()).hexdigest() + ".json")
        if offline:
            if not snapshot.is_file():
                raise PackageInstallError("catalog is not cached for offline use")
            data = snapshot.read_bytes()
        else:
            with _read_url(location) as source:
                data = source.read(_MAX_CATALOG + 1)
        if len(data) > _MAX_CATALOG:
            raise PackageInstallError("catalog exceeds its metadata limit")
        self.data = json.loads(data)
        if self.data.get("schema_version") != 2:
            raise PackageInstallError("catalog needs delivery schema 2")
        self.runtime = self.data.get("runtime")
        self.version = self.runtime["python_version"] if self.runtime else None
        self.abi = self.runtime["abi"] if self.runtime else self.data.get("abi")
        self.ports = {port["reference"]: port for port in self.data.get("ports", [])}
        self.packages = {}
        for package in self.data["packages"]:
            name = canonicalize_name(package["name"])
            if name in self.packages:
                raise PackageInstallError("curated catalog contains multiple versions of " + name)
            self.packages[name] = package
        if len(self.packages) > 256:
            raise PackageInstallError("catalog exceeds its package limit")
        self.providers = {}
        for provider in self.data["native"]:
            if provider["name"] in self.providers:
                raise PackageInstallError("catalog contains duplicate native provider names")
            self.providers[provider["name"]] = provider
        self.markers = {
            "implementation_name": "cpython",
            "implementation_version": self.version,
            "os_name": "posix",
            "platform_machine": "wasm32",
            "platform_python_implementation": "CPython",
            "platform_release": "",
            "platform_system": "WASI",
            "platform_version": "",
            "python_full_version": self.version,
            "python_version": ".".join(self.version.split(".")[:2]) if self.version else "",
            "sys_platform": "wasi",
            "extra": "",
        }
        if not offline:
            snapshot.parent.mkdir(exist_ok=True)
            from ._catalog_cache import atomic_bytes

            atomic_bytes(snapshot, data)

    def asset(self, record: dict) -> Path:
        """Fetch an immutable asset once; failed transfers never become cache hits."""
        path = _inside(self.cache, record["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        size = record["size"]
        if type(size) is not int or not 0 < size <= _MAX_ASSET:
            raise PackageInstallError("asset exceeds its download limit")
        with path.with_name(path.name + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.is_file():
                startup_logger.info("asset: disk cache hit name=%s", path.name)
                return path
            if self.offline:
                raise PackageInstallError("selected asset is not cached for offline use: " + path.name)
            from ._catalog_cache import staged_file

            with phase("asset.download", asset=path.name, bytes=size), staged_file(path) as temporary:
                with _read_url(record["url"]) as source, temporary.open("wb") as output:
                    total = 0
                    while chunk := source.read(1024**2):
                        total += len(chunk)
                        if total > size:
                            raise PackageInstallError("asset exceeds its declared download size")
                        output.write(chunk)
                    if total != size:
                        raise PackageInstallError("asset transfer ended early")
        return path

    def bundle(self) -> Path:
        if self.runtime is None:
            raise PackageInstallError("catalog has no Python runtime")
        archive = self.asset(self.runtime)
        key = _relative(self.runtime["path"]).parts[-2]
        target = self.cache / "runtimes" / key
        target.parent.mkdir(exist_ok=True)
        with target.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (target / "runtime").is_dir():
                startup_logger.info("runtime: extracted cache hit")
                return target / "runtime"
            with tempfile.TemporaryDirectory(prefix=".runtime-", dir=target.parent) as temporary:
                stage = Path(temporary) / "unpacked"
                stage.mkdir()
                with phase("runtime.extract"):
                    _extract(archive, stage, allowed_roots=frozenset({"runtime"}))
                stage.rename(target)
        return target / "runtime"

    def matching_companions(self, record: dict) -> list[dict]:
        """Select host-native assets before downloading any bytes from the catalog."""
        if not self.trusted_precompiled:
            return []
        from . import _native

        return [
            item
            for item in record.get("companions", [])
            if item["target"] == _native.precompiled_wasm_target()
            and item["build_id"] == _native.precompiled_wasm_build_id()
        ]

    def _register_companion(self, source: bytes, companion: dict, *, executable: bool = False) -> None:
        from . import _native

        name = Path(companion["path"]).name
        with phase("precompiled.register", asset=name):
            outcome = _native.register_trusted_precompiled_wasm_file(
                source,
                str(self.asset(companion)),
                executable=executable,
            )
        startup_logger.info("precompiled: %s asset=%s (compatibility checked on first use)", outcome, name)

    def register_precompiled(self, source: Path, record: dict, *, executable: bool) -> None:
        """Load a host companion only when the caller trusts this catalog's producer bytes.

        Wasmtime incompatibility leaves the raw Wasm compilation path available.
        The trust opt-in guarantees the exact source/kind association; this is not
        a native-code verification or a guest filesystem operation.
        """
        companions = self.matching_companions(record)
        if len(companions) > 1:
            raise PackageInstallError("multiple matching native companions")
        if companions:
            self._register_companion(source.read_bytes(), companions[0], executable=executable)

    def register_wheel_companions(self, wheel: Path, record: dict, declared: set[str]) -> None:
        """Associate trusted sidecars only with extensions declared in the inspected wheel."""
        companions = self.matching_companions(record)
        if not companions:
            return
        with zipfile.ZipFile(wheel) as archive:
            selected = set()
            for companion in companions:
                source = _relative(companion["source"]).as_posix()
                if source not in declared or source in selected:
                    raise PackageInstallError("invalid or duplicate wheel companion source")
                selected.add(source)
                self._register_companion(archive.read(source), companion)

    def resolve(self, requirements: str | Sequence[str]) -> list[dict]:
        if self.runtime is None:
            raise PackageInstallError("catalog has no Python runtime")
        if isinstance(requirements, str):
            requirements = (requirements,)
        if not isinstance(requirements, Sequence) or isinstance(requirements, bytes):
            raise TypeError("pypi_packages must be package specs or a sequence of specs")
        if len(requirements) > 256:
            raise ValueError("pypi_packages accepts at most 256 package specs")
        pending = deque()
        for text in requirements:
            if not isinstance(text, str):
                raise TypeError("package specs must be strings")
            try:
                pending.append(Requirement(text))
            except InvalidRequirement as error:
                raise PackageInstallError("invalid package requirement") from error
        selected, active = {}, {}
        while pending:
            requirement = pending.popleft()
            if requirement.url is not None:
                raise PackageInstallError("curated catalogs do not support direct URL requirements")
            if requirement.marker and not requirement.marker.evaluate(self.markers):
                continue
            name = canonicalize_name(requirement.name)
            package = self.packages.get(name)
            if package is None:
                raise PackageInstallError("package is absent from the curated catalog: " + name)
            version = Version(package["version"])
            if not requirement.specifier.contains(version, prereleases=True):
                raise PackageInstallError(f"{name}=={version} does not satisfy {requirement.specifier}")
            if package["requires_python"] and not SpecifierSet(package["requires_python"]).contains(self.version):
                raise PackageInstallError("package requires a different Python version: " + name)
            extras = {canonicalize_name(extra) for extra in requirement.extras}
            if not extras <= {canonicalize_name(extra) for extra in package["extras"]}:
                raise PackageInstallError("unknown package extra for " + name)
            prior = active.get(name)
            if prior is not None and extras <= prior:
                continue
            active[name] = (prior or set()) | extras
            selected[name] = package
            for text in package["requires_dist"]:
                dependency = Requirement(text)
                if dependency.marker and not any(
                    dependency.marker.evaluate({**self.markers, "extra": extra}) for extra in {"", *active[name]}
                ):
                    continue
                dependency.marker = None
                pending.append(dependency)
        return [selected[name] for name in sorted(selected)]

    def native_closure(self, names: set[str]) -> list[dict]:
        selected = {}
        pending = list(names)
        while pending:
            name = pending.pop()
            if name in selected:
                continue
            if name not in self.providers:
                raise PackageInstallError("missing native provider: " + name)
            provider = self.providers[name]
            if provider["abi"] not in {None, self.abi}:
                raise PackageInstallError("native provider has an incompatible ABI")
            selected[name] = provider
            pending.extend(provider["native_dependencies"])
        return [selected[name] for name in sorted(selected)]

    def install(self, runtime: CPythonRuntime, environment: Environment, requirements: str | Sequence[str]) -> None:
        packages = self.resolve(requirements)
        names = set(self.runtime["native_dependencies"])
        for package in packages:
            names.update(package["native_dependencies"])
            if package["abi"] not in {None, self.abi}:
                raise PackageInstallError("wheel has an incompatible ABI")
        providers = self.native_closure(names)
        with tempfile.TemporaryDirectory(prefix=".install-", dir=self.cache) as temporary:
            staging = Path(temporary)
            site = staging / runtime.site_packages.lstrip("/")
            site.mkdir(parents=True)
            count, unpacked = 0, 0
            for package in packages:
                if package["name"] in environment._catalog_packages:
                    continue
                path = self.asset(package)
                with phase("wheel.prepare", package=package["name"]):
                    inspection = _inspect_wheel(
                        path,
                        name=package["name"],
                        version=package["version"],
                        abi=self.abi,
                        curated=True,
                        verify_digests=False,
                    )
                    count += inspection.file_count
                    unpacked += inspection.uncompressed_bytes
                    if count > _MAX_FILES or unpacked > _MAX_UNPACKED:
                        raise PackageInstallError("selected wheels exceed their staging limit")
                    self.register_wheel_companions(path, package, set(inspection.artifacts))
                    unpack_wheel(path, site)
            create_entry_points(site)
            _stage_console_scripts(site, staging, runtime.venv)
            assets = self.assets_for([*packages, *providers])
            with phase("native.stage", providers=len(providers), assets=len(assets)):
                stage_catalog_items(self, environment, providers, assets, staging)
            with phase("packages.mount"):
                _mount_package_tree(environment, staging, "/", normalize_modes=True)
        environment._catalog_packages.update(package["name"] for package in packages)
        environment._catalog_native.update(provider["name"] for provider in providers)
        environment._catalog_assets.update(item["path"] for item in assets)

    def ports_for(self, records: Sequence[dict]) -> set[str]:
        """Follow the catalog's declared target/runtime port dependencies."""
        selected = {record["port"] for record in records if "port" in record}
        pending = list(selected)
        while pending:
            for reference in self.ports.get(pending.pop(), {}).get("dependencies", []):
                if reference not in selected:
                    selected.add(reference)
                    pending.append(reference)
        return selected

    def assets_for(self, records: Sequence[dict], *, development: bool = False) -> list[dict]:
        selected = self.ports_for(records)
        return [
            item
            for item in self.data.get("assets", [])
            if item["port"] in selected and (development or item.get("usage") != "development")
        ]

    def install_tools(self, environment: Environment, tools: str | Sequence[str]) -> None:
        """Install selected tool ports and their declared closure in one atomic mount."""
        if isinstance(tools, str):
            tools = (tools,)
        if not isinstance(tools, Sequence) or isinstance(tools, bytes) or not 1 <= len(tools) <= 128:
            raise ValueError("tools must contain 1..128 catalog tool names")
        for name in tools:
            if name not in self.providers or self.providers[name]["kind"] != "tools":
                raise PackageInstallError("tool is absent from the curated catalog: " + str(name))
        names = set(tools)
        ports = self.ports_for(self.native_closure(names))
        names.update(
            item["name"] for item in self.providers.values() if item["kind"] == "tools" and item.get("port") in ports
        )
        providers = self.native_closure(names)
        assets = self.assets_for(providers, development=True)
        with tempfile.TemporaryDirectory(prefix=".tools-", dir=self.cache) as temporary:
            staging = Path(temporary)
            stage_catalog_items(self, environment, providers, assets, staging)
            replacements = {item["destination"] for item in providers if item["kind"] == "tools"}
            commands = staging / "usr/bin"
            if commands.is_dir():
                replacements.update(
                    "/usr/bin/" + path.name
                    for path in commands.iterdir()
                    if path.is_file() and path.stat().st_mode & 0o111
                )
            # The mount policy admits only original base-image commands at these
            # destinations; an ordinary guest file still causes an atomic conflict.
            _mount_package_tree(
                environment,
                staging,
                "/",
                normalize_modes=True,
                replace_builtin_tools=sorted(replacements),
            )
        environment._catalog_native.update(provider["name"] for provider in providers)
        environment._catalog_assets.update(item["path"] for item in assets)


def stage_catalog_items(
    catalog: Catalog, environment: Environment, providers: Sequence[dict], assets: Sequence[dict], staging: Path
) -> None:
    for item in [
        *(provider for provider in providers if provider["name"] not in environment._catalog_native),
        *(asset for asset in assets if asset["path"] not in environment._catalog_assets),
    ]:
        source = catalog.asset(item)
        if item["kind"] == "archive":
            unpack_files(source, staging)
            continue
        if item["kind"] in {"tools", "shared_libraries"}:
            data = source.read_bytes()
            if item["kind"] == "shared_libraries" or data.startswith(b"\0asm"):
                _verify_wasm(data, catalog.abi, max_bytes=128 * 1024**2)
                catalog.register_precompiled(source, item, executable=item["kind"] == "tools")
        destination = staging / _relative(item["destination"].removeprefix("/"))
        if destination.exists():
            if destination.is_file() and destination.read_bytes() == source.read_bytes():
                continue
            raise PackageInstallError("catalog installation paths conflict: " + item["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o755 if item["kind"] == "tools" else 0o644)


def unpack_files(path: Path, staging: Path) -> None:
    """Stage a declared file bundle within guest paths, including conflict checks."""
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > _MAX_FILES or sum(member.file_size for member in members) > _MAX_UNPACKED:
            raise PackageInstallError("catalog files exceed their staging limit")
        for member in members:
            target = staging / _relative(member.filename.rstrip("/"))
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if (member.external_attr >> 16) & 0o170000 not in {0, 0o100000}:
                raise PackageInstallError("catalog files must be regular files")
            data = archive.read(member)
            if target.exists():
                if target.is_file() and target.read_bytes() == data:
                    continue
                raise PackageInstallError("catalog installation paths conflict: " + member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if member.external_attr >> 16 & 0o111 else 0o644)


def unpack_wheel(path: Path, site: Path) -> None:
    """Relocate wheel data and reject conflicting installed files."""
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            relative = _relative(member.filename.rstrip("/"))
            if relative.name.endswith(".cwasm"):
                continue
            if relative.parts[0].endswith(".data"):
                if len(relative.parts) < 3:
                    continue
                relative = Path("bin" if relative.parts[1] == "scripts" else "").joinpath(*relative.parts[2:])
            target = site / relative
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            data = archive.read(member)
            if "bin" == relative.parts[0] and data.startswith((b"#!python\n", b"#!pythonw\n")):
                data = b"#!/usr/bin/python\n" + data.partition(b"\n")[2]
            if target.exists():
                if target.is_file() and target.read_bytes() == data:
                    continue
                raise PackageInstallError("wheel paths conflict: " + str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if relative.parts[0] == "bin" or member.external_attr >> 16 & 0o111 else 0o644)


def create_entry_points(site: Path) -> None:
    for metadata in sorted(site.glob("*.dist-info/entry_points.txt")):
        config = configparser.ConfigParser(interpolation=None)
        config.optionxform = str
        config.read_string(metadata.read_text())
        if not config.has_section("console_scripts"):
            continue
        for name, value in config.items("console_scripts"):
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", name) is None or name in {"python", "python3", "activate"}:
                raise PackageInstallError("invalid console script name")
            match = re.fullmatch(r"([A-Za-z_][\w.]*):([A-Za-z_][\w.]*)(?:\s*\[.*\])?", value.strip())
            if match is None:
                raise PackageInstallError("unsupported console entry point")
            scripts = site / "bin"
            scripts.mkdir(exist_ok=True)
            target = scripts / name
            if target.exists():
                continue
            module, function = match.groups()
            target.write_text(
                "#!/usr/bin/python\nimport importlib, sys\n"
                f"entry = importlib.import_module({module!r})\n"
                f"for part in {function.split('.')!r}: entry = getattr(entry, part)\n"
                "sys.exit(entry())\n"
            )
            target.chmod(0o755)
