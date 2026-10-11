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

    def __init__(self, location: str | Path, *, cache_dir: Path | None = None, offline: bool = False):
        self.cache = (
            cache_dir or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "shellsim/ports"
        ).resolve()
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
        if self.data.get("schema_version") != 2 or "runtime" not in self.data:
            raise PackageInstallError("catalog needs delivery schema 2 and a runtime asset")
        self.runtime = self.data["runtime"]
        self.version = self.runtime["python_version"]
        self.abi = self.runtime["abi"]
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
            "python_version": ".".join(self.version.split(".")[:2]),
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
                return path
            if self.offline:
                raise PackageInstallError("selected asset is not cached for offline use: " + path.name)
            from ._catalog_cache import staged_file

            with staged_file(path) as temporary:
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
        archive = self.asset(self.runtime)
        key = _relative(self.runtime["path"]).parts[-2]
        target = self.cache / "runtimes" / key
        target.parent.mkdir(exist_ok=True)
        with target.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (target / "runtime").is_dir():
                return target / "runtime"
            with tempfile.TemporaryDirectory(prefix=".runtime-", dir=target.parent) as temporary:
                stage = Path(temporary) / "unpacked"
                stage.mkdir()
                _extract(archive, stage, allowed_roots=frozenset({"runtime"}))
                stage.rename(target)
        return target / "runtime"

    def resolve(self, requirements: str | Sequence[str]) -> list[dict]:
        if isinstance(requirements, str):
            requirements = (requirements,)
        if not isinstance(requirements, Sequence) or isinstance(requirements, bytes):
            raise TypeError("pypi_packages must be package specs or a sequence of specs")
        if not 1 <= len(requirements) <= 256:
            raise ValueError("pypi_packages needs 1..256 package specs")
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
                unpack_wheel(path, site)
            create_entry_points(site)
            _stage_console_scripts(site, staging, runtime.venv)
            for provider in providers:
                if provider["name"] in environment._catalog_native:
                    continue
                source = self.asset(provider)
                _verify_wasm(source.read_bytes(), self.abi)
                destination = staging / _relative(provider["destination"].removeprefix("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                destination.chmod(0o755 if provider["kind"] == "tools" else 0o644)
            _mount_package_tree(environment, staging, "/", normalize_modes=True)
        environment._catalog_packages.update(package["name"] for package in packages)
        environment._catalog_native.update(provider["name"] for provider in providers)


def unpack_wheel(path: Path, site: Path) -> None:
    """Relocate wheel data and reject conflicting installed files."""
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            relative = _relative(member.filename.rstrip("/"))
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
