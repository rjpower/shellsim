"""Resolve a curated native tool/library catalog and atomically mount verified exports.

The host supplies an explicit local catalog. Selection never builds missing releases
or searches ambient host libraries. Native programs execute only inside the guest.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from ._api import Environment, _NativeInstallation

_MAX_JSON = 1024 * 1024
_MAX_PACKAGES = 256
_MAX_NAMES = 64
_MAX_SEARCH = 2048
_MAX_FILES = 10_000
# Leave room for a compiler, its SDK and scientific dependencies in one install.
_MAX_BYTES = 1024**3
_MAX_RELEASE_BYTES = 2 * 1024**3
_KINDS = {"devel", "runtime", "build-tool"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_INSTALL_ROOTS = ("/usr/bin/", "/usr/lib/", "/usr/local/", "/usr/share/", "/opt/", "/lib/", "/tcc/", "/wasi-sysroot/")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON:
        raise ValueError("native package metadata is missing, linked or too large")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("native package metadata must be an object")
    return value


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value or len(value) > 4096:
        raise ValueError("invalid native artifact path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts) or path.as_posix() != value:
        raise ValueError("native artifact path must be canonical and relative")
    return path


def _destination(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith(_INSTALL_ROOTS):
        raise ValueError("native export is outside an installation prefix")
    relative = _relative(value[1:])
    return PurePosixPath("/") / relative


def _requirement(value: str) -> Requirement:
    if not isinstance(value, str) or len(value) > 4096:
        raise ValueError("invalid native package requirement")
    requirement = Requirement(value)
    if requirement.url or requirement.extras or requirement.marker:
        raise ValueError("native requirements do not support URLs, extras or Python markers")
    return requirement


@dataclass(frozen=True)
class _Dependency:
    requirement: Requirement
    kind: str
    linked: bool = True


@dataclass(frozen=True)
class _Package:
    name: str
    version: Version
    kind: str
    artifact: PurePosixPath
    digest: str
    destinations: dict[str, PurePosixPath]
    dependencies: tuple[_Dependency, ...]
    recipe_name: str
    recipe_port: str
    profile: str
    toolchain: str
    abi: str | None
    target: str
    directories: dict[str, PurePosixPath]


def _acyclic(selected: dict[str, _Package]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> bool:
        if name in visiting:
            return False
        if name in visited:
            return True
        visiting.add(name)
        for dependency in selected[name].dependencies:
            if not visit(canonicalize_name(dependency.requirement.name)):
                return False
        visiting.remove(name)
        visited.add(name)
        return True

    return all(visit(name) for name in selected)


def _resolve(
    packages: dict[str, tuple[_Package, ...]],
    specs: Sequence[str],
    valid_closure: Callable[[dict[str, _Package]], bool],
) -> dict[str, _Package]:
    if len(specs) > _MAX_NAMES:
        raise ValueError("too many native package requests")
    constraints: dict[str, list[tuple[Requirement, str | None]]] = {}
    for value in specs:
        requirement = _requirement(value)
        constraints.setdefault(canonicalize_name(requirement.name), []).append((requirement, None))
    attempts = 0

    def search(
        selected: dict[str, _Package], needed: dict[str, list[tuple[Requirement, str | None]]]
    ) -> dict[str, _Package] | None:
        nonlocal attempts
        attempts += 1
        if attempts > _MAX_SEARCH or len(needed) > _MAX_NAMES:
            raise ValueError("native dependency resolution exceeds its bounded search")
        domains = {}
        for name, requirements in needed.items():
            candidates = tuple(
                package
                for package in packages.get(name, ())
                if all(
                    package.version in requirement.specifier and (kind is None or package.kind == kind)
                    for requirement, kind in requirements
                )
            )
            if name in selected:
                if selected[name] not in candidates:
                    return None
            elif not candidates:
                return None
            else:
                domains[name] = candidates
        if not domains:
            return selected if _acyclic(selected) and valid_closure(selected) else None
        name = min(domains, key=lambda name: (len(domains[name]), name))
        for package in domains[name]:
            next_selected = {**selected, name: package}
            next_needed = {key: list(value) for key, value in needed.items()}
            for dependency in package.dependencies:
                dependency_name = canonicalize_name(dependency.requirement.name)
                next_needed.setdefault(dependency_name, []).append((dependency.requirement, dependency.kind))
            result = search(next_selected, next_needed)
            if result is not None:
                return result
        return None

    selected = search({}, constraints)
    if selected is None:
        raise ValueError("no compatible native package closure exists in the catalog")
    return selected


def _metadata(root: Path, package: _Package, target: str) -> tuple[Path, dict[str, Any]]:
    target = package.target
    prefix = root
    for part in package.artifact.parts:
        prefix /= part
        if prefix.is_symlink():
            raise ValueError("native artifact path contains a link")
    manifest = _json(prefix / "artifact.json")
    unsigned = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if manifest.get("artifact_sha256") != package.digest or _digest(unsigned) != package.digest:
        raise ValueError("native artifact manifest identity differs from the catalog")
    recipe = manifest["inputs"]["recipe"]
    if manifest["inputs"].get("recipe_sha256") != _digest(recipe):
        raise ValueError("native recipe identity is inconsistent")
    if "source" in recipe and manifest["inputs"].get("source_sha256") != recipe["source"]["sha256"]:
        raise ValueError("native source identity is inconsistent")
    toolchain = manifest["inputs"]["toolchain"]
    profile = toolchain.get("profile")
    legacy_profile = (
        isinstance(profile, dict) and profile.get("name") == package.profile and profile.get("target") == target
    )
    graph_profile = (
        isinstance(toolchain.get("cohort"), str)
        and _HASH.fullmatch(toolchain["cohort"]) is not None
        and toolchain.get("target") == target
        and toolchain.get("abi") == package.abi == package.profile
    )
    if not legacy_profile and not graph_profile:
        raise ValueError("native toolchain profile differs from its recipe")
    if "sdk" in recipe and toolchain.get("sdk") != recipe["sdk"]:
        raise ValueError("native SDK identity differs from its recipe")
    if (
        recipe["target"] != target
        or Version(recipe["version"]) != package.version
        or recipe["name"] != package.recipe_name
        or recipe["target_profile"] != package.profile
        or _digest(manifest["inputs"]["toolchain"]) != package.toolchain
        or recipe.get("abi") != package.abi
    ):
        raise ValueError("native artifact source name, version, cohort or toolchain differs from the catalog")
    return prefix, manifest


def _artifact(root: Path, package: _Package, target: str) -> tuple[Path, dict[str, Any]]:
    prefix, manifest = _metadata(root, package, target)
    recipe = manifest["inputs"]["recipe"]
    files = manifest["files"]
    exports = recipe["exports"]
    directories = manifest.get("directories", {})
    if (
        not isinstance(directories, dict)
        or len(directories) > _MAX_FILES
        or directories != recipe.get("empty_directories", {})
        or set(directories) != set(package.directories)
    ):
        raise ValueError("native artifact directory exports differ from catalog or recipe")
    for name, mode in directories.items():
        path = prefix.joinpath(*_relative(name).parts)
        if mode != 0o755 or path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise ValueError("native artifact empty directory is missing or invalid")
    if not isinstance(files, dict) or not isinstance(exports, dict):
        raise ValueError("invalid native artifact exports")
    if any(
        not isinstance(group, list) or any(not isinstance(name, str) for name in group) for group in exports.values()
    ):
        raise ValueError("native export groups must be lists of paths")
    exported = [name for group in exports.values() for name in group]
    if len(exported) > _MAX_FILES or len(set(exported)) != len(exported) or set(exported) != set(files):
        raise ValueError("native artifact exports differ from its file manifest")
    if set(package.destinations) != set(files):
        raise ValueError("catalog must stage every verified native export")
    seen = set()
    total = 0
    visited_paths = 0
    pending = [prefix]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            visited_paths += 1
            if visited_paths > _MAX_FILES + 1:
                raise ValueError("native artifact contains too many paths")
            if path.is_symlink():
                raise ValueError("native artifact contains a link")
            if path.is_dir():
                pending.append(path)
                if len(pending) + len(seen) > _MAX_FILES:
                    raise ValueError("native artifact contains too many paths")
                continue
            name = path.relative_to(prefix).as_posix()
            if name == "artifact.json":
                continue
            if (
                not path.is_file()
                or name not in files
                or not isinstance(files[name], str)
                or _HASH.fullmatch(files[name]) is None
            ):
                raise ValueError("native artifact contains undeclared or invalid files")
            _relative(name)
            total += path.stat().st_size
            if total > _MAX_BYTES or len(seen) >= _MAX_FILES:
                raise ValueError("native artifact exceeds bounded file limits")
            if hashlib.sha256(path.read_bytes()).hexdigest() != files[name]:
                raise ValueError("native artifact file differs from its manifest")
            seen.add(name)
    if seen != set(files):
        raise ValueError("native artifact has missing exports")
    return prefix, manifest


def _coherent(
    root: Path,
    target: str,
    selected: dict[str, _Package],
    metadata: dict[tuple[str, Version], dict[str, Any]],
    installed: dict[str, str],
) -> bool:
    if any(name in installed and installed[name] != package.digest for name, package in selected.items()):
        return False
    for package in selected.values():
        key = package.name, package.version
        if key not in metadata:
            metadata[key] = _metadata(root, package, target)[1]
        manifest = metadata[key]
        for linked, field, identities in (
            (True, "target_dependencies", "dependency_artifacts"),
            (False, "runtime_dependencies", "runtime_artifacts"),
        ):
            declared = manifest["inputs"]["recipe"].get(field, [])
            pinned = manifest["inputs"].get(identities, {})
            edges = [edge for edge in package.dependencies if edge.linked == linked]
            dependencies = {
                selected[canonicalize_name(edge.requirement.name)].recipe_port: selected[
                    canonicalize_name(edge.requirement.name)
                ]
                for edge in edges
            }
            if len(dependencies) != len(edges):
                raise ValueError("native catalog has duplicate provider identities")
            ports = [dependency["port"] for dependency in declared]
            if len(set(ports)) != len(ports) or set(ports) != set(dependencies) or set(pinned) != set(ports):
                raise ValueError("native catalog omits or adds a verified dependency edge")
            for dependency in declared:
                provider = dependencies[dependency["port"]]
                if (
                    provider.digest != pinned[dependency["port"]]
                    or provider.version != Version(dependency["version"])
                    or (linked and provider.target != package.target)
                    or ("target_profile" in dependency and provider.profile != dependency["target_profile"])
                ):
                    return False
    return True


def verify_release_catalog(catalog_path: Path) -> set[str]:
    """Validate every listed native release and return its exact regular files.

    Each candidate must have a satisfiable, correctly pinned dependency closure;
    different versions are checked separately and need not coexist in one guest.
    """
    catalog_path = Path(catalog_path)
    root = catalog_path.parent
    if catalog_path.name != "catalog.json" or root.is_symlink() or catalog_path.is_symlink():
        raise ValueError("native release needs a regular catalog.json")
    seen: set[str] = set()
    seen_directories: set[str] = set()
    visited = total = 0
    for path in root.rglob("*"):
        visited += 1
        if visited > _MAX_FILES * 2 or path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("native release contains too many, linked or special paths")
        if path.is_file():
            total += path.stat().st_size
            if total > _MAX_RELEASE_BYTES or len(seen) >= _MAX_FILES:
                raise ValueError("native release exceeds bounded file limits")
            seen.add(path.relative_to(root).as_posix())
        else:
            seen_directories.add(path.relative_to(root).as_posix())
    universe = _NativePackageUniverse(catalog_path)
    expected = {"catalog.json"}
    expected_directories = set()
    metadata: dict[tuple[str, Version], dict[str, Any]] = {}
    for candidates in universe._packages.values():
        for package in candidates:
            prefix, manifest = _artifact(universe.root, package, universe.target)
            base = prefix.relative_to(universe.root)
            expected_directories.update((base / name).as_posix() for name in manifest.get("directories", {}))
            expected.add((base / "artifact.json").as_posix())
            expected.update((base / name).as_posix() for name in manifest["files"])
            _resolve(
                universe._packages,
                [f"{package.name}=={package.version}"],
                lambda selected, expected=package: selected.get(expected.name) is expected
                and _coherent(universe.root, universe.target, selected, metadata, {}),
            )
    expected_directories.update(
        str(parent)
        for name in expected | expected_directories
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    )
    if seen != expected or seen_directories != expected_directories:
        raise ValueError("native release has missing or undeclared files")
    return expected


class _NativePackageUniverse:
    """Install compatible versions from an explicit, bounded local native catalog.

    Specs use ordinary package names and version ranges, such as
    ``["make>=4.4,<5", "zlib-devel==1.3.1"]``. Dependency kinds distinguish
    development files, runtime libraries and guest build tools. Catalog releases
    must already contain verified artifacts; unavailable versions raise an error.
    """

    def __init__(self, catalog_path: str | Path):
        self.catalog_path = Path(catalog_path).resolve()
        self.root = self.catalog_path.parent
        catalog = _json(self.catalog_path)
        if catalog.get("format") != 1 or catalog.get("target") not in {"wasm32-wasip1", "wasm32-wasip1-threads"}:
            raise ValueError("unsupported native catalog format or target")
        self.target = catalog["target"]
        records = catalog.get("packages")
        if not isinstance(records, list) or len(records) > _MAX_PACKAGES:
            raise ValueError("invalid or oversized native catalog")
        packages: dict[str, list[_Package]] = {}
        for record in records:
            requirement = _requirement(record["name"])
            if requirement.specifier:
                raise ValueError("native catalog names must be bare package names")
            name = canonicalize_name(requirement.name)
            version = Version(record["version"])
            kind = record["kind"]
            digest = record["artifact_sha256"]
            if kind not in _KINDS or not isinstance(digest, str) or _HASH.fullmatch(digest) is None:
                raise ValueError("invalid native catalog kind or identity")
            destinations = {
                _relative(source).as_posix(): _destination(destination)
                for source, destination in record["destinations"].items()
            }
            if len(destinations) > _MAX_FILES:
                raise ValueError("too many native catalog exports")
            dependencies = tuple(
                _Dependency(_requirement(dependency["requirement"]), dependency["kind"], dependency.get("linked", True))
                for dependency in record.get("dependencies", ())
            )
            if len(dependencies) > _MAX_NAMES or any(
                dependency.kind not in _KINDS or not isinstance(dependency.linked, bool) for dependency in dependencies
            ):
                raise ValueError("invalid native catalog dependency kind")
            recipe_name = record["recipe_name"]
            recipe_port = record.get("recipe_port", recipe_name)
            profile = record["target_profile"]
            toolchain = record["toolchain_sha256"]
            if (
                not all(
                    isinstance(value, str) and 0 < len(value) <= 4096 for value in (recipe_name, recipe_port, profile)
                )
                or not isinstance(toolchain, str)
                or _HASH.fullmatch(toolchain) is None
            ):
                raise ValueError("invalid native recipe name, profile or toolchain identity")
            directory_exports = record.get("directories", {})
            if not isinstance(directory_exports, dict) or len(directory_exports) > _MAX_FILES:
                raise ValueError("invalid native catalog directory exports")
            package = _Package(
                name,
                version,
                kind,
                _relative(record["artifact"]),
                digest,
                destinations,
                dependencies,
                recipe_name,
                recipe_port,
                profile,
                toolchain,
                record.get("abi"),
                record.get("target", self.target),
                {
                    _relative(source).as_posix(): _destination(destination)
                    for source, destination in directory_exports.items()
                },
            )
            if package.target not in {"wasm32-wasip1", "wasm32-wasip1-threads"}:
                raise ValueError("unsupported native artifact target")
            candidates = packages.setdefault(name, [])
            if any(existing.version == version or existing.kind != kind for existing in candidates):
                raise ValueError("duplicate native release or ambiguous package kind")
            candidates.append(package)
        if len(packages) > _MAX_NAMES:
            raise ValueError("too many native catalog names")
        self._packages = {
            name: tuple(sorted(values, key=lambda package: package.version, reverse=True))
            for name, values in packages.items()
        }

    @classmethod
    def from_release(
        cls, descriptor: str | Path, *, cache_dir: str | Path | None = None, offline: bool = False
    ) -> _NativePackageUniverse:
        """Load a sealed native catalog without selecting a Python runtime or host resolver."""
        from ._native_release import materialize_native

        return cls(
            materialize_native(
                Path(descriptor), cache_dir=Path(cache_dir) if cache_dir is not None else None, offline=offline
            )
        )

    def install(self, environment: Environment, specs: str | Sequence[str]) -> dict[str, str]:
        """Verify and stage the complete dependency closure before one atomic VFS mount.

        Catalog destinations explicitly own installation paths, including replacement
        of built-in command placeholders. Guest working directories are never exports.
        Return the selected package versions, including transitive dependencies.
        """
        with environment._native_install_lock:
            return self._install_locked(environment, specs)

    def _install_locked(self, environment: Environment, specs: str | Sequence[str]) -> dict[str, str]:
        installed = environment._native_installation
        metadata: dict[tuple[str, Version], dict[str, Any]] = {}
        selected = _resolve(
            self._packages,
            [specs] if isinstance(specs, str) else specs,
            lambda choices: _coherent(self.root, self.target, choices, metadata, installed.artifacts),
        )
        identities = {name: package.digest for name, package in selected.items()}
        if any(
            name in installed.artifacts and installed.artifacts[name] != identity
            for name, identity in identities.items()
        ):
            raise ValueError("native package conflicts with an immutable installed artifact")
        if len(installed.artifacts.keys() | identities.keys()) > _MAX_NAMES:
            raise ValueError("cumulative native installation exceeds its package bound")
        files: dict[PurePosixPath, tuple[Path, str, int]] = {}
        directories: dict[str, int] = {}
        total = 0
        for name in sorted(selected):
            package = selected[name]
            prefix, manifest = _artifact(self.root, package, self.target)
            for source, destination in package.directories.items():
                directories[str(destination)] = manifest["directories"][source]
            for source, destination in package.destinations.items():
                payload = prefix / source
                digest = manifest["files"][source]
                mode = 0o755 if source in manifest["inputs"]["recipe"]["exports"].get("tools", ()) else 0o644
                if destination in files:
                    if files[destination][1:] != (digest, mode):
                        raise ValueError("native packages have conflicting destination identities")
                    continue
                total += payload.stat().st_size
                if total > _MAX_BYTES or len(files) >= _MAX_FILES:
                    raise ValueError("native package closure exceeds bounded VFS import limits")
                files[destination] = (payload, digest, mode)
        cumulative = dict(installed.files)
        cumulative_directories = installed.directories | directories
        if len(cumulative_directories) > _MAX_FILES:
            raise ValueError("cumulative native directory installation exceeds its bound")
        for destination, (payload, identity, mode) in files.items():
            if str(destination) in cumulative and cumulative[str(destination)][:2] != (identity, mode):
                raise ValueError("native export conflicts with an immutable installed destination")
            cumulative[str(destination)] = (identity, mode, payload.stat().st_size)
        if len(cumulative) > _MAX_FILES or sum(item[2] for item in cumulative.values()) > _MAX_BYTES:
            raise ValueError("cumulative native installation exceeds its export bound")
        for directory in cumulative_directories:
            path = PurePosixPath(directory)
            if any(str(item) in cumulative for item in (path, *path.parents)):
                raise ValueError("native directory conflicts with an installed file")
        with tempfile.TemporaryDirectory(prefix="shellsim-native-packages-") as temporary:
            staging = Path(temporary)
            for destination, mode in sorted(directories.items()):
                target = staging / PurePosixPath(destination).relative_to("/")
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(mode)
            for destination, (source, digest, mode) in sorted(files.items()):
                if str(destination) in installed.files:
                    if hashlib.sha256(environment.read_file(str(destination))).hexdigest() != digest:
                        raise ValueError("installed native export bytes changed")
                    continue
                target = staging / destination.relative_to("/")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                target.chmod(mode)
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError("native export changed while staging")
            environment._native.mount_package_tree(str(staging))
        environment._native_installation = _NativeInstallation(
            installed.artifacts | identities, cumulative, cumulative_directories
        )
        return {name: str(package.version) for name, package in sorted(selected.items())}
