"""Select registry pins from uv.lock without using its host wheel artifacts.

uv exports the lock's dependency, extra and group graph in an isolated project.
The selected pins are resolved again for the fixed WASI interpreter and staged
by the same verified, atomic package path as explicit PyPI specs.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from packaging.markers import InvalidMarker, Marker
from packaging.specifiers import InvalidSpecifier, SpecifierSet

from ._api import Environment
from ._cpython_universe import _name, _run_uv, _Runtime, install
from .pypi import PackageInstallError

_MAX_LOCK_BYTES = 2 * 1024 * 1024
_MAX_EXPORT_BYTES = 128 * 1024
_MAX_PACKAGES = 512
_MAX_SELECTIONS = 32
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*\Z")
_PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)(?:\s+;\s+(.+))?\Z")
_GUEST_MARKERS = {
    "implementation_name": "cpython",
    "implementation_version": "3.13.7",
    "os_name": "posix",
    "platform_machine": "wasm32",
    "platform_python_implementation": "CPython",
    "platform_release": "0.0.0",
    "platform_system": "wasi",
    "platform_version": "0.0.0",
    "python_full_version": "3.13.7",
    "python_version": "3.13",
    "sys_platform": "wasi",
    "extra": "",
}


def _read_lock(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_LOCK_BYTES:
        raise PackageInstallError("uv.lock is missing, linked, or exceeds 2 MiB")
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError as error:
            raise PackageInstallError("Python 3.9/3.10 hosts need shellsim[pypi] for TOML parsing") from error
    try:
        with path.open("rb") as stream:
            data = stream.read(_MAX_LOCK_BYTES + 1)
        if len(data) > _MAX_LOCK_BYTES:
            raise PackageInstallError("uv.lock exceeds 2 MiB")
        lock = tomllib.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise PackageInstallError("uv.lock is not valid TOML") from error
    if not isinstance(lock, dict) or lock.get("version") != 1:
        raise PackageInstallError("unsupported uv.lock format version")
    return lock, data


def _selection(values: Sequence[str], *, available: dict[str, Any], label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence) or len(values) > _MAX_SELECTIONS:
        raise PackageInstallError(f"{label} must be a bounded sequence of names")
    selected = set()
    for value in values:
        if not isinstance(value, str) or len(value) > 128 or _LABEL.fullmatch(value) is None:
            raise PackageInstallError(f"invalid {label} name")
        if value not in available:
            raise PackageInstallError(f"uv.lock has no {label} named {value}")
        selected.add(value)
    return tuple(sorted(selected))


def _source_packages(lock: dict[str, Any], *, project_mounted: bool) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages or len(packages) > _MAX_PACKAGES:
        raise PackageInstallError("uv.lock has no packages or exceeds 512 packages")
    roots = []
    versions = set()
    for package in packages:
        if not isinstance(package, dict):
            raise PackageInstallError("invalid uv.lock package")
        name = _name(package.get("name"))
        version = package.get("version")
        source = package.get("source")
        if not isinstance(version, str) or len(version) > 128 or _VERSION.fullmatch(version) is None:
            raise PackageInstallError("invalid uv.lock package version")
        if not isinstance(source, dict) or len(source) != 1:
            raise PackageInstallError(f"{name}: uv.lock package has an invalid source")
        if "registry" in source and isinstance(source["registry"], str) and 0 < len(source["registry"]) <= 2048:
            versions.add((name, version))
        elif source in ({"virtual": "."}, {"editable": "."}):
            roots.append(package)
        else:
            raise PackageInstallError(f"{name}: VCS, URL, local, and editable dependencies cannot be installed in WASI")
    if len(roots) != 1:
        raise PackageInstallError("uv.lock must identify exactly one root project")
    if not project_mounted:
        raise PackageInstallError("uv.lock root source must be separately mounted; pass project_mounted=True")
    return roots[0], versions


def _check_guest_target(lock: dict[str, Any], version: str) -> str:
    python = lock.get("requires-python")
    if not isinstance(python, str) or not python or len(python) > 256 or any(ord(char) < 32 for char in python):
        raise PackageInstallError("uv.lock has an invalid Python requirement")
    try:
        if not SpecifierSet(python).contains(version):
            raise PackageInstallError(f"uv.lock requires Python {python}, but the WASI guest is {version}")
    except InvalidSpecifier as error:
        raise PackageInstallError("uv.lock has an invalid Python requirement") from error
    supported = lock.get("supported-markers")
    if supported is not None:
        if not isinstance(supported, list) or len(supported) > 128:
            raise PackageInstallError("uv.lock has invalid supported environment markers")
        matches = False
        for marker in supported:
            if not isinstance(marker, str) or not 0 < len(marker) <= 2048:
                raise PackageInstallError("uv.lock has invalid supported environment markers")
            try:
                matches |= Marker(marker).evaluate(_GUEST_MARKERS)
            except InvalidMarker as error:
                raise PackageInstallError("uv.lock has invalid supported environment markers") from error
        if not matches:
            raise PackageInstallError("uv.lock supported environments exclude the WASI guest")
    return python


def _write_project(directory: Path, python: str, root: dict[str, Any]) -> None:
    extras = root.get("optional-dependencies", {})
    groups = root.get("dev-dependencies", {})
    if (
        not isinstance(extras, dict)
        or not isinstance(groups, dict)
        or len(extras) > _MAX_SELECTIONS
        or len(groups) > _MAX_SELECTIONS
        or any(not isinstance(name, str) or _LABEL.fullmatch(name) is None for name in (*extras, *groups))
    ):
        raise PackageInstallError("uv.lock has invalid extra or dependency group metadata")
    lines = [
        "[project]",
        f"name = {json.dumps(root['name'])}",
        f"version = {json.dumps(root['version'])}",
        f"requires-python = {json.dumps(python)}",
        "dependencies = []",
    ]
    if extras:
        lines.extend(["", "[project.optional-dependencies]"])
        lines.extend(f"{json.dumps(name)} = []" for name in sorted(extras))
    if groups:
        lines.extend(["", "[dependency-groups]"])
        lines.extend(f"{json.dumps(name)} = []" for name in sorted(groups))
    (directory / "pyproject.toml").write_text("\n".join(lines) + "\n")


def _exported(path: Path, locked_versions: set[tuple[str, str]]) -> tuple[str, ...]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_EXPORT_BYTES:
        raise PackageInstallError("uv exported an oversized or invalid requirements file")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError as error:
        raise PackageInstallError("uv exported non-UTF-8 requirements") from error
    if len(lines) > _MAX_PACKAGES:
        raise PackageInstallError("uv exported too many package pins")
    for line in lines:
        match = _PIN.fullmatch(line)
        if match is None or len(line) > 2048 or "\\" in line or "#" in line:
            raise PackageInstallError("uv.lock export contains a non-registry or unpinned requirement")
        if (_name(match[1]), match[2]) not in locked_versions:
            raise PackageInstallError("uv.lock export contains a version absent from its package table")
    return tuple(lines)


def install_lock(
    runtime: _Runtime,
    environment: Environment,
    path: Path,
    *,
    extras: Sequence[str],
    groups: Sequence[str],
    project_mounted: bool,
) -> None:
    """Export selected registry pins, re-resolve on WASI, then stage atomically."""
    if runtime.uv is None:
        raise PackageInstallError("uv.lock installation requires the patched uv executable")
    lock, lock_bytes = _read_lock(path)
    python = _check_guest_target(lock, runtime.version)
    root, locked_versions = _source_packages(lock, project_mounted=project_mounted)
    available_extras = root.get("optional-dependencies", {})
    available_groups = root.get("dev-dependencies", {})
    if not isinstance(available_extras, dict) or not isinstance(available_groups, dict):
        raise PackageInstallError("uv.lock has invalid extra or dependency group metadata")
    extras = _selection(extras, available=available_extras, label="extra")
    groups = _selection(groups, available=available_groups, label="group")
    with tempfile.TemporaryDirectory(prefix="shellsim-uv-lock-") as temporary:
        work = Path(temporary)
        (work / "uv.lock").write_bytes(lock_bytes)
        _write_project(work, python, root)
        output = work / "requirements.txt"
        args = [
            "export",
            "--frozen",
            "--offline",
            "--no-build",
            "--no-default-groups",
            "--no-hashes",
            "--no-header",
            "--no-annotate",
            "--no-emit-project",
            "--output-file",
            str(output),
        ]
        for extra in extras:
            args.extend(("--extra", extra))
        for group in groups:
            args.extend(("--group", group))
        _run_uv(runtime.uv, args, cwd=work)
        requirements = _exported(output, locked_versions)
        if requirements:
            install(runtime, environment, requirements, locked_versions=locked_versions)
