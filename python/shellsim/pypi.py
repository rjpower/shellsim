"""Resolve trusted PyPI requirements on the host and stage compatible files in the VFS."""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Protocol, Sequence

from . import _native

_VERSION_CLAUSE = r"(?:==|!=|<=|>=|<|>|~=)\s*[A-Za-z0-9.*+!_-]+"
_REQUIREMENT = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9._,-]+\])?"
    rf"(?:\s*{_VERSION_CLAUSE}(?:\s*,\s*{_VERSION_CLAUSE})*)?"
)
_NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib", ".a", ".wasm")
_BUNDLED_DISTRIBUTIONS = {"numpy": "2.5.3"}
_SITE_PACKAGES = "/usr/lib/python3.14/site-packages"


class PackageInstallError(RuntimeError):
    """A requested distribution cannot be resolved or loaded by shellsim Python."""


class _PackageMounter(Protocol):
    def mount_package_tree(
        self, host_root: str, destination_root: str, replace_builtin_tools: Sequence[str]
    ) -> None: ...


class _StagingEnvironment(Protocol):
    _native: _PackageMounter


def _mount_package_tree(
    environment: _StagingEnvironment,
    target: Path,
    destination: str,
    *,
    normalize_modes: bool = False,
    replace_builtin_tools: Sequence[str] = (),
) -> None:
    """Apply one package transaction with deterministic Python payload permissions.

    Native catalog exports retain their explicit modes. Python data is readable and
    executable entry points remain executable, regardless of the host umask.
    """
    if normalize_modes:
        for path in target.rglob("*"):
            if path.is_symlink():
                raise PackageInstallError("staged package set contains links")
            mode = 0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644
            path.chmod(mode)
    try:
        environment._native.mount_package_tree(str(target), destination, list(replace_builtin_tools))
    except _native.PackageConflictError as error:
        raise PackageInstallError(str(error)) from error


def install_pypi(environment: _StagingEnvironment, requirement: str) -> None:
    """Install a trusted PyPI requirement and its pure Python dependencies into site-packages.

    The host's ``uv`` performs resolution and any source builds. The guest receives only copied
    files after every resolved distribution has passed compatibility checks. Source builds may
    execute package build code in the host process environment.
    """
    if not isinstance(requirement, str):
        raise TypeError("requirement must be str")
    if _REQUIREMENT.fullmatch(requirement) is None:
        raise ValueError("requirement must name a PyPI distribution with optional extras or version")

    with tempfile.TemporaryDirectory(prefix="shellsim-pypi-") as temporary:
        target = Path(temporary) / "site-packages"
        command = [
            "uv",
            "pip",
            "install",
            "--target",
            str(target),
            "--python",
            "3.14",
            "--python-version",
            "3.14",
            "--link-mode",
            "copy",
            "--",
            requirement,
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
        except FileNotFoundError as error:
            raise PackageInstallError("uv is required to install PyPI packages on the host") from error
        except subprocess.TimeoutExpired as error:
            raise PackageInstallError(f"timed out resolving {requirement}") from error
        if completed.returncode:
            detail = completed.stderr.strip()[-8000:]
            raise PackageInstallError(f"cannot resolve {requirement}: {detail}")

        bundled_count = _remove_bundled_distributions(target)
        blockers = _compatibility_blockers(target, bundled_count)
        if blockers:
            raise PackageInstallError("shellsim Python cannot stage incompatible distributions: " + "; ".join(blockers))
        _check_import_collisions(target)

        _mount_package_tree(environment, target, _SITE_PACKAGES, normalize_modes=True)


def _check_import_collisions(target: Path) -> None:
    imports = set()
    for entry in target.iterdir():
        if entry.is_dir() and not entry.name.endswith((".dist-info", ".data")):
            imports.add(entry.name)
        elif entry.is_file() and entry.suffix == ".py":
            imports.add(entry.stem)
    collisions = sorted(name for name in imports if _native.is_bundled_python_module(name))
    if collisions:
        raise PackageInstallError("wheel import names conflict with bundled modules: " + ", ".join(collisions))


def _remove_bundled_distributions(target: Path) -> int:
    """Use a bundled distribution only when uv resolved the exact modeled version."""
    omitted = 0
    for distribution in sorted(target.glob("*.dist-info")):
        metadata = distribution / "METADATA"
        if not metadata.is_file() or metadata.is_symlink() or metadata.stat().st_size > 65536:
            continue
        headers = {}
        for line in metadata.read_text().splitlines():
            if not line:
                break
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key.lower()] = value
        name = re.sub(r"[-_.]+", "-", headers.get("name", "").lower())
        if _BUNDLED_DISTRIBUTIONS.get(name) != headers.get("version"):
            continue

        record = distribution / "RECORD"
        if not record.is_file() or record.is_symlink():
            raise PackageInstallError(f"{distribution.name}: cannot omit bundled distribution without RECORD")
        if record.stat().st_size > 1024 * 1024:
            raise PackageInstallError(f"{distribution.name}: oversized RECORD")
        paths = []
        target_root = target.resolve()
        with record.open(newline="") as contents:
            for row in csv.reader(contents):
                if not row:
                    continue
                if len(paths) >= 10000:
                    raise PackageInstallError(f"{distribution.name}: too many RECORD files")
                relative = PurePosixPath(row[0])
                if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                    raise PackageInstallError(f"{distribution.name}: invalid RECORD path")
                if relative.parts[0] not in {"numpy", "numpy.libs", distribution.name} and relative.as_posix() not in {
                    "bin/f2py",
                    "bin/numpy-config",
                }:
                    raise PackageInstallError(f"{distribution.name}: RECORD includes another distribution's file")
                path = target.joinpath(*relative.parts)
                if not path.resolve().is_relative_to(target_root) or not path.is_file():
                    raise PackageInstallError(f"{distribution.name}: invalid RECORD file {relative}")
                paths.append(path)
        if record not in paths:
            raise PackageInstallError(f"{distribution.name}: RECORD does not list itself")
        for path in paths:
            path.unlink()
        directories = {
            parent for path in paths for parent in path.parents if parent != target and target in parent.parents
        }
        for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
        if distribution.exists():
            raise PackageInstallError(f"{distribution.name}: unlisted files remain in bundled distribution")
        omitted += 1
    return omitted


def _compatibility_blockers(target: Path, bundled_count: int = 0, *, python_version: str = "3.14") -> list[str]:
    distributions = sorted(target.glob("*.dist-info"))
    if not distributions and not bundled_count:
        return ["uv installed no wheel metadata"]
    blockers = []
    for distribution in distributions:
        wheel = distribution / "WHEEL"
        if wheel.is_symlink() or not wheel.is_file():
            blockers.append(f"{distribution.name}: missing WHEEL metadata")
            continue
        if wheel.stat().st_size > 65536:
            blockers.append(f"{distribution.name}: oversized WHEEL metadata")
            continue
        metadata = wheel.read_text()
        tags = [line[5:].strip() for line in metadata.splitlines() if line.startswith("Tag: ")]
        if "Root-Is-Purelib: true" not in metadata:
            blockers.append(f"{distribution.name}: wheel is not purelib")
        if not any(
            tag.endswith("-none-any")
            and {"py3", "py" + python_version.replace(".", "")}.intersection(tag.split("-")[0].split("."))
            for tag in tags
        ):
            blockers.append(f"{distribution.name}: incompatible wheel tags {', '.join(tags) or '(none)'}")
    paths = []
    for path in target.rglob("*"):
        paths.append(path)
        if len(paths) > 10000:
            blockers.append("distribution contains more than 10,000 paths")
            break
    if any(path.is_symlink() for path in paths):
        blockers.append("distribution contains symbolic links")
    native = sorted(
        path.relative_to(target).as_posix()
        for path in paths
        if path.name.lower().endswith(_NATIVE_SUFFIXES) or ".so." in path.name.lower()
    )
    if native:
        blockers.append("native files: " + ", ".join(native[:8]) + (" ..." if len(native) > 8 else ""))
    return blockers
