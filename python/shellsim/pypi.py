"""Validate Python wheel files and mount them into the guest filesystem."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, Sequence

_VERSION_CLAUSE = r"(?:==|!=|<=|>=|<|>|~=)\s*[A-Za-z0-9.*+!_-]+"
_REQUIREMENT = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9._,-]+\])?"
    rf"(?:\s*{_VERSION_CLAUSE}(?:\s*,\s*{_VERSION_CLAUSE})*)?"
)
_NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib", ".a", ".wasm")


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
    from . import _native

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


def _compatibility_blockers(target: Path, *, python_version: str) -> list[str]:
    distributions = sorted(target.glob("*.dist-info"))
    if not distributions:
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
