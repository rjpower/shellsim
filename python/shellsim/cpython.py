"""Explicitly mount and run a source-built CPython WASI bundle.

Build bundles with ``uv run ports/cpython/build.py``. Host filesystem access is
confined to this trusted setup API; interpreter execution uses the virtual WASI
process boundary and the environment's cumulative resource limits.
"""

from __future__ import annotations

import email
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence, Union

from ._api import Environment, RunResult, SimulationError
from .pypi import _REQUIREMENT, PackageInstallError, _compatibility_blockers

_EXECUTABLE = "/usr/bin/python3.wasm"
_SITE_PACKAGES = "/usr/lib/python3.13/site-packages"
_MAX_BYTES = 128 * 1024 * 1024
_MAX_FILES = 10000


class CPythonRuntime:
    """A verified CPython 3.13.7 bundle, separate from shellsim's Python VM.

    The bundle is a trusted host build directory containing ``manifest.json``
    and ``rootfs``. Keep that directory unchanged while mounting it.
    """

    def __init__(
        self,
        bundle: Union[str, Path],
        *,
        universe: Union[str, Path, None] = None,
        uv: Union[str, Path, None] = None,
        venv: Union[str, Path, None] = None,
    ) -> None:
        self.bundle = Path(bundle)
        self.universe = Path(universe) if universe is not None else None
        self.uv = Path(uv).resolve() if uv is not None else None
        if (self.universe is None) != (self.uv is None):
            raise ValueError("a CPython package universe requires an explicit patched uv executable")
        manifest_path = self.bundle / "manifest.json"
        if manifest_path.stat().st_size > 1024 * 1024:
            raise ValueError("CPython manifest exceeds 1 MiB")
        self.manifest = json.loads(manifest_path.read_text())
        recipe = self.manifest["recipe"]
        if recipe["version"] != "3.13.7" or recipe["target"] != "wasm32-wasip1" or recipe["prefix"] != "/usr":
            raise ValueError("unsupported CPython bundle version, target, or prefix")
        if self.manifest["site_packages"] != _SITE_PACKAGES:
            raise ValueError("unsupported CPython site-packages path")
        self.version = recipe["version"]
        if self.manifest.get("dynamic_abi"):
            raw_venv = os.fspath(venv) if venv is not None else "/work/.venv"
            if (
                not isinstance(raw_venv, str)
                or not raw_venv.startswith("/")
                or raw_venv in {"/", "/usr", "/bin"}
                or raw_venv.startswith(("/usr/", "/bin/", "/dev/", "/proc/"))
                or "\0" in raw_venv
                or any(part in {".", "..", ""} for part in raw_venv[1:].split("/"))
            ):
                raise ValueError("CPython venv must have a separate absolute VFS path")
            self.venv = raw_venv
            self.site_packages = self.venv + "/lib/python3.13/site-packages"
        else:
            if venv is not None:
                raise ValueError("a task venv requires a dynamic CPython bundle")
            self.venv = None
            self.site_packages = _SITE_PACKAGES
        self.builtin_modules = tuple(self.manifest.get("builtin_modules", ()))
        self._verify()

    def _verify(self) -> None:
        root = self.bundle / "rootfs"
        if root.is_symlink() or not root.is_dir():
            raise ValueError("CPython rootfs must be a real directory")
        files = self.manifest["files"]
        if not isinstance(files, dict) or len(files) > _MAX_FILES:
            raise ValueError("invalid CPython file manifest")
        seen = set()
        size = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("CPython bundle contains symbolic links")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("CPython bundle contains a special file")
            name = "/" + path.relative_to(root).as_posix()
            size += path.stat().st_size
            if size > _MAX_BYTES or len(seen) >= _MAX_FILES:
                raise ValueError("CPython bundle exceeds its size limit")
            if name not in files or hashlib.sha256(path.read_bytes()).hexdigest() != files[name]:
                raise ValueError(f"CPython bundle integrity failure: {name}")
            seen.add(name)
        if seen != set(files):
            raise ValueError("CPython bundle has missing files")
        binary = root / _EXECUTABLE.lstrip("/")
        with binary.open("rb") as source:
            if source.read(8) != b"\0asm\x01\0\0\0":
                raise ValueError("CPython executable is not a core Wasm module")
        if not binary.stat().st_mode & 0o111:
            raise ValueError("CPython executable has no execute permission")

    def mount(self, environment: Environment) -> None:
        """Mount the verified bundle and select its VFS Python launchers."""
        self._verify()
        environment._native.mount_cpython(str(self.bundle / "rootfs"), self.venv)

    def run(self, environment: Environment, argv: Sequence[str], *, stdin: bytes = b"") -> RunResult:
        """Run interpreter arguments such as ``('-c', 'print(42)')`` in the VFS."""
        if isinstance(argv, (str, bytes)) or any(not isinstance(arg, str) for arg in argv):
            raise TypeError("argv must be a sequence of str")
        executable = self.venv + "/bin/python" if self.venv is not None else _EXECUTABLE
        command = " ".join(shlex.quote(arg) for arg in [executable, *argv])
        prefix = "PYTHONDONTWRITEBYTECODE=1 "
        if self.venv is None:
            prefix += "PYTHONHOME=/usr "
        return environment.run(prefix + command, stdin)

    def _stage(self, environment: Environment, target: Path) -> None:
        """Reject file conflicts before atomic import; the environment must be idle."""
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            destination = self.site_packages + "/" + path.relative_to(target).as_posix()
            try:
                existing = environment.read_file(destination)
            except SimulationError as error:
                if "No such file" not in str(error):
                    raise
            else:
                if existing != path.read_bytes():
                    raise PackageInstallError(f"package file would overwrite an existing VFS file: {destination}")
        environment.mount(target, self.site_packages)

    def install_pypi(self, environment: Environment, requirement: str) -> None:
        """Resolve packages for the bundle's Python and WASI ABI.

        A dynamic bundle requires an explicit local universe and patched uv.
        Static bundles retain their existing pure-wheel installation path.
        """
        if not isinstance(requirement, str):
            raise TypeError("requirement must be str")
        if _REQUIREMENT.fullmatch(requirement) is None:
            raise ValueError("requirement must name a PyPI distribution with optional extras or version")
        self._verify()
        if self.manifest.get("dynamic_abi"):
            if self.universe is None or self.uv is None:
                raise PackageInstallError("dynamic CPython packages require a local universe and patched uv executable")
            from ._cpython_universe import install

            install(self, environment, requirement)
            return
        if self.universe is not None:
            raise PackageInstallError("a package universe requires a dynamic CPython bundle")
        with tempfile.TemporaryDirectory(prefix="shellsim-cpython-pypi-") as temp:
            target = Path(temp) / "site-packages"
            target.mkdir()
            constraints = []
            for port in self.manifest.get("native_ports", ()):
                name = port["name"]
                version = port["version"]
                modules = port.get("builtin_modules")
                if (
                    re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
                    or re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", version) is None
                    or not isinstance(modules, list)
                    or not modules
                    or any(
                        not isinstance(module, str)
                        or not all(part.isidentifier() for part in module.split("."))
                        or module not in self.builtin_modules
                        for module in modules
                    )
                ):
                    raise PackageInstallError("invalid builtin native provider manifest")
                directory = f"{name}-{version}.dist-info"
                if port.get("dist_info") != self.site_packages + "/" + directory:
                    raise PackageInstallError("invalid builtin native provider metadata path")
                source = self.bundle / "rootfs" / port["dist_info"].lstrip("/")
                if not source.is_dir() or not (source / "METADATA").is_file():
                    raise PackageInstallError("builtin native provider metadata is missing from the bundle")
                metadata = email.message_from_bytes((source / "METADATA").read_bytes())
                if (
                    metadata["Name"] != name
                    or metadata["Version"] != version
                    or metadata.get_all("Requires-Dist", []) != port["requires_dist"]
                ):
                    raise PackageInstallError("builtin native provider metadata differs from its recipe")
                # Seed the exact verified metadata already mounted with this provider; an
                # installer must not replace its source identity or distribution version.
                shutil.copytree(source, target / directory)
                constraints.append(f"{name}=={version}")
            command = [
                "uv",
                "pip",
                "install",
                "--target",
                str(target),
                "--python",
                "3.13",
                "--python-version",
                "3.13",
                "--link-mode",
                "copy",
            ]
            if constraints:
                constraint_file = Path(temp) / "native-constraints.txt"
                constraint_file.write_text("\n".join(constraints) + "\n")
                command += ["--constraint", str(constraint_file)]
            command += ["--", requirement]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
            except FileNotFoundError as error:
                raise PackageInstallError("uv is required to install PyPI packages on the host") from error
            except subprocess.TimeoutExpired as error:
                raise PackageInstallError(f"timed out resolving {requirement}") from error
            if completed.returncode:
                raise PackageInstallError(f"cannot resolve {requirement}: {completed.stderr.strip()[-8000:]}")
            blockers = _compatibility_blockers(target, python_version="3.13")
            if blockers:
                raise PackageInstallError(
                    "CPython WASI cannot stage incompatible distributions: " + "; ".join(blockers)
                )
            self._stage(environment, target)

    def install_wheel(self, environment: Environment, wheel: Union[str, Path]) -> None:
        """Stage a pure Python wheel without resolving dependencies or running builds.

        Native extensions require a source recipe linked into the interpreter.
        Host native wheels cannot be loaded by a WASI command.
        """
        path = Path(wheel)
        if path.stat().st_size > _MAX_BYTES:
            raise PackageInstallError("wheel exceeds 128 MiB")
        with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory(prefix="shellsim-cpython-wheel-") as temp:
            members = archive.infolist()
            if len(members) > _MAX_FILES or sum(member.file_size for member in members) > _MAX_BYTES:
                raise PackageInstallError("wheel exceeds its file or size limit")
            metadata = [member for member in members if member.filename.endswith(".dist-info/WHEEL")]
            if len(metadata) != 1 or metadata[0].file_size > 65536:
                raise PackageInstallError("wheel must contain one bounded WHEEL metadata file")
            text = archive.read(metadata[0]).decode("utf-8")
            tags = [line[5:].strip() for line in text.splitlines() if line.startswith("Tag: ")]
            if "Root-Is-Purelib: true" not in text or not any(
                tag.endswith("-none-any") and {"py3", "py313"}.intersection(tag.split("-")[0].split("."))
                for tag in tags
            ):
                raise PackageInstallError("CPython WASI requires a pure Python 3.13 wheel or a native source recipe")
            target = Path(temp)
            seen = set()
            for member in members:
                relative = PurePosixPath(member.filename)
                mode = member.external_attr >> 16
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or "\\" in member.filename
                    or stat.S_ISLNK(mode)
                    or relative in seen
                ):
                    raise PackageInstallError("wheel contains an invalid or duplicate path")
                seen.add(relative)
                if any(part.endswith(".data") for part in relative.parts):
                    raise PackageInstallError("wheel data relocation is not supported")
                if (
                    member.filename.lower().endswith((".so", ".pyd", ".dll", ".dylib", ".a", ".wasm"))
                    or ".so." in member.filename.lower()
                ):
                    raise PackageInstallError("native wheel files require a WASI source recipe")
                destination = target.joinpath(*relative.parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
            self._stage(environment, target)
