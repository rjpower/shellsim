"""Explicitly mount and run a source-built CPython WASI bundle.

Build bundles with ``uv run --no-project python -m ports python/cpython --store PATH``. Host filesystem access is
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
from .pypi import _REQUIREMENT, PackageInstallError, _compatibility_blockers, _mount_package_tree

_EXECUTABLE = "/usr/bin/python3.wasm"
_SITE_PACKAGES = "/usr/lib/python3.13/site-packages"
_MAX_BYTES = 128 * 1024 * 1024
_MAX_FILES = 10000
_MAX_REQUIREMENTS = 256
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_PYTHON_PACKAGE_PLATFORM = "wasm32-wasip1"
_DYNAMIC_RUNTIME_PROFILES = {
    ("wasm32-wasip1", "shellsim-wasi-sdk34-cpython3137-v2"),
    ("wasm32-wasip1-threads", "shellsim-wasi-sdk34-cpython3137-threads-v3"),
}


def _requirements(value: Union[str, Sequence[str]]) -> tuple[str, ...]:
    if isinstance(value, str):
        requirements = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        if len(value) > _MAX_REQUIREMENTS:
            raise ValueError("requirements must contain between 1 and 256 PyPI specs")
        requirements = tuple(value)
    else:
        raise TypeError("requirements must be a PyPI spec or a sequence of specs")
    if not requirements or len(requirements) > _MAX_REQUIREMENTS:
        raise ValueError("requirements must contain between 1 and 256 PyPI specs")
    for requirement in requirements:
        if not isinstance(requirement, str):
            raise TypeError("each requirement must be str")
        if len(requirement) > 2048 or _REQUIREMENT.fullmatch(requirement) is None:
            raise ValueError("each requirement must name a PyPI distribution with optional extras or version")
    return requirements


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
        with manifest_path.open("rb") as source:
            manifest_bytes = source.read(_MAX_MANIFEST_BYTES + 1)
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise ValueError("CPython manifest exceeds 4 MiB")
        self.manifest = json.loads(manifest_bytes)
        recipe = self.manifest["recipe"]
        if recipe["version"] != "3.13.7" or recipe["prefix"] != "/usr":
            raise ValueError("unsupported CPython bundle version, target, or prefix")
        target = recipe["target"]
        abi = self.manifest.get("dynamic_abi")
        if not isinstance(target, str) or (abi is not None and not isinstance(abi, str)):
            raise ValueError("invalid CPython runtime target or dynamic ABI")
        if abi is None:
            if target != _PYTHON_PACKAGE_PLATFORM:
                raise ValueError("unsupported static CPython bundle target")
        elif (target, abi) not in _DYNAMIC_RUNTIME_PROFILES:
            raise ValueError("unsupported CPython runtime target and dynamic ABI")
        if "dynamic_abi" in recipe and recipe["dynamic_abi"] != abi:
            raise ValueError("CPython recipe and runtime dynamic ABI differ")
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
                or len(raw_venv) > 4096
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

    @classmethod
    def from_release(
        cls,
        descriptor: Union[str, Path],
        *,
        cache_dir: Union[str, Path, None] = None,
        venv: Union[str, Path, None] = None,
        offline: bool = False,
    ) -> CPythonRuntime:
        """Load a trusted local release descriptor and verify its cached cohort.

        The descriptor must be supplied by the caller until release assets and a
        matching pin are published with shellsim. Downloads run only on the host.
        """
        from ._cpython_release import materialize

        bundle, universe, uv = materialize(
            Path(descriptor), cache_dir=Path(cache_dir) if cache_dir is not None else None, offline=offline
        )
        return cls(bundle, universe=universe, uv=uv, venv=venv)

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
        if environment._cpython_runtime is not None and environment._cpython_runtime is not self:
            raise SimulationError("a different CPython runtime is already mounted")
        self._verify()
        environment._native.mount_cpython(str(self.bundle / "rootfs"), self.venv)
        environment._cpython_runtime = self

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
        _mount_package_tree(environment, target, self.site_packages, normalize_modes=True)

    def install_pypi(self, environment: Environment, requirement: Union[str, Sequence[str]]) -> None:
        """Resolve one or more specs together for the bundle's Python and WASI ABI.

        A dynamic bundle requires an explicit local universe and patched uv.
        Static bundles retain their existing pure-wheel installation path.
        """
        requirements = _requirements(requirement)
        self._verify()
        if self.manifest.get("dynamic_abi"):
            if self.universe is None or self.uv is None:
                raise PackageInstallError("dynamic CPython packages require a local universe and patched uv executable")
            from ._cpython_universe import install

            install(self, environment, requirements)
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
            command += ["--", *requirements]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
            except FileNotFoundError as error:
                raise PackageInstallError("uv is required to install PyPI packages on the host") from error
            except subprocess.TimeoutExpired as error:
                raise PackageInstallError("timed out resolving PyPI requirements") from error
            if completed.returncode:
                raise PackageInstallError("cannot resolve PyPI requirements: " + completed.stderr.strip()[-8000:])
            blockers = _compatibility_blockers(target, python_version="3.13")
            if blockers:
                raise PackageInstallError(
                    "CPython WASI cannot stage incompatible distributions: " + "; ".join(blockers)
                )
            self._stage(environment, target)

    def install_lock(
        self,
        environment: Environment,
        lock: Union[str, Path],
        *,
        extras: Sequence[str] = (),
        groups: Sequence[str] = (),
        project_mounted: bool = False,
    ) -> None:
        """Install selected ``uv.lock`` dependencies for the WASI guest target.

        Set ``project_mounted`` when the root project source is separately mounted
        into the VFS. Local and VCS dependency packages are unsupported.
        """
        self._verify()
        if not self.manifest.get("dynamic_abi") or self.universe is None or self.uv is None:
            raise PackageInstallError("uv.lock installation requires a dynamic bundle, local universe, and patched uv")
        from ._cpython_lock import install_lock

        install_lock(self, environment, Path(lock), extras=extras, groups=groups, project_mounted=project_mounted)

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
                destination.chmod(0o755 if mode & 0o111 else 0o644)
            self._stage(environment, target)
