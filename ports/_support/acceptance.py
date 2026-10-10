"""Run declared port probes in fresh public Shellsim environments.

The graph runner supplies a sealed release and exact port selection. Each probe
uses a fresh guest, so package installation and execution are both part of the
result. Native probes are compiled only by an admitted target SDK and their
Wasm output runs only inside the guest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from ports._support.graph import Port
from ports._support.wasm import mark_abi
from ports.native.dependencies import target_environment

if TYPE_CHECKING:
    from shellsim import Limits

    from ports._support.sdk_products import MaterializedSDK

_MAX_TESTS = 32
_MAX_SOURCE_BYTES = 16 * 1024**2
_MAX_TEST_LIMITS = {"cpu": 1_000_000_000_000, "memory": 16 * 1024**3, "disk": 2 * 1024**3}


@dataclass(frozen=True)
class AcceptanceRequest:
    """One graph port, sealed public release and private proof directory."""

    port: Port
    descriptor: Path
    output: Path
    install_kind: Literal["pypi", "native", "pypi+native"]
    sdk_context: MaterializedSDK | None = None
    dependency_sysroot: Path | None = None
    limits: Limits | None = None


@dataclass(frozen=True)
class AcceptanceResult:
    """Hash-bound guest outcome and resource result for one declared probe."""

    script: str
    kind: str
    source_sha256: str
    artifact_sha256: str | None
    fixture_sha256: dict[str, str]
    returncode: int
    stop_reason: str | None
    stdout_sha256: str
    stderr_sha256: str
    usage: dict[str, int]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _test_limits(recipe: dict) -> Limits | None:
    """Admit explicit guest budgets for checks that exceed environment defaults."""
    values = recipe.get("test_limits")
    if values is None:
        return None
    if (
        not isinstance(values, dict)
        or not values
        or any(
            name not in _MAX_TEST_LIMITS or type(value) is not int or not 0 < value <= _MAX_TEST_LIMITS[name]
            for name, value in values.items()
        )
    ):
        raise ValueError("test_limits must contain bounded positive cpu, memory or disk integers")
    from shellsim import Limits

    return Limits(**values)


def _bounded_digest(path: Path, limit: int) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            size += len(chunk)
            if size > limit:
                raise ValueError("native acceptance link input exceeds size limit")
            digest.update(chunk)
    return digest.hexdigest()


def _source(port: Port, raw: object) -> tuple[Path, bytes]:
    if not isinstance(raw, str) or not raw or len(raw) > 4096 or "\\" in raw or "\0" in raw:
        raise ValueError("acceptance source must be a relative port path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts) or relative.as_posix() != raw:
        raise ValueError("acceptance source escapes its port")
    path = port.directory.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(port.directory.resolve()):
        raise ValueError("acceptance source is not a regular port file")
    with path.open("rb") as stream:
        data = stream.read(_MAX_SOURCE_BYTES + 1)
    if len(data) > _MAX_SOURCE_BYTES:
        raise ValueError("acceptance source exceeds its size limit")
    return path, data


def _fixtures(port: Port, raw: object) -> dict[str, bytes]:
    """Admit bounded test inputs below /work before materializing a release."""
    if not isinstance(raw, list) or len(raw) > 32:
        raise ValueError("acceptance fixtures must be a bounded list")
    files = {}
    size = 0
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"source", "destination"}:
            raise ValueError("acceptance fixture needs source and destination")
        destination = item["destination"]
        if not isinstance(destination, str) or not destination or len(destination) > 4096:
            raise ValueError("acceptance fixture destination is invalid")
        relative = PurePosixPath(destination)
        if (
            "\\" in destination
            or "\0" in destination
            or relative.is_absolute()
            or any(part in {".", ".."} for part in relative.parts)
            or relative.as_posix() != destination
            or destination.startswith("shellsim-acceptance")
        ):
            raise ValueError("acceptance fixture destination escapes its work directory")
        if destination in files:
            raise ValueError("acceptance fixture destination is duplicated")
        _, data = _source(port, item["source"])
        size += len(data)
        if size > _MAX_SOURCE_BYTES:
            raise ValueError("acceptance fixtures exceed their aggregate size limit")
        files[destination] = data
    return files


def _native_command(
    sdk_context: MaterializedSDK,
    dependency_sysroot: Path,
    source: Path,
    destination: Path,
    link_inputs: list[str],
    sdk_link_inputs: list[str],
    include_directories: list[str],
    link_flags: list[str] | None = None,
) -> tuple[str, ...]:
    if not sdk_context.has_frontend:
        raise ValueError("native acceptance needs the admitted standard Clang frontend")
    if not dependency_sysroot.is_absolute():
        raise ValueError("native acceptance dependency sysroot must be absolute")
    link_flags = [] if link_flags is None else link_flags
    if (
        not isinstance(link_flags, list)
        or len(link_flags) > 32
        or any(
            not isinstance(flag, str)
            or len(flag) > 4096
            or re.fullmatch(r"-Wl,--wrap=[A-Za-z_][A-Za-z0-9_]*(?:,--wrap=[A-Za-z_][A-Za-z0-9_]*)*", flag) is None
            for flag in link_flags
        )
    ):
        raise ValueError("native acceptance link_flags must be bounded linker wrapper switches")
    includes = []
    include_root = dependency_sysroot / "usr/local"
    for raw in include_directories:
        if not isinstance(raw, str) or not raw or len(raw) > 4096 or "\\" in raw or "\0" in raw:
            raise ValueError("native acceptance include directory is invalid")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts) or relative.as_posix() != raw:
            raise ValueError("native acceptance include directory escapes dependency sysroot")
        path = include_root.joinpath(*relative.parts)
        if not path.is_dir() or not path.resolve().is_relative_to(include_root.resolve()):
            raise ValueError("native acceptance include directory is missing or linked outside dependency sysroot")
        includes.append("-I" + str(path))
    links = []
    for raw in link_inputs:
        if not isinstance(raw, str) or not raw or len(raw) > 4096 or "\\" in raw or "\0" in raw:
            raise ValueError("native acceptance link input is invalid")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw:
            raise ValueError("native acceptance link input escapes dependency sysroot")
        path = include_root.joinpath(*relative.parts)
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(include_root.resolve())
            or path.suffix not in {".a", ".so"}
            or path.stat().st_size > 128 * 1024**2
        ):
            raise ValueError("native acceptance link input is missing or unsupported")
        links.append(str(path))
    sdk_context_root = sdk_context.sysroot.root / "sysroot"
    for raw in sdk_link_inputs:
        if not isinstance(raw, str) or not raw or len(raw) > 4096 or "\\" in raw or "\0" in raw:
            raise ValueError("native acceptance SDK link input is invalid")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw:
            raise ValueError("native acceptance SDK link input escapes verified sysroot")
        path = sdk_context_root.joinpath(*relative.parts)
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(sdk_context_root.resolve())
            or path.suffix not in {".a", ".so"}
            or path.stat().st_size > 128 * 1024**2
        ):
            raise ValueError("native acceptance SDK link input is missing or unsupported")
        expected = sdk_context.sysroot.contents["artifacts"].get("sysroot/" + raw)
        if not isinstance(expected, str) or _bounded_digest(path, 128 * 1024**2) != expected:
            raise ValueError("native acceptance SDK link input differs from verified sysroot")
        links.append(str(path))
    return (
        str(sdk_context.compiler()),
        *sdk_context.compiler_flags,
        *sdk_context.linker_flags,
        *sdk_context.executable_flags,
        "-I" + str(dependency_sysroot / "usr/local/include"),
        *includes,
        "-L" + str(dependency_sysroot / "usr/local/lib"),
        str(source),
        *(("-Wl,-Bdynamic",) if any(Path(path).suffix == ".so" for path in links) else ()),
        *links,
        *link_flags,
        "-o",
        str(destination),
    )


def accept_port(request: AcceptanceRequest) -> tuple[AcceptanceResult, ...]:
    """Require every declared probe to pass in a newly installed guest.

    Missing or malformed tests fail before release materialization. The caller
    owns graph and artifact publication; this function writes only proof files.
    """
    tests = request.port.recipe.get("tests")
    if not isinstance(tests, list) or not 1 <= len(tests) <= _MAX_TESTS:
        raise ValueError("port needs between one and 32 declared acceptance probes")
    if request.install_kind not in {"pypi", "native", "pypi+native"}:
        raise ValueError("acceptance installation kind is invalid")
    limits = _test_limits(request.port.recipe)
    if request.limits is not None:
        limits = request.limits
    if request.output.exists() or request.output.is_symlink():
        raise ValueError("acceptance proof output already exists")
    descriptor = request.descriptor
    if descriptor.is_symlink() or not descriptor.is_file() or descriptor.stat().st_size > 1024 * 1024:
        raise ValueError("acceptance release descriptor is missing or oversized")
    descriptor_sha = _digest(descriptor.read_bytes())
    prepared = []
    for test in tests:
        if not isinstance(test, dict) or test.get("kind") not in {"python", "native", "shell"}:
            raise ValueError("acceptance test kind is invalid")
        kind = test["kind"]
        if kind == "shell" and request.install_kind not in {"native", "pypi+native"}:
            raise ValueError("shell acceptance needs a native port selection")
        if kind == "native" and request.install_kind not in {"native", "pypi+native"}:
            raise ValueError("native acceptance needs a native port selection")
        path, data = _source(request.port, test.get("source" if kind == "native" else "script"))
        link_inputs = test.get("link_inputs", [])
        sdk_link_inputs = test.get("sdk_link_inputs", [])
        include_directories = test.get("include_directories", [])
        link_flags = test.get("link_flags", [])
        args = test.get("args", [])
        fixtures = _fixtures(request.port, test.get("files", []))
        if (
            not isinstance(args, list)
            or len(args) > 32
            or any(not isinstance(arg, str) or len(arg) > 4096 or "\0" in arg for arg in args)
        ):
            raise ValueError("acceptance arguments are invalid")
        if kind == "native":
            if request.sdk_context is None or request.dependency_sysroot is None:
                raise ValueError("native acceptance needs a verified SDK and dependency sysroot")
            if (
                not isinstance(link_inputs, list)
                or not isinstance(sdk_link_inputs, list)
                or not isinstance(include_directories, list)
                or len(link_inputs) > 256
                or len(sdk_link_inputs) > 256
                or len(include_directories) > 256
            ):
                raise ValueError("native acceptance link inputs and include directories must be lists")
            _native_command(
                request.sdk_context,
                request.dependency_sysroot,
                path,
                request.output / "check.wasm",
                link_inputs,
                sdk_link_inputs,
                include_directories,
                link_flags,
            )
        elif any(
            key in test for key in ("link_inputs", "sdk_link_inputs", "include_directories", "link_flags", "libraries")
        ):
            raise ValueError("Python acceptance does not take native compiler options")
        if "libraries" in test:
            raise ValueError("native acceptance requires exact link_inputs rather than -l search")
        prepared.append(
            (kind, path, data, link_inputs, sdk_link_inputs, include_directories, link_flags, args, fixtures)
        )

    request.output.mkdir(parents=True)
    from shellsim import Environment

    from ports._support.native_catalog import installation

    name = request.port.name if request.install_kind == "pypi" else installation(request.port)[0]
    spec = f"{name}=={request.port.version}"
    results = []
    with tempfile.TemporaryDirectory(prefix=".release-cache-", dir=request.output) as cache_dir:
        for index, (
            kind,
            path,
            data,
            link_inputs,
            sdk_link_inputs,
            include_directories,
            link_flags,
            args,
            fixtures,
        ) in enumerate(prepared):
            proof = request.output / f"probe-{index}"
            proof.mkdir()
            if kind == "native":
                assert request.sdk_context is not None and request.dependency_sysroot is not None
                artifact = proof / "check.wasm"
                command = _native_command(
                    request.sdk_context,
                    request.dependency_sysroot,
                    path,
                    artifact,
                    link_inputs,
                    sdk_link_inputs,
                    include_directories,
                    link_flags,
                )
                with (proof / "build.log").open("wb") as log:
                    subprocess.run(
                        command,
                        cwd=proof,
                        env=target_environment(request.sdk_context.sdk.root),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                mark_abi(artifact, request.sdk_context.dynamic_abi.encode())
                (proof / "build-command.json").write_text(json.dumps(command) + "\n")
                with artifact.open("rb") as stream:
                    wasm_header = stream.read(8)
                if artifact.stat().st_size > 128 * 1024**2 or wasm_header != b"\0asm\x01\0\0\0":
                    raise ValueError("native acceptance compiler did not produce bounded core Wasm")
            setup = {"pypi" if request.install_kind == "pypi" else "tools": [spec]}
            if request.install_kind == "pypi+native":
                setup["pypi"] = [f"{request.port.name}=={request.port.version}"]
            kwargs = {"limits": limits} if limits is not None else {}
            env = Environment.from_release(descriptor, cache_dir=Path(cache_dir), **setup, **kwargs)
            guest_path = "/work/shellsim-acceptance" + {"python": ".py", "shell": ".sh", "native": ".wasm"}[kind]
            env.write_file(
                guest_path,
                artifact.read_bytes() if kind == "native" else data,
                mode=0o755 if kind == "native" else 0o644,
            )
            for destination, contents in fixtures.items():
                env.write_file("/work/" + destination, contents, mode=0o644)
            guest_command = {"python": "python ", "shell": "sh ", "native": ""}[kind] + shlex.quote(guest_path)
            result = env.run(" ".join((guest_command, *(shlex.quote(arg) for arg in args))))
            (proof / "stdout").write_bytes(result.stdout)
            (proof / "stderr").write_bytes(result.stderr)
            outcome = AcceptanceResult(
                script=path.relative_to(request.port.directory).as_posix(),
                kind=kind,
                fixture_sha256={name: _digest(contents) for name, contents in fixtures.items()},
                source_sha256=_digest(data),
                artifact_sha256=_digest(artifact.read_bytes()) if kind == "native" else None,
                returncode=result.returncode,
                stop_reason=result.stop_reason,
                stdout_sha256=_digest(result.stdout),
                stderr_sha256=_digest(result.stderr),
                usage=dataclasses.asdict(result.usage),
            )
            results.append(outcome)
            (proof / "result.json").write_text(json.dumps(dataclasses.asdict(outcome), sort_keys=True, indent=2) + "\n")
            if result.returncode != 0 or result.stop_reason is not None:
                raise RuntimeError(f"port acceptance failed: {path} (exit {result.returncode}, {result.stop_reason})")
    (request.output / "receipt.json").write_text(
        json.dumps(
            {
                "port": request.port.reference,
                "recipe_sha256": request.port.digest,
                "release_sha256": descriptor_sha,
                "installed_requirement": spec,
                "install_kind": request.install_kind,
                "sdk_context": request.sdk_context.identity if request.sdk_context is not None else None,
                "test_limits": dataclasses.asdict(limits) if limits is not None else None,
                "results": [dataclasses.asdict(item) for item in results],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    return tuple(results)
