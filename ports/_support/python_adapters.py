"""Build verified Python wheel artifacts for the ports graph runner.

The runner admits source and recipe identities and owns cache publication. An
adapter only writes into its private staging prefix and returns command records.
"""

from __future__ import annotations

import base64
import csv
import email
import hashlib
import io
import json
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ports._support.native_adapters import NativeBuildCommand, NativeBuildContext
from ports._support.pure_wheel import verified_files
from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import needed_libraries
from ports.native.dependencies import target_environment

_MAX_PURE_WHEEL = 64 * 1024**2
_MAX_EXTENSION_SOURCE = 64 * 1024**2
_MAX_EXTENSION_FILES = 256
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_DIST_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class PureWheelBuildRequest:
    """An admitted upstream wheel and recipe, with a private output directory."""

    source: Path
    staging_prefix: Path
    recipe: Mapping[str, Any]


@dataclass(frozen=True)
class PythonBuildOutput:
    """Unpublished wheel staging and the commands run to produce it."""

    staging_prefix: Path
    commands: tuple[NativeBuildCommand, ...]


@dataclass(frozen=True)
class CPythonBuildContext:
    """Headers and runtime identity admitted by the pinned build cohort."""

    include_dir: Path
    generated_config_dir: Path
    version: str
    abi: str
    runtime_target: str
    wheel_platform: str
    receipt: Mapping[str, str]


@dataclass(frozen=True)
class ExtensionBuildRequest:
    """Recipe-driven extension build against an admitted target CPython cohort."""

    context: NativeBuildContext
    cpython: CPythonBuildContext
    recipe: Mapping[str, Any]


def build_pure_wheel(request: PureWheelBuildRequest) -> PythonBuildOutput:
    """Stage the unchanged pinned wheel after checking metadata and contents."""
    source = request.source
    if source.is_symlink() or not source.is_file():
        raise ValueError("pure wheel source must be a regular file")
    with source.open("rb") as stream:
        data = stream.read(_MAX_PURE_WHEEL + 1)
    if len(data) > _MAX_PURE_WHEEL:
        raise ValueError("pure wheel exceeds build bounds")
    verified_files(source, dict(request.recipe), source=data)
    wheels = request.staging_prefix / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    output = wheels / source.name
    if output.exists() or output.is_symlink():
        raise ValueError("pure wheel staging path already exists")
    output.write_bytes(data)
    return PythonBuildOutput(request.staging_prefix, ())


def _source_file(root: Path, raw: str) -> Path:
    """Resolve a recipe path without letting source trees escape their admission root."""
    if not isinstance(raw, str) or not raw or "\\" in raw or "\0" in raw:
        raise ValueError("invalid extension source path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("extension source path escapes source root")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("extension source must be a regular admitted file")
    if path.stat().st_size > _MAX_EXTENSION_SOURCE:
        raise ValueError("extension source exceeds build bounds")
    return path


def _dependency_path(root: Path, raw: str, *, directory: bool) -> Path:
    """Admit an exact file or include directory from the merged native closure."""
    if not isinstance(raw, str) or not raw or len(raw) > 4096 or "\\" in raw or "\0" in raw:
        raise ValueError("invalid extension dependency path")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != raw:
        raise ValueError("extension dependency path escapes merged sysroot")
    prefix = root / "usr/local"
    path = prefix.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(prefix.resolve()):
        raise ValueError("extension dependency path escapes merged sysroot")
    if directory:
        if not path.is_dir():
            raise ValueError("declared extension include directory is missing")
    elif not path.is_file() or path.stat().st_size > _MAX_EXTENSION_SOURCE:
        raise ValueError("declared extension link input is missing or oversized")
    return path


def _wheel(stage: Path, destination: Path, record: str) -> None:
    """Seal staged wheel files with deterministic member order and RECORD."""
    rows = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        rows.append((path.relative_to(stage).as_posix(), "sha256=" + digest, len(data)))
    rows.append((record, "", ""))
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    (stage / record).write_text(output.getvalue())
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                member = zipfile.ZipInfo(path.relative_to(stage).as_posix(), (1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o644 << 16
                archive.writestr(member, path.read_bytes())


def build_extension(request: ExtensionBuildRequest) -> PythonBuildOutput:
    """Compile a declared C/C++ extension and stage one ABI-marked native wheel.

    The graph runner verifies source, compiler and CPython receipts before this
    call. This adapter checks recipe-to-receipt coherence and writes only to its
    private build/staging paths; the runner decides publication.
    """
    context, cpython, recipe = request.context, request.cpython, request.recipe
    name, version, abi = recipe["name"], recipe["version"], recipe["abi"]
    build = recipe["build"]
    module = build["module"]
    if not _DIST_NAME.fullmatch(name) or not re.fullmatch(r"[0-9][A-Za-z0-9_.+!-]*", version):
        raise ValueError("invalid extension distribution identity")
    if not _IDENTIFIER.fullmatch(module) or build.get("adapter") != "python-extension":
        raise ValueError("invalid extension module or adapter")
    if (
        cpython.version.split(".")[:2] != ["3", "13"]
        or cpython.abi != abi
        or cpython.runtime_target != context.target
        or recipe["target"] != context.target
        or cpython.wheel_platform != "wasm32_wasip1"
        or ("target_profile" in recipe and recipe["target_profile"] != abi)
    ):
        raise ValueError("extension target or ABI differs from admitted CPython")
    if not context.shared_library_flags:
        raise ValueError("extension build needs admitted shared-library link flags")
    paths = (context.source, context.build, context.staging_prefix, cpython.include_dir, cpython.generated_config_dir)
    if any(not path.is_absolute() for path in paths):
        raise ValueError("extension build paths must be absolute")
    raw_sources = build["sources"]
    if not isinstance(raw_sources, list) or not 1 <= len(raw_sources) <= _MAX_EXTENSION_FILES:
        raise ValueError("invalid extension source list")
    sources = [_source_file(context.source, raw) for raw in raw_sources]
    if len(set(sources)) != len(sources) or any(path.suffix not in {".c", ".cc", ".cpp", ".cxx"} for path in sources):
        raise ValueError("invalid or repeated extension translation unit")
    metadata = _source_file(context.source, build["metadata"])
    message = email.message_from_bytes(metadata.read_bytes())
    if message["Name"] != name or message["Version"] != version:
        raise ValueError("extension source metadata differs from recipe")
    if sorted(message.get_all("Requires-Dist", [])) != sorted(recipe.get("requires_dist", [])):
        raise ValueError("extension dependencies differ from recipe")
    licenses = build.get("licenses", [])
    if not isinstance(licenses, list) or len(licenses) > _MAX_EXTENSION_FILES:
        raise ValueError("invalid extension license list")
    license_files = [_source_file(context.source, raw) for raw in licenses]
    defines = build.get("defines", [])
    if not isinstance(defines, list) or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*(?:=[A-Za-z0-9_]+)?", item) for item in defines
    ):
        raise ValueError("invalid extension preprocessor definitions")
    dependencies = build.get("native_dependencies", [])
    if not isinstance(dependencies, list) or any(not isinstance(item, str) or not item for item in dependencies):
        raise ValueError("invalid extension native dependency list")
    declared_ports = recipe.get("target_dependencies", [])
    native_ports = (
        [
            item["port"]
            for item in declared_ports
            if isinstance(item, dict) and isinstance(item.get("port"), str) and item["port"].startswith("native/")
        ]
        if isinstance(declared_ports, list)
        else []
    )
    if (
        not isinstance(declared_ports, list)
        or any(not isinstance(item, dict) or not isinstance(item.get("port"), str) for item in declared_ports)
        or set(native_ports) != set(context.dependencies)
        or len(native_ports) != len(context.dependencies)
    ):
        raise ValueError("extension native dependencies differ from the admitted graph")
    include_entries = build.get("include_directories", [])
    link_entries = build.get("link_inputs", [])
    if (
        not isinstance(include_entries, list)
        or not isinstance(link_entries, list)
        or len(include_entries) > _MAX_EXTENSION_FILES
        or len(link_entries) > _MAX_EXTENSION_FILES
    ):
        raise ValueError("invalid extension native build input list")
    includes = [_dependency_path(context.dependency_sysroot, raw, directory=True) for raw in include_entries]
    links = [_dependency_path(context.dependency_sysroot, raw, directory=False) for raw in link_entries]
    if len(set(includes)) != len(includes) or len(set(links)) != len(links):
        raise ValueError("repeated extension native build input")
    if any(path.suffix not in {".a", ".so"} for path in links):
        raise ValueError("extension link input must be an archive or shared library")
    linked_shared = {path.name for path in links if path.suffix == ".so"}
    if not linked_shared.issubset(set(dependencies)):
        raise ValueError("extension shared link input lacks a declared native dependency")
    if context.build.exists() or context.staging_prefix.exists():
        raise ValueError("extension build output already exists")
    context.build.mkdir(parents=True)
    wheel_stage = context.build / "wheel-root"
    wheel_stage.mkdir()
    (context.staging_prefix / "wheels").mkdir(parents=True)
    environment = target_environment(context.sdk)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    commands = []
    objects = []
    with (context.build / "build.log").open("wb") as log:
        for index, source in enumerate(sources):
            obj = context.build / f"source-{index}.o"
            compiler = context.target_tools["cxx"] if source.suffix != ".c" else context.target_tools["cc"]
            command = NativeBuildCommand(
                (
                    str(compiler),
                    *context.compiler_flags,
                    "-fPIC",
                    "-I" + str(cpython.include_dir),
                    "-I" + str(cpython.generated_config_dir),
                    *("-I" + str(path) for path in includes),
                    *("-D" + item for item in defines),
                    "-c",
                    str(source),
                    "-o",
                    str(obj),
                ),
                context.build,
            )
            subprocess.run(
                command.argv, cwd=command.directory, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
            )
            commands.append(command)
            objects.append(obj)
        extension = wheel_stage / f"{module}.so"
        linker = (
            context.target_tools["cxx"] if any(path.suffix != ".c" for path in sources) else context.target_tools["cc"]
        )
        command = NativeBuildCommand(
            (
                str(linker),
                *context.compiler_flags,
                *context.linker_flags,
                *context.shared_library_flags,
                "-Wl,--export=PyInit_" + module,
                "-Wl,--export-all,--fatal-warnings",
                *(str(obj) for obj in objects),
                *(str(path) for path in links),
                "-o",
                str(extension),
                *(str(path) for path in context.shared_library_inputs),
            ),
            context.build,
        )
        subprocess.run(
            command.argv, cwd=command.directory, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
        )
        commands.append(command)
    mark_abi(extension, abi.encode())
    if sorted(needed_libraries(extension)) != sorted(dependencies):
        raise ValueError("compiled extension dependency closure differs from recipe")
    dist_name = name.replace("-", "_")
    info = wheel_stage / f"{dist_name}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_bytes(metadata.read_bytes())
    (info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: shellsim-python-extension\n"
        "Root-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1\n"
    )
    if license_files:
        license_root = info / "licenses"
        license_root.mkdir()
        for license_file in license_files:
            destination = license_root / license_file.name
            if destination.exists():
                raise ValueError("extension license basenames conflict")
            destination.write_bytes(license_file.read_bytes())
    manifest = {
        "schema_version": 1,
        "name": name,
        "version": version,
        "abi": abi,
        "recipe": dict(recipe),
        "cpython_receipt": dict(cpython.receipt),
        "artifacts": [
            {
                "path": f"{module}.so",
                "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
                "native_dependencies": dependencies,
            }
        ],
    }
    (info / "shellsim-native.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    output = context.staging_prefix / "wheels" / f"{dist_name}-{version}-cp313-cp313-wasm32_wasip1.whl"
    _wheel(wheel_stage, output, f"{dist_name}-{version}.dist-info/RECORD")
    return PythonBuildOutput(context.staging_prefix, tuple(commands))
