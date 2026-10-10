"""Build upstream Python projects through an offline pinned PEP 517 backend.

Backend wheels are graph inputs, not additions to the global host cohort.
The admitted host interpreter runs build logic against target CPython sysconfig
and compiler wrappers. The resulting upstream wheel layout and metadata are
preserved; Wasm admission and RECORD sealing happen before graph publication.
"""

from __future__ import annotations

import ast
import email
import json
import re
import shlex
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Mapping

import tomllib

from ports._support.native_adapters import (
    NativeBuildCommand,
    NativeBuildContext,
    build_environment,
    compiler_wrapper_text,
)
from ports._support.pure_wheel import verified_files, verified_host_files
from ports._support.python_adapters import CPythonBuildContext, PythonBuildOutput, _wheel
from ports._support.python_meson import extension_destination, shared_providers
from ports._support.store import file_hash, relative_path
from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import validate_provider_signatures


@dataclass(frozen=True)
class BackendWheel:
    path: Path
    recipe: Mapping


@dataclass(frozen=True)
class PEP517BuildRequest:
    context: NativeBuildContext
    cpython: CPythonBuildContext
    sysconfig_data: Path
    recipe: Mapping
    backend_wheels: tuple[BackendWheel, ...]


def target_sysconfig(request: PEP517BuildRequest) -> dict:
    """Overlay actual target ABI values with admitted tools and header paths."""
    path = request.sysconfig_data
    if path.is_symlink() or path.stat().st_size > 1024**2:
        raise ValueError("target sysconfig is missing or oversized")
    statements = ast.parse(path.read_text()).body
    assignments = [node for node in statements if isinstance(node, ast.Assign)]
    if len(assignments) != 1 or [node.id for node in assignments[0].targets] != ["build_time_vars"]:
        raise ValueError("unsupported target sysconfig data")
    values = ast.literal_eval(assignments[0].value)
    if not isinstance(values, dict) or any(
        not isinstance(key, str) or not isinstance(value, (str, int)) for key, value in values.items()
    ):
        raise ValueError("unsupported target sysconfig binding")
    context, python = request.context, request.cpython
    wrappers = context.build / "adapter-tools"
    cc, cxx = (shlex.quote(str(wrappers / role)) for role in ("cc", "cxx"))
    values.update(
        {
            "CC": cc,
            "CXX": cxx,
            "LDSHARED": cc + " -shared",
            "LDCXXSHARED": cxx + " -shared",
            "BLDSHARED": cc + " -shared",
            "AR": str(context.target_tools["ar"]),
            "ARFLAGS": "rcs",
            "CFLAGS": "-O2",
            "CXXFLAGS": "-O2",
            "CPPFLAGS": "",
            "LDFLAGS": "",
            "CCSHARED": "-fPIC",
            "INCLUDEPY": str(python.include_dir),
            "CONFINCLUDEPY": str(python.generated_config_dir),
            "LIBDIR": "/usr/local/lib",
            "LIBPL": "/usr/local/lib",
            "prefix": "/usr/local",
            "exec_prefix": "/usr/local",
        }
    )
    return values


def _backend_paths(source: Path, declaration: Mapping) -> list[str]:
    paths = declaration.get("backend-path", [])
    if not isinstance(paths, list) or len(paths) > 16:
        raise ValueError("invalid PEP 517 backend-path")
    result = []
    for raw in paths:
        path = source if raw == "." else source / relative_path(raw)
        if not path.is_dir() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("PEP 517 backend-path escapes admitted source")
        result.append(str(path.resolve()))
    return result


def _unpack_backends(wheels: tuple[BackendWheel, ...], destination: Path) -> None:
    if not 1 <= len(wheels) <= 64:
        raise ValueError("backend wheel closure must contain between one and 64 wheels")
    destination.mkdir()
    size = 0
    for wheel in wheels:
        recipe = dict(wheel.recipe)
        files = (
            verified_host_files(wheel.path, recipe)
            if recipe["build"]["adapter"] == "host-wheel"
            else verified_files(wheel.path, recipe)
        )
        for name, data in files.items():
            path = destination / name
            if name.endswith("/"):
                continue
            if path.exists() or any(part.endswith(".data") for part in PurePosixPath(name).parts):
                raise ValueError("overlapping or unsupported backend wheel layout")
            size += len(data)
            if size > 256 * 1024**2:
                raise ValueError("backend wheel closure exceeds its byte bound")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def _stage_wheel(path: Path, stage: Path, request: PEP517BuildRequest) -> None:
    if path.is_symlink() or path.stat().st_size > 64 * 1024**2:
        raise ValueError("backend wheel is missing or oversized")
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) > 16384 or sum(info.file_size for info in archive.infolist()) > 256 * 1024**2:
            raise ValueError("backend wheel exceeds output bounds")
        for info in archive.infolist():
            name = relative_path(info.filename.rstrip("/"))
            if any(part.endswith(".data") for part in PurePosixPath(name).parts):
                raise ValueError("backend wheel .data relocation is unsupported")
            destination = stage / name
            if destination.exists() or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("backend wheel has duplicate or linked members")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))
    recipe = request.recipe
    dist = recipe["name"].replace("-", "_") + "-" + recipe["version"] + ".dist-info"
    metadata = email.message_from_bytes((stage / dist / "METADATA").read_bytes())
    wheel = email.message_from_bytes((stage / dist / "WHEEL").read_bytes())
    if (
        metadata["Name"] != recipe["name"]
        or metadata["Version"] != recipe["version"]
        or metadata["Requires-Python"] != recipe.get("requires_python")
        or sorted(metadata.get_all("Requires-Dist", [])) != sorted(recipe.get("requires_dist", []))
    ):
        raise ValueError("upstream wheel metadata differs from recipe")
    pure = wheel["Root-Is-Purelib"] == "true"
    tags = wheel.get_all("Tag", [])
    python_tag = "cp" + "".join(request.cpython.version.split(".")[:2])
    if (pure and (not tags or any(not tag.endswith("-none-any") for tag in tags))) or (
        not pure and tags != [python_tag + "-" + python_tag + "-" + request.cpython.wheel_platform]
    ):
        raise ValueError("backend wheel does not declare the admitted target tag")
    providers = shared_providers(request.context.dependencies)
    artifacts = []
    for member in sorted(stage.rglob("*")):
        if not member.is_file():
            continue
        if member.suffix == ".so":
            if pure:
                raise ValueError("pure backend wheel contains a native extension")
            relative = member.relative_to(stage).as_posix()
            canonical = stage / extension_destination(relative)
            if canonical != member:
                if canonical.exists():
                    raise ValueError("backend extension destinations overlap")
                member.rename(canonical)
                member = canonical
            mark_abi(member, recipe["abi"].encode())
            needed = validate_provider_signatures(member, providers)
            artifacts.append(
                {
                    "path": member.relative_to(stage).as_posix(),
                    "sha256": file_hash(member),
                    "native_dependencies": needed,
                }
            )
        elif member.suffix.lower() in {".pyd", ".dll", ".dylib", ".a", ".wasm"} or member.read_bytes().startswith(
            (b"MZ", b"\x7fELF", b"\0asm")
        ):
            raise ValueError("backend wheel contains unsupported native output")
    if not artifacts and not pure:
        raise ValueError("PEP 517 native project produced no extensions")
    if pure:
        (stage / dist / "RECORD").unlink()
        return
    manifest = {
        "schema_version": 1,
        "name": recipe["name"],
        "version": recipe["version"],
        "abi": recipe["abi"],
        "recipe": dict(recipe),
        "cpython_receipt": dict(request.cpython.receipt),
        "artifacts": artifacts,
    }
    (stage / dist / "shellsim-native.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (stage / dist / "RECORD").unlink()


def build_pep517(request: PEP517BuildRequest) -> PythonBuildOutput:
    """Execute upstream hooks offline, then admit and reseal their wheel."""
    context, recipe = request.context, request.recipe
    if (
        request.cpython.abi != recipe.get("abi", request.cpython.abi)
        or request.cpython.runtime_target != context.target
        or recipe.get("target", context.target) != context.target
    ):
        raise ValueError("PEP 517 target differs from admitted CPython")
    if context.build.exists() or context.staging_prefix.exists():
        raise ValueError("PEP 517 build output already exists")
    project_file = context.source / "pyproject.toml"
    project = tomllib.loads(project_file.read_text()) if project_file.exists() else {}
    declaration = project.get(
        "build-system", {"build-backend": "setuptools.build_meta:__legacy__", "requires": ["setuptools>=40.8.0"]}
    )
    if not isinstance(declaration, dict):
        raise ValueError("PEP 517 build-system must be a table")
    requirements = declaration.get("requires")
    if (
        not isinstance(requirements, list)
        or len(requirements) > 64
        or any(not isinstance(value, str) or len(value) > 4096 for value in requirements)
    ):
        raise ValueError("PEP 517 build-system requires must be a bounded list of requirements")
    backend = declaration.get("build-backend", "setuptools.build_meta:__legacy__")
    if not isinstance(backend, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_.]*(?::[A-Za-z_][A-Za-z0-9_.]*)?", backend
    ):
        raise ValueError("invalid PEP 517 build-backend name")
    declaration = {**declaration, "build-backend": backend}
    backend_paths = _backend_paths(context.source, declaration)
    settings = recipe["build"].get("config_settings", {})
    if (
        not isinstance(settings, dict)
        or len(settings) > 64
        or any(
            not isinstance(key, str) or not isinstance(value, (str, list)) or len(str(value)) > 4096
            for key, value in settings.items()
        )
    ):
        raise ValueError("invalid PEP 517 config settings")
    context.build.mkdir(parents=True)
    imports = context.build / "backend-imports"
    _unpack_backends(request.backend_wheels, imports)
    tools = context.build / "adapter-tools"
    tools.mkdir()
    response = Path(__file__).with_name("compiler_response.py").read_text()
    compiler_context = replace(
        context,
        compiler_flags=(
            *context.compiler_flags,
            "-I" + str(request.cpython.include_dir),
            "-I" + str(request.cpython.generated_config_dir),
        ),
    )
    for role in ("cc", "cxx"):
        path = tools / role
        path.write_text(compiler_wrapper_text(compiler_context, role, response))
        path.chmod(0o755)
    configuration = context.build / "configuration"
    configuration.mkdir()
    values = target_sysconfig(request)
    (configuration / "_shellsim_target_sysconfig.py").write_text("build_time_vars = " + repr(values) + "\n")
    output = context.build / "upstream-wheels"
    output.mkdir()
    payload = {
        "backend_paths": backend_paths,
        "imports": str(imports),
        "configuration": str(configuration),
        "requires": declaration["requires"],
        "backend": declaration["build-backend"],
        "config_settings": settings,
        "python_version": [int(value) for value in request.cpython.version.split(".")[:2]],
        "output": str(output),
        "response": str(context.build / "backend-response.json"),
    }
    arguments = context.build / "backend-request.json"
    arguments.write_text(json.dumps(payload, sort_keys=True))
    driver = context.build / "pep517-runner.py"
    shutil.copyfile(Path(__file__).with_name("pep517_runner.py"), driver)
    environment = build_environment(context, {})
    environment.update(
        {
            "_PYTHON_SYSCONFIGDATA_NAME": "_shellsim_target_sysconfig",
            "_PYTHON_HOST_PLATFORM": request.cpython.wheel_platform,
            "SETUPTOOLS_USE_DISTUTILS": "local",
            "PIP_NO_INDEX": "1",
        }
    )
    command = NativeBuildCommand(
        (str(context.host_tools["python"]), "-I", "-S", str(driver), str(arguments)), context.source
    )
    with (context.build / "backend.log").open("wb") as log:
        subprocess.run(
            command.argv, cwd=command.directory, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    result = json.loads(Path(payload["response"]).read_text())
    filename = relative_path(result["wheel"])
    if "/" in filename or not filename.endswith(".whl"):
        raise ValueError("backend wheel filename escapes output")
    stage = context.staging_prefix / "wheel-root"
    stage.mkdir(parents=True)
    _stage_wheel(output / filename, stage, request)
    wheels = context.staging_prefix / "wheels"
    wheels.mkdir()
    dist = recipe["name"].replace("-", "_") + "-" + recipe["version"] + ".dist-info"
    _wheel(stage, wheels / filename, dist + "/RECORD")
    provenance = {
        "target_sysconfig_sha256": file_hash(request.sysconfig_data),
        "target_overlay": values,
        "backend_wheels": [
            {"name": wheel.recipe["name"], "version": wheel.recipe["version"], "sha256": file_hash(wheel.path)}
            for wheel in request.backend_wheels
        ],
        "additional_requires": result["additional_requires"],
        "compiler_wrappers": {role: (tools / role).read_text() for role in ("cc", "cxx")},
        "environment": environment,
    }
    (context.staging_prefix / "pep517-receipt.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
    return PythonBuildOutput(context.staging_prefix, (command,))
