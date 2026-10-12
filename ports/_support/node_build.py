"""Execute one port adapter with already materialized inputs; never schedule dependencies."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from ports._support.build import apply_patch
from ports._support.graph import Port
from ports._support.local_sources import stage_local_sources
from ports._support.native_adapters import CompilerCacheLauncher, NativeBuildContext
from ports._support.native_artifacts import NativeArtifact, merge_dependency_sysroot, seal_native_install
from ports._support.python_adapters import CPythonBuildContext
from ports._support.sdk_products import MaterializedSDK
from ports._support.store import extract, fetch, file_hash, identity, relative_path
from ports.api import BuildContext, build_port

logger = logging.getLogger(__name__)


def _run_builder(context: BuildContext):
    try:
        return build_port(context)
    except subprocess.CalledProcessError:
        if context.native is not None:
            path = context.native.build / "build.log"
            if path.is_file():
                with path.open("rb") as log:
                    log.seek(max(0, path.stat().st_size - 8192))
                    logger.error("%s build log:\n%s", context.port.reference, log.read().decode(errors="replace"))
        raise


def _native_target(sdk_context: MaterializedSDK):
    """Use the same resolved products for sealing and acceptance closure checks."""
    from ports._support.native_artifacts import NativeTarget

    return NativeTarget(
        sdk_context.target, sdk_context.dynamic_abi, sdk_context.dynamic_abi, sdk_context.toolchain_receipt
    )


def _local_file(port: Port, name: object) -> Path:
    path = port.directory / relative_path(name)
    if not path.resolve().is_relative_to(port.directory.resolve()):
        raise ValueError("recipe input escapes its port directory")
    file_hash(path)
    return path


def _admit_recipe(port: Port) -> dict[str, str]:
    """Admit metadata and local source inputs without importing build code."""
    from ports._support.build import check_build_scripts
    from ports.api import _HELPER_FILES

    if port.recipe.get("build_system") not in _HELPER_FILES:
        raise ValueError("unsupported build system: " + port.reference)
    check_build_scripts(port.recipe, port.directory)
    files = {}
    for item in port.recipe.get("patches", []):
        path = _local_file(port, item["file"])
        actual = file_hash(path)
        if actual != item["sha256"]:
            raise ValueError("recipe patch differs: " + str(path))
        files[item["file"]] = actual
    files.update(port.recipe.get("inputs", {}))
    return files


def _executable_sdk_link_inputs(sdk_context: MaterializedSDK, build: Mapping) -> tuple[Path, ...]:
    """Resolve declared executable archives only from the verified platform product."""
    declared = build.get("executable_sdk_link_inputs", [])
    if not isinstance(declared, list) or len(declared) > 32:
        raise ValueError("executable SDK link inputs must be a bounded array")
    if not declared:
        return ()
    root = sdk_context.sysroot.root / "sysroot"
    inputs = []
    for raw in declared:
        name = relative_path(raw)
        if len(name) > 4096 or not name.endswith(".a") or name in inputs:
            raise ValueError("executable SDK link input must be a unique archive path")
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
            or path.stat().st_size > 128 * 1024**2
        ):
            raise ValueError("executable SDK archive is missing or unsupported")
        expected = sdk_context.sysroot.contents["artifacts"].get("sysroot/" + name)
        if not isinstance(expected, str) or file_hash(path) != expected:
            raise ValueError("executable SDK archive differs from admitted platform")
        inputs.append(name)
    return tuple(root / name for name in inputs)


def _backend_wheels(port: Port, selected: Mapping[str, Port], results: Mapping[str, Path]):
    """Follow exact backend dependency edges, including runtime imports."""
    from ports._support.python_pep517 import BackendWheel

    pending = [
        item.recipe
        for item in port.dependencies
        if item.kind == "build" and selected[item.recipe].recipe["build_system"] != "llvm-host"
    ]
    visited, wheels = set(), []
    while pending:
        reference = pending.pop()
        if reference in visited:
            continue
        visited.add(reference)
        dependency = selected[reference]
        if dependency.recipe["build_system"] not in {"pure-wheel", "host-wheel"}:
            raise ValueError("PEP 517 backend dependency is not a pinned wheel")
        paths = list((results[reference] / "wheels").glob("*.whl"))
        if len(paths) != 1:
            raise ValueError("backend dependency must supply one verified wheel")
        wheels.append(BackendWheel(paths[0], dependency.recipe))
        pending.extend(item.recipe for item in dependency.dependencies)
    return tuple(wheels)


def _source_exports(port: Port, context: NativeBuildContext) -> None:
    """Stage declared source files, such as licenses omitted by upstream install."""
    declarations = port.recipe.get("source_exports", [])
    if not isinstance(declarations, list) or len(declarations) > 256:
        raise ValueError("source exports must be a bounded list")
    for item in declarations:
        source = context.source / relative_path(item["source"])
        if not source.resolve().is_relative_to(context.source.resolve()):
            raise ValueError("source export escapes admitted sources")
        file_hash(source, limit=16 * 1024**2)
        destination = context.staging_prefix / "usr/local" / relative_path(item["destination"])
        if destination.exists() or destination.is_symlink():
            raise ValueError("source export conflicts with installed file")
        if not destination.parent.resolve().is_relative_to(context.staging_prefix.resolve()):
            raise ValueError("source export escapes staging prefix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


@dataclass(frozen=True)
class ProductWorkspace:
    work: Path
    result: Path


def produce_port(
    ports: Path,
    port: Port,
    cache: Path,
    key: str,
    sdk,
    selected: Mapping[str, Port],
    results: Mapping[str, Path],
    *,
    jobs: int,
    offline: bool = False,
    compiler_cache: CompilerCacheLauncher | None = None,
    retained_workspace: Path | None = None,
) -> Path:
    """Produce a single result; callers own planning, readiness and canonical publication."""
    result = cache / "results" / key
    if (result / "build-receipt.json").is_file():
        return result
    work_root = cache / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=key[:12] + "-", dir=work_root))
    slot = ProductWorkspace(work=work, result=work / "result")
    slot.result.mkdir()
    recipe, system = port.recipe, port.recipe["build_system"]
    build_sdk = sdk
    pure = system in {"pure-wheel", "host-wheel", "source-tree"}
    native = {}
    for reference, root in results.items():
        if (root / "native/artifact.json").is_file():
            dependency = selected[reference]
            native[str(PurePosixPath(reference).parts[0]) + "/" + dependency.name] = NativeArtifact(
                root / "native", json.loads((root / "native/artifact.json").read_text())
            )
    if not pure:
        target = _native_target(build_sdk)
        compiler, sysroot = build_sdk.llvm, build_sdk.sysroot
    executable_inputs = _executable_sdk_link_inputs(build_sdk, recipe) if not pure else ()
    source = (
        stage_local_sources(ports, recipe["source"], slot.work / "source")
        if "files" in recipe["source"]
        else fetch(recipe["source"], cache / "sources", offline=offline)
    )
    if pure:
        if system == "source-tree" and "files" not in recipe["source"]:
            source = extract(source, slot.work / "source", subdirectory=recipe["source"]["subdirectory"])
            for patch in recipe.get("patches", []):
                apply_patch(source, _local_file(port, patch["file"]), patch["sha256"])
        _run_builder(
            BuildContext(
                port,
                source,
                slot.result,
                sdk=build_sdk,
                jobs=jobs,
                compiler_cache=compiler_cache,
                offline=offline,
                build_artifacts={item.port: results[item.recipe] for item in port.dependencies if item.kind == "build"},
            )
        )
    else:
        if system != "llvm-guest" and "files" not in recipe["source"]:
            source = extract(source, slot.work / "source", subdirectory=recipe["source"]["subdirectory"])
        if system != "llvm-guest":
            for patch in recipe.get("patches", []):
                apply_patch(source, _local_file(port, patch["file"]), patch["sha256"])
        direct = {}
        for item in port.dependencies:
            if item.kind != "target":
                continue
            if item.port in native:
                direct[item.port] = native[item.port]
            elif item.port.startswith("native/"):
                raise ValueError("native dependency has no sealed artifact: " + item.port)
        workspace, build_directory = slot.work, slot.work / "build"
        if system == "llvm-guest":
            from ports.toolchain.llvm.guest import workspace_compatibility

            compatibility = workspace_compatibility(
                recipe,
                compiler,
                sysroot.root / "sysroot",
                build_sdk.sdk.root,
                {name: artifact.prefix for name, artifact in direct.items()},
                {name: tool.path for name, tool in build_sdk.host_tools.items()},
                build_sdk.target,
            )
            build_directory = (
                retained_workspace.resolve()
                if retained_workspace is not None
                else cache.resolve() / "workspaces/llvm-guest" / identity(compatibility) / "build"
            )
            workspace = build_directory.parent
        prefix = workspace / "dependencies"
        if system == "llvm-guest" and prefix.exists():
            shutil.rmtree(prefix)
        merge_dependency_sysroot(direct, native, prefix, target)
        context = NativeBuildContext(
            source=source,
            build=build_directory,
            staging_prefix=slot.work / "install",
            sdk=build_sdk.sdk.root,
            compiler_prefix=build_sdk.llvm.root,
            sysroot=build_sdk.sysroot.root / "sysroot",
            target=build_sdk.target,
            compiler_flags=(*build_sdk.compiler_flags, "-I" + str(prefix / "usr/local/include"))
            if direct
            else build_sdk.compiler_flags,
            linker_flags=(*build_sdk.linker_flags, "-L" + str(prefix / "usr/local/lib"))
            if direct
            else build_sdk.linker_flags,
            dependencies={name: artifact.prefix for name, artifact in direct.items()},
            host_tools={name: tool.path for name, tool in build_sdk.host_tools.items()},
            target_tools={name: tool.path for name, tool in build_sdk.target_tools.items()},
            dependency_sysroot=prefix,
            abi=build_sdk.dynamic_abi,
            compiler_resource_directory=build_sdk.compiler_resource_directory,
            linker=build_sdk.linker,
            shared_library_flags=build_sdk.shared_library_flags,
            executable_flags=build_sdk.executable_flags,
            shared_library_inputs=(build_sdk.compiler_runtime_archive,),
            executable_link_inputs=executable_inputs,
            compiler_cache=compiler_cache,
        )
        python_context = None
        if system.startswith("python-"):
            python = build_sdk.python
            python_context = CPythonBuildContext(
                python.source_root / "Include",
                python.generated_config_dir,
                python.version,
                python.dynamic_abi,
                python.runtime_target,
                python.wheel_platform,
                {
                    "source": python.source_sha256,
                    "headers": python.headers_sha256,
                    "pyconfig": python.pyconfig_sha256,
                    "runtime": python.runtime_manifest_sha256,
                },
            )
        output = _run_builder(
            BuildContext(
                port,
                source,
                slot.result,
                sdk=build_sdk,
                native=context,
                cpython=python_context,
                backend_wheels=_backend_wheels(port, selected, results) if system == "python-pep517" else (),
                jobs=jobs,
                offline=offline,
                compiler_cache=compiler_cache,
                build_artifacts={item.port: results[item.recipe] for item in port.dependencies if item.kind == "build"},
                workspace=retained_workspace,
            )
        )
        _source_exports(port, context)
        for receipt_name in ("pep517-receipt.json",):
            if (output.staging_prefix / receipt_name).is_file():
                shutil.copyfile(output.staging_prefix / receipt_name, slot.result / receipt_name)
        if (output.staging_prefix / "wheels").exists():
            shutil.copytree(output.staging_prefix / "wheels", slot.result / "wheels")
        if recipe.get("exports") or recipe.get("export_directories"):
            # Data follows delivery edges; the ABI envelope covers native providers.
            linked_recipe = {
                **recipe,
                "runtime_dependencies": [
                    item for item in recipe.get("runtime_dependencies", []) if item["port"] in native
                ],
            }
            seal_native_install(
                linked_recipe,
                port.directory,
                output.staging_prefix,
                slot.result / "native",
                target,
                direct,
                native,
                {
                    item.port: native[item.port]
                    for item in port.dependencies
                    if item.kind == "runtime" and item.port in native
                },
            )

    (slot.result / "build-receipt.json").write_text(json.dumps({"key": key, "recipe": recipe}))
    result.parent.mkdir(parents=True, exist_ok=True)
    slot.result.rename(result)
    shutil.rmtree(work)
    return result
