"""Build recipe dependency graphs through admitted adapters and a verified store.

Recipes own package choices. This driver dispatches by build system, publishes
only complete results, and records exact dependency and compiler identities.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Mapping, Sequence

from ports._support.build import apply_patch
from ports._support.graph import Graph, Port, guest_graph, plan
from ports._support.native_adapters import (
    NativeAdapter,
    NativeBuildContext,
    build_native,
    compilation_driver_inputs,
    native_build_request,
)
from ports._support.python_adapters import (
    CPythonBuildContext,
    ExtensionBuildRequest,
    PureWheelBuildRequest,
    build_extension,
    build_host_wheel,
    build_pure_wheel,
)
from ports._support.store import build_slot, extract, fetch, file_hash, relative_path

_COMMON_BUILD_MODULES = ("runner.py", "graph.py", "store.py", "sdk.py", "sdk_products.py", "local_sources.py")
_NATIVE_BUILD_MODULES = (
    "native_adapters.py",
    "compiler_response.py",
    "native_artifacts.py",
    "wasm.py",
    "wasm_metadata.py",
    "native/dependencies.py",
    "toolchain/runtime_profile.py",
)
_PURE_BUILD_MODULES = ("python_adapters.py", "pure_wheel.py")

if TYPE_CHECKING:
    from ports._support.sdk_products import MaterializedSDK


@dataclass(frozen=True)
class GraphBuild:
    """Dependency-first graph and immutable result directories by recipe reference."""

    graph: Graph
    results: dict[str, Path]
    sdks: dict[str, MaterializedSDK]


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
    build = port.recipe.get("build", {})
    adapter = build.get("adapter")
    if adapter in {"llvm-host", "wasi-sysroot"}:
        expected_role = "host-tool" if adapter == "llvm-host" else "target-platform"
        if port.role != expected_role:
            raise ValueError("toolchain adapter role differs")
        path = _local_file(port, build["producer_recipe"])
        if file_hash(path) != build["producer_sha256"]:
            raise ValueError("toolchain producer recipe differs")
        from ports._support.build import check_build_scripts

        check_build_scripts(json.loads(path.read_text()), port.directory)
        return {build["producer_recipe"]: build["producer_sha256"]}
    if adapter not in {
        "pure-wheel",
        "host-wheel",
        "python-pep517",
        "python-extension",
        "python-meson",
        "llvm-guest",
        "llvm-guest-sdk",
        *(item.value for item in NativeAdapter),
    }:
        raise ValueError(f"recipe has no supported build adapter: {port.reference}")
    if (
        adapter not in {"pure-wheel", "host-wheel", "llvm-guest", "llvm-guest-sdk"}
        and "files" not in port.recipe["source"]
    ):
        relative_path(port.recipe["source"]["subdirectory"])
    if adapter == "host-wheel" and port.role != "host-tool":
        raise ValueError("host-wheel adapter requires a host-only role")
    files = {}
    for item in [*port.recipe.get("patches", []), *build.get("hooks", [])]:
        path = _local_file(port, item["file"])
        actual = file_hash(path)
        if actual != item["sha256"]:
            raise ValueError(f"recipe patch or hook differs: {path}")
        files[item["file"]] = actual
    for hook in build.get("hooks", []):
        if hook.get("phase") not in {"before_build", "after_install"}:
            raise ValueError("unsupported port hook phase")
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


def _hooks(port: Port, phase: str, context: NativeBuildContext, sdk_context: MaterializedSDK) -> None:
    """Run pinned port-local Python hooks with an explicit JSON context argument."""
    from ports.native.dependencies import target_environment

    hooks = [hook for hook in port.recipe["build"].get("hooks", []) if hook["phase"] == phase]
    if not hooks:
        return
    context.build.parent.mkdir(parents=True, exist_ok=True)
    path = context.build.parent / "hook-context.json"
    path.write_text(json.dumps(asdict(context), default=str, sort_keys=True) + "\n")
    environment = target_environment(context.sdk)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PATH"] = os.pathsep.join(sorted({str(tool.path.parent) for tool in sdk_context.host_tools.values()}))
    for index, hook in enumerate(hooks):
        with (context.build.parent / f"hook-{phase}-{index}.log").open("wb") as log:
            subprocess.run(
                [str(sdk_context.tool("python")), str(_local_file(port, hook["file"])), str(path)],
                cwd=context.source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def _native_driver_inputs(context: NativeBuildContext, *, build: dict, jobs: int | None) -> dict:
    return compilation_driver_inputs(native_build_request(context, build, jobs))


def _backend_wheels(port: Port, selected: Mapping[str, Port], results: Mapping[str, Path]):
    """Follow exact backend dependency edges, including runtime imports."""
    from ports._support.python_pep517 import BackendWheel

    pending = [
        item.recipe
        for item in port.dependencies
        if item.kind == "build" and item.recipe != "toolchain/llvm/host-recipe.json"
    ]
    visited, wheels = set(), []
    while pending:
        reference = pending.pop()
        if reference in visited:
            continue
        visited.add(reference)
        dependency = selected[reference]
        if dependency.recipe["build"]["adapter"] not in {"pure-wheel", "host-wheel"}:
            raise ValueError("PEP 517 backend dependency is not a pinned wheel")
        paths = list((results[reference] / "wheels").glob("*.whl"))
        if len(paths) != 1:
            raise ValueError("backend dependency must supply one verified wheel")
        wheels.append(BackendWheel(paths[0], dependency.recipe))
        pending.extend(item.recipe for item in dependency.dependencies)
    return tuple(wheels)


def _build_implementation(support: Path, adapter: str, *, stdlib: bool = False) -> dict[str, str]:
    """Bind cached outputs to the code that actually stages and seals them."""
    modules = [*_COMMON_BUILD_MODULES]
    if adapter in {"pure-wheel", "host-wheel"}:
        modules.extend(_PURE_BUILD_MODULES)
    elif adapter in {"llvm-guest", "llvm-guest-sdk"}:
        modules.extend(("build.py", "toolchain/llvm/guest.py", *_NATIVE_BUILD_MODULES))
    elif adapter in {"python-extension", "python-meson", "python-pep517"}:
        modules.extend(("build.py", "python_adapters.py", *_NATIVE_BUILD_MODULES))
        if adapter == "python-meson":
            modules.append("python_meson.py")
        elif adapter == "python-pep517":
            modules.extend(("python_pep517.py", "pep517_runner.py", "python_meson.py", "pure_wheel.py"))
    else:
        modules.extend(("build.py", *_NATIVE_BUILD_MODULES))
    if adapter in {"meson", "python-meson"}:
        modules.extend(("meson_adapter.py", "meson_workspace.py"))
    elif adapter == "cmake":
        modules.append("cmake_adapter.py")
    elif adapter in {"configure-make", "plain-make"}:
        modules.append("make_adapter.py")
    if stdlib:
        modules.append("python/cpython/graph_assembly.py")
    return {
        name: file_hash(
            support.parent / name if name.startswith(("native/", "toolchain/", "python/")) else support / name
        )
        for name in modules
    }


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


def build_graph(
    ports: Path,
    requests: Sequence[str],
    sdk_context: MaterializedSDK,
    store: Path,
    *,
    offline: bool = False,
    jobs: int | None = None,
    default_sdk: str = "default",
    workspaces: Mapping[str, Path] | None = None,
) -> GraphBuild:
    """Build each selected recipe once, reusing only byte-verified cache entries."""
    from ports._support.native_artifacts import (
        NativeArtifact,
        merge_dependency_sysroot,
        seal_native_install,
    )

    if jobs is not None and not 1 <= jobs <= 16:
        raise ValueError("build jobs must be between one and sixteen")
    graph = plan(
        ports, requests, target_profile=sdk_context.dynamic_abi if sdk_context else None, default_sdk=default_sdk
    )
    selected = {port.reference: port for port in graph.ports}
    for reference in workspaces or {}:
        if reference not in selected or selected[reference].recipe["build"]["adapter"] not in {
            "llvm-guest",
            "meson",
            "python-meson",
        }:
            raise ValueError("explicit workspace requires a selected persistent producer: " + reference)
    local_inputs = {port.reference: _admit_recipe(port) for port in graph.ports}
    from ports._support.local_sources import local_source_files, stage_local_sources

    for port in graph.ports:
        if "files" in port.recipe.get("source", {}):
            local_source_files(ports, port.recipe["source"])
    for port in graph.ports:
        if port.recipe["build"]["adapter"] in {"pure-wheel", "host-wheel", "llvm-host", "wasi-sysroot"}:
            continue
        if (
            len(
                [
                    item
                    for item in port.dependencies
                    if item.kind == "build" and item.recipe == "toolchain/llvm/host-recipe.json"
                ]
            )
            != 1
        ):
            raise ValueError(f"native build requires an explicit host compiler dependency: {port.reference}")
        if len([item for item in port.dependencies if item.kind == "platform"]) != 1:
            raise ValueError(f"native build requires an explicit target platform dependency: {port.reference}")
    support = Path(__file__).parent
    implementations = {
        port.reference: _build_implementation(
            support, port.recipe["build"]["adapter"], stdlib=port.recipe["build"].get("output") == "stdlib"
        )
        for port in graph.ports
        if port.recipe["build"]["adapter"] not in {"llvm-host", "wasi-sysroot"}
    }
    results: dict[str, Path] = {}
    native: dict[str, NativeArtifact] = {}
    keys: dict[str, str] = {}
    products = {}
    sdks = {}
    for port in graph.ports:
        adapter = port.recipe["build"]["adapter"]
        if adapter in {"llvm-host", "wasi-sysroot"}:
            from ports._support.sdk_products import verify_product
            from ports._support.store import identity

            product = sdk_context.llvm if adapter == "llvm-host" else sdk_context.sysroot
            verify_product(product)
            producer_path = _local_file(port, port.recipe["build"]["producer_recipe"])
            if product.contents["identity"]["recipe"] != json.loads(producer_path.read_text()):
                raise ValueError("graph product producer differs from declared recipe")
            products[port.reference] = product
            results[port.reference] = product.root
            keys[port.reference] = identity({"recipe": port.digest, "product": product.sha256})
            continue
        build_sdk = sdk_context
        if adapter not in {"pure-wheel", "host-wheel"}:
            from ports._support.sdk_products import resolved_toolchain

            sysroot = next(products[item.recipe] for item in port.dependencies if item.kind == "platform")
            compiler = next(
                products[item.recipe]
                for item in port.dependencies
                if item.kind == "build" and item.recipe == "toolchain/llvm/host-recipe.json"
            )
            build_sdk = resolved_toolchain(sdk_context, compiler, sysroot)
            target = _native_target(build_sdk)
        sdks[port.reference] = build_sdk
        executable_inputs = (
            _executable_sdk_link_inputs(build_sdk, port.recipe["build"])
            if adapter not in {"pure-wheel", "host-wheel"}
            else ()
        )
        inputs = {
            "recipe_sha256": port.digest,
            "local_inputs": local_inputs[port.reference],
            "implementation": implementations[port.reference],
            "dependencies": {
                dependency.kind + ":" + dependency.port: keys[dependency.recipe] for dependency in port.dependencies
            },
        }
        if port.sdk_selection is not None:
            inputs["sdk_selection"] = asdict(port.sdk_selection)
        if adapter not in {"pure-wheel", "host-wheel"}:
            inputs["sdk"] = build_sdk.identity
            inputs["host_tools"] = {
                name: {"sha256": tool.sha256, "receipt": tool.receipt_sha256}
                for name, tool in sdk_context.host_tools.items()
                if name != "uv"
            }
            if adapter in {"python-extension", "python-pep517", "python-meson"}:
                inputs["python"] = {
                    "source": sdk_context.python.source_sha256,
                    "headers": sdk_context.python.headers_sha256,
                    "pyconfig": sdk_context.python.pyconfig_sha256,
                }
        if adapter not in {"pure-wheel", "host-wheel", "llvm-guest-sdk"}:
            inputs["target_flags"] = {
                "compiler": list(build_sdk.compiler_flags),
                "linker": list(build_sdk.linker_flags),
                "shared_library": list(build_sdk.shared_library_flags),
                "shared_library_inputs": [
                    {
                        "path": str(build_sdk.compiler_runtime_archive),
                        "sha256": file_hash(build_sdk.compiler_runtime_archive),
                    }
                ],
                "executable": list(build_sdk.executable_flags),
                "executable_link_inputs": [
                    {"path": str(path), "sha256": file_hash(path)} for path in executable_inputs
                ],
            }
        with build_slot(store, inputs) as slot, ExitStack() as workspace_stack:
            keys[port.reference] = slot.key
            print(
                f"ports: result cache {'hit' if slot.cached else 'miss'}: {port.reference}", file=sys.stderr, flush=True
            )
            if not slot.cached:
                recipe, build = port.recipe, port.recipe["build"]
                source = (
                    stage_local_sources(ports, recipe["source"], slot.work / "source")
                    if "files" in recipe["source"]
                    else fetch(recipe["source"], store / "sources", offline=offline)
                )
                if adapter in {"pure-wheel", "host-wheel"}:
                    producer = build_pure_wheel if adapter == "pure-wheel" else build_host_wheel
                    producer(PureWheelBuildRequest(source, slot.result, recipe))
                else:
                    if adapter != "llvm-guest" and "files" not in recipe["source"]:
                        source = extract(source, slot.work / "source", subdirectory=recipe["source"]["subdirectory"])
                    if adapter != "llvm-guest":
                        for patch in recipe.get("patches", []):
                            apply_patch(source, _local_file(port, patch["file"]), patch["sha256"])
                    for dependency in port.dependencies:
                        if (
                            dependency.kind == "target"
                            and dependency.port.startswith("native/")
                            and dependency.port not in native
                        ):
                            raise ValueError(f"native dependency has no sealed artifact: {dependency.port}")
                    direct = {
                        dependency.port: native[dependency.port]
                        for dependency in port.dependencies
                        if dependency.kind == "target" and dependency.port in native
                    }
                    workspace = slot.work
                    build_directory = workspace / "build"
                    if adapter == "llvm-guest":
                        from ports._support.store import identity
                        from ports.toolchain.llvm.guest import workspace_compatibility

                        compatibility = workspace_compatibility(
                            recipe,
                            compiler,
                            sysroot.root / "sysroot",
                            sdk_context.sdk.root,
                            {name: artifact.prefix for name, artifact in direct.items()},
                            {name: tool.path for name, tool in sdk_context.host_tools.items()},
                            build_sdk.target,
                        )
                        build_directory = (
                            workspaces[port.reference].resolve()
                            if workspaces is not None and port.reference in workspaces
                            else store.resolve() / "workspaces" / "llvm-guest" / identity(compatibility) / "build"
                        )
                        workspace = build_directory.parent
                    prefix = workspace / "dependencies"
                    if adapter == "llvm-guest" and prefix.exists():
                        shutil.rmtree(prefix)
                    merge_dependency_sysroot(direct, native, prefix, target)
                    context = NativeBuildContext(
                        source=source,
                        build=build_directory,
                        staging_prefix=slot.work / "install",
                        sdk=build_sdk.sdk.root,
                        compiler_prefix=build_sdk.llvm.root,
                        sysroot=build_sdk.sysroot.root / "sysroot",
                        target=sdk_context.target,
                        compiler_flags=(*build_sdk.compiler_flags, "-I" + str(prefix / "usr/local/include"))
                        if direct
                        else build_sdk.compiler_flags,
                        linker_flags=(*build_sdk.linker_flags, "-L" + str(prefix / "usr/local/lib"))
                        if direct
                        else build_sdk.linker_flags,
                        dependencies={name: artifact.prefix for name, artifact in direct.items()},
                        host_tools={name: tool.path for name, tool in sdk_context.host_tools.items()},
                        target_tools={name: tool.path for name, tool in build_sdk.target_tools.items()},
                        dependency_sysroot=prefix,
                        abi=build_sdk.dynamic_abi,
                        compiler_resource_directory=build_sdk.compiler_resource_directory,
                        linker=build_sdk.linker,
                        shared_library_flags=build_sdk.shared_library_flags,
                        executable_flags=build_sdk.executable_flags,
                        shared_library_inputs=(build_sdk.compiler_runtime_archive,),
                        executable_link_inputs=executable_inputs,
                    )
                    if adapter in {"python-extension", "python-meson", "python-pep517"}:
                        python = sdk_context.python
                        if python is None:
                            raise ValueError("Python extension requires admitted CPython headers and runtime")
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
                        if adapter == "python-meson":
                            from ports._support.python_meson import (
                                PythonMesonBuildRequest,
                                build_python_meson,
                                python_meson_driver_inputs,
                            )

                            host_packages = {}
                            for item in build.get("host_header_packages", {}).values():
                                tool = sdk_context.host_tools[item["tool"]]
                                if tool.receipt_path is None:
                                    raise ValueError("Python Meson host headers require a complete package receipt")
                                proof = json.loads(tool.receipt_path.read_text())
                                host_packages[item["tool"]] = (tool.receipt_path.parent / proof["root"]).resolve()
                    if adapter in {"meson", "python-meson"} and workspaces and port.reference in workspaces:
                        from ports._support.meson_workspace import retained_meson

                        if build.get("hooks"):
                            raise ValueError(
                                "retained Meson workspaces require source patches rather than mutable hooks"
                            )
                        compilation = {
                            name: build[name]
                            for name in (
                                "configure_args",
                                "configure_environment",
                                "cross_properties",
                                "dependency_properties",
                                "host_header_packages",
                                "install_prefix",
                            )
                            if name in build
                        }
                        host_code = {}
                        for name in (
                            "python",
                            "meson",
                            "ninja",
                            "cython",
                            "f2py",
                            "pybind11-config",
                            "pkg-config",
                            "sh",
                        ):
                            if name in sdk_context.host_tools:
                                tool = sdk_context.host_tools[name]
                                host_code[name] = {"path": str(tool.path), "sha256": tool.sha256}
                                if tool.receipt_path is not None:
                                    proof = json.loads(tool.receipt_path.read_text())
                                    host_code[name]["files"] = proof["files"]
                        product_inputs = {
                            "compiler": build_sdk.llvm.sha256,
                            "platform": build_sdk.sysroot.sha256,
                            "sdk": build_sdk.sdk.sha256,
                        }
                        if sdk_context.python is not None:
                            product_inputs["python"] = {
                                "source": sdk_context.python.source_sha256,
                                "headers": sdk_context.python.headers_sha256,
                                "pyconfig": sdk_context.python.pyconfig_sha256,
                            }
                        context = workspace_stack.enter_context(
                            retained_meson(
                                context,
                                workspaces[port.reference],
                                compilation,
                                product_inputs,
                                host_code,
                                driver_inputs=(
                                    partial(
                                        python_meson_driver_inputs,
                                        cpython=python_context,
                                        recipe=recipe,
                                        host_packages=host_packages,
                                    )
                                )
                                if adapter == "python-meson"
                                else partial(_native_driver_inputs, build=build, jobs=jobs),
                            )
                        )
                        # The sealed result inventories this exact compilation receipt.
                        shutil.copyfile(
                            context.build.parent / ".meson-workspace.json",
                            slot.result / "meson-workspace-receipt.json",
                        )
                    _hooks(port, "before_build", context, sdk_context)
                    if build["adapter"] in {"python-extension", "python-meson", "python-pep517"}:
                        if adapter == "python-meson":
                            output = build_python_meson(
                                PythonMesonBuildRequest(context, python_context, recipe, host_packages)
                            )
                        elif adapter == "python-pep517":
                            from ports._support.python_pep517 import PEP517BuildRequest, build_pep517

                            version = ".".join(python.version.split(".")[:2])
                            configs = [
                                name
                                for name in sdk_context.runtime.contents["files"]
                                if name.startswith(f"/usr/lib/python{version}/_sysconfigdata_") and name.endswith(".py")
                            ]
                            if len(configs) != 1:
                                raise ValueError("target runtime must export one admitted sysconfig data file")
                            config = python.runtime_bundle / "rootfs" / configs[0].lstrip("/")
                            if file_hash(config) != sdk_context.runtime.contents["files"][configs[0]]:
                                raise ValueError("target sysconfig differs from admitted runtime")
                            output = build_pep517(
                                PEP517BuildRequest(
                                    context,
                                    python_context,
                                    config,
                                    recipe,
                                    _backend_wheels(port, selected, results),
                                )
                            )
                            shutil.copyfile(
                                output.staging_prefix / "pep517-receipt.json", slot.result / "pep517-receipt.json"
                            )
                        else:
                            output = build_extension(ExtensionBuildRequest(context, python_context, recipe))
                        _hooks(port, "after_install", context, sdk_context)
                        if adapter == "python-meson" or build.get("output", "wheel") == "wheel":
                            shutil.copytree(output.staging_prefix / "wheels", slot.result / "wheels")
                        if (adapter == "python-meson" or build.get("output") == "stdlib") and (
                            recipe.get("exports") or recipe.get("export_directories")
                        ):
                            seal_native_install(
                                recipe,
                                port.directory,
                                output.staging_prefix,
                                slot.result / "native",
                                target,
                                direct,
                                native,
                                {item.port: native[item.port] for item in port.dependencies if item.kind == "runtime"},
                            )
                    else:
                        if adapter == "llvm-guest-sdk":
                            from ports.toolchain.llvm.guest import install_guest_sdk

                            output = install_guest_sdk(context, recipe)
                        elif adapter == "llvm-guest":
                            from ports.toolchain.llvm.guest import build_guest

                            output = build_guest(context, recipe, compiler, jobs=jobs)
                        else:
                            output = build_native(native_build_request(context, build, jobs))
                        _hooks(port, "after_install", context, sdk_context)
                        _source_exports(port, context)
                        seal_native_install(
                            recipe,
                            port.directory,
                            output.staging_prefix,
                            slot.result / "native",
                            target,
                            direct,
                            native,
                            {item.port: native[item.port] for item in port.dependencies if item.kind == "runtime"},
                        )
        result = store.resolve() / "results" / keys[port.reference]
        results[port.reference] = result
        if (result / "native").exists():
            from ports.native.dependencies import verify_artifact

            native[PurePosixPath(port.reference).parts[0] + "/" + port.name] = NativeArtifact(
                result / "native", verify_artifact(result / "native")
            )
    return GraphBuild(graph, results, sdks)


def publish_graph(build: GraphBuild, sdk_context: MaterializedSDK, output: Path) -> Path:
    """Seal wheels and shared providers into one locally installable release.

    Existing catalog and release validators enforce runtime compatibility and
    the complete shared-provider closure before publishing the directory.
    """
    from ports._support.catalog import compose
    from ports._support.native_catalog import publish_native_catalog
    from ports._support.wasm_metadata import needed_libraries
    from ports.native.dependencies import verify_artifact
    from ports.python.cpython.release import build_release

    if sdk_context.python is None:
        raise ValueError("graph release requires a CPython runtime")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ports-catalog-", dir=output.parent) as temporary:
        runtime = sdk_context.python.runtime_bundle
        incorporated_providers = frozenset()
        stdlib_ports = [
            port for port in guest_graph(build.graph).ports if port.recipe["build"].get("output") == "stdlib"
        ]
        if stdlib_ports:
            from ports._support.native_artifacts import NativeArtifact
            from ports.python.cpython.graph_assembly import assemble_stdlib

            artifacts = {
                port.reference.split("/", 1)[0] + "/" + port.name: NativeArtifact(
                    build.results[port.reference] / "native", verify_artifact(build.results[port.reference] / "native")
                )
                for port in guest_graph(build.graph).ports
                if (build.results[port.reference] / "native").is_dir()
            }
            modules = {
                port.reference.split("/", 1)[0] + "/" + port.name: artifacts[
                    port.reference.split("/", 1)[0] + "/" + port.name
                ]
                for port in stdlib_ports
            }
            runtime, incorporated_providers = assemble_stdlib(
                runtime,
                modules,
                artifacts,
                _native_target(build.sdks[stdlib_ports[0].reference]),
                Path(temporary) / "runtime",
                runtime_manifest_sha256=sdk_context.python.runtime_manifest_sha256,
            )
        raw = Path(temporary) / "raw"
        (raw / "wheels").mkdir(parents=True)
        (raw / "providers").mkdir()
        packages, providers = [], []
        for port in guest_graph(build.graph).ports:
            if port.role in {"host-tool", "target-platform"}:
                continue
            result = build.results[port.reference]
            for wheel in sorted((result / "wheels").glob("*.whl")):
                destination = raw / "wheels" / wheel.name
                if destination.exists():
                    raise ValueError("graph wheel filenames conflict")
                shutil.copyfile(wheel, destination)
                packages.append(
                    {
                        "name": port.name,
                        "version": port.version,
                        "wheel": "wheels/" + wheel.name,
                        "sha256": file_hash(wheel),
                    }
                )
            if (result / "native").exists():
                artifact = verify_artifact(result / "native")
                for relative in artifact["inputs"]["recipe"]["exports"].get("shared_libraries", []):
                    path = result / "native" / relative
                    if port.recipe["build"].get("output") == "stdlib":
                        continue
                    if path.name in incorporated_providers:
                        if file_hash(path) != file_hash(runtime / "rootfs/lib" / path.name):
                            raise ValueError("graph provider conflicts with assembled stdlib runtime")
                        # Wheels and other providers still need an explicit catalog closure.
                    destination = raw / "providers" / path.name
                    if destination.exists():
                        raise ValueError("graph shared provider names conflict")
                    shutil.copyfile(path, destination)
                    providers.append(
                        {
                            "name": path.name,
                            "path": "providers/" + path.name,
                            "destination": "/lib/" + path.name,
                            "sha256": file_hash(path),
                            "native_dependencies": needed_libraries(path),
                        }
                    )
        catalog = {
            "schema_version": 1,
            "abi": sdk_context.dynamic_abi,
            "target": "wasm32-wasip1",
            "python_version": sdk_context.python.version,
            "packages": packages,
            "native_providers": providers,
        }
        (raw / "catalog.json").write_text(json.dumps(catalog, sort_keys=True, indent=2) + "\n")
        combined = compose(runtime, [raw], Path(temporary) / "catalog")
        native_catalog = publish_native_catalog(guest_graph(build.graph), build.results, Path(temporary) / "native")
        return build_release(runtime, combined, sdk_context.tool("uv"), output, native_catalog=native_catalog)


def accept_graph(build: GraphBuild, sdk_context: MaterializedSDK, descriptor: Path, output: Path) -> None:
    """Run every selected port's declared guest checks against the graph release."""
    from ports._support.acceptance import AcceptanceRequest, accept_port
    from ports._support.native_artifacts import NativeArtifact, merge_dependency_sysroot
    from ports.native.dependencies import verify_artifact

    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    native = {}
    for port in guest_graph(build.graph).ports:
        if port.role in {"host-tool", "target-platform"}:
            continue
        prefix = build.results[port.reference] / "native"
        if prefix.exists():
            native[port.reference.split("/", 1)[0] + "/" + port.name] = NativeArtifact(prefix, verify_artifact(prefix))
    for port in guest_graph(build.graph).ports:
        if port.role in {"host-tool", "target-platform"}:
            continue
        build_sdk = build.sdks[port.reference]
        proof = output / Path(port.reference).with_suffix("")
        identity = port.reference.split("/", 1)[0] + "/" + port.name
        kind = "native" if identity in native else "pypi"
        if kind == "native" and (build.results[port.reference] / "wheels").exists():
            kind = "pypi+native"
        dependencies = None
        if kind in {"native", "pypi+native"}:
            if identity not in native:
                raise ValueError(f"native port has no sealed artifact: {port.reference}")
            if any(test["kind"] == "native" for test in port.recipe.get("tests", [])):
                dependencies = proof.parent / (proof.name + "-dependencies")
                target = _native_target(build_sdk)
                merge_dependency_sysroot({identity: native[identity]}, native, dependencies, target)
        accept_port(AcceptanceRequest(port, descriptor, proof, kind, build_sdk, dependencies))
    (output / "graph.json").write_text(
        json.dumps(
            {
                "release_sha256": file_hash(descriptor),
                "resolved_toolchains": {
                    reference: resolved.toolchain_receipt for reference, resolved in sorted(build.sdks.items())
                },
                "recipes": {port.reference: port.digest for port in build.graph.ports},
                "builds": {name: path.name for name, path in build.results.items()},
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    from ports._support.sdk import materialize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipes", nargs="+")
    parser.add_argument("--ports", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sdk", default="default", help="default SDK for unopinionated target ports")
    parser.add_argument("--host-seed", type=Path, help="explicit native host tools; no target products")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        action="append",
        default=[],
        metavar="RECIPE=PATH",
        help="actual compatible retained Ninja build directory; producer state lives in its parent",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--jobs", type=int, help="override each recipe's build parallelism")
    parser.add_argument("--output", type=Path, help="seal a locally installable graph release")
    parser.add_argument("--check", action="store_true", help="install and run every declared guest probe")
    args = parser.parse_args()
    if args.check and args.output is None:
        parser.error("--check requires --output")
    workspaces = {}
    for declaration in args.workspace:
        reference, separator, directory = declaration.partition("=")
        if not separator or not reference or not directory:
            parser.error("--workspace requires RECIPE=PATH")
        reference = relative_path(reference)
        if not reference.endswith(".json"):
            reference += "/recipe.json"
        if reference in workspaces:
            parser.error("duplicate --workspace recipe")
        workspaces[reference] = Path(directory)
    graph = plan(args.ports, args.recipes, default_sdk=args.sdk)
    needs_target = any(port.recipe["build"]["adapter"] not in {"pure-wheel", "host-wheel"} for port in graph.ports)
    needs_python = args.output is not None or any(
        port.recipe["build"]["adapter"] in {"python-extension", "python-pep517", "python-meson"} for port in graph.ports
    )
    if args.check:
        for port in guest_graph(graph).ports:
            if port.role in {"host-tool", "target-platform"}:
                continue
            tests = port.recipe.get("tests")
            if not isinstance(tests, list) or not 1 <= len(tests) <= 32:
                raise ValueError(f"port has no declared guest acceptance: {port.reference}")
            for test in tests:
                if not isinstance(test, dict) or test.get("kind") not in {"python", "native", "shell"}:
                    raise ValueError("unsupported guest test kind")
                _local_file(port, test.get("source" if test["kind"] == "native" else "script"))
    sdk_context = (
        materialize(
            args.ports,
            graph,
            args.store,
            default=args.sdk,
            host_seed=args.host_seed,
            python=needs_python,
            offline=args.offline,
        )
        if needs_target or needs_python
        else None
    )

    result = build_graph(
        args.ports,
        args.recipes,
        sdk_context,
        args.store,
        offline=args.offline,
        jobs=args.jobs,
        default_sdk=args.sdk,
        workspaces=workspaces,
    )
    if args.output is not None:
        if args.check:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix=".ports-check-", dir=args.output.parent))
            try:
                descriptor = publish_graph(result, sdk_context, work / "release")
                accept_graph(result, sdk_context, descriptor, work / "release/acceptance")
                (work / "release").rename(args.output)
            except Exception as error:
                error.add_note(f"graph acceptance work retained at {work}")
                raise
            work.rmdir()
        else:
            publish_graph(result, sdk_context, args.output)
        print(args.output / "release.json")
    print(json.dumps({name: str(path) for name, path in result.results.items()}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
