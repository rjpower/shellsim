"""Build recipe dependency graphs through admitted adapters and a verified store.

Static recipes own graph choices. Port functions build admitted inputs; this
driver publishes complete results with exact dependency and compiler identities.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Mapping, Sequence

from ports._support.build import apply_patch
from ports._support.graph import Graph, Port, guest_graph, plan
from ports._support.native_adapters import (
    CompilerCacheLauncher,
    NativeBuildContext,
)
from ports._support.python_adapters import (
    CPythonBuildContext,
)
from ports._support.store import build_slot, extract, fetch, file_hash, relative_path

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
    admitted_predecessors: Mapping[str, str] | None = None,
    compiler_cache: CompilerCacheLauncher | None = None,
) -> GraphBuild:
    """Execute admitted port builders and publish byte-verified immutable results.

    Transferred predecessors must have their current key and a verified cached
    result. They never build on a miss, including when implementation bytes change.
    """
    from ports._support.local_sources import stage_local_sources
    from ports._support.native_artifacts import NativeArtifact, merge_dependency_sysroot, seal_native_install
    from ports._support.sdk_products import resolved_toolchain, verify_product
    from ports._support.store import identity, verify
    from ports.api import BuildContext, build_port, implementation

    if jobs is not None and not 1 <= jobs <= 16:
        raise ValueError("build jobs must be between one and sixteen")
    graph = plan(
        ports, requests, target_profile=sdk_context.dynamic_abi if sdk_context else None, default_sdk=default_sdk
    )
    selected = {port.reference: port for port in graph.ports}
    if set(admitted_predecessors or {}) - selected.keys():
        raise ValueError("admitted predecessor is outside the requested graph")
    for reference in workspaces or {}:
        if reference not in selected or selected[reference].recipe["build_system"] not in {
            "llvm-guest",
            "meson",
            "python-meson",
        }:
            raise ValueError("explicit workspace requires a selected persistent producer: " + reference)
    local_inputs = {port.reference: _admit_recipe(port) for port in graph.ports}
    implementations = {port.reference: implementation(port) for port in graph.ports}
    product_systems = {"llvm-host", "wasi-sysroot", "sdk-tooling", "cpython-threaded", "uv-host"}
    results, native, keys, products, sdks = {}, {}, {}, {}, {}
    for port in graph.ports:
        recipe, system = port.recipe, port.recipe["build_system"]
        if system in product_systems:
            product = {
                "llvm-host": sdk_context.llvm,
                "wasi-sysroot": sdk_context.sysroot,
                "sdk-tooling": sdk_context.sdk,
                "cpython-threaded": sdk_context.cpython_manifest,
            }.get(system)
            if system == "uv-host":
                from ports._support.sdk_products import Receipt, read_json

                resolver = sdk_context.host_tools["uv"]
                product = Receipt(
                    resolver.path.parent,
                    resolver.receipt_path,
                    resolver.receipt_sha256,
                    read_json(resolver.receipt_path),
                )
            if product is None:
                raise ValueError("selected SDK product was not materialized")
            if system in {"llvm-host", "wasi-sysroot"}:
                verify_product(product)
            products[port.reference] = product
            results[port.reference] = product.root
            keys[port.reference] = identity({"recipe": port.digest, "product": product.sha256})
            if port.reference in (admitted_predecessors or {}):
                if admitted_predecessors[port.reference] != keys[port.reference]:
                    raise ValueError("admitted predecessor key differs: " + port.reference)
            continue
        build_sdk = sdk_context
        pure = system in {"pure-wheel", "host-wheel", "source-tree"}
        if not pure:
            platform_edges = [item for item in port.dependencies if item.kind == "platform"]
            compiler_edges = [
                item
                for item in port.dependencies
                if item.kind == "build" and selected[item.recipe].recipe["build_system"] == "llvm-host"
            ]
            if len(platform_edges) != 1 or len(compiler_edges) != 1:
                raise ValueError("native build requires one explicit compiler and platform dependency")
            compiler, sysroot = products[compiler_edges[0].recipe], products[platform_edges[0].recipe]
            build_sdk = resolved_toolchain(sdk_context, compiler, sysroot)
            target = _native_target(build_sdk)
        sdks[port.reference] = build_sdk
        executable_inputs = _executable_sdk_link_inputs(build_sdk, recipe) if not pure else ()
        driver_files = ["runner.py", "store.py"]
        if "files" in recipe["source"]:
            driver_files.append("local_sources.py")
        if recipe.get("patches"):
            driver_files.append("build.py")
        inputs = {
            "recipe_sha256": port.digest,
            "local_inputs": local_inputs[port.reference],
            "implementation": implementations[port.reference],
            "driver": {name: file_hash(Path(__file__).with_name(name)) for name in driver_files},
            "dependencies": {item.kind + ":" + item.port: keys[item.recipe] for item in port.dependencies},
        }
        if port.sdk_selection is not None:
            inputs["sdk_selection"] = asdict(port.sdk_selection)
        if not pure:
            if compiler_cache is not None:
                from ports._support.native_adapters import compiler_cache_identity

                inputs["compiler_cache"] = compiler_cache_identity(compiler_cache)
            inputs["sdk"] = build_sdk.identity
            inputs["host_tools"] = {
                name: {"sha256": tool.sha256, "receipt": tool.receipt_sha256}
                for name, tool in build_sdk.host_tools.items()
                if name != "uv"
            }
            inputs["target_flags"] = {
                "compiler": build_sdk.compiler_flags,
                "linker": build_sdk.linker_flags,
                "shared_library": build_sdk.shared_library_flags,
                "executable": build_sdk.executable_flags,
                "compiler_runtime": file_hash(build_sdk.compiler_runtime_archive),
                "executable_link_inputs": {str(path): file_hash(path) for path in executable_inputs},
            }
            if system.startswith("python-"):
                python = build_sdk.python
                if python is None:
                    raise ValueError("Python builder requires admitted CPython headers and runtime")
                inputs["python"] = {
                    "source": python.source_sha256,
                    "headers": python.headers_sha256,
                    "pyconfig": python.pyconfig_sha256,
                }
        # Cache receipts use JSON arrays for the typed toolchain argument tuples.
        inputs = json.loads(json.dumps(inputs, sort_keys=True))
        if port.reference in (admitted_predecessors or {}):
            expected = admitted_predecessors[port.reference]
            if expected != identity(inputs):
                raise ValueError("admitted predecessor key differs: " + port.reference)
            predecessor = store.resolve() / "results" / expected
            if not predecessor.is_dir():
                raise ValueError("admitted predecessor result is missing: " + port.reference)
            verify(predecessor, inputs)
        with build_slot(store, inputs) as slot:
            keys[port.reference] = slot.key
            print(
                f"ports: result cache {'hit' if slot.cached else 'miss'}: {port.reference}", file=sys.stderr, flush=True
            )
            if not slot.cached:
                source = (
                    stage_local_sources(ports, recipe["source"], slot.work / "source")
                    if "files" in recipe["source"]
                    else fetch(recipe["source"], store / "sources", offline=offline)
                )
                if pure:
                    if system == "source-tree" and "files" not in recipe["source"]:
                        source = extract(source, slot.work / "source", subdirectory=recipe["source"]["subdirectory"])
                        for patch in recipe.get("patches", []):
                            apply_patch(source, _local_file(port, patch["file"]), patch["sha256"])
                    build_port(BuildContext(port, source, slot.result, sdk=build_sdk, jobs=jobs))
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
                            workspaces[port.reference].resolve()
                            if workspaces and port.reference in workspaces
                            else store.resolve() / "workspaces/llvm-guest" / identity(compatibility) / "build"
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
                    output = build_port(
                        BuildContext(
                            port,
                            source,
                            slot.result,
                            sdk=build_sdk,
                            native=context,
                            cpython=python_context,
                            backend_wheels=_backend_wheels(port, selected, results)
                            if system == "python-pep517"
                            else (),
                            jobs=jobs,
                            workspace=workspaces.get(port.reference) if workspaces else None,
                        )
                    )
                    _source_exports(port, context)
                    for receipt_name in ("pep517-receipt.json",):
                        if (output.staging_prefix / receipt_name).is_file():
                            shutil.copyfile(output.staging_prefix / receipt_name, slot.result / receipt_name)
                    if (output.staging_prefix / "wheels").exists():
                        shutil.copytree(output.staging_prefix / "wheels", slot.result / "wheels")
                    if recipe.get("exports") or recipe.get("export_directories"):
                        seal_native_install(
                            recipe,
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
        stdlib_ports = [port for port in guest_graph(build.graph).ports if port.recipe.get("output") == "stdlib"]
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
                    if port.recipe.get("output") == "stdlib":
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
        reference, _, variant = port.reference.partition(":")
        proof = output / Path(reference).with_suffix("")
        if variant:
            proof /= variant
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
    parser.add_argument("--backend", choices=("local", "buildomatic", "iris"), default="local")
    parser.add_argument("--jobs", type=int, help="override each recipe's build parallelism")
    parser.add_argument("--output", type=Path, help="seal a locally installable graph release")
    parser.add_argument("--check", action="store_true", help="install and run every declared guest probe")
    args = parser.parse_args()
    if args.backend != "local" and args.output is not None:
        parser.error(
            "distributed builds emit cache manifests; use ports.buildomatic.ports.publish_manifest for releases"
        )
    if args.check and args.output is None:
        parser.error("--check requires --output")
    workspaces = {}
    for declaration in args.workspace:
        reference, separator, directory = declaration.partition("=")
        if not separator or not reference or not directory:
            parser.error("--workspace requires RECIPE=PATH")
        from ports._support.graph import canonical_reference

        reference = canonical_reference(args.ports, reference)
        if reference in workspaces:
            parser.error("duplicate --workspace recipe")
        workspaces[reference] = Path(directory)
    graph = plan(args.ports, args.recipes, default_sdk=args.sdk)
    needs_target = any(
        port.recipe["build_system"] not in {"pure-wheel", "host-wheel", "source-tree"} for port in graph.ports
    )
    needs_python = args.output is not None or any(
        port.recipe["build_system"]
        in {"python-extension", "python-pep517", "python-meson", "cpython-threaded", "uv-host"}
        for port in graph.ports
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

    execute = build_graph
    if args.backend != "local":
        from ports.buildomatic.ports import run_graph

        execute = run_graph
    result = execute(
        args.ports,
        args.recipes,
        sdk_context,
        args.store,
        offline=args.offline,
        jobs=args.jobs,
        default_sdk=args.sdk,
        workspaces=workspaces,
        **({"backend": args.backend} if args.backend != "local" else {}),
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
