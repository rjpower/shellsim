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
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Mapping, Sequence

from ports._support.build import apply_patch
from ports._support.graph import Graph, Port, plan
from ports._support.native_adapters import NativeAdapter, NativeBuildContext, NativeBuildRequest, build_native
from ports._support.python_adapters import (
    CPythonBuildContext,
    ExtensionBuildRequest,
    PureWheelBuildRequest,
    build_extension,
    build_pure_wheel,
)
from ports._support.store import build_slot, extract, fetch, file_hash, relative_path

_COMMON_BUILD_MODULES = ("runner.py", "graph.py", "store.py", "cohort.py", "local_sources.py")
_NATIVE_BUILD_MODULES = (
    "native_adapters.py",
    "native_artifacts.py",
    "wasm.py",
    "wasm_metadata.py",
    "native/dependencies.py",
)
_PURE_BUILD_MODULES = ("python_adapters.py", "pure_wheel.py")

if TYPE_CHECKING:
    from ports._support.cohort import BuildCohort, CompilerBootstrap, PlatformBootstrap


@dataclass(frozen=True)
class GraphBuild:
    """Dependency-first graph and immutable result directories by recipe reference."""

    graph: Graph
    results: dict[str, Path]


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
        "python-extension",
        "llvm-guest",
        "llvm-guest-sdk",
        *(item.value for item in NativeAdapter),
    }:
        raise ValueError(f"recipe has no supported build adapter: {port.reference}")
    if adapter not in {"pure-wheel", "llvm-guest", "llvm-guest-sdk"} and "files" not in port.recipe["source"]:
        relative_path(port.recipe["source"]["subdirectory"])
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


def _hooks(port: Port, phase: str, context: NativeBuildContext, cohort: BuildCohort) -> None:
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
    environment["PATH"] = os.pathsep.join(sorted({str(tool.path.parent) for tool in cohort.host_tools.values()}))
    for index, hook in enumerate(hooks):
        with (context.build.parent / f"hook-{phase}-{index}.log").open("wb") as log:
            subprocess.run(
                [str(cohort.tool("python")), str(_local_file(port, hook["file"])), str(path)],
                cwd=context.source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def _strings(build: dict, field: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    values = build.get(field, list(default))
    if not isinstance(values, list) or len(values) > 256 or any(not isinstance(value, str) for value in values):
        raise ValueError(f"adapter {field} must be a bounded string array")
    return tuple(values)


def _build_implementation(support: Path, adapter: str) -> dict[str, str]:
    """Bind cached outputs to the code that actually stages and seals them."""
    modules = [*_COMMON_BUILD_MODULES]
    if adapter == "pure-wheel":
        modules.extend(_PURE_BUILD_MODULES)
    elif adapter in {"llvm-guest", "llvm-guest-sdk"}:
        modules.extend(("build.py", "toolchain/llvm/guest.py", *_NATIVE_BUILD_MODULES))
    elif adapter == "python-extension":
        modules.extend(("build.py", "python_adapters.py", *_NATIVE_BUILD_MODULES))
    else:
        modules.extend(("build.py", *_NATIVE_BUILD_MODULES))
    return {
        name: file_hash(support.parent / name if name.startswith(("native/", "toolchain/")) else support / name)
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
    cohort: BuildCohort,
    store: Path,
    *,
    offline: bool = False,
    jobs: int | None = None,
    bootstrap: CompilerBootstrap | None = None,
    platform_bootstrap: PlatformBootstrap | None = None,
    workspaces: Mapping[str, Path] | None = None,
) -> GraphBuild:
    """Build each selected recipe once, reusing only byte-verified cache entries."""
    from ports._support.native_artifacts import (
        NativeArtifact,
        NativeTarget,
        merge_dependency_sysroot,
        seal_native_install,
    )

    if jobs is not None and not 1 <= jobs <= 16:
        raise ValueError("build jobs must be between one and sixteen")
    graph = plan(ports, requests, target_profile=cohort.dynamic_abi)
    selected = {port.reference: port for port in graph.ports}
    for reference in workspaces or {}:
        if reference not in selected or selected[reference].recipe["build"]["adapter"] != "llvm-guest":
            raise ValueError("explicit workspace requires a selected persistent producer: " + reference)
    local_inputs = {port.reference: _admit_recipe(port) for port in graph.ports}
    from ports._support.local_sources import local_source_files, stage_local_sources

    for port in graph.ports:
        if "files" in port.recipe.get("source", {}):
            local_source_files(ports, port.recipe["source"])
    for port in graph.ports:
        if port.recipe["build"]["adapter"] in {"pure-wheel", "llvm-host", "wasi-sysroot"}:
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
        adapter: _build_implementation(support, adapter)
        for adapter in {port.recipe["build"]["adapter"] for port in graph.ports} - {"llvm-host", "wasi-sysroot"}
    }
    results: dict[str, Path] = {}
    native: dict[str, NativeArtifact] = {}
    keys: dict[str, str] = {}
    products = {}
    target = NativeTarget(
        cohort.target,
        cohort.dynamic_abi,
        cohort.dynamic_abi,
        {"cohort": cohort.identity, "target": cohort.target, "abi": cohort.dynamic_abi},
    )
    for port in graph.ports:
        adapter = port.recipe["build"]["adapter"]
        if adapter in {"llvm-host", "wasi-sysroot"}:
            from ports._support.cohort import resolve_compiler, resolve_platform, verify_product
            from ports._support.store import identity

            product = (
                resolve_compiler(cohort, bootstrap)
                if adapter == "llvm-host"
                else resolve_platform(cohort, products["toolchain/llvm/host-recipe.json"], platform_bootstrap)
            )
            verify_product(product)
            producer_path = _local_file(port, port.recipe["build"]["producer_recipe"])
            if product.contents["identity"]["recipe"] != json.loads(producer_path.read_text()):
                raise ValueError("graph product producer differs from declared recipe")
            products[port.reference] = product
            results[port.reference] = product.root
            keys[port.reference] = identity({"recipe": port.digest, "product": product.sha256})
            continue
        build_cohort = cohort
        if adapter != "pure-wheel":
            from ports._support.cohort import resolved_toolchain

            sysroot = next(products[item.recipe] for item in port.dependencies if item.kind == "platform")
            compiler = next(
                products[item.recipe]
                for item in port.dependencies
                if item.kind == "build" and item.recipe == "toolchain/llvm/host-recipe.json"
            )
            build_cohort = resolved_toolchain(cohort, compiler, sysroot)
        inputs = {
            "recipe_sha256": port.digest,
            "local_inputs": local_inputs[port.reference],
            "implementation": implementations[adapter],
            "cohort": cohort.identity,
            "dependencies": {
                dependency.kind + ":" + dependency.port: keys[dependency.recipe] for dependency in port.dependencies
            },
        }
        if adapter not in {"pure-wheel", "llvm-guest-sdk"}:
            inputs["target_flags"] = {
                "compiler": list(build_cohort.compiler_flags),
                "linker": list(build_cohort.linker_flags),
                "shared_library": list(build_cohort.shared_library_flags),
                "executable": list(build_cohort.executable_flags),
            }
        with build_slot(store, inputs) as slot:
            keys[port.reference] = slot.key
            if not slot.cached:
                recipe, build = port.recipe, port.recipe["build"]
                source = (
                    stage_local_sources(ports, recipe["source"], slot.work / "source")
                    if "files" in recipe["source"]
                    else fetch(recipe["source"], store / "sources", offline=offline)
                )
                if build["adapter"] == "pure-wheel":
                    build_pure_wheel(PureWheelBuildRequest(source, slot.result, recipe))
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
                    if adapter == "llvm-guest":
                        from ports._support.store import identity

                        compatibility = {
                            "source": recipe["source"]["sha256"],
                            "patches": recipe.get("patches", []),
                            "compiler": {
                                name: compiler.contents["artifacts"]["bin/" + name]
                                for name in (
                                    "clang",
                                    "llvm-ar",
                                    "llvm-objcopy",
                                    "llvm-tblgen",
                                    "llvm-min-tblgen",
                                    "clang-tblgen",
                                )
                            },
                            "platform": {
                                path: digest
                                for path, digest in sysroot.contents["artifacts"].items()
                                if path.startswith("sysroot/include/")
                            },
                            "resources": {
                                path: digest
                                for path, digest in cohort.sdk.contents.items()
                                if path.startswith("lib/clang/23/include/")
                            },
                            "dependencies": {
                                name: {
                                    path: digest
                                    for path, digest in artifact.manifest["files"].items()
                                    if path.startswith("include/")
                                }
                                for name, artifact in direct.items()
                            },
                            "host_tools": {name: cohort.host_tools[name].sha256 for name in ("cmake", "ninja")},
                        }
                        workspace = (
                            workspaces[port.reference].resolve()
                            if workspaces is not None and port.reference in workspaces
                            else store.resolve() / "workspaces" / "llvm-guest" / identity(compatibility)
                        )
                    prefix = workspace / "dependencies"
                    if adapter == "llvm-guest" and prefix.exists():
                        shutil.rmtree(prefix)
                    merge_dependency_sysroot(direct, native, prefix, target)
                    context = NativeBuildContext(
                        source=source,
                        build=workspace / "build",
                        staging_prefix=slot.work / "install",
                        sdk=build_cohort.sdk.root,
                        compiler_prefix=build_cohort.llvm.root,
                        sysroot=build_cohort.sysroot.root / "sysroot",
                        target=cohort.target,
                        compiler_flags=(*build_cohort.compiler_flags, "-I" + str(prefix / "usr/local/include"))
                        if direct
                        else build_cohort.compiler_flags,
                        linker_flags=(*build_cohort.linker_flags, "-L" + str(prefix / "usr/local/lib"))
                        if direct
                        else build_cohort.linker_flags,
                        dependencies={name: artifact.prefix for name, artifact in direct.items()},
                        host_tools={name: tool.path for name, tool in cohort.host_tools.items()},
                        target_tools={name: tool.path for name, tool in build_cohort.target_tools.items()},
                        dependency_sysroot=prefix,
                        shared_library_flags=build_cohort.shared_library_flags,
                        executable_flags=build_cohort.executable_flags,
                    )
                    _hooks(port, "before_build", context, cohort)
                    if build["adapter"] == "python-extension":
                        python = cohort.python
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
                        output = build_extension(ExtensionBuildRequest(context, python_context, recipe))
                        _hooks(port, "after_install", context, cohort)
                        shutil.copytree(output.staging_prefix / "wheels", slot.result / "wheels")
                    else:
                        if adapter == "llvm-guest-sdk":
                            from ports.toolchain.llvm.guest import install_guest_sdk

                            output = install_guest_sdk(context, recipe)
                        elif adapter == "llvm-guest":
                            from ports.toolchain.llvm.guest import build_guest

                            output = build_guest(context, recipe, compiler)
                        else:
                            output = build_native(
                                NativeBuildRequest(
                                    NativeAdapter(build["adapter"]),
                                    context,
                                    _strings(build, "configure_args"),
                                    _strings(build, "build_targets"),
                                    _strings(build, "install_targets", ("install",)),
                                    jobs if jobs is not None else build.get("jobs", 1),
                                    PurePosixPath(build.get("install_prefix", "/usr/local")),
                                    build.get("configure_environment", {}),
                                    _strings(build, "build_args"),
                                )
                            )
                        _hooks(port, "after_install", context, cohort)
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
    return GraphBuild(graph, results)


def publish_graph(build: GraphBuild, cohort: BuildCohort, output: Path) -> Path:
    """Seal wheels and shared providers into one locally installable release.

    Existing catalog and release validators enforce runtime compatibility and
    the complete shared-provider closure before publishing the directory.
    """
    from ports._support.catalog import compose
    from ports._support.native_catalog import publish_native_catalog
    from ports._support.wasm_metadata import needed_libraries
    from ports.native.dependencies import verify_artifact
    from ports.python.cpython.release import build_release

    if cohort.python is None:
        raise ValueError("graph release requires a CPython runtime")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ports-catalog-", dir=output.parent) as temporary:
        raw = Path(temporary) / "raw"
        (raw / "wheels").mkdir(parents=True)
        (raw / "providers").mkdir()
        packages, providers = [], []
        for port in build.graph.ports:
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
            "abi": cohort.dynamic_abi,
            "target": "wasm32-wasip1",
            "python_version": cohort.python.version,
            "packages": packages,
            "native_providers": providers,
        }
        (raw / "catalog.json").write_text(json.dumps(catalog, sort_keys=True, indent=2) + "\n")
        combined = compose(cohort.python.runtime_bundle, [raw], Path(temporary) / "catalog")
        native_catalog = publish_native_catalog(build.graph, build.results, Path(temporary) / "native")
        return build_release(
            cohort.python.runtime_bundle, combined, cohort.tool("uv"), output, native_catalog=native_catalog
        )


def accept_graph(build: GraphBuild, cohort: BuildCohort, descriptor: Path, output: Path) -> None:
    """Run every selected port's declared guest checks against the graph release."""
    from ports._support.acceptance import AcceptanceRequest, accept_port
    from ports._support.native_artifacts import NativeArtifact, NativeTarget, merge_dependency_sysroot
    from ports.native.dependencies import verify_artifact

    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    native = {}
    for port in build.graph.ports:
        if port.role in {"host-tool", "target-platform"}:
            continue
        prefix = build.results[port.reference] / "native"
        if prefix.exists():
            native[port.reference.split("/", 1)[0] + "/" + port.name] = NativeArtifact(prefix, verify_artifact(prefix))
    target = NativeTarget(
        cohort.target,
        cohort.dynamic_abi,
        cohort.dynamic_abi,
        {"cohort": cohort.identity, "target": cohort.target, "abi": cohort.dynamic_abi},
    )
    for port in build.graph.ports:
        if port.role in {"host-tool", "target-platform"}:
            continue
        proof = output / Path(port.reference).with_suffix("")
        identity = port.reference.split("/", 1)[0] + "/" + port.name
        kind = "native" if identity in native else "pypi"
        dependencies = None
        if kind == "native":
            if identity not in native:
                raise ValueError(f"native port has no sealed artifact: {port.reference}")
            if any(test["kind"] == "native" for test in port.recipe.get("tests", [])):
                dependencies = proof.parent / (proof.name + "-dependencies")
                merge_dependency_sysroot({identity: native[identity]}, native, dependencies, target)
        accept_port(AcceptanceRequest(port, descriptor, proof, kind, cohort, dependencies))
    (output / "graph.json").write_text(
        json.dumps(
            {
                "release_sha256": file_hash(descriptor),
                "cohort": cohort.identity,
                "recipes": {port.reference: port.digest for port in build.graph.ports},
                "builds": {name: path.name for name, path in build.results.items()},
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    from ports._support.cohort import load_cohort

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipes", nargs="+")
    parser.add_argument("--ports", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--platform-bootstrap", type=Path, help="explicit pinned SDK/libc archives and producer workspace"
    )
    parser.add_argument("--bootstrap", type=Path, help="explicit pinned host LLVM seed inputs")
    parser.add_argument(
        "--workspace",
        action="append",
        default=[],
        metavar="RECIPE=PATH",
        help="explicit compatible persistent producer workspace",
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
    cohort = load_cohort(args.cohort)
    if args.check:
        for port in plan(args.ports, args.recipes, target_profile=cohort.dynamic_abi).ports:
            if port.role in {"host-tool", "target-platform"}:
                continue
            tests = port.recipe.get("tests")
            if not isinstance(tests, list) or not 1 <= len(tests) <= 32:
                raise ValueError(f"port has no declared guest acceptance: {port.reference}")
            for test in tests:
                if not isinstance(test, dict) or test.get("kind") not in {"python", "native", "shell"}:
                    raise ValueError("unsupported guest test kind")
                _local_file(port, test.get("source" if test["kind"] == "native" else "script"))
    from ports._support.cohort import load_bootstrap, load_platform_bootstrap

    result = build_graph(
        args.ports,
        args.recipes,
        cohort,
        args.store,
        offline=args.offline,
        jobs=args.jobs,
        bootstrap=load_bootstrap(args.bootstrap) if args.bootstrap else None,
        platform_bootstrap=load_platform_bootstrap(args.platform_bootstrap) if args.platform_bootstrap else None,
        workspaces=workspaces,
    )
    if args.output is not None:
        if args.check:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix=".ports-check-", dir=args.output.parent))
            try:
                descriptor = publish_graph(result, cohort, work / "release")
                accept_graph(result, cohort, descriptor, work / "release/acceptance")
                (work / "release").rename(args.output)
            except Exception as error:
                error.add_note(f"graph acceptance work retained at {work}")
                raise
            work.rmdir()
        else:
            publish_graph(result, cohort, args.output)
        print(args.output / "release.json")
    print(json.dumps({name: str(path) for name, path in result.results.items()}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
