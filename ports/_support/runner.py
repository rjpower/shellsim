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
from typing import TYPE_CHECKING, Sequence

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

_COMMON_BUILD_MODULES = ("runner.py", "graph.py", "store.py")
_NATIVE_BUILD_MODULES = (
    "native_adapters.py",
    "native_artifacts.py",
    "wasm.py",
    "wasm_metadata.py",
    "native/dependencies.py",
)
_PURE_BUILD_MODULES = ("python_adapters.py", "pure_wheel.py")

if TYPE_CHECKING:
    from ports._support.cohort import BuildCohort


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
    if adapter not in {"pure-wheel", "python-extension", *(item.value for item in NativeAdapter)}:
        raise ValueError(f"recipe has no supported build adapter: {port.reference}")
    if adapter != "pure-wheel":
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
    elif adapter == "python-extension":
        modules.extend(("build.py", "python_adapters.py", *_NATIVE_BUILD_MODULES))
    else:
        modules.extend(("build.py", *_NATIVE_BUILD_MODULES))
    return {
        name: file_hash(support.parent / name if name.startswith("native/") else support / name) for name in modules
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
    local_inputs = {port.reference: _admit_recipe(port) for port in graph.ports}
    if any(port.recipe["build"]["adapter"] != "pure-wheel" for port in graph.ports):
        cohort.compiler()  # Refuse an incomplete cohort before fetching any source.
    support = Path(__file__).parent
    implementations = {
        adapter: _build_implementation(support, adapter)
        for adapter in {port.recipe["build"]["adapter"] for port in graph.ports}
    }
    results: dict[str, Path] = {}
    native: dict[str, NativeArtifact] = {}
    keys: dict[str, str] = {}
    target = NativeTarget(
        cohort.target,
        cohort.dynamic_abi,
        cohort.dynamic_abi,
        {"cohort": cohort.identity, "target": cohort.target, "abi": cohort.dynamic_abi},
    )
    for port in graph.ports:
        adapter = port.recipe["build"]["adapter"]
        inputs = {
            "recipe_sha256": port.digest,
            "local_inputs": local_inputs[port.reference],
            "implementation": implementations[adapter],
            "cohort": cohort.identity,
            "dependencies": {dependency.port: keys[dependency.recipe] for dependency in port.dependencies},
        }
        if adapter != "pure-wheel":
            inputs["target_flags"] = {
                "compiler": list(cohort.compiler_flags),
                "linker": list(cohort.linker_flags),
                "shared_library": list(cohort.shared_library_flags),
                "executable": list(cohort.executable_flags),
            }
        with build_slot(store, inputs) as slot:
            keys[port.reference] = slot.key
            if not slot.cached:
                recipe, build = port.recipe, port.recipe["build"]
                source = fetch(recipe["source"], store / "sources", offline=offline)
                if build["adapter"] == "pure-wheel":
                    build_pure_wheel(PureWheelBuildRequest(source, slot.result, recipe))
                else:
                    source = extract(source, slot.work / "source", subdirectory=recipe["source"]["subdirectory"])
                    for patch in recipe.get("patches", []):
                        apply_patch(source, _local_file(port, patch["file"]), patch["sha256"])
                    for dependency in port.dependencies:
                        if dependency.port.startswith("native/") and dependency.port not in native:
                            raise ValueError(f"native dependency has no sealed artifact: {dependency.port}")
                    direct = {
                        dependency.port: native[dependency.port]
                        for dependency in port.dependencies
                        if dependency.port in native
                    }
                    prefix = slot.work / "dependencies"
                    merge_dependency_sysroot(direct, native, prefix, target)
                    context = NativeBuildContext(
                        source=source,
                        build=slot.work / "build",
                        staging_prefix=slot.work / "install",
                        sdk=cohort.sdk.root,
                        compiler_prefix=cohort.llvm.root,
                        sysroot=cohort.sysroot.root / "sysroot",
                        target=cohort.target,
                        compiler_flags=cohort.compiler_flags,
                        linker_flags=cohort.linker_flags,
                        dependencies={name: artifact.prefix for name, artifact in direct.items()},
                        host_tools={name: tool.path for name, tool in cohort.host_tools.items()},
                        target_tools={name: tool.path for name, tool in cohort.target_tools.items()},
                        dependency_sysroot=prefix,
                        shared_library_flags=cohort.shared_library_flags,
                        executable_flags=cohort.executable_flags,
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
                        output = build_native(
                            NativeBuildRequest(
                                NativeAdapter(build["adapter"]),
                                context,
                                _strings(build, "configure_args"),
                                _strings(build, "build_targets"),
                                _strings(build, "install_targets", ("install",)),
                                jobs if jobs is not None else build.get("jobs", 1),
                                PurePosixPath(build.get("install_prefix", "/usr/local")),
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
                        )
        result = store.resolve() / "results" / keys[port.reference]
        results[port.reference] = result
        if (result / "native").exists():
            from ports.native.dependencies import verify_artifact

            native["native/" + port.name] = NativeArtifact(result / "native", verify_artifact(result / "native"))
    return GraphBuild(graph, results)


def publish_graph(build: GraphBuild, cohort: BuildCohort, output: Path) -> Path:
    """Seal wheels and shared providers into one locally installable release.

    Existing catalog and release validators enforce runtime compatibility and
    the complete shared-provider closure before publishing the directory.
    """
    from ports._support.catalog import compose
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
        return build_release(cohort.python.runtime_bundle, combined, cohort.tool("uv"), output)


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
        prefix = build.results[port.reference] / "native"
        if prefix.exists():
            native["native/" + port.name] = NativeArtifact(prefix, verify_artifact(prefix))
    target = NativeTarget(
        cohort.target,
        cohort.dynamic_abi,
        cohort.dynamic_abi,
        {"cohort": cohort.identity, "target": cohort.target, "abi": cohort.dynamic_abi},
    )
    for port in build.graph.ports:
        proof = output / Path(port.reference).with_suffix("")
        kind = "native" if port.reference.startswith("native/") else "pypi"
        dependencies = None
        if kind == "native":
            if "native/" + port.name not in native:
                raise ValueError(f"native port has no sealed artifact: {port.reference}")
            dependencies = proof.parent / (proof.name + "-dependencies")
            merge_dependency_sysroot(
                {"native/" + port.name: native["native/" + port.name]}, native, dependencies, target
            )
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
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--jobs", type=int, help="override each recipe's build parallelism")
    parser.add_argument("--output", type=Path, help="seal a locally installable graph release")
    parser.add_argument("--check", action="store_true", help="install and run every declared guest probe")
    args = parser.parse_args()
    if args.check and args.output is None:
        parser.error("--check requires --output")
    cohort = load_cohort(args.cohort)
    if args.check:
        for port in plan(args.ports, args.recipes, target_profile=cohort.dynamic_abi).ports:
            tests = port.recipe.get("tests")
            if not isinstance(tests, list) or not 1 <= len(tests) <= 32:
                raise ValueError(f"port has no declared guest acceptance: {port.reference}")
            for test in tests:
                if not isinstance(test, dict) or test.get("kind") not in {"python", "native"}:
                    raise ValueError("unsupported guest test kind")
                _local_file(port, test.get("script" if test["kind"] == "python" else "source"))
    result = build_graph(args.ports, args.recipes, cohort, args.store, offline=args.offline, jobs=args.jobs)
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
