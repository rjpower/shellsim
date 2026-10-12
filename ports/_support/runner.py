"""Build recipe dependency graphs through admitted adapters and a verified store.

Static recipes own graph choices. Port functions build admitted inputs; this
driver publishes complete results with exact dependency and compiler identities.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from ports._support.graph import Graph, guest_graph
from ports._support.node_build import _native_target
from ports._support.store import file_hash

if TYPE_CHECKING:
    from ports._support.sdk_products import MaterializedSDK


@dataclass(frozen=True)
class GraphBuild:
    """Dependency-first graph and immutable result directories by recipe reference."""

    graph: Graph
    results: dict[str, Path]
    sdks: dict[str, MaterializedSDK]


def build_graph(
    ports: Path, requests: Sequence[str], sdk_context, store: Path, *, offline=False, jobs=None, default_sdk="default"
) -> GraphBuild:
    """Materialize shared-workflow results; SDK context is only for release/acceptance helpers."""
    from ports.buildomatic.artifacts import install
    from ports.buildomatic.backends.local import LocalBackend
    from ports.buildomatic.nodes import Operation
    from ports.buildomatic.remote_store import StorageStore
    from ports.buildomatic.workflow import run

    canonical = StorageStore(str((store / "store").resolve()))
    result = run(
        ports,
        requests,
        canonical,
        store,
        LocalBackend(canonical, store / "execution"),
        offline=offline,
        jobs=jobs or 16,
        sdk_name=default_sdk,
    )
    paths = {
        node.port.reference: install(canonical, result.records[node.reference]["artifact"], store / "artifacts")
        for node in result.plan.nodes
        if node.operation is Operation.BUILD and node.reference in result.records
    }
    return GraphBuild(result.plan.graph, paths, dict.fromkeys(paths, sdk_context))


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


def main(arguments: Sequence[str] | None = None) -> None:
    from ports.buildomatic.cli import main as ports_main

    ports_main(["build", *(sys.argv[1:] if arguments is None else arguments)])


if __name__ == "__main__":
    main()
