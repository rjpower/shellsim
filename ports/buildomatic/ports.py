"""Transport admitted recipe closures through Buildomatic's generic action DAG.

Preparation fetches pinned sources. Workers receive enumerated code and complete
predecessor results, run the ordinary driver offline, and return verified trees.
SDK producers are imported products, never remote bootstrap actions.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from ports._support.graph import Graph, plan
from ports._support.native_adapters import CompilerCacheLauncher, compiler_cache_identity
from ports._support.runner import GraphBuild, _admit_recipe, build_graph
from ports._support.store import fetch, file_hash, identity, relative_path, verify
from ports.api import implementation

if TYPE_CHECKING:
    from ports._support.sdk_products import MaterializedSDK
    from ports.buildomatic import BuildRequest, BuildResult, Store, Worker
    from ports.buildomatic.backends.iris import IrisBackend

PRODUCT_SYSTEMS = frozenset({"llvm-host", "wasi-sysroot", "sdk-tooling", "cpython-threaded", "uv-host"})


@dataclass(frozen=True)
class PreparedGraph:
    """Immutable request plus caller-side metadata needed for release acceptance."""

    graph: Graph
    request: BuildRequest
    action_ids: dict[str, str]
    sdk_context: MaterializedSDK | None
    product_results: dict[str, Path]
    ports: Path
    jobs: int | None
    default_sdk: str
    compiler_cache: CompilerCacheLauncher | None
    worker_identity: dict[str, str]


def _product_results(graph: Graph, sdk: MaterializedSDK | None) -> tuple[dict[str, Path], dict[str, str]]:
    """Bind producer nodes to the same admitted products used by the local runner."""
    results, keys = {}, {}
    for port in graph.ports:
        system = port.recipe["build_system"]
        if system not in PRODUCT_SYSTEMS:
            continue
        if sdk is None:
            raise ValueError("producer nodes require an imported admitted SDK")
        product = {
            "llvm-host": sdk.llvm,
            "wasi-sysroot": sdk.sysroot,
            "sdk-tooling": sdk.sdk,
            "cpython-threaded": sdk.cpython_manifest,
        }.get(system)
        if system == "uv-host":
            from ports._support.sdk_products import Receipt, read_json

            tool = sdk.host_tools["uv"]
            product = Receipt(tool.path.parent, tool.receipt_path, tool.receipt_sha256, read_json(tool.receipt_path))
        if product is None:
            raise ValueError("selected SDK product was not materialized")
        results[port.reference] = product.root
        keys[port.reference] = identity({"recipe": port.digest, "product": product.sha256})
    return results, keys


def code_files(ports: Path, graph: Graph, *, portable: bool = False) -> dict[str, Path]:
    """Enumerate code, recipes and declared inputs, without scanning a worktree.

    Python imports expand the shared runtime closure. Dynamic port helpers come
    only from implementation declarations. Optional backend imports in the CLI
    do not belong to worker code. Every selected input is regular and unlinked.
    """
    shared = Path(__file__).resolve().parents[1]
    ports = ports.resolve()
    files: dict[str, Path] = {}

    def add(name: str, path: Path, boundary: Path) -> None:
        name = relative_path(name)
        if not path.resolve(strict=True).is_relative_to(boundary):
            raise ValueError("code input escapes its admitted tree")
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(boundary)):
            raise ValueError("linked code inputs are unsupported")
        file_hash(path, limit=16 * 1024**2)
        if name in files and file_hash(files[name]) != file_hash(path):
            raise ValueError("selected code conflicts with the shared runtime: " + name)
        files[name] = path

    for name in ("api.py", "_support/runner.py", "_support/store.py", "_support/local_sources.py"):
        add(name, shared / name, shared)
    if portable:
        add("buildomatic/portable.py", shared / "buildomatic/portable.py", shared)
    for port in graph.ports:
        _admit_recipe(port)
        recipe_path = port.reference.partition(":")[0]
        add(recipe_path, ports / recipe_path, ports)
        if port.sdk_selection is not None:
            add(port.sdk_selection.reference, ports / port.sdk_selection.reference, ports)
        for name in implementation(port):
            if name == "builder":
                path = port.directory / "build.py"
                add(path.relative_to(ports).as_posix(), path, ports)
            elif name.startswith("local:"):
                name = name.removeprefix("local:")
                add(name, ports / name, ports)
            else:
                add(name, shared / name, shared)
        for patch in port.recipe.get("patches", []):
            path = port.directory / relative_path(patch["file"])
            add(path.relative_to(ports).as_posix(), path, ports)
        for item in port.recipe.get("build_scripts", []):
            path = port.directory / relative_path(item["file"])
            add(path.relative_to(ports).as_posix(), path, ports)
        for name in port.recipe.get("inputs", {}):
            add(name, shared / relative_path(name), shared)
        for item in port.recipe.get("source", {}).get("files", []):
            add(item["path"], ports / relative_path(item["path"]), ports)
    # The default alias is needed when a recipe selected the named default SDK.
    if any(port.sdk_selection for port in graph.ports) and (ports / "sdks/default.json").exists():
        add("sdks/default.json", ports / "sdks/default.json", ports)
    pending, inspected = list(files), set()
    while pending:
        name = pending.pop()
        if name in inspected or not name.endswith(".py"):
            continue
        inspected.add(name)
        package = "ports." + name.removesuffix(".py").replace("/", ".")
        for node in ast.walk(ast.parse(files[name].read_bytes())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parent = package.split(".")[: -node.level]
                    module = ".".join([*parent, module]).rstrip(".")
                modules = [module, *(module + "." + alias.name for alias in node.names)]
            for module in modules:
                if not module.startswith("ports."):
                    continue
                if module.startswith("ports.buildomatic") and not (portable and module == "ports.buildomatic.portable"):
                    continue
                if ".tests" in module:
                    raise ValueError("worker code cannot import test modules")
                relative = module.removeprefix("ports.").replace(".", "/")
                candidates = (relative + ".py", relative + "/__init__.py")
                for candidate in candidates:
                    if candidate in files:
                        pending.append(candidate)
                        break
                    path = shared / candidate
                    if path.is_file():
                        add(candidate, path, shared)
                        pending.append(candidate)
                        break
    if "_support/producer_migration.py" in files:
        add("_support/producer-migration-v1.json", shared / "_support/producer-migration-v1.json", shared)
    return dict(sorted(files.items()))


def _stage_code(files: Mapping[str, Path], destination: Path) -> None:
    """Copy exactly the manifest, verifying byte and executable-mode preservation."""
    for name, source in files.items():
        output = destination / "ports" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        expected = file_hash(source)
        shutil.copyfile(source, output)
        output.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
        if file_hash(output) != expected:
            raise ValueError("code input changed during preparation: " + name)
    worker = Path(__file__).with_name("port_worker.py")
    shutil.copyfile(worker, destination / "port_worker.py")
    if file_hash(destination / "port_worker.py") != file_hash(worker):
        raise ValueError("worker implementation changed during preparation")


def prepare_graph(
    ports: Path,
    requests: Sequence[str],
    sdk_context: MaterializedSDK | None,
    ports_store: Path,
    blob_store: Store,
    *,
    offline: bool = False,
    jobs: int | None = None,
    default_sdk: str = "default",
    max_workers: int = 1,
    compiler_cache: CompilerCacheLauncher | None = None,
    build_key: str | None = None,
    worker_identity: Mapping[str, str] | None = None,
) -> PreparedGraph:
    """Fetch sources and seal one generic action for every canonical consumer."""
    from ports._support.local_sources import local_source_files
    from ports.buildomatic import Action, BuildRequest, InputMount, ResourceLimits, capture_tree

    graph = plan(
        ports, requests, target_profile=sdk_context.dynamic_abi if sdk_context else None, default_sdk=default_sdk
    )
    if jobs is not None and not 1 <= jobs <= 16:
        raise ValueError("build jobs must be between one and sixteen")
    worker_identity = dict(worker_identity or {})
    if worker_identity.keys() - {"config_sha256", "task_image", "service_id"} or any(
        not isinstance(value, str) or len(value) > 4096 or "\0" in value for value in worker_identity.values()
    ):
        raise ValueError("unsupported public worker identity")
    products, product_keys = _product_results(graph, sdk_context)
    consumers = [port for port in graph.ports if port.recipe["build_system"] not in PRODUCT_SYSTEMS]
    if not consumers:
        raise ValueError("Buildomatic requires at least one consumer recipe")
    action_ids = {port.reference: "port-" + hashlib.sha256(port.reference.encode()).hexdigest() for port in consumers}
    ports_store.mkdir(parents=True, exist_ok=True)
    actions = []
    with tempfile.TemporaryDirectory(prefix=".buildomatic-prepare-", dir=ports_store) as temporary:
        root = Path(temporary)
        code = root / "code"
        code.mkdir()
        _stage_code(code_files(ports, graph, portable=sdk_context is not None), code)
        transported = plan(
            code / "ports",
            graph.roots,
            target_profile=sdk_context.dynamic_abi if sdk_context else None,
            default_sdk=default_sdk,
        )
        if {port.reference: port.digest for port in transported.ports} != {
            port.reference: port.digest for port in graph.ports
        }:
            raise ValueError("recipe closure changed during preparation")
        code_bundle = capture_tree(code, blob_store)
        sdk_mounts = ()
        if sdk_context is not None:
            from ports.buildomatic.portable import export_sdk

            export_sdk(sdk_context, root / "sdk")
            sdk_mounts = (
                InputMount(
                    "sdk",
                    capture_tree(
                        root / "sdk",
                        blob_store,
                        limits=ResourceLimits(output_bytes=32 * 1024**3, max_files=300_000),
                    ),
                ),
            )
        closures: dict[str, set[str]] = {}
        for port in graph.ports:
            closures[port.reference] = {
                reference
                for dependency in port.dependencies
                for reference in (dependency.recipe, *closures[dependency.recipe])
            }
        for port in consumers:
            sources = root / action_ids[port.reference]
            sources.mkdir()
            source = port.recipe["source"]
            if "files" in source:
                local_source_files(ports, source)
            else:
                fetched = fetch(source, ports_store / "sources", offline=offline)
                destination = sources / "sources" / source["sha256"] / fetched.name
                destination.parent.mkdir(parents=True)
                shutil.copyfile(fetched, destination)
                if file_hash(destination) != source["sha256"]:
                    raise ValueError("source changed during preparation")
            closure = closures[port.reference]
            specification = {
                "schema_version": 1,
                "reference": port.reference,
                "default_sdk": default_sdk,
                "jobs": jobs,
                "predecessors": {name: action_ids[name] for name in sorted(closure) if name in action_ids},
                "products": {name: product_keys[name] for name in sorted(closure) if name in product_keys},
                "compiler_cache": compiler_cache_identity(compiler_cache, verify_executable=False)
                if compiler_cache is not None
                else None,
                "worker_identity": worker_identity,
                "recipes": {
                    selected.reference: selected.digest
                    for selected in graph.ports
                    if selected.reference in closure or selected.reference == port.reference
                },
            }
            (sources / "node.json").write_text(json.dumps(specification, sort_keys=True))
            actions.append(
                Action(
                    action_ids[port.reference],
                    ("python", "-s", "-B", "inputs/code/port_worker.py"),
                    dependencies=tuple(specification["predecessors"].values()),
                    inputs=(
                        InputMount("code", code_bundle),
                        InputMount("recipe", capture_tree(sources, blob_store)),
                        *sdk_mounts,
                    ),
                    env=(("PYTHONDONTWRITEBYTECODE", "1"),),
                )
            )
    request = BuildRequest(
        build_key
        if build_key is not None
        else identity({"actions": [asdict(action) for action in actions], "max_workers": max_workers}),
        tuple(actions),
        max_workers=max_workers,
    )
    return PreparedGraph(
        graph, request, action_ids, sdk_context, products, ports, jobs, default_sdk, compiler_cache, worker_identity
    )


def collect_graph(prepared: PreparedGraph, result: BuildResult, blob_store: Store, ports_store: Path) -> GraphBuild:
    """Verify every returned tree before atomically publishing caller cache entries."""
    from ports.buildomatic import BuildState, NodeState, extract_tree, request_id

    if result.state is not BuildState.SUCCEEDED:
        raise RuntimeError("Buildomatic graph did not succeed: " + str(result))
    admitted = {}
    nodes = {node.action_id: node for node in result.nodes}
    if result.request_id != request_id(prepared.request):
        raise ValueError("returned request identity differs from the prepared request")
    if len(nodes) != len(result.nodes) or set(nodes) != set(prepared.action_ids.values()):
        raise ValueError("returned action closure differs from the prepared request")
    if any(node.state is not NodeState.SUCCEEDED for node in result.nodes):
        raise ValueError("successful graph contains an unsuccessful recipe")
    ports_store.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".buildomatic-collect-", dir=ports_store) as temporary:
        for reference, action_id in prepared.action_ids.items():
            node = nodes[action_id]
            if node.bundle is None:
                raise ValueError("successful recipe has no result bundle")
            destination = Path(temporary) / action_id
            extract_tree(node.bundle, blob_store, destination)
            metadata = json.loads((destination / "node.json").read_text())
            if metadata["reference"] != reference:
                raise ValueError("returned recipe reference differs")
            tree = destination / "result"
            receipt = json.loads((tree / "build-receipt.json").read_text())
            receipt = verify(tree, receipt["inputs"])
            if receipt["key"] != metadata["key"]:
                raise ValueError("returned result key differs")
            published = ports_store.resolve() / "results" / receipt["key"]
            published.parent.mkdir(parents=True, exist_ok=True)
            if published.exists() or published.is_symlink():
                verify(published, receipt["inputs"])
            else:
                tree.rename(published)
            admitted[reference] = published.name
    # Recompute identities using caller code and original SDK paths. This also
    # supplies exactly the SDK contexts consumed by existing guest acceptance.
    build = build_graph(
        prepared.ports,
        prepared.graph.roots,
        prepared.sdk_context,
        ports_store,
        offline=True,
        jobs=prepared.jobs,
        default_sdk=prepared.default_sdk,
        admitted_predecessors=admitted,
        compiler_cache=prepared.compiler_cache,
    )
    directory = ports_store / "buildomatic/manifests"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "request_id": result.request_id,
        "roots": prepared.graph.roots,
        "default_sdk": prepared.default_sdk,
        "jobs": prepared.jobs,
        "recipes": {port.reference: port.digest for port in build.graph.ports},
        "results": admitted,
        "bundles": {reference: nodes[action_id].bundle.digest for reference, action_id in prepared.action_ids.items()},
        "compiler_cache": compiler_cache_identity(prepared.compiler_cache, verify_executable=False)
        if prepared.compiler_cache is not None
        else None,
        "worker_identity": prepared.worker_identity,
    }
    path = directory / (result.request_id + ".json")
    with tempfile.NamedTemporaryFile(mode="w", prefix=".manifest-", dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    print("ports: checked cache manifest: " + str(path), file=sys.stderr)
    return build


def publish_manifest(
    manifest: Path,
    ports: Path,
    sdk_context: MaterializedSDK,
    store: Path,
    output: Path,
    *,
    check: bool = False,
) -> Path:
    """Explicitly publish verified cached results, optionally after guest acceptance.

    SDK products must already be admitted by the caller. Missing or changed cache
    entries fail before publication and require a separate build request.
    """
    from ports._support.runner import accept_graph, publish_graph

    if manifest.stat().st_size > 1024**2:
        raise ValueError("cache manifest exceeds its size bound")
    value = json.loads(manifest.read_text())
    if value["schema_version"] != 1:
        raise ValueError("unsupported cache manifest schema")
    graph = plan(ports, value["roots"], default_sdk=value["default_sdk"])
    if {port.reference: port.digest for port in graph.ports} != value["recipes"]:
        raise ValueError("cache manifest recipes differ from the current graph")
    consumers = {port.reference for port in graph.ports if port.recipe["build_system"] not in PRODUCT_SYSTEMS}
    if set(value["results"]) != consumers:
        raise ValueError("cache manifest result closure differs")
    cache = value["compiler_cache"]
    launcher = (
        None
        if cache is None
        else CompilerCacheLauncher(Path(cache["path"]), cache["sha256"], tuple(cache["environment"].items()))
    )
    build = build_graph(
        ports,
        graph.roots,
        sdk_context,
        store,
        offline=True,
        jobs=value["jobs"],
        default_sdk=value["default_sdk"],
        admitted_predecessors=value["results"],
        compiler_cache=launcher,
    )
    if not check:
        return publish_graph(build, sdk_context, output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".ports-check-", dir=output.parent))
    try:
        descriptor = publish_graph(build, sdk_context, work / "release")
        accept_graph(build, sdk_context, descriptor, work / "release/acceptance")
        (work / "release").rename(output)
    except Exception as error:
        error.add_note(f"graph acceptance work retained at {work}")
        raise
    work.rmdir()
    return output / "release.json"


def run_graph(
    ports: Path,
    requests: Sequence[str],
    sdk_context: MaterializedSDK | None,
    store: Path,
    *,
    offline: bool = False,
    jobs: int | None = None,
    default_sdk: str = "default",
    workspaces: Mapping[str, Path] | None = None,
    backend: str = "buildomatic",
    blob_store: Store | None = None,
    workers: Mapping[str, Worker] | None = None,
    max_workers: int = 1,
    compiler_cache: CompilerCacheLauncher | None = None,
    remote_backend: IrisBackend | None = None,
    iris_service: Path | None = None,
    build_key: str | None = None,
    worker_identity: Mapping[str, str] | None = None,
) -> GraphBuild:
    """Run through the core ready queue; optional Iris imports occur only here."""
    from ports.buildomatic import BuildState, Coordinator, LocalStore, ResourceLimits, WorkerExecutor

    if workspaces:
        raise ValueError("Buildomatic actions require private build workspaces")
    if backend not in {"buildomatic", "iris"}:
        raise ValueError("unsupported Buildomatic backend")
    if backend == "iris":
        if remote_backend is None and iris_service is not None:
            if iris_service.stat().st_size > 64 * 1024:
                raise ValueError("Iris connection descriptor exceeds its size bound")
            connection = json.loads(iris_service.read_text())
            if set(connection) != {
                "schema_version",
                "job_id",
                "prefix",
                "cache_prefix",
                "controller_url",
                "cluster_name",
                "workspace",
                "config_sha256",
                "task_image",
                "service_id",
            }:
                raise ValueError("unsupported Iris connection fields")
            if type(connection["schema_version"]) is not int or connection["schema_version"] != 1:
                raise ValueError("unsupported Iris connection schema")
            from iris.cli.connect import open_iris_client
            from iris.cluster.types import JobName, Namespace

            from ports.buildomatic.backends.iris import IrisBackend

            with open_iris_client(
                cluster_name=connection["cluster_name"],
                workspace=Path(connection["workspace"]),
            ) as client:
                remote = IrisBackend(
                    client,
                    connection["controller_url"],
                    str(Namespace.from_job_id(JobName.from_wire(connection["job_id"]))),
                    prefix=connection["prefix"],
                    cache_prefix=connection["cache_prefix"],
                )
                return run_graph(
                    ports,
                    requests,
                    sdk_context,
                    store,
                    offline=offline,
                    jobs=jobs,
                    default_sdk=default_sdk,
                    backend="iris",
                    blob_store=remote.store,
                    remote_backend=remote,
                    max_workers=max_workers,
                    compiler_cache=compiler_cache,
                    build_key=build_key,
                    worker_identity={
                        name: connection[name]
                        for name in ("config_sha256", "task_image", "service_id")
                    },
                )
        if blob_store is None or remote_backend is None:
            raise ValueError("Iris builds require a connected remote service and its blob store")
    else:
        blob_store = LocalStore(store / "buildomatic/blobs") if blob_store is None else blob_store
        if workers is None:
            limits = (
                ResourceLimits(output_bytes=40 * 1024**3, max_files=400_000)
                if sdk_context is not None
                else ResourceLimits()
            )
            workers = {
                f"local-{index}": WorkerExecutor(
                    blob_store, store / f"buildomatic/worker-{index}", max_running=1, limits=limits
                )
                for index in range(max_workers)
            }
    prepared = prepare_graph(
        ports,
        requests,
        sdk_context,
        store,
        blob_store,
        offline=offline,
        jobs=jobs,
        default_sdk=default_sdk,
        max_workers=max_workers,
        compiler_cache=compiler_cache,
        build_key=build_key,
        worker_identity=worker_identity,
    )
    if backend == "iris":
        build_id = remote_backend.submit(prepared.request)
        print("ports: accepted remote build: " + build_id, file=sys.stderr, flush=True)
    else:
        coordinator = Coordinator(blob_store, "ports-" + identity(prepared.request.idempotency_key), workers)
        coordinator.submit(prepared.request)
    while True:
        result = remote_backend.get(build_id) if backend == "iris" else coordinator.tick()
        if result.state in {BuildState.SUCCEEDED, BuildState.FAILED, BuildState.CANCELLED}:
            break
        time.sleep(0.05)
    build = collect_graph(prepared, result, blob_store, store)
    if backend == "iris":
        remote_backend.acknowledge(build_id)
    else:
        coordinator.acknowledge()
    return build


def main() -> None:
    """Publish from a checked cache manifest and an explicitly imported SDK."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish", help="publish a release from verified cache results")
    publish.add_argument("manifest", type=Path)
    publish.add_argument("--ports", type=Path, default=Path(__file__).resolve().parents[1])
    publish.add_argument(
        "--sdk-descriptor", type=Path, required=True, help="portable SDK descriptor; never builds SDK products"
    )
    publish.add_argument("--store", type=Path, required=True)
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--check", action="store_true")
    args = parser.parse_args()
    from ports.buildomatic.portable import import_sdk, original_root_bindings

    sdk = import_sdk(
        args.sdk_descriptor,
        args.store / "sdk-import",
        original_bindings=True,
        bindings=original_root_bindings(args.sdk_descriptor),
    )
    print(publish_manifest(args.manifest, args.ports, sdk, args.store, args.output, check=args.check))


if __name__ == "__main__":
    main()
