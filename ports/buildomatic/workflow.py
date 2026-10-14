"""Plan one ports DAG and dispatch identical node requests locally or to Iris."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from ports._support.acceptance import _fixtures, _source
from ports._support.graph import Graph, Port, guest_graph, plan
from ports._support.node_build import _admit_recipe
from ports._support.producer_policy import policy
from ports._support.sdk import SDKDefinition, definition
from ports._support.store import file_hash
from ports.api import implementation
from ports.buildomatic import (
    Action,
    BuildRequest,
    BuildResult,
    BuildState,
    InputMount,
    LocalStore,
    NodeState,
    capture_tree,
)
from ports.buildomatic.artifacts import write_reference
from ports.buildomatic.cache import BuildCache, compiler_identity, delivery_key, node_key, runtime_delivery_key
from ports.buildomatic.delivery import delivery_graph
from ports.buildomatic.nodes import Input, Node, Operation
from ports.buildomatic.remote_store import StorageStore
from ports.buildomatic.transport import PRODUCT_SYSTEMS, _stage_code, code_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Plan:
    ports: Path
    graph: Graph
    nodes: tuple[Node, ...]
    build_roots: tuple[str, ...]
    delivery_roots: tuple[str, ...]
    check_roots: tuple[str, ...]
    sdk: SDKDefinition | None


@dataclass(frozen=True)
class Result:
    plan: Plan
    records: dict[str, dict]


def build_recipe(port: Port) -> dict:
    return {
        key: value
        for key, value in port.recipe.items()
        if key not in {"tests", "test_limits", "test_dependencies", "runtime_dependencies"}
    }


def compile_plan(ports: Path, requests: Sequence[str], sdk_name: str = "default") -> Plan:
    """Expand build and runtime producers without fetching installed products."""
    ports = ports.resolve()
    initial = plan(ports, requests, default_sdk=sdk_name)
    sdk = (
        definition(ports, initial, sdk_name)
        if any(p.sdk_selection for p in initial.ports) or (ports / "sdks/default.json").is_file()
        else None
    )
    python = any(Path(p.reference).parts[0] == "python" and p.role != "host-tool" for p in initial.ports)
    runtime = sdk.metadata.get("runtime") if sdk and python else None
    roots = [*requests, *(runtime["dependencies"] if runtime else [])]
    graph = plan(ports, roots, default_sdk=sdk.name if sdk else sdk_name)
    descriptors = {product.port.reference: product for product in sdk.products} if sdk else {}
    products = {product.name: product for product in sdk.products} if sdk else {}
    required = {p.reference for p in graph.ports if p.recipe["build_system"] in PRODUCT_SYSTEMS}
    if runtime:
        required.add(products[runtime["producer"]].port.reference)
    for port in graph.ports:
        if port.recipe["build_system"].startswith("python-"):
            required.add(products["cpython"].port.reference)

    def require(reference):
        descriptor = descriptors[reference]
        for name in descriptor.dependencies:
            dependency = products[name].port.reference
            if dependency not in required:
                required.add(dependency)
                require(dependency)

    for reference in list(required):
        require(reference)
    nodes, builds = [], {}
    for descriptor in sdk.products if sdk else ():
        if descriptor.port.reference not in required:
            continue
        inputs = tuple(
            Input(name, products[name].port.reference, builds[products[name].port.reference].key)
            for name in descriptor.dependencies
        )
        key = node_key(
            "build",
            policy=policy(descriptor.port.recipe),
            implementation=implementation(descriptor.port),
            inputs={item.name: item.key for item in inputs},
        )
        node = Node(
            descriptor.port.reference,
            Operation.BUILD,
            key,
            descriptor.port,
            inputs,
            {"name": descriptor.name, "recipe": descriptor.recipe, "dependencies": descriptor.dependencies},
        )
        builds[node.reference] = node
        nodes.append(node)
    selected = {p.reference: p for p in graph.ports}

    def ancestors(port, *, runtime_edges=False):
        found = set()
        visited = set()
        pending = [
            (edge.recipe, runtime_edges or edge.kind == "build")
            for edge in port.dependencies
            if edge.kind != "test" and (runtime_edges or edge.kind != "runtime")
        ]
        while pending:
            reference, include_runtime = pending.pop()
            if (reference, include_runtime) in visited:
                continue
            visited.add((reference, include_runtime))
            found.add(reference)
            pending.extend(
                (edge.recipe, include_runtime or edge.kind == "build")
                for edge in selected[reference].dependencies
                if edge.kind != "test" and (include_runtime or edge.kind != "runtime")
            )
        return found

    for port in graph.ports:
        if port.reference in builds:
            continue
        inherited = ancestors(port)
        roles = (
            (
                "compiler",
                "platform",
                "tooling",
                *(("cpython",) if port.recipe["build_system"].startswith("python-") else ()),
            )
            if port.recipe["build_system"] not in {"pure-wheel", "host-wheel", "source-tree"}
            else ()
        )
        contextual = {products[name].port.reference for name in roles}
        inputs = [Input(ref, ref, builds[ref].key) for ref in sorted(inherited - contextual)]
        inputs.extend(
            Input(
                "sdk:" + name,
                builds[products[name].port.reference].reference,
                builds[products[name].port.reference].key,
            )
            for name in roles
        )
        key = node_key(
            "build",
            recipe=build_recipe(port),
            local_inputs=_admit_recipe(port),
            implementation=implementation(port),
            inputs={item.name: item.key for item in inputs},
            patch_driver=file_hash(Path(__file__).resolve().parents[1] / "_support/build.py")
            if port.recipe.get("patches")
            else None,
        )
        node = Node(port.reference, Operation.BUILD, key, port, tuple(inputs))
        builds[node.reference] = node
        nodes.append(node)
    guest_ports = {port.reference for port in guest_graph(graph).ports}
    test_ports = {edge.recipe for port in graph.ports for edge in port.dependencies if edge.kind == "test"}
    delivery_ports = guest_graph(Graph((*graph.roots, *sorted(test_ports)), graph.ports))
    deliveries = {}
    for port in delivery_ports.ports:
        if port.role in {"host-tool", "target-platform"} or port.reference in descriptors:
            continue
        product = builds[port.reference]
        inputs = (Input("product", product.reference, product.key),)
        bytecode_runtime = None
        if (
            runtime
            and port.recipe.get("output") != "stdlib"
            and (port.recipe["build_system"] == "pure-wheel" or port.recipe["build_system"].startswith("python-"))
        ):
            bytecode_runtime = builds[products[runtime["producer"]].port.reference]
            inputs += (Input("bytecode_runtime", bytecode_runtime.reference, bytecode_runtime.key),)
        key = delivery_key(product.key, port=port, bytecode_runtime=bytecode_runtime.key if bytecode_runtime else None)
        node = Node(port.reference + ":delivery", Operation.DELIVERY, key, port, inputs)
        deliveries[port.reference] = node
        nodes.append(node)
    runtime_ports = (
        {p.reference for p in plan(ports, runtime["dependencies"], default_sdk=sdk.name).ports} if runtime else set()
    )
    runtime_node = None
    if runtime:
        product = builds[products[runtime["producer"]].port.reference]
        inputs = (Input("product", product.reference, product.key),)
        key = runtime_delivery_key(product.key)
        runtime_node = Node(product.reference + ":delivery", Operation.RUNTIME, key, product.port, inputs)
        nodes.append(runtime_node)
    check_roots = []
    for port in graph.ports:
        if port.reference not in deliveries or port.reference not in guest_ports:
            continue
        tested = {edge.recipe for edge in port.dependencies if edge.kind == "test"}
        check_ports = {port.reference, *ancestors(port, runtime_edges=True), *tested}
        for reference in tested:
            check_ports.update(ancestors(selected[reference], runtime_edges=True))
        inputs = [
            Input("package:" + ref, node.reference, node.key)
            for ref, node in deliveries.items()
            if ref in check_ports or ref in runtime_ports
        ]
        if runtime_node:
            inputs.append(Input("runtime", runtime_node.reference, runtime_node.key))
        for index, test in enumerate(port.recipe.get("tests", [])):
            probe = None
            if test["kind"] == "native":
                probe_inputs = (
                    *(
                        Input("product:" + ref, builds[ref].reference, builds[ref].key)
                        for ref in sorted({port.reference, *ancestors(port)})
                        if ref not in descriptors
                    ),
                    *(
                        Input(
                            "sdk:" + name,
                            builds[products[name].port.reference].reference,
                            builds[products[name].port.reference].key,
                        )
                        for name in ("compiler", "platform", "tooling")
                    ),
                )
                probe_key = node_key(
                    "probe",
                    command=file_hash(Path(__file__).resolve().parents[1] / "_support/acceptance.py"),
                    sdk_binding=file_hash(Path(__file__).resolve().parents[1] / "_support/sdk_products.py"),
                    implementation=file_hash(Path(__file__).with_name("checks.py")),
                    test=test,
                    source=file_hash(port.directory / test["source"]),
                    inputs={item.name: item.key for item in probe_inputs},
                )
                probe = Node(
                    port.reference + ":probe-" + str(index), Operation.PROBE, probe_key, port, probe_inputs, test=test
                )
                nodes.append(probe)
            test_inputs = tuple(inputs) + ((Input("probe", probe.reference, probe.key),) if probe else ())
            key = node_key(
                "check",
                implementation=file_hash(Path(__file__).with_name("checks.py")),
                test=test,
                source=file_hash(port.directory / test.get("script", test.get("source"))),
                limits=port.recipe.get("test_limits", {}),
                inputs={item.name: item.key for item in test_inputs},
                **(
                    {"fixtures": {item["source"]: file_hash(port.directory / item["source"]) for item in test["files"]}}
                    if test.get("files")
                    else {}
                ),
            )
            node = Node(port.reference + ":check-" + str(index), Operation.CHECK, key, port, test_inputs, test=test)
            nodes.append(node)
            check_roots.append(node.reference)
    return Plan(
        ports.resolve(),
        graph,
        tuple(nodes),
        tuple(n.reference for ref, n in deliveries.items() if ref in guest_ports)
        + ((runtime_node.reference,) if runtime_node else ())
        + tuple(ref for ref in graph.roots if ref not in deliveries),
        tuple(n.reference for ref, n in deliveries.items() if ref in guest_ports)
        + ((runtime_node.reference,) if runtime_node else ()),
        tuple(check_roots),
        sdk,
    )


def records_for(plan: Plan, store: StorageStore) -> dict[str, dict | None]:
    with ThreadPoolExecutor(max_workers=16) as executor:
        cache = BuildCache(store)
        values = executor.map(lambda node: cache.get(node.key), plan.nodes)
        return {node.reference: value for node, value in zip(plan.nodes, values, strict=True)}


def prepare(
    plan: Plan,
    roots: Sequence[str],
    store: StorageStore,
    state: Path,
    backend,
    *,
    jobs: int,
    max_workers: int,
    offline: bool = False,
    timeout: float = 14400,
    allow_build: bool = True,
    records: dict[str, dict | None] | None = None,
) -> BuildRequest | None:
    """Prune cached outputs before code transport or worker allocation."""
    records = records_for(plan, store) if records is None else records
    nodes = {node.reference: node for node in plan.nodes}
    missing = {}

    def visit(reference):
        node = nodes[reference]
        if records[node.reference] is not None:
            return
        if reference in missing:
            return
        if not allow_build and node.operation is Operation.BUILD:
            raise ValueError("missing built input; run ports build: " + node.reference)
        for item in node.inputs:
            visit(item.reference)
        missing[reference] = node

    for root in roots:
        visit(root)
    if not missing:
        return None
    logger.info("Preparing %d actions and uploading their code and recipe inputs", len(missing))
    local = LocalStore(state / "transport")
    state.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=state) as temporary:
        root = Path(temporary)
        code = root / "code"
        code.mkdir()
        ports = plan.ports
        files = code_files(
            ports,
            Graph(
                plan.graph.roots,
                tuple({p.reference: p for p in (*plan.graph.ports, *(n.port for n in missing.values()))}.values()),
            ),
            extra_files=("buildomatic/checks.py",)
            if any(n.operation in {Operation.CHECK, Operation.PROBE} for n in missing.values())
            else (),
        )
        for name in ("_support/node_build.py",):
            files[name] = Path(__file__).resolve().parents[1] / name
        if plan.sdk:
            for product in plan.sdk.products:
                files[str(product.recipe_path.relative_to(ports))] = product.recipe_path
        for node in missing.values():
            if node.test:
                _fixtures(node.port, node.test.get("files", []))
                for name in [
                    node.test.get("script", node.test.get("source")),
                    *(item["source"] for item in node.test.get("files", [])),
                ]:
                    path, _ = _source(node.port, name)
                    files[str(path.relative_to(ports))] = path
        _stage_code(files, code)
        for module in Path(__file__).parent.rglob("*.py"):
            if "tests" in module.parts:
                continue
            target = code / "ports/buildomatic" / module.relative_to(Path(__file__).parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(module, target)
        code_bundle = capture_tree(code, local)
        bundles = [code_bundle]
        actions = []
        for node in missing.values():
            recipe = root / node.action_id
            recipe.mkdir()
            spec = node.specification(store.prefix, jobs, offline)
            spec.update(
                {
                    "sdk_name": plan.sdk.name if plan.sdk else "default",
                    "target": plan.sdk.metadata["target"] if plan.sdk else None,
                    "abi": plan.sdk.metadata["abi"] if plan.sdk else None,
                }
            )
            if node.operation in {Operation.DELIVERY, Operation.RUNTIME}:
                compiler = compiler_identity(node.port if node.operation is Operation.DELIVERY else None)
                if compiler is not None:
                    spec["compiler"] = compiler
            write_reference(recipe / "node.json", spec)
            bundle = capture_tree(recipe, local)
            bundles.append(bundle)
            actions.append(
                Action(
                    node.action_id,
                    ("python", "-m", "ports.buildomatic.node_worker"),
                    tuple(
                        dict.fromkeys(
                            nodes[item.reference].action_id for item in node.inputs if item.reference in missing
                        )
                    ),
                    (InputMount("code", code_bundle), InputMount("recipe", bundle)),
                    (("PYTHONPATH", "inputs/code"),),
                    timeout,
                    max_attempts=1,
                )
            )
        backend.upload_bundles(local, bundles)
    return BuildRequest(uuid4().hex, tuple(actions), max_workers)


def wait_for_build(backend, build_id: str, nodes: Sequence[Node]) -> BuildResult:
    """Report observed node transitions and cancel this submission on interruption."""
    labels = {node.action_id: node.operation.value + " " + node.reference.replace("/recipe.json", "") for node in nodes}
    observed = {}
    next_report = time.monotonic() + 10
    terminal = {BuildState.SUCCEEDED, BuildState.FAILED, BuildState.CANCELLED}
    try:
        while True:
            outcome = backend.get(build_id)
            for node in outcome.nodes:
                if observed.get(node.action_id) == node.state:
                    continue
                observed[node.action_id] = node.state
                label = labels[node.action_id]
                if node.state is NodeState.RUNNING:
                    logger.info("Starting %s", label)
                elif node.state is NodeState.SUCCEEDED:
                    logger.info("Finished %s", label)
                elif node.state is NodeState.FAILED:
                    logger.error("Failed %s: %s", label, node.error)
                elif node.state is NodeState.CANCELLED:
                    logger.info("Cancelled %s", label)
            now = time.monotonic()
            if now >= next_report or outcome.state in terminal:
                counts = Counter(node.state for node in outcome.nodes)
                logger.info(
                    "Progress: %d/%d finished, %d running, %d pending, %d failed, %d blocked",
                    counts[NodeState.SUCCEEDED],
                    len(outcome.nodes),
                    counts[NodeState.RUNNING],
                    counts[NodeState.PENDING],
                    counts[NodeState.FAILED],
                    counts[NodeState.BLOCKED],
                )
                next_report = now + 10
            if outcome.state in terminal:
                return outcome
            time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("Cancelling submitted build %s", build_id)
        backend.cancel(build_id)
        raise


def run(
    ports: Path,
    requests: Sequence[str],
    store: StorageStore,
    state: Path,
    backend,
    *,
    operation: Operation = Operation.BUILD,
    sdk_name: str = "default",
    jobs: int = 16,
    max_workers: int = 1,
    offline: bool = False,
) -> Result:
    logger.info("Reading graph for %d roots from %s", len(requests), ports)
    compiled = compile_plan(ports, requests, sdk_name)
    logger.info(
        "Graph: %d recipes, %d nodes; checking cached products in %s",
        len(compiled.graph.ports),
        len(compiled.nodes),
        store.prefix,
    )
    known = records_for(compiled, store)
    roots = {
        Operation.BUILD: compiled.build_roots,
        Operation.DELIVERY: compiled.delivery_roots,
        Operation.CHECK: compiled.check_roots,
    }[operation]
    request = prepare(
        compiled,
        roots,
        store,
        state,
        backend,
        jobs=jobs,
        max_workers=max_workers,
        offline=offline,
        allow_build=operation is not Operation.DELIVERY,
        records=known,
    )
    logger.info("Planned %d nodes, %d actions", len(compiled.nodes), len(request.actions) if request else 0)
    if request:
        build = backend.submit(request)
        logger.info("Submitted %s: %d actions, up to %d workers", build, len(request.actions), request.max_workers)
        outcome = wait_for_build(backend, build, compiled.nodes)
        if outcome.state is not BuildState.SUCCEEDED:
            raise RuntimeError(f"{operation.value} {outcome.state.value}\n{backend.get_logs(build)}")
        backend.acknowledge(build)
    else:
        logger.info("All requested outputs are cached; no work dispatched")
    if request:
        refreshed = records_for(
            Plan(
                compiled.ports,
                compiled.graph,
                tuple(node for node in compiled.nodes if known[node.reference] is None),
                (),
                (),
                (),
                compiled.sdk,
            ),
            store,
        )
        known.update(refreshed)
    records = {reference: record for reference, record in known.items() if record is not None}
    products = {
        node.port.reference: record
        for node in compiled.nodes
        if node.operation in {Operation.BUILD, Operation.DELIVERY}
        and (record := records.get(node.reference)) is not None
        and "delivery" in record
    }
    runtime = next(
        (
            records[node.reference]["runtime"]
            for node in compiled.nodes
            if node.operation is Operation.RUNTIME and node.reference in records
        ),
        None,
    )
    write_reference(
        state / "graph.json",
        delivery_graph(
            compiled.graph, products, abi=compiled.sdk.metadata["abi"] if compiled.sdk else None, runtime=runtime
        ),
    )
    write_reference(
        state / "results.json",
        {
            "roots": list(requests),
            "sdk_name": compiled.sdk.name if compiled.sdk else sdk_name,
            "prefix": store.prefix,
            "products": products,
        },
    )
    logger.info("Saved graph.json and results.json in %s", state)
    return Result(compiled, records)
