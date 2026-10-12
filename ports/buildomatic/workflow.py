"""Plan one ports DAG and dispatch identical node requests locally or to Iris."""

from __future__ import annotations

import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from ports._support.graph import Graph, Port, guest_graph, plan
from ports._support.node_build import _admit_recipe
from ports._support.producer_policy import policy
from ports._support.sdk import SDKDefinition, definition
from ports._support.store import file_hash, identity
from ports.api import implementation
from ports.buildomatic import Action, BuildRequest, BuildState, InputMount, LocalStore, capture_tree
from ports.buildomatic.artifacts import register, resolve, write_reference
from ports.buildomatic.delivery import DeliveryStore, delivery_graph
from ports.buildomatic.delivery_products import delivery_key
from ports.buildomatic.legacy import compatible_driver, compatible_packaging
from ports.buildomatic.nodes import Input, Node, Operation
from ports.buildomatic.remote_store import StorageStore
from ports.buildomatic.sdk_artifacts import sdk_inputs, sdk_inputs_match
from ports.buildomatic.transport import PRODUCT_SYSTEMS, _stage_code, code_files


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
        key: value for key, value in port.recipe.items() if key not in {"tests", "test_limits", "runtime_dependencies"}
    }


def compile_plan(ports: Path, requests: Sequence[str], sdk_name: str = "default") -> Plan:
    """Expand build and runtime producers without fetching installed products."""
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
        key = identity(
            {
                "operation": "build",
                "policy": policy(descriptor.port.recipe),
                "implementation": implementation(descriptor.port),
                "inputs": {item.name: item.key for item in inputs},
            }
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
            if runtime_edges or edge.kind != "runtime"
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
                if include_runtime or edge.kind != "runtime"
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
        key = identity(
            {
                "operation": "build",
                "adapter_driver": file_hash(Path(__file__).resolve().parents[1] / "_support/node_build.py"),
                "sdk_binding": file_hash(Path(__file__).resolve().parents[1] / "_support/sdk_products.py")
                if port.recipe["build_system"] not in {"pure-wheel", "host-wheel", "source-tree"}
                else None,
                "recipe": build_recipe(port),
                "local_inputs": _admit_recipe(port),
                "implementation": implementation(port),
                "inputs": {item.name: item.key for item in inputs},
                "patch_driver": file_hash(Path(__file__).resolve().parents[1] / "_support/build.py")
                if port.recipe.get("patches")
                else None,
            }
        )
        node = Node(port.reference, Operation.BUILD, key, port, tuple(inputs))
        builds[node.reference] = node
        nodes.append(node)
    deliveries = {}
    for port in guest_graph(graph).ports:
        if port.role in {"host-tool", "target-platform"} or port.reference in descriptors:
            continue
        product = builds[port.reference]
        inputs = (Input("product", product.reference, product.key),)
        key = delivery_key(product.key)
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
        key = identity(
            {
                "operation": "runtime",
                "implementation": file_hash(Path(__file__).with_name("delivery_products.py")),
                "inputs": {item.name: item.key for item in inputs},
            }
        )
        runtime_node = Node(product.reference + ":delivery", Operation.RUNTIME, key, product.port, inputs)
        nodes.append(runtime_node)
    check_roots = []
    for port in graph.ports:
        if port.reference not in deliveries:
            continue
        inputs = [
            Input("package:" + ref, node.reference, node.key)
            for ref, node in deliveries.items()
            if ref in {port.reference, *ancestors(port, runtime_edges=True)} or ref in runtime_ports
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
                probe_key = identity(
                    {
                        "operation": "probe",
                        "command": file_hash(Path(__file__).resolve().parents[1] / "_support/acceptance.py"),
                        "sdk_binding": file_hash(Path(__file__).resolve().parents[1] / "_support/sdk_products.py"),
                        "implementation": file_hash(Path(__file__).with_name("checks.py")),
                        "test": test,
                        "source": file_hash(port.directory / test["source"]),
                        "inputs": {item.name: item.key for item in probe_inputs},
                    }
                )
                probe = Node(
                    port.reference + ":probe-" + str(index), Operation.PROBE, probe_key, port, probe_inputs, test=test
                )
                nodes.append(probe)
            test_inputs = tuple(inputs) + ((Input("probe", probe.reference, probe.key),) if probe else ())
            key = identity(
                {
                    "operation": "check",
                    "implementation": file_hash(Path(__file__).with_name("checks.py")),
                    "test": test,
                    "source": file_hash(port.directory / test.get("script", test.get("source"))),
                    "limits": port.recipe.get("test_limits", {}),
                    "inputs": {item.name: item.key for item in test_inputs},
                }
            )
            node = Node(port.reference + ":check-" + str(index), Operation.CHECK, key, port, test_inputs, test=test)
            nodes.append(node)
            check_roots.append(node.reference)
    return Plan(
        ports.resolve(),
        graph,
        tuple(nodes),
        tuple(n.reference for n in deliveries.values())
        + ((runtime_node.reference,) if runtime_node else ())
        + tuple(ref for ref in graph.roots if ref not in deliveries),
        tuple(n.reference for n in deliveries.values()) + ((runtime_node.reference,) if runtime_node else ()),
        tuple(check_roots),
        sdk,
    )


def records_for(plan: Plan, store: StorageStore) -> dict[str, dict | None]:
    with ThreadPoolExecutor(max_workers=16) as executor:
        values = executor.map(lambda node: resolve(store, node.record_key), plan.nodes)
        return {node.reference: value for node, value in zip(plan.nodes, values, strict=True)}


def normalize_cached(
    plan: Plan, store: StorageStore, records: dict[str, dict | None] | None = None
) -> dict[str, dict | None]:
    """Admit qualified existing output references once, without moving their bytes."""
    records = records_for(plan, store) if records is None else records
    missing_builds = any(records[node.reference] is None for node in plan.nodes if node.operation is Operation.BUILD)
    old_sdk = resolve(store, "sdk/" + plan.sdk.name) if plan.sdk and missing_builds else None
    compatible_roles = set()
    if old_sdk:
        current = sdk_inputs(plan.ports, plan.sdk.name)
        for descriptor in plan.sdk.products:
            role = descriptor.name
            recorded = old_sdk["inputs"]["producers"][role]
            selected = current["producers"][role]
            if all(name in compatible_roles for name in descriptor.dependencies) and sdk_inputs_match(
                {"definition": {}, "producers": {role: recorded}}, {"definition": {}, "producers": {role: selected}}
            ):
                compatible_roles.add(role)
    legacy_keys = {}
    for port in plan.graph.ports:
        system = port.recipe["build_system"]
        if system in PRODUCT_SYSTEMS:
            if not old_sdk:
                continue
            role = {
                "llvm-host": "compiler",
                "wasi-sysroot": "platform",
                "sdk-tooling": "tooling",
                "cpython-threaded": "cpython",
                "uv-host": "resolver",
            }[system]
            if role in compatible_roles and role in old_sdk["products"]:
                legacy_keys[port.reference] = old_sdk["products"][role]["receipt_id"]
            continue
        if (
            system not in {"pure-wheel", "host-wheel", "source-tree"}
            and not {"compiler", "platform", "tooling"} <= compatible_roles
        ):
            continue
        if any(edge.recipe not in legacy_keys for edge in port.dependencies):
            continue
        inputs = {
            "recipe": port.digest,
            "local_inputs": _admit_recipe(port),
            "implementation": implementation(port),
            "driver": {
                name: file_hash(Path(__file__).resolve().parents[1] / name)
                for name in (["_support/build.py"] if port.recipe.get("patches") else [])
            },
            "sdk": old_sdk["identity"]
            if old_sdk
            and {"compiler", "platform", "tooling"} <= compatible_roles
            and system not in {"pure-wheel", "host-wheel", "source-tree"}
            else None,
            "dependencies": {edge.recipe: legacy_keys[edge.recipe] for edge in port.dependencies},
        }
        if system in {"python-extension", "python-meson", "python-pep517", "python-rust"}:
            if not old_sdk or "cpython" not in compatible_roles or "cpython" not in old_sdk["products"]:
                continue
            inputs["python"] = old_sdk["products"]["cpython"]["receipt_id"]
        legacy_keys[port.reference] = identity(inputs)
    for node in plan.nodes:
        if records[node.reference] is not None:
            continue
        record = None
        if node.producer and old_sdk and node.producer["name"] in compatible_roles:
            record = old_sdk["products"].get(node.producer["name"])
            if record:
                record = {**record, "reference": node.port.reference, "build_key": node.key}
        elif node.operation is Operation.BUILD and node.port.reference in legacy_keys and compatible_driver(node.port):
            record = resolve(store, "port/" + legacy_keys[node.port.reference])
        elif node.operation is Operation.DELIVERY:
            product = records[node.inputs[0].reference]
            if product and (
                product.get("delivery_key") == node.key
                or "delivery_key" not in product
                and compatible_packaging()
                and "delivery" in product
            ):
                record = {**product, "delivery_key": node.key}
        elif node.operation is Operation.RUNTIME:
            product = records[node.inputs[0].reference]
            if product and compatible_packaging():
                record = DeliveryStore(store.prefix).record("runtime/" + product["receipt_id"])
        if record:
            register(store, node.record_key, record)
            records[node.reference] = record
    return records


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
                for name in [
                    node.test.get("script", node.test.get("source")),
                    *(item["source"] for item in node.test.get("files", [])),
                ]:
                    path = node.port.directory / name
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
    compiled = compile_plan(ports, requests, sdk_name)
    known = normalize_cached(compiled, store)
    roots = {
        Operation.BUILD: compiled.build_roots,
        Operation.DELIVERY: compiled.delivery_roots,
        Operation.CHECK: compiled.check_roots,
    }[operation]
    if operation is Operation.CHECK:
        for reference in roots:
            known[reference] = None
    request = prepare(
        compiled,
        roots,
        store,
        state,
        backend,
        jobs=jobs,
        max_workers=max_workers,
        offline=offline,
        allow_build=operation is Operation.BUILD,
        records=known,
    )
    print(
        f"buildomatic: {len(compiled.nodes)} planned nodes, {len(request.actions) if request else 0} actions",
        flush=True,
    )
    if request:
        build = backend.submit(request)
        while (outcome := backend.get(build)).state not in {
            BuildState.SUCCEEDED,
            BuildState.FAILED,
            BuildState.CANCELLED,
        }:
            time.sleep(0.2)
        if outcome.state is not BuildState.SUCCEEDED:
            raise RuntimeError(str(outcome) + "\n" + str(backend.get_logs(build)))
        backend.acknowledge(build)
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
    return Result(compiled, records)
