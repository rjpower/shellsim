"""Plan port actions from canonical metadata and return artifact references.

The coordinator needs recipes and small SDK/product records. Workers fetch
SDK and predecessor archives directly; collection never imports their bytes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from ports._support.graph import Graph, plan
from ports._support.runner import _admit_recipe
from ports._support.store import file_hash, identity
from ports.api import implementation
from ports.buildomatic import Action, BuildRequest, BuildState, InputMount, LocalStore, capture_tree
from ports.buildomatic.artifacts import write_reference
from ports.buildomatic.delivery import delivery_graph
from ports.buildomatic.ports import PRODUCT_SYSTEMS, _stage_code, code_files
from ports.buildomatic.sdk_artifacts import sdk_inputs, sdk_inputs_match


@dataclass(frozen=True)
class StoreBuild:
    """Products remain in the canonical store until explicitly requested."""

    graph: Graph
    sdk: dict | None
    results: dict[str, dict]


def _resolve_products(backend, keys: dict[str, str]) -> dict[str, dict]:
    """Resolve catalog metadata with bounded concurrency, without fetching products."""
    if not keys:
        return {}
    with ThreadPoolExecutor(max_workers=min(16, len(keys))) as executor:
        records = executor.map(backend.resolve_artifact, ("port/" + key for key in keys.values()))
        return {reference: record for reference, record in zip(keys, records) if record is not None}


def consumer_sdk(manifest: dict | None, graph: Graph) -> dict | None:
    """Send references only to SDK components consumed by this port closure."""
    pure = {"pure-wheel", "host-wheel", "source-tree"}
    if manifest is None or all(port.recipe["build_system"] in pure for port in graph.ports):
        return None
    roles = {"compiler", "platform", "tooling"}
    if any(
        port.recipe["build_system"].startswith("python-") or port.recipe["build_system"] == "cpython-threaded"
        for port in graph.ports
    ):
        roles.add("cpython")
    if any(port.recipe["build_system"] == "uv-host" for port in graph.ports):
        roles.add("resolver")
    return {
        **manifest,
        "products": {role: component for role, component in manifest["products"].items() if role in roles},
    }


def prepare(
    ports: Path,
    requests: Sequence[str],
    sdk: dict | None,
    local: LocalStore,
    backend,
    prefix: str,
    scratch: Path,
    jobs: int,
    max_workers: int,
    *,
    sdk_name: str = "default",
    build_key: str | None = None,
    port_timeout: float = 14400,
) -> tuple[Graph, BuildRequest | None, dict, dict]:
    """Check catalog keys and submit whole-port misses with reference inputs."""
    if not 1 <= jobs <= 256 or not 1 <= max_workers <= 32:
        raise ValueError("build jobs need 1..256 and workers need 1..32")
    ports = ports.resolve()
    graph = plan(ports, requests, default_sdk=sdk_name, target_profile=sdk["dynamic_abi"] if sdk else None)
    keys, results, actions, action_ids = {}, {}, [], {}
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=scratch) as temporary:
        root = Path(temporary)
        code = root / "code"
        code.mkdir()
        files = code_files(ports, graph)
        if sdk:
            from ports._support.sdk import definition

            for product in definition(ports, graph, sdk_name).products:
                relative = product.recipe_path.relative_to(ports.resolve())
                files[str(relative)] = product.recipe_path
        # Store consumers also need the selected producer policies and their
        # supporting code to interpret SDK metadata on the worker.
        _stage_code(files, code)
        for module in Path(__file__).parent.rglob("*.py"):
            if "tests" in module.relative_to(Path(__file__).parent).parts:
                continue
            destination = code / "ports/buildomatic" / module.relative_to(Path(__file__).parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(module, destination)
        code_bundle = capture_tree(code, local)
        bundles = [code_bundle]
        for port in graph.ports:
            system = port.recipe["build_system"]
            if system in PRODUCT_SYSTEMS:
                role = {
                    "llvm-host": "compiler",
                    "wasi-sysroot": "platform",
                    "sdk-tooling": "tooling",
                    "cpython-threaded": "cpython",
                    "uv-host": "resolver",
                }[system]
                keys[port.reference] = sdk["products"][role]["receipt_id"]
                continue
            inputs = {
                "recipe": port.digest,
                "local_inputs": _admit_recipe(port),
                "implementation": implementation(port),
                "driver": {
                    name: file_hash(Path(__file__).resolve().parents[1] / name)
                    for name in (["_support/build.py"] if port.recipe.get("patches") else [])
                },
                "sdk": sdk["identity"] if sdk and system not in {"pure-wheel", "host-wheel", "source-tree"} else None,
                "dependencies": {edge.recipe: keys[edge.recipe] for edge in port.dependencies},
            }
            if system in {"python-extension", "python-meson", "python-pep517", "python-rust"}:
                if sdk is None or "cpython" not in sdk["products"]:
                    raise ValueError("Python extensions require an SDK bootstrapped with --python")
                inputs["python"] = sdk["products"]["cpython"]["receipt_id"]
            key = identity(inputs)
            keys[port.reference] = key

        # Keys depend on predecessor keys, not on their catalog records. Resolve
        # the complete key set concurrently before choosing the missing actions.
        results.update(
            _resolve_products(
                backend,
                {
                    port.reference: keys[port.reference]
                    for port in graph.ports
                    if port.recipe["build_system"] not in PRODUCT_SYSTEMS
                },
            )
        )

        for port in graph.ports:
            if port.recipe["build_system"] in PRODUCT_SYSTEMS:
                continue
            existing = results.get(port.reference)
            if existing is not None and ("delivery" in existing or port.role in {"host-tool", "target-platform"}):
                continue
            action_id = ("delivery-" if existing is not None else "port-") + hashlib.sha256(
                port.reference.encode()
            ).hexdigest()
            action_ids[port.reference] = action_id
            predecessors = {}
            # The ordinary driver walks the whole transitive graph. Supply
            # references to every consumer predecessor in topological order.
            pending = [edge.recipe for edge in port.dependencies]
            closure = set()
            selected = {node.reference: node for node in graph.ports}
            while pending:
                reference = pending.pop()
                if reference in closure:
                    continue
                closure.add(reference)
                pending.extend(edge.recipe for edge in selected[reference].dependencies)
            for reference in sorted(closure):
                if reference in results:
                    predecessors[reference] = results[reference]
                elif reference in action_ids:
                    predecessors[reference] = {"action_id": action_ids[reference]}
            recipe = root / action_id
            recipe.mkdir()
            write_reference(
                recipe / "node.json",
                {
                    "reference": port.reference,
                    "sdk": None
                    if existing is not None
                    else consumer_sdk(
                        sdk, Graph((port.reference,), tuple(selected[name] for name in closure | {port.reference}))
                    ),
                    "sdk_name": sdk_name,
                    "prefix": prefix,
                    "jobs": jobs,
                    "build_key": keys[port.reference],
                    "predecessors": predecessors,
                    **({"existing": existing} if existing is not None else {}),
                },
            )
            recipe_bundle = capture_tree(recipe, local)
            bundles.append(recipe_bundle)
            actions.append(
                Action(
                    action_id,
                    ("python", "-m", "ports.buildomatic.store_worker"),
                    dependencies=()
                    if existing is not None
                    else tuple(value["action_id"] for value in predecessors.values() if "action_id" in value),
                    inputs=(InputMount("code", code_bundle), InputMount("recipe", recipe_bundle)),
                    env=(("PYTHONPATH", "inputs/code"),),
                    timeout_seconds=port_timeout,
                    max_attempts=1,
                )
            )
        if sdk is not None and "cpython" in sdk["products"]:
            component = sdk["products"]["cpython"]
            runtime_record = backend.resolve_artifact("runtime/" + component["receipt_id"])
            if runtime_record is None or backend.resolve_artifact("runtime/" + sdk_name) != runtime_record:
                recipe = root / "runtime"
                recipe.mkdir()
                write_reference(
                    recipe / "node.json",
                    {
                        "prefix": prefix,
                        "sdk_name": sdk_name,
                        "component": {"receipt_id": component["receipt_id"], "artifact": component["artifact"]},
                    },
                )
                recipe_bundle = capture_tree(recipe, local)
                bundles.append(recipe_bundle)
                actions.append(
                    Action(
                        "runtime-export",
                        ("python", "-m", "ports.buildomatic.runtime_worker"),
                        inputs=(InputMount("code", code_bundle), InputMount("recipe", recipe_bundle)),
                        env=(("PYTHONPATH", "inputs/code"),),
                        timeout_seconds=3600,
                        max_attempts=1,
                    )
                )
        if actions:
            backend.upload_bundles(local, bundles)
    request = (
        BuildRequest(
            identity(
                {"actions": [asdict(action) for action in actions], "workers": max_workers, "build_key": build_key}
            ),
            tuple(actions),
            max_workers,
        )
        if actions
        else None
    )
    return graph, request, results, keys


def run(
    ports: Path,
    requests: Sequence[str],
    sdk_name: str,
    service: Path,
    scratch: Path,
    *,
    jobs: int = 16,
    max_workers: int = 1,
    build_key: str | None = None,
    port_timeout: float = 14400,
) -> StoreBuild:
    """Resolve and build on an existing Iris service without local SDK files."""
    from ports.buildomatic.backends.iris import backend_context_factory

    scratch.mkdir(parents=True, exist_ok=True)
    descriptor = json.loads(service.read_text())
    prefix = descriptor.get("cache_prefix") or descriptor["prefix"]
    factory = backend_context_factory(
        job_id=descriptor["job_id"],
        cluster_name=descriptor["cluster_name"],
        controller_url=descriptor["controller_url"],
        workspace=Path(descriptor["workspace"]),
        prefix=descriptor["prefix"],
        cache_prefix=prefix,
    )
    if sdk_name == "default":
        sdk_name = json.loads((ports / "sdks/default.json").read_text())["sdk"]
    with factory() as backend:
        initial = plan(ports, requests, default_sdk=sdk_name)
        needs_sdk = any(
            port.recipe["build_system"] not in {"pure-wheel", "host-wheel", "source-tree"} for port in initial.ports
        )
        sdk = backend.resolve_artifact("sdk/" + sdk_name)
        if needs_sdk and sdk is None:
            raise ValueError("SDK cache miss: explicitly bootstrap " + sdk_name)
        if needs_sdk and sdk is not None and not sdk_inputs_match(sdk["inputs"], sdk_inputs(ports, sdk_name)):
            raise ValueError("SDK build inputs changed: explicitly bootstrap " + sdk_name)
        local = LocalStore(scratch / "transport")
        print(f"buildomatic: checking product cache for {len(initial.ports)} graph nodes", flush=True)
        graph, request, results, keys = prepare(
            ports,
            requests,
            sdk,
            local,
            backend,
            prefix,
            scratch,
            jobs,
            max_workers,
            sdk_name=sdk_name,
            build_key=build_key,
            port_timeout=port_timeout,
        )
        actions = request.actions if request else ()
        backfills = sum(action.id.startswith("delivery-") for action in actions)
        runtime_exports = sum(action.id == "runtime-export" for action in actions)
        print(
            f"buildomatic: product cache: {len(results)} hits, "
            f"{len(actions) - backfills - runtime_exports} builds, {backfills} asset backfills, {runtime_exports} runtime exports",
            flush=True,
        )
        if request is not None:
            build_id = backend.submit(request)
            print("buildomatic: submitted " + build_id, flush=True)
            while True:
                outcome = backend.get(build_id)
                if outcome.state in {BuildState.SUCCEEDED, BuildState.FAILED, BuildState.CANCELLED}:
                    break
                time.sleep(1)
            if outcome.state is not BuildState.SUCCEEDED:
                raise RuntimeError(str(outcome) + "\n" + str(backend.get_logs(build_id)))
            results.update(
                _resolve_products(
                    backend,
                    {
                        port.reference: keys[port.reference]
                        for port in graph.ports
                        if port.recipe["build_system"] not in PRODUCT_SYSTEMS
                        and (
                            port.reference not in results
                            or (
                                "delivery" not in results[port.reference]
                                and port.role not in {"host-tool", "target-platform"}
                            )
                        )
                    },
                )
            )
            backend.acknowledge(build_id)
        runtime_key = sdk["products"]["cpython"]["receipt_id"] if sdk and "cpython" in sdk["products"] else sdk_name
        runtime_record = backend.resolve_artifact("runtime/" + runtime_key)
        runtime = runtime_record["runtime"] if runtime_record is not None else None
    result = StoreBuild(graph, sdk, results)
    write_reference(
        scratch / "results.json",
        {"roots": graph.roots, "sdk": sdk, "sdk_name": sdk_name, "products": results, "prefix": prefix},
    )
    write_reference(
        scratch / "graph.json", delivery_graph(graph, results, abi=sdk["dynamic_abi"] if sdk else None, runtime=runtime)
    )
    return result
