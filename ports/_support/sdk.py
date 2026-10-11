"""Materialize the pinned SDK product graph from verified cache or host seeds.

Only trusted host builds use these paths. Target products are never bootstrap
requirements: missing products reach the existing pinned producers in dependency
order. Imported inventories are references to immutable products, not new receipts.
"""

from __future__ import annotations

import fcntl
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ports._support.graph import Graph, Port, _document, canonical_reference
from ports._support.producer_policy import load_policy, policy, verify_policy
from ports._support.producer_policy import metadata as producer_metadata
from ports._support.sdk_products import (
    MaterializedSDK,
    Receipt,
    Tool,
    admit_host_tools,
    admit_sdk,
    file_hash,
    json_hash,
    read_json,
    receipt,
    tool_reference,
    verify_cpython_recipe,
    verify_files,
    verify_product,
)
from ports._support.store import fetch, relative_path
from ports.api import BuildContext, ProductBuildOutput, build_port, implementation


class PrerequisiteError(ValueError):
    """A supported producer is missing an explicit native host seed."""


@dataclass(frozen=True)
class SDKProduct:
    """A pinned existing producer and its logical product dependencies."""

    name: str
    adapter: str
    recipe_path: Path
    recipe: dict
    dependencies: tuple[str, ...]
    port: Port


@dataclass(frozen=True)
class SDKDefinition:
    """SDK metadata and product nodes ordered before their consumers."""

    name: str
    reference: str
    metadata: dict
    products: tuple[SDKProduct, ...]


@dataclass(frozen=True)
class HostSeed:
    """Native executables only; no target platform or runtime can be supplied."""

    tools: Mapping[str, Tool]
    compiler_tools: Mapping[str, Tool]
    python_helper: Tool | None


def definition(ports: Path, graph: Graph, default: str = "default") -> SDKDefinition:
    """Validate every pinned producer and edge before any source fetch or build."""
    references = {port.sdk_selection.reference for port in graph.ports if port.sdk_selection is not None}
    if len(references) > 1:
        raise ValueError("linked graph selects incompatible SDKs")
    if references:
        reference = references.pop()
    else:
        if default == "default":
            _, _, alias = _document(ports, "sdks/default.json")
            default = alias["sdk"]
        reference = "sdks/" + relative_path(default) + ".json"
    _, _, metadata = _document(ports, reference)
    if metadata["host"] != "linux-x86_64" or sys.platform != "linux" or platform.machine() != "x86_64":
        raise PrerequisiteError("SDK producers currently require an x86-64 Linux host")
    declarations = metadata["products"]
    if not isinstance(declarations, dict) or not 1 <= len(declarations) <= 32:
        raise ValueError("SDK product graph must have one to thirty-two nodes")
    nodes = {}
    for name, declaration in declarations.items():
        if set(declaration) != {"adapter", "recipe", "dependencies"}:
            raise ValueError("SDK product declaration fields differ")
        port_reference = canonical_reference(ports, declaration["recipe"])
        directory, selected = producer_metadata(port_reference, ports)
        if selected["build_system"] != declaration["adapter"]:
            raise ValueError("SDK product build system differs from its port")
        recipe = load_policy(port_reference, ports)
        port = Port(
            port_reference,
            directory,
            selected["name"],
            selected["version"],
            "",
            (),
            selected,
            variant=port_reference.partition(":")[2],
        )
        path = directory / "recipe.json"
        dependencies = declaration["dependencies"]
        if not isinstance(dependencies, list) or len(dependencies) > 32 or len(set(dependencies)) != len(dependencies):
            raise ValueError("invalid SDK product dependencies")
        nodes[name] = SDKProduct(name, declaration["adapter"], path, recipe, tuple(dependencies), port)
    expected_edges = {
        "sdk-tooling": (),
        "llvm-host": (),
        "wasi-sysroot": ("compiler", "tooling"),
        "cpython-threaded": ("compiler", "platform", "tooling"),
        "uv-host": (),
    }
    for node in nodes.values():
        if node.adapter not in expected_edges or set(node.dependencies) != set(expected_edges[node.adapter]):
            raise ValueError("SDK producer dependencies differ from its supported interface")
    if (
        metadata["target"] != nodes["cpython"].recipe["target"]
        or metadata["abi"] != nodes["cpython"].recipe["dynamic_abi"]
    ):
        raise ValueError("SDK target or ABI differs from its pinned producers")
    ordered, visiting, done = [], set(), set()

    def visit(name: str) -> None:
        if name in done:
            return
        if name in visiting or name not in nodes:
            raise ValueError("SDK product graph has a cycle or missing dependency")
        visiting.add(name)
        for dependency in nodes[name].dependencies:
            visit(dependency)
        visiting.remove(name)
        done.add(name)
        ordered.append(nodes[name])

    for name in nodes:
        visit(name)
    return SDKDefinition(Path(reference).stem, reference, metadata, tuple(ordered))


def compatible_inputs(record: dict, current: dict) -> bool:
    """Reuse admitted prebuilt binaries independently of seeds for missing nodes.

    Their original build tools remain in the immutable producer receipt. Import
    admission binds source policy and dependency products, without claiming that
    currently configured host seeds were the tools which built those binaries.
    """
    recorded = record["inputs"]
    origin = record.get("origin")
    if isinstance(origin, dict) and origin.get("kind") == "reviewed-authoring-migration-v1":
        fields = ("producer_policy", "implementation", "dependencies")
        return all(recorded[field] == current[field] for field in fields)
    return recorded == current


def product_reference(product: Receipt) -> dict:
    """Retain the original receipt digest and resolved immutable inventory root."""
    return {"root": str(product.root), "manifest": str(product.path), "sha256": product.sha256}


def _publish_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".preparing")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def load_seed(path: Path | None) -> HostSeed:
    """Admit native executable bindings separately from all target products."""
    if path is None:
        return HostSeed({}, {}, None)
    value = read_json(path)
    if set(value) != {"schema_version", "host", "tools", "compiler_tools", "python_helper"}:
        raise ValueError("host seed fields differ; target products cannot be host seeds")
    if value["schema_version"] != 1 or value["host"] != "linux-x86_64":
        raise PrerequisiteError("unsupported host seed platform")
    tools = admit_host_tools(path.parent, value["tools"])
    compiler_tools = {}
    if value["compiler_tools"] and set(value["compiler_tools"]) != {"cc", "cxx", "cmake", "ninja"}:
        raise ValueError("host compiler seed requires cc, cxx, cmake and ninja")
    for name, binding in value["compiler_tools"].items():
        compiler_tools[name] = _seed_tool(path.parent, binding)
    helper = _seed_tool(path.parent, value["python_helper"]) if value["python_helper"] is not None else None
    return HostSeed(tools, compiler_tools, helper)


def _seed_tool(base: Path, binding: dict) -> Tool:
    if set(binding) != {"path", "sha256"}:
        raise ValueError("native seed executable fields differ")
    path = (base / binding["path"]).resolve()
    if file_hash(path) != binding["sha256"]:
        raise ValueError("native seed executable changed")
    return Tool(path, binding["sha256"])


def verify_node(node: SDKProduct, product: Receipt) -> None:
    """Verify original producer provenance and consumed byte inventories."""
    if node.adapter == "sdk-tooling":
        from ports.toolchain.wasi_threads.dynamic import digest, sdk_tooling

        if (
            digest(product.contents) != node.recipe["sdk_tooling_digest"]
            or sdk_tooling(product.root) != product.contents
        ):
            raise ValueError("SDK tooling inventory differs")
        return
    if node.adapter == "cpython-threaded":
        verify_cpython_recipe(product.contents["recipe"])
        verify_files(
            product.root / "rootfs", {name.lstrip("/"): value for name, value in product.contents["files"].items()}
        )
        return
    if node.adapter == "uv-host":
        verify_policy(product.contents["recipe"], node.port.reference)
        if not product.contents["build"]["locked"]:
            raise ValueError("resolver producer differs")
        executable = product.contents["executable"]
        if file_hash(product.root / executable["file"]) != executable["sha256"]:
            raise ValueError("resolver executable inventory differs")
        return
    if node.adapter not in {"llvm-host", "wasi-sysroot"}:
        raise ValueError("unsupported SDK producer: " + node.adapter)
    verify_policy(product.contents["identity"]["recipe"], node.port.reference)
    verify_product(product)


def verify_dependencies(node: SDKProduct, product: Receipt, products: Mapping[str, Receipt]) -> None:
    """Bind embedded producer provenance to the exact selected dependency receipts."""
    if node.adapter == "wasi-sysroot":
        recorded = product.contents["identity"]
        if (
            recorded["compiler"] != products["compiler"].contents
            or recorded["sdk_tooling"] != products["tooling"].contents
        ):
            raise ValueError("platform dependency receipt differs")
    elif node.adapter == "cpython-threaded":
        if product.contents["build_profile"]["sysroot"] != products["platform"].contents:
            raise ValueError("CPython platform dependency receipt differs")


def _produce(
    node: SDKProduct, products: Mapping[str, Receipt], seed: HostSeed, store: Path, work: Path, offline: bool
) -> Receipt:
    """Call the existing producer once its declared products are available."""
    from ports._support.build import check_build_scripts

    check_build_scripts(node.recipe, node.recipe_path.parent)
    if node.adapter == "llvm-host" and set(seed.compiler_tools) != {"cc", "cxx", "cmake", "ninja"}:
        raise PrerequisiteError("missing host compiler seed: bind cc, cxx, cmake and ninja with --host-seed")
    if node.adapter == "wasi-sysroot":
        _require_tools(seed, ("cmake", "ninja"))
    if node.adapter == "cpython-threaded":
        _require_tools(seed, ("make",))
        if seed.python_helper is None:
            raise PrerequisiteError("missing native CPython build helper: bind python_helper with --host-seed")
    if node.adapter == "uv-host" and offline:
        raise PrerequisiteError("offline resolver product is missing; materialize the pinned resolver online first")
    sources = {}
    if node.adapter in {"sdk-tooling", "wasi-sysroot"}:
        names = ("sdk",) if node.adapter == "sdk-tooling" else ("sdk", "wasi_libc")
        sources = {name: fetch(node.recipe[name], store / "sources", offline=offline) for name in names}
    elif node.adapter != "uv-host":
        sources["source"] = fetch(node.recipe["source"], store / "sources", offline=offline)
    source = next(iter(sources.values()), work)
    output = build_port(
        BuildContext(
            node.port,
            source,
            work,
            sources=sources,
            product_dependencies=products,
            host_seed=seed,
            work=work,
            offline=offline,
        )
    )
    if not isinstance(output, ProductBuildOutput):
        raise ValueError("SDK builder must return an unpublished product")
    return Receipt(output.root, output.manifest, file_hash(output.manifest), read_json(output.manifest))


def producer_work(node: SDKProduct, inputs: dict, seed: HostSeed, store: Path) -> Path:
    """Retain compatible LLVM Ninja state; isolate non-resumable attempts.

    LLVM owns patch/configuration admission and immutable product sealing. Other
    producers require absent outputs, so failed attempts stay inspectable while a
    retry receives a fresh output beneath the same input identity directory.
    """
    root = store.resolve() / "sdk-work" / node.name
    if node.adapter == "llvm-host":
        compatibility = {
            "source": node.recipe["source"],
            "tools": {
                name: {"path": str(tool.path), "sha256": tool.sha256} for name, tool in seed.compiler_tools.items()
            },
        }
        return root / "workspaces" / json_hash(compatibility)
    import tempfile

    attempts = root / json_hash(inputs)
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=attempts))
    return attempt / "output"


def _require_tools(seed: HostSeed, names: tuple[str, ...]) -> None:
    missing = set(names) - seed.tools.keys()
    if missing:
        raise PrerequisiteError("missing native host tool bindings: " + ", ".join(sorted(missing)))


def node_inputs(node: SDKProduct, products: Mapping[str, Receipt], seed: HostSeed) -> dict:
    """Invalidate only the producer inputs consumed by this product node."""
    tools = {}
    helper = None
    if node.adapter == "llvm-host":
        tools = seed.compiler_tools
    elif node.adapter == "wasi-sysroot":
        tools = {name: seed.tools[name] for name in ("cmake", "ninja") if name in seed.tools}
    elif node.adapter == "cpython-threaded":
        tools = {name: seed.tools[name] for name in ("make",) if name in seed.tools}
        helper = seed.python_helper.sha256 if seed.python_helper else None
    return {
        "producer_policy": json_hash(policy(node.port.recipe)),
        "implementation": implementation(node.port),
        "dependencies": {name: products[name].sha256 for name in node.dependencies},
        "host_tools": {name: {"sha256": tool.sha256, "receipt": tool.receipt_sha256} for name, tool in tools.items()},
        "python_helper": helper,
    }


def admit_cached_sdk(
    ports: Path,
    graph: Graph,
    store: Path,
    *,
    default: str = "default",
    host_seed: Path | None = None,
) -> MaterializedSDK:
    """Read and admit a complete retained SDK without producing or publishing.

    All five product records must match current inputs and verified dependency
    receipts. Missing or stale inputs fail rather than entering a producer.
    Products, CPython runtime and host tools keep their original inventory roots;
    no locks, directories, registry records or materialized manifests are written.
    The caller must provide a quiescent retained product store.
    """
    from ports._support.build import check_build_scripts

    sdk = definition(ports, graph, default)
    missing = {"compiler", "platform", "tooling", "cpython", "resolver"} - {node.name for node in sdk.products}
    if missing:
        raise ValueError("SDK omits required products: " + ", ".join(sorted(missing)))
    index = store.resolve() / "sdk-products" / sdk.name
    seed_path = host_seed or (index / "host-seed.json" if (index / "host-seed.json").exists() else None)
    seed = load_seed(seed_path)
    products: dict[str, Receipt] = {}
    for node in sdk.products:
        check_build_scripts(node.recipe, node.recipe_path.parent)
        inputs = node_inputs(node, products, seed)
        record = read_json(index / (node.name + ".json"))
        if type(record["schema_version"]) is not int or record["schema_version"] != 1:
            raise ValueError("unsupported retained SDK product record")
        if not compatible_inputs(record, inputs):
            raise ValueError("retained SDK product inputs differ: " + node.name)
        product = receipt(index, record["product"])
        verify_node(node, product)
        verify_dependencies(node, product, products)
        products[node.name] = product
    resolver = products["resolver"]
    executable = resolver.contents["executable"]
    tools = dict(seed.tools)
    tools["uv"] = Tool(resolver.root / executable["file"], executable["sha256"], resolver.path, resolver.sha256)
    entries = {
        name: {
            "provider": "llvm",
            "path": "bin/" + executable,
            "sha256": file_hash(products["compiler"].root / "bin" / executable),
        }
        for name, executable in {
            "cc": "clang",
            "cxx": "clang++",
            "ar": "llvm-ar",
            "ranlib": "llvm-ranlib",
            "strip": "llvm-strip",
        }.items()
    }
    return admit_sdk(
        products["tooling"],
        products["compiler"],
        products["platform"],
        products["cpython"],
        products["cpython"],
        tools,
        entries,
        target=sdk.metadata["target"],
        abi=sdk.metadata["abi"],
    )


def materialize(
    ports: Path,
    graph: Graph,
    store: Path,
    *,
    default: str = "default",
    host_seed: Path | None = None,
    python: bool = False,
    offline: bool = False,
) -> MaterializedSDK:
    """Serialize publication of SDK products, then return concrete driver inputs."""
    sdk = definition(ports, graph, default)
    index = store.resolve() / "sdk-products" / sdk.name
    index.mkdir(parents=True, exist_ok=True)
    with (index / "materialize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _materialize(sdk, store, host_seed, python, offline)


def _materialize(
    sdk: SDKDefinition, store: Path, host_seed: Path | None, python: bool, offline: bool
) -> MaterializedSDK:
    """Resolve SDK products, compile missing nodes and emit a materialized manifest."""
    index = store.resolve() / "sdk-products" / sdk.name
    seed_path = host_seed or (index / "host-seed.json" if (index / "host-seed.json").exists() else None)
    seed = load_seed(seed_path)
    required = {"compiler", "platform", "tooling"}
    if python:
        required.update(("cpython", "resolver"))
    nodes = {node.name: node for node in sdk.products}
    pending = list(required)
    while pending:
        name = pending.pop()
        if name not in nodes:
            raise ValueError("SDK omits required product: " + name)
        for dependency in nodes[name].dependencies:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    products, records = {}, {}
    for node in sdk.products:
        if node.name not in required:
            continue
        from ports._support.build import check_build_scripts

        check_build_scripts(node.recipe, node.recipe_path.parent)
        inputs = node_inputs(node, products, seed)
        path = index / (node.name + ".json")
        record = read_json(path) if path.exists() else None
        if record is not None and compatible_inputs(record, inputs):
            product = receipt(path.parent, record["product"])
            verify_node(node, product)
            verify_dependencies(node, product, products)
            print("ports: SDK product cache hit: " + node.name, file=sys.stderr, flush=True)
        else:
            print("ports: SDK product cache miss: " + node.name, file=sys.stderr, flush=True)
            product = _produce(node, products, seed, store.resolve(), producer_work(node, inputs, seed, store), offline)
            verify_node(node, product)
            verify_dependencies(node, product, products)
            record = {
                "schema_version": 1,
                "product": product_reference(product),
                "inputs": inputs,
                "origin": "producer",
            }
            _publish_json(path, record)
        products[node.name], records[node.name] = product, record
    runtime = None
    if python:
        # Explicit stdlib graph selections assemble from the package-free product.
        # Imported assembled runtimes remain preserved in their original records.
        runtime = products["cpython"]
    tools = dict(seed.tools)
    if python:
        resolver = products["resolver"]
        executable = resolver.contents["executable"]
        tools["uv"] = Tool(resolver.root / executable["file"], executable["sha256"], resolver.path, resolver.sha256)
    entries = {
        name: {
            "provider": "llvm",
            "path": "bin/" + executable,
            "sha256": file_hash(products["compiler"].root / "bin" / executable),
        }
        for name, executable in {
            "cc": "clang",
            "cxx": "clang++",
            "ar": "llvm-ar",
            "ranlib": "llvm-ranlib",
            "strip": "llvm-strip",
        }.items()
    }
    result = admit_sdk(
        products["tooling"],
        products["compiler"],
        products["platform"],
        products.get("cpython"),
        runtime,
        tools,
        entries,
        target=sdk.metadata["target"],
        abi=sdk.metadata["abi"],
    )
    _publish_json(
        store.resolve() / "materialized-sdks" / (json_hash({"sdk": result.identity, "products": records}) + ".json"),
        {
            "schema_version": 1,
            "definition": sdk.reference,
            "identity": result.identity,
            "products": records,
            "host_tools": {name: tool_reference(tool) for name, tool in tools.items()},
        },
    )
    return result
