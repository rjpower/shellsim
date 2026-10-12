"""Execute one graph node; dependency selection and scheduling belong to the planner."""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ports._support.graph import _read
from ports._support.native_adapters import CompilerCacheLauncher
from ports._support.node_build import produce_port
from ports._support.sdk import SDKProduct, _produce, installed_seed, producer_work
from ports._support.sdk_products import Receipt
from ports._support.store import identity
from ports.buildomatic.artifacts import install, register, resolve, upload, write_reference
from ports.buildomatic.delivery import DeliveryStore
from ports.buildomatic.delivery_products import delivery_key, publish_product, publish_runtime
from ports.buildomatic.nodes import Node, Operation, decode
from ports.buildomatic.remote_store import StorageStore
from ports.buildomatic.sdk_artifacts import load_sdk


def launcher() -> CompilerCacheLauncher | None:
    executable = shutil.which("sccache")
    return (
        CompilerCacheLauncher(
            Path(executable), "", tuple((k, v) for k, v in os.environ.items() if k.startswith("SCCACHE_"))
        )
        if executable
        else None
    )


def input_records(node: Node, store: StorageStore) -> dict:
    def read(item):
        record = resolve(store, "node/" + item.key)
        if record is None:
            raise ValueError("missing declared input: " + item.reference)
        return item.name, record

    with ThreadPoolExecutor(max_workers=16) as pool:
        return dict(pool.map(read, node.inputs))


def receipt(record: dict, store: StorageStore, cache: Path) -> Receipt:
    root = install(store, record["artifact"], cache / "artifacts")
    path = root / ".buildomatic-receipt.json"
    write_reference(path, record["receipt"])
    return Receipt(root, path, record["receipt_id"], record["receipt"])


def build(node: Node, records: dict, store: StorageStore, cache: Path, specification: dict) -> dict:
    if node.producer is not None:
        producer = node.producer
        descriptor = SDKProduct(
            producer["name"],
            node.port.recipe["build_system"],
            node.port.directory / "recipe.json",
            producer["recipe"],
            tuple(producer["dependencies"]),
            node.port,
        )
        products = {name: receipt(records[name], store, cache) for name in descriptor.dependencies}
        seed = installed_seed(
            python_version=descriptor.recipe["version"] if descriptor.adapter == "cpython-threaded" else None
        )
        inputs = {"key": node.key}
        product = _produce(
            descriptor,
            products,
            seed,
            cache,
            producer_work(descriptor, inputs, seed, cache),
            specification["offline"],
            jobs=specification["jobs"],
            compiler_cache=launcher(),
        )
        return {
            "reference": node.port.reference,
            "build_key": node.key,
            "artifact": upload(store, product.root, cache / "archives"),
            "receipt": product.contents,
            "receipt_id": product.sha256,
        }
    ports = Path(__file__).resolve().parents[1]
    selected = {node.port.reference: node.port}
    results, components = {}, {}
    for name, record in records.items():
        if name.startswith("sdk:"):
            components[name[4:]] = record
            reference = record["reference"]
            selected[reference] = _read(ports, reference, specification["sdk_name"])
            results[reference] = install(store, record["artifact"], cache / "artifacts")
        else:
            reference = record["reference"]
            selected[reference] = _read(ports, reference, specification["sdk_name"])
            results[reference] = install(store, record["artifact"], cache / "artifacts")
    sdk = (
        load_sdk(
            store,
            {"products": components, "target": specification["target"], "dynamic_abi": specification["abi"]},
            cache / "artifacts",
        )
        if components
        else None
    )
    root = produce_port(
        ports,
        node.port,
        cache / "ports",
        node.key,
        sdk,
        selected,
        results,
        jobs=specification["jobs"],
        offline=specification["offline"],
        compiler_cache=launcher(),
    )
    return {
        "reference": node.port.reference,
        "build_key": node.key,
        "artifact": upload(store, root, cache / "archives"),
        "delivery_key": delivery_key(node.key),
        "delivery": publish_product(
            node.port, root, node.key, DeliveryStore(store.prefix), record_key=delivery_key(node.key)
        ),
    }


def delivery(node: Node, records: dict, store: StorageStore, cache: Path) -> dict:
    product = records["product"]
    if product.get("delivery_key") == node.key or node.port.recipe["build_system"] == "pure-wheel":
        return product
    root = install(store, product["artifact"], cache / "artifacts")
    return {
        **product,
        "delivery_key": node.key,
        "delivery": publish_product(
            node.port, root, product["build_key"], DeliveryStore(store.prefix), record_key=node.key
        ),
    }


def runtime(node: Node, records: dict, store: StorageStore, cache: Path) -> dict:
    product = records["product"]
    root = install(store, product["artifact"], cache / "artifacts")
    raw_key = identity(
        {"operation": "runtime", "format_version": 1, "inputs": {item.name: item.key for item in node.inputs}}
    )
    return {"runtime": publish_runtime(root, raw_key, DeliveryStore(store.prefix), record_key=node.key)}


def execute(node: Node, store: StorageStore, cache: Path, specification: dict) -> dict:
    if node.operation is not Operation.CHECK and (cached := resolve(store, node.record_key)) is not None:
        return cached
    records = input_records(node, store)
    if node.operation is Operation.BUILD:
        result = build(node, records, store, cache, specification)
    elif node.operation is Operation.DELIVERY:
        result = delivery(node, records, store, cache)
    elif node.operation is Operation.RUNTIME:
        result = runtime(node, records, store, cache)
    else:
        from ports.buildomatic.checks import execute_check

        result = execute_check(node, records, store, cache, specification)
    register(store, node.record_key, result)
    return result


def main() -> None:
    from rigging.log_setup import configure_logging

    configure_logging()
    specification = json.loads((Path(os.environ["BUILD_INPUT_recipe"]) / "node.json").read_text())
    node = decode(specification, Path(__file__).resolve().parents[1])
    store = StorageStore(specification["prefix"])
    cache = Path(os.environ.get("BUILDOMATIC_CACHE", "/app/.buildomatic"))
    cache.mkdir(parents=True, exist_ok=True)
    result = execute(node, store, cache, specification)
    write_reference(Path(os.environ["BUILD_OUTPUT_DIR"]) / "result.json", result)


if __name__ == "__main__":
    main()
