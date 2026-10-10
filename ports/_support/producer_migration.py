"""Explicit, frozen admission of products from the reviewed authoring migration.

This command changes cache index bindings only. Product receipts and inventories
retain their original identities. Ordinary lookup never invokes migration, and
changes to current producer code cannot use this reviewed equivalence record.
"""

from __future__ import annotations

import fcntl
from pathlib import Path

from ports._support.graph import Graph
from ports._support.sdk import (
    _publish_json,
    definition,
    load_seed,
    node_inputs,
    verify_dependencies,
    verify_node,
)
from ports._support.sdk_products import json_hash, read_json, receipt
from ports.api import implementation


def admit_migration(node, recorded_policy: str, product) -> None:
    """Require known original policy and this migration's reviewed implementation."""
    registry = read_json(Path(__file__).with_name("producer-migration-v1.json"))
    entry = registry["products"][node.name]
    if recorded_policy not in entry["source_policy_sha256"]:
        raise ValueError("producer policy is not covered by the reviewed migration")
    if json_hash(implementation(node.port)) != entry["implementation_sha256"]:
        raise ValueError("producer implementation differs from the reviewed migration")
    recorded = product.contents.get("recipe", product.contents.get("identity", {}).get("recipe"))
    if recorded is not None and json_hash(recorded) not in entry["receipt_policy_sha256"]:
        raise ValueError("original producer implementation is not covered by the reviewed migration")
    verify_node(node, product)


def migrate_products(store: Path, *, ports: Path | None = None) -> Path:
    """Verify all existing entries before atomically replacing their index bindings."""
    root = ports or Path(__file__).resolve().parents[1]
    sdk = definition(root, Graph((), ()))
    index = store.resolve() / "sdk-products" / sdk.name
    if not index.is_dir():
        raise ValueError("SDK cache to migrate does not exist")
    with (index / "materialize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        seed = load_seed(index / "host-seed.json" if (index / "host-seed.json").exists() else None)
        products, updates = {}, {}
        for node in sdk.products:
            path = index / (node.name + ".json")
            if not path.exists():
                continue
            record = read_json(path)
            if any(name not in products for name in node.dependencies):
                raise ValueError("SDK migration requires its recorded dependency products")
            product = receipt(index, record["product"])
            current = node_inputs(node, products, seed)
            if record["inputs"] == current:
                verify_node(node, product)
                verify_dependencies(node, product, products)
                products[node.name] = product
                continue
            if record["inputs"]["dependencies"] != current["dependencies"]:
                raise ValueError("SDK migration dependency receipt identities differ")
            admit_migration(node, record["inputs"]["producer_policy"], product)
            verify_dependencies(node, product, products)
            updated = {
                **record,
                "inputs": current,
                "origin": {
                    "kind": "reviewed-authoring-migration-v1",
                    "previous": record["origin"],
                    "source_policy_sha256": record["inputs"]["producer_policy"],
                },
            }
            updates[path] = updated
            products[node.name] = product
        for path, updated in updates.items():
            _publish_json(path, updated)
    return index


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    args = parser.parse_args()
    print(migrate_products(args.store))


if __name__ == "__main__":
    main()
