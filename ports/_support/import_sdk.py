"""Import historical cohort products into the graph store after verification.

The historical descriptor is accepted only by this migration command. Ordinary
graph builds resolve pinned SDK products and never read a cohort descriptor.
"""

from pathlib import Path

from ports._support.sdk_products import (
    admit_host_tools,
    admit_sdk,
    file_hash,
    read_json,
    receipt,
)


def load_legacy_cohort(path: Path, *, expected_sha256: str | None = None):
    """Verify a setup descriptor and every consumed target receipt and header.

    Relative roots resolve against the explicitly supplied setup descriptor.
    Local setup may reference external immutable build directories. Exported
    target cohorts use only contained roots; host tools remain explicit bindings.
    """
    path = path.resolve()
    if expected_sha256 is not None and file_hash(path) != expected_sha256:
        raise ValueError("build cohort descriptor digest differs")
    value = read_json(path)
    if set(value) != {
        "schema_version",
        "target",
        "dynamic_abi",
        "sdk",
        "llvm",
        "sysroot",
        "cpython",
        "runtime",
        "host_tools",
        "target_tools",
    }:
        raise ValueError("build cohort descriptor fields differ")
    if (
        value["schema_version"] != 1
        or value["target"] != "wasm32-wasip1-threads"
        or value["dynamic_abi"] != "shellsim-wasi-sdk34-cpython3137-threads-v3"
    ):
        raise ValueError("unsupported build cohort profile")
    sdk, llvm, sysroot = (receipt(path.parent, value[name]) for name in ("sdk", "llvm", "sysroot"))
    cpython = receipt(path.parent, value["cpython"]) if value["cpython"] is not None else None
    runtime = receipt(path.parent, value["runtime"]) if value["runtime"] is not None else None
    return admit_sdk(
        sdk,
        llvm,
        sysroot,
        cpython,
        runtime,
        admit_host_tools(path.parent, value["host_tools"]),
        value["target_tools"],
        target=value["target"],
        abi=value["dynamic_abi"],
    )


def import_products(path: Path, store: Path) -> Path:
    """Verify and register existing inventories as optional SDK product cache hits."""
    import json

    from ports._support.graph import Graph
    from ports._support.sdk import _publish_json, definition, product_reference, verify_node
    from ports._support.sdk_products import json_hash, tool_reference

    context = load_legacy_cohort(path)
    ports = Path(__file__).resolve().parents[1]
    sdk = definition(ports, Graph((), ()))
    products = {"tooling": context.sdk, "compiler": context.llvm, "platform": context.sysroot}
    if context.cpython_manifest is not None:
        products["cpython"] = context.cpython_manifest
    resolver = context.host_tools.get("uv")
    if resolver is not None:
        products["resolver"] = receipt(
            resolver.receipt_path.parent,
            {
                "root": str(resolver.path.parent),
                "manifest": str(resolver.receipt_path),
                "sha256": resolver.receipt_sha256,
            },
        )
    index = store.resolve() / "sdk-products" / sdk.name
    for node in sdk.products:
        if node.name not in products:
            continue
        product = products[node.name]
        verify_node(node, product)
        record = {
            "schema_version": 1,
            "product": product_reference(product),
            "inputs": {
                "producer_policy": json_hash(node.recipe),
                "dependencies": {name: products[name].sha256 for name in node.dependencies},
            },
            "origin": {"kind": "verified-legacy-import", "descriptor_sha256": file_hash(path)},
        }
        if node.name == "cpython":
            record["runtime"] = product_reference(context.runtime)
        destination = index / (node.name + ".json")
        if destination.exists() and json.loads(destination.read_text()) != record:
            raise FileExistsError(destination)
        _publish_json(destination, record)
    seed = {
        "schema_version": 1,
        "host": "linux-x86_64",
        "tools": {name: tool_reference(tool) for name, tool in context.host_tools.items() if name != "uv"},
        "compiler_tools": {},
        "python_helper": None,
    }
    _publish_json(index / "host-seed.json", seed)
    return index


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-cohort", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    print(import_products(args.legacy_cohort, args.store))


if __name__ == "__main__":
    main()
