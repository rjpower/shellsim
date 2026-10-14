"""Product identities, asset addresses and completed-node records.

Packaging revisions change derived products without invalidating raw builds.
Only Wasmtime-dependent products include the serialization ABI identity. Completion
records retain their existing addresses so this cleanup reuses cached builds.
"""

from __future__ import annotations

from ports._support.graph import Port
from ports._support.store import identity, relative_path
from ports.buildomatic.artifacts import register, resolve
from ports.buildomatic.contracts import Store

# Change when packaging changes its output or installation semantics.
DELIVERY_REVISION = 5
RUNTIME_REVISION = 3


def node_key(operation: str, **inputs) -> str:
    return identity({"operation": operation, **inputs})


def record_key(key: str) -> str:
    return "node/" + relative_path(key)


def asset_path(key: str, name: str) -> str:
    return f"assets/{relative_path(key)}/{relative_path(name)}"


def compiler_identity(port: Port | None = None) -> dict[str, str] | None:
    """Key native companions on load compatibility; pure wheels and data have no ABI key."""
    if port is not None and port.recipe["build_system"] in {"pure-wheel", "source-tree", "llvm-guest-sdk"}:
        return None
    from shellsim import _native

    return {"build_id": _native.precompiled_wasm_build_id(), "target": _native.precompiled_wasm_target()}


def delivery_key(product_key: str, *, port: Port) -> str:
    compiler = compiler_identity(port)
    return node_key(
        "delivery",
        product=product_key,
        format_version=DELIVERY_REVISION,
        **({"wasmtime": compiler} if compiler is not None else {}),
    )


def file_bundle_key(product_key: str) -> str:
    """File bundles depend on packaging rules, never on the Wasmtime compiler."""
    return node_key("files", product=product_key, format_version=DELIVERY_REVISION)


def runtime_delivery_key(product_key: str) -> str:
    return node_key("runtime", product=product_key, format_version=RUNTIME_REVISION, wasmtime=compiler_identity())


def runtime_archive_key(product_key: str) -> str:
    return node_key("runtime", format_version=RUNTIME_REVISION, inputs={"product": product_key})


class BuildCache:
    """Read metadata without fetching bytes; commit only completed node outputs."""

    def __init__(self, store: Store):
        self.store = store

    def get(self, key: str) -> dict | None:
        return resolve(self.store, record_key(key))

    def put(self, key: str, result: dict) -> None:
        register(self.store, record_key(key), result)
