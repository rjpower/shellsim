"""Install selected SDK components from canonical artifact references."""

from __future__ import annotations

from pathlib import Path

from ports._support.graph import Graph
from ports._support.producer_policy import policy
from ports._support.sdk import definition, installed_seed
from ports._support.sdk_products import Receipt, Tool, admit_sdk
from ports.api import implementation
from ports.buildomatic.artifacts import install, write_reference
from ports.buildomatic.contracts import Store


def sdk_inputs(ports: Path, name: str) -> dict:
    """Track producer recipes and build code without inspecting host tools."""
    sdk = definition(ports, Graph((), ()), name)
    return {
        "definition": sdk.metadata,
        "producers": {
            node.name: {"policy": policy(node.port.recipe), "implementation": implementation(node.port)}
            for node in sdk.products
        },
    }


def load_sdk(store: Store, manifest: dict, cache: Path):
    """Fetch only components listed in the SDK manifest, without host imports."""
    products = {}
    for role, component in manifest["products"].items():
        root = install(store, component["artifact"], cache)
        receipt_path = root / ".buildomatic-receipt.json"
        write_reference(receipt_path, component["receipt"])
        products[role] = Receipt(root, receipt_path, component["receipt_id"], component["receipt"])
    compiler = products["compiler"]
    # sccache probes Clang without the port's command-line target. This compiler
    # contains only WebAssembly, so give its standard config the SDK default.
    for driver in ("clang", "clang++"):
        (compiler.root / "bin" / (driver + ".cfg")).write_text("--target=" + manifest["target"] + "\n")
    entries = {}
    for name, executable in {
        "cc": "clang",
        "cxx": "clang++",
        "ar": "llvm-ar",
        "ranlib": "llvm-ranlib",
        "strip": "llvm-strip",
    }.items():
        path = "bin/" + executable
        real = "bin/" + compiler.contents.get("symlinks", {}).get(path, executable)
        entries[name] = {"provider": "llvm", "path": path, "sha256": compiler.contents["artifacts"][real]}
    tools = dict(installed_seed().tools)
    if resolver := products.get("resolver"):
        executable = resolver.contents["executable"]
        tools["uv"] = Tool(resolver.root / executable["file"], executable["sha256"], resolver.path, resolver.sha256)
    return admit_sdk(
        products["tooling"],
        compiler,
        products["platform"],
        products.get("cpython"),
        products.get("cpython"),
        tools,
        entries,
        target=manifest["target"],
        abi=manifest["dynamic_abi"],
    )
