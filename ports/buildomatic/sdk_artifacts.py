"""Install selected SDK components from canonical artifact references."""

from __future__ import annotations

from pathlib import Path

from ports._support.sdk import installed_seed
from ports._support.sdk_products import Receipt, Tool, admit_sdk
from ports.buildomatic.artifacts import install, write_reference
from ports.buildomatic.contracts import Store


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
    python = products.get("cpython")
    tools = dict(installed_seed(python_version=python.contents["recipe"]["version"] if python else None).tools)
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
