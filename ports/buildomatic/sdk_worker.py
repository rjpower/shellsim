"""Bootstrap an SDK on its build worker and publish store references.

The coordinator supplies code and configuration. It never receives an installed
SDK. Normal installed tools bootstrap the compiler; sccache and Ninja state
remain on the worker for retries.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

from ports._support.graph import Graph
from ports._support.native_adapters import CompilerCacheLauncher
from ports._support.sdk import installed_seed, materialize
from ports._support.sdk_products import Tool
from ports.buildomatic.artifacts import register, resolve, upload, write_reference
from ports.buildomatic.remote_store import RemoteStore
from ports.buildomatic.sdk_artifacts import sdk_inputs


def bootstrap(
    ports: Path,
    prefix: str,
    name: str,
    cache: Path,
    output: Path,
    jobs: int,
    *,
    python: bool = False,
    resolver: bool = False,
) -> dict:
    """Build missing products and publish one small SDK manifest to the store."""
    store = RemoteStore(prefix)
    key = "sdk/" + name
    inputs = sdk_inputs(ports, name)
    cached = resolve(store, key)
    required = {"compiler", "platform", "tooling"}
    if python:
        required.add("cpython")
    if resolver:
        required.add("resolver")
    if cached and cached.get("inputs") == inputs and required <= cached["products"].keys():
        write_reference(output, cached)
        return cached
    cache.mkdir(parents=True, exist_ok=True)
    seed = replace(installed_seed(), python_helper=Tool(Path("/opt/buildomatic/venv/bin/python"), ""))
    launcher = CompilerCacheLauncher(
        Path(shutil.which("sccache")),
        "",
        tuple((key, value) for key, value in os.environ.items() if key.startswith("SCCACHE_")),
    )
    os.environ.setdefault("CARGO_TARGET_DIR", str(cache.parent / "cargo-build"))
    sdk = materialize(
        ports,
        Graph((), ()),
        cache,
        default=name,
        seed=seed,
        jobs=jobs,
        compiler_cache=launcher,
        python=python,
        resolver=resolver,
    )
    products = {}
    selected = {"tooling": sdk.sdk, "compiler": sdk.llvm, "platform": sdk.sysroot}
    if python:
        selected["cpython"] = sdk.cpython_manifest
    if resolver:
        from ports._support.sdk_products import receipt

        record = json.loads((cache / "sdk-products" / name / "resolver.json").read_text())
        selected["resolver"] = receipt(cache / "sdk-products" / name, record["product"])
    for role, product in selected.items():
        if cached and role in cached["products"] and cached["products"][role]["receipt_id"] == product.sha256:
            products[role] = cached["products"][role]
            continue
        print("buildomatic: upload SDK component " + role, flush=True)
        products[role] = {
            "artifact": upload(store, product.root, cache / "archives"),
            "receipt": product.contents,
            "receipt_id": product.sha256,
        }
    result = {
        "schema_version": 2,
        "inputs": inputs,
        "identity": sdk.identity,
        "target": sdk.target,
        "dynamic_abi": sdk.dynamic_abi,
        "products": products,
    }
    register(store, key, result)
    write_reference(output, result)
    print("buildomatic: SDK available at " + prefix + "/" + key, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--sdk", default="default")
    parser.add_argument("--jobs", type=int, default=128)
    parser.add_argument("--python", action="store_true", help="include CPython headers and runtime")
    parser.add_argument("--resolver", action="store_true", help="also build the patched host WASI resolver")
    parser.add_argument("--cache", type=Path, default=Path("/app/.buildomatic/sdk-cache"))
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("BUILD_OUTPUT_DIR", ".")) / "sdk.json")
    args = parser.parse_args()
    # Resolve the default alias so all clients use the canonical SDK name.
    ports = Path(__file__).resolve().parents[1]
    name = args.sdk
    if name == "default":
        name = json.loads((ports / "sdks/default.json").read_text())["sdk"]
    bootstrap(
        ports,
        args.prefix,
        name,
        args.cache,
        args.output,
        args.jobs,
        python=args.python or args.resolver,
        resolver=args.resolver,
    )


if __name__ == "__main__":
    main()
