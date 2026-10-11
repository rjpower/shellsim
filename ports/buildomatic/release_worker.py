"""Assemble a public runtime release beside cached SDK and port products.

Only release assets leave the worker. The compiler and its workspaces stay in
the canonical store and worker cache, just as they do for ordinary port builds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ports._support.graph import plan
from ports._support.runner import GraphBuild, publish_graph
from ports.buildomatic.artifacts import install
from ports.buildomatic.ports import _product_results
from ports.buildomatic.remote_store import RemoteStore
from ports.buildomatic.sdk_artifacts import load_sdk


def main() -> None:
    specification = json.loads((Path(os.environ["BUILD_INPUT_recipe"]) / "node.json").read_text())
    ports = Path(__file__).resolve().parents[1]
    graph = plan(ports, specification["roots"], default_sdk=specification["sdk_name"])
    store = RemoteStore(specification["prefix"])
    cache = Path("/app/.buildomatic/artifacts")
    sdk = load_sdk(store, specification["sdk"], cache)
    results, _ = _product_results(graph, sdk)
    for reference, product in specification["products"].items():
        results[reference] = install(store, product["artifact"], cache)
    build = GraphBuild(graph, results, {port.reference: sdk for port in graph.ports})
    descriptor = publish_graph(build, sdk, Path(os.environ["BUILD_OUTPUT_DIR"]) / "release")
    print("buildomatic: release assembled " + str(descriptor), flush=True)


if __name__ == "__main__":
    main()
