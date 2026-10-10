"""Execute one admitted recipe offline, refusing predecessor cache misses.

This script runs from the transported code root. It imports no backend SDK and
publishes only a complete verified selected result plus its canonical key.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def execute(code: Path, recipe: Path, inputs: Path, output: Path, work: Path) -> None:
    """Preload verified predecessors, import an SDK, and build one selected node."""
    sys.path.insert(0, str(code))
    from ports._support.graph import plan
    from ports._support.native_adapters import CompilerCacheLauncher
    from ports._support.runner import build_graph
    from ports._support.store import relative_path, verify

    specification = json.loads((recipe / "node.json").read_text())
    if specification["schema_version"] != 1:
        raise ValueError("unsupported port action schema")
    graph = plan(code / "ports", [specification["reference"]], default_sdk=specification["default_sdk"])
    if {port.reference: port.digest for port in graph.ports} != specification["recipes"]:
        raise ValueError("transported recipe closure differs from the admitted graph")
    store = work / "store"
    store.mkdir(parents=True, exist_ok=False)
    if (recipe / "sources").exists():
        shutil.copytree(recipe / "sources", store / "sources")
    admitted = dict(specification["products"])
    for reference, action_id in specification["predecessors"].items():
        dependency = inputs / relative_path(action_id)
        metadata = json.loads((dependency / "node.json").read_text())
        if metadata["reference"] != reference:
            raise ValueError("transferred predecessor reference differs")
        result = dependency / "result"
        receipt = json.loads((result / "build-receipt.json").read_text())
        receipt = verify(result, receipt["inputs"])
        if receipt["key"] != metadata["key"]:
            raise ValueError("transferred predecessor key differs")
        admitted[reference] = receipt["key"]
        destination = store / "results" / receipt["key"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(result, destination)
        verify(destination, receipt["inputs"])
    sdk = None
    if (inputs / "sdk").exists():
        from ports.buildomatic.portable import import_sdk, original_root_bindings

        descriptor = inputs / "sdk/sdk.json"
        sdk = import_sdk(descriptor, work / "sdk", original_bindings=True, bindings=original_root_bindings(descriptor))
    cache = specification["compiler_cache"]
    launcher = (
        None
        if cache is None
        else CompilerCacheLauncher(Path(cache["path"]), cache["sha256"], tuple(cache["environment"].items()))
    )
    build = build_graph(
        code / "ports",
        [specification["reference"]],
        sdk,
        store,
        offline=True,
        jobs=specification["jobs"],
        default_sdk=specification["default_sdk"],
        admitted_predecessors=admitted,
        compiler_cache=launcher,
    )
    result = build.results[specification["reference"]]
    receipt = json.loads((result / "build-receipt.json").read_text())
    verify(result, receipt["inputs"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(result, output / "result")
    verify(output / "result", receipt["inputs"])
    (output / "node.json").write_text(
        json.dumps({"reference": specification["reference"], "key": receipt["key"]}, sort_keys=True)
    )


def main() -> None:
    """Use only the private workspace and explicit core input/output mounts."""
    execute(
        Path(os.environ["BUILD_INPUT_code"]),
        Path(os.environ["BUILD_INPUT_recipe"]),
        Path.cwd() / "inputs",
        Path(os.environ["BUILD_OUTPUT_DIR"]),
        Path.cwd() / "port-work",
    )


if __name__ == "__main__":
    main()
