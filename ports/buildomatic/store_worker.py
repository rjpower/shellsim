"""Build a port using remote references and publish its result directly."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ports._support.native_adapters import CompilerCacheLauncher
from ports._support.runner import build_graph
from ports.buildomatic.artifacts import install, register, upload, write_reference
from ports.buildomatic.remote_store import RemoteStore
from ports.buildomatic.sdk_artifacts import load_sdk


def main() -> None:
    specification = json.loads((Path(os.environ["BUILD_INPUT_recipe"]) / "node.json").read_text())
    store = RemoteStore(specification["prefix"])
    cache = Path("/app/.buildomatic/artifacts")
    sdk = load_sdk(store, specification["sdk"], cache) if specification["sdk"] else None
    work = Path("/app/.buildomatic/ports")
    work.mkdir(parents=True, exist_ok=True)
    admitted = {}
    for reference, predecessor in specification["predecessors"].items():
        if "action_id" in predecessor:
            predecessor = json.loads((Path("inputs") / predecessor["action_id"] / "result.json").read_text())
        root = install(store, predecessor["artifact"], cache)
        receipt = json.loads((root / "build-receipt.json").read_text())
        destination = work / "results" / receipt["key"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        if not destination.exists():
            shutil.copytree(root, destination)
        admitted[reference] = receipt["key"]
    cache_settings = tuple((key, value) for key, value in os.environ.items() if key.startswith("SCCACHE_"))
    launcher = CompilerCacheLauncher(Path("/usr/local/bin/sccache"), "", cache_settings)
    build = build_graph(
        Path(__file__).resolve().parents[1],
        [specification["reference"]],
        sdk,
        work,
        jobs=specification["jobs"],
        admitted_predecessors=admitted,
        compiler_cache=launcher,
        default_sdk=specification["sdk_name"],
    )
    root = build.results[specification["reference"]]
    result = {
        "reference": specification["reference"],
        "build_key": specification["build_key"],
        "artifact": upload(store, root, work / "archives"),
    }
    register(store, "port/" + specification["build_key"], result)
    write_reference(Path(os.environ["BUILD_OUTPUT_DIR"]) / "result.json", result)


if __name__ == "__main__":
    main()
