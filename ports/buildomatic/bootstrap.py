"""Build SDK components on an existing Iris service and return store metadata.

The client transports declared producer code. Compiler products and native
workspaces stay on workers; adding Python or the resolver reuses those products.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from ports._support.graph import Graph, plan
from ports._support.sdk import definition
from ports._support.store import identity
from ports.buildomatic import Action, BuildRequest, BuildState, InputMount, LocalStore, capture_tree
from ports.buildomatic.artifacts import write_reference
from ports.buildomatic.backends.iris import backend_context_factory
from ports.buildomatic.ports import _stage_code, code_files
from ports.buildomatic.sdk_artifacts import sdk_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iris-service", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True, help="client transport and metadata directory")
    parser.add_argument("--ports", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sdk", default="default")
    parser.add_argument("--jobs", type=int, default=128)
    parser.add_argument("--python", action="store_true")
    parser.add_argument("--resolver", action="store_true")
    parser.add_argument("--build-key", help="new request identity after a failed action")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 256:
        parser.error("--jobs requires 1..256")
    sdk = definition(args.ports, Graph((), ()), args.sdk)
    graph = plan(args.ports, [node.port.reference for node in sdk.products], default_sdk=sdk.name)
    descriptor = json.loads(args.iris_service.read_text())
    prefix = descriptor.get("cache_prefix") or descriptor["prefix"]
    args.store.mkdir(parents=True, exist_ok=True)
    factory = backend_context_factory(
        job_id=descriptor["job_id"],
        cluster_name=descriptor["cluster_name"],
        controller_url=descriptor["controller_url"],
        workspace=Path(descriptor["workspace"]),
        prefix=descriptor["prefix"],
        cache_prefix=prefix,
    )
    required = {"compiler", "platform", "tooling"}
    if args.python or args.resolver:
        required.add("cpython")
    if args.resolver:
        required.add("resolver")
    with factory() as backend:
        cached = backend.resolve_artifact("sdk/" + sdk.name)
        if cached and cached["inputs"] == sdk_inputs(args.ports, sdk.name) and required <= cached["products"].keys():
            write_reference(args.store / "sdk.json", cached)
            print("buildomatic: SDK cache hit " + sdk.name)
            return
    local = LocalStore(args.store / "transport")
    with tempfile.TemporaryDirectory(prefix="sdk-code-", dir=args.store) as temporary:
        root = Path(temporary)
        _stage_code(code_files(args.ports, graph), root)
        for module in Path(__file__).parent.rglob("*.py"):
            if "tests" in module.relative_to(Path(__file__).parent).parts:
                continue
            destination = root / "ports/buildomatic" / module.relative_to(Path(__file__).parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(module, destination)
        bundle = capture_tree(root, local)
    argv = [
        "python",
        "-m",
        "ports.buildomatic.sdk_worker",
        "--prefix",
        prefix,
        "--sdk",
        sdk.name,
        "--jobs",
        str(args.jobs),
    ]
    if args.python:
        argv.append("--python")
    if args.resolver:
        argv.append("--resolver")
    action = Action(
        "sdk",
        tuple(argv),
        inputs=(InputMount("code", bundle),),
        env=(("PYTHONPATH", "inputs/code"),),
        timeout_seconds=14400,
        max_attempts=1,
    )
    request = BuildRequest(identity({"action": asdict(action), "build_key": args.build_key}), (action,))
    with factory() as backend:
        backend.upload_bundles(local, [bundle])
        build_id = backend.submit(request)
        write_reference(args.store / "request.json", {"build_id": build_id})
        print("buildomatic: submitted " + build_id, flush=True)
        while (result := backend.get(build_id)).state not in {
            BuildState.SUCCEEDED,
            BuildState.FAILED,
            BuildState.CANCELLED,
        }:
            time.sleep(1)
        if result.state is not BuildState.SUCCEEDED:
            raise RuntimeError(str(backend.get_logs(build_id)))
        manifest = backend.resolve_artifact("sdk/" + sdk.name)
        write_reference(args.store / "sdk.json", manifest)
        backend.acknowledge(build_id)
        print("buildomatic: SDK available at " + prefix + "/sdk/" + sdk.name)


if __name__ == "__main__":
    main()
