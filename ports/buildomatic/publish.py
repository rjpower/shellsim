"""Explicitly fetch products or run native guest checks after a remote build.

Build commands return references. This workflow fetches bytes only when a caller
requests an installation or a guest probe; compiler/SDK work stays on Iris.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from ports._support.graph import plan
from ports._support.store import identity
from ports.buildomatic import Action, BuildRequest, BuildState, InputMount, LocalStore, capture_tree, extract_tree
from ports.buildomatic.artifacts import install
from ports.buildomatic.backends.iris import backend_context_factory
from ports.buildomatic.ports import _stage_code, code_files


def run_guest(executable: Path, libraries: Path, output: Path, *, cpu: int, memory: int) -> None:
    """Run saved probe bytes locally so budget changes need no remote work."""
    from shellsim import Environment

    env = Environment(cpu=cpu, memory=memory)
    env.mkdir("/lib", parents=True)
    env.write_file("/work/probe.wasm", executable.read_bytes(), mode=0o755)
    for library in libraries.glob("*.so"):
        env.write_file("/lib/" + library.name, library.read_bytes(), mode=0o755)
    outcome = env.run("/work/probe.wasm")
    print(outcome.stdout.decode(), end="")
    (output / "guest.stdout").write_bytes(outcome.stdout)
    (output / "guest.stderr").write_bytes(outcome.stderr)
    if outcome.returncode:
        raise RuntimeError(outcome.stderr.decode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "check", "guest"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--iris-service", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=100_000_000_000)
    parser.add_argument("--memory", type=int, default=8 * 1024**3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "guest":
        run_guest(args.output / "acceptance.wasm", args.output / "lib", args.output, cpu=args.cpu, memory=args.memory)
        return
    if args.results is None or args.iris_service is None:
        parser.error("fetch and check require --results and --iris-service")
    references = json.loads(args.results.read_text())
    service = json.loads(args.iris_service.read_text())
    factory = backend_context_factory(
        job_id=service["job_id"],
        cluster_name=service["cluster_name"],
        controller_url=service["controller_url"],
        workspace=Path(service["workspace"]),
        prefix=service["prefix"],
        cache_prefix=service.get("cache_prefix"),
    )
    with factory() as backend:
        if args.command == "fetch":
            for name, product in references["products"].items():
                print(name, install(backend.store, product["artifact"], args.output))
            return
        ports = Path(__file__).resolve().parents[1]
        graph = plan(ports, list(references["products"]), default_sdk=references["sdk_name"])
        local = LocalStore(args.output / "transport")
        selected = {port.reference: port for port in graph.ports}
        for port in graph.ports:
            if port.reference not in references["products"]:
                continue
            for test_index, test in enumerate(port.recipe["tests"]):
                if test["kind"] != "native":
                    raise ValueError("this check workflow currently supports native probes")
                with tempfile.TemporaryDirectory(prefix="probe-", dir=args.output) as temporary:
                    root = Path(temporary)
                    code = root / "code"
                    code.mkdir()
                    files = code_files(ports, graph)
                    if references["sdk"]:
                        from ports._support.sdk import definition

                        for product in definition(ports, graph, references["sdk_name"]).products:
                            files[str(product.recipe_path.relative_to(ports))] = product.recipe_path
                    _stage_code(files, code)
                    probe = port.directory / test["source"]
                    destination = code / "ports" / probe.relative_to(ports)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(probe, destination)

                    for module in Path(__file__).parent.rglob("*.py"):
                        if "tests" in module.relative_to(Path(__file__).parent).parts:
                            continue
                        destination = code / "ports/buildomatic" / module.relative_to(Path(__file__).parent)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(module, destination)
                    recipe = root / "recipe"
                    recipe.mkdir()
                    spec = {
                        "reference": port.reference,
                        "test": test,
                        "sdk": references["sdk"],
                        "product": references["products"][port.reference],
                        "prefix": references["prefix"],
                    }
                    closure, pending = {port.reference}, [port.reference]
                    while pending:
                        node = selected[pending.pop()]
                        for edge in node.dependencies:
                            if edge.recipe not in closure:
                                closure.add(edge.recipe)
                                pending.append(edge.recipe)
                    spec["products"] = {
                        name: references["products"][name] for name in sorted(closure) if name in references["products"]
                    }
                    (recipe / "node.json").write_text(json.dumps(spec))
                    code_bundle, recipe_bundle = capture_tree(code, local), capture_tree(recipe, local)
                    backend.upload_bundles(local, [code_bundle, recipe_bundle])
                    action = Action(
                        "probe",
                        ("python", "-m", "ports.buildomatic.native_check_worker"),
                        inputs=(InputMount("code", code_bundle), InputMount("recipe", recipe_bundle)),
                        env=(("PYTHONPATH", "inputs/code"),),
                        max_attempts=1,
                    )
                    build_id = backend.submit(
                        BuildRequest(identity({"spec": spec, "code": code_bundle.digest}), (action,))
                    )
                    while (result := backend.get(build_id)).state not in {
                        BuildState.SUCCEEDED,
                        BuildState.FAILED,
                        BuildState.CANCELLED,
                    }:
                        time.sleep(1)
                    if result.state is not BuildState.SUCCEEDED:
                        raise RuntimeError(str(backend.get_logs(build_id)))
                    bundle = result.nodes[0].bundle
                    backend.download_bundles(local, [bundle])
                    output = root / "result"
                    extract_tree(bundle, local, output)
                    executable = (output / "acceptance.wasm").read_bytes()
                    (args.output / "acceptance.wasm").write_bytes(executable)
                    if (output / "lib").is_dir():
                        shutil.copytree(output / "lib", args.output / "lib", dirs_exist_ok=True)
                    run_guest(
                        args.output / "acceptance.wasm",
                        args.output / "lib",
                        args.output,
                        cpu=args.cpu,
                        memory=args.memory,
                    )
                    evidence = args.output / (port.name + "-" + str(test_index))
                    evidence.mkdir(exist_ok=True)
                    for filename in ("acceptance.wasm", "guest.stdout", "guest.stderr"):
                        shutil.copyfile(args.output / filename, evidence / filename)
                    backend.acknowledge(build_id)


if __name__ == "__main__":
    main()
