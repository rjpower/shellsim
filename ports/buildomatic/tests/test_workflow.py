"""Exercise the public shared build/export loop and pruning of build-only inputs."""

import json
from pathlib import Path

from ports.__main__ import main
from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic.artifacts import register
from ports.buildomatic.backends.local import LocalBackend
from ports.buildomatic.delivery_products import delivery_key
from ports.buildomatic.nodes import Operation
from ports.buildomatic.remote_store import StorageStore
from ports.buildomatic.workflow import compile_plan, normalize_cached, prepare, run


def test_shared_cli_build_exports_incrementally_and_rebuilds_dependents(tmp_path, monkeypatch):
    ports, state = tmp_path / "ports", tmp_path / "state"
    source = state / "execution/cache/ports"
    recipe_path, recipe = _wheel_port(ports, source, "base", [])
    _wheel_port(ports, source, "app", ["base"])
    arguments = ["--ports", str(ports), "--state", str(state), "--offline"]
    main(["build", "python/app", *arguments])
    graph = json.loads((state / "graph.json").read_text())
    assert {package["name"] for package in graph["packages"]} == {"base", "app"}
    store = StorageStore(json.loads((state / "results.json").read_text())["prefix"])
    for package in graph["packages"]:
        assert (Path(store.prefix) / package["path"]).is_file()
    main(["export", *arguments, "--url-prefix", "https://ports.example"])
    first = json.loads((state / "export/catalog.json").read_text())
    assert len(first["packages"]) == 2

    def reject(*args, **kwargs):
        raise AssertionError("cache hit dispatched work")

    with monkeypatch.context() as guarded:
        guarded.setattr(LocalBackend, "submit", reject)
        main(["build", "python/app", *arguments])
        main(["export", *arguments, "--url-prefix", "https://ports.example"])
    recipe_path.write_text(json.dumps({**recipe, "features": {"changed": True}}))
    main(["build", "python/app", *arguments])
    changed = json.loads((state / "graph.json").read_text())
    assert {item["build_key"] for item in changed["ports"]}.isdisjoint({item["build_key"] for item in graph["ports"]})


def test_cached_consumer_prunes_build_tool_but_keeps_runtime_outputs(tmp_path):
    ports = tmp_path / "ports"
    for name in ("builder", "runtime", "app"):
        _wheel_port(ports, tmp_path / "sources", name, [])
    path = ports / "python/app/recipe.json"
    recipe = json.loads(path.read_text())
    recipe["build_dependencies"] = [{"port": "python/builder", "version": "1"}]
    recipe["runtime_dependencies"] = [{"port": "python/runtime", "version": "1"}]
    path.write_text(json.dumps(recipe))
    compiled = compile_plan(ports, ["python/app"])
    store = StorageStore((tmp_path / "canonical").as_uri())
    app = next(node for node in compiled.nodes if node.port.name == "app" and node.operation is Operation.BUILD)
    register(
        store,
        app.record_key,
        {
            "reference": app.reference,
            "delivery_key": delivery_key(app.key),
            "delivery": {"build_key": app.key, "wheels": [], "native": []},
        },
    )
    backend = LocalBackend(store, tmp_path / "execution")
    request = prepare(
        compiled,
        compiled.build_roots,
        store,
        tmp_path / "client",
        backend,
        jobs=8,
        max_workers=2,
        records=normalize_cached(compiled, store),
    )
    runtime = next(node for node in compiled.nodes if node.port.name == "runtime" and node.operation is Operation.BUILD)
    assert runtime.action_id in {action.id for action in request.actions}
    builder = next(node for node in compiled.nodes if node.port.name == "builder" and node.operation is Operation.BUILD)
    assert builder.action_id not in {action.id for action in request.actions}
    register(
        store,
        runtime.record_key,
        {
            "reference": runtime.reference,
            "delivery_key": delivery_key(runtime.key),
            "delivery": {"build_key": runtime.key, "wheels": [], "native": []},
        },
    )
    run(ports, ["python/app"], store, tmp_path / "export-client", backend, operation=Operation.DELIVERY)
    exported = json.loads((tmp_path / "export-client/graph.json").read_text())
    assert {node["reference"] for node in exported["ports"]} == {app.reference, runtime.reference}
