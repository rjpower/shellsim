"""Exercise the public shared build/export loop and pruning of build-only inputs."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ports.__main__ import main
from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic import BuildResult, BuildState, NodeResult, NodeState, workflow
from ports.buildomatic.artifacts import register
from ports.buildomatic.backends.iris import AutomaticIrisBackend, IrisConfig
from ports.buildomatic.backends.local import LocalBackend
from ports.buildomatic.delivery_products import delivery_key
from ports.buildomatic.nodes import Operation
from ports.buildomatic.remote_store import StorageStore
from ports.buildomatic.workflow import compile_plan, normalize_cached, prepare, run, wait_for_build


def test_shared_cli_build_exports_incrementally_and_rebuilds_dependents(tmp_path, monkeypatch):
    ports, state = tmp_path / "ports", tmp_path / "state"
    source = state / "execution/cache/ports"
    recipe_path, recipe = _wheel_port(ports, source, "base", [])
    _wheel_port(ports, source, "app", ["base"])
    arguments = ["--ports", str(ports), "--state", str(state), "--offline"]
    main(["build", "--all", *arguments])
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
        main(["build", "--all", *arguments])
        main(["export", *arguments, "--url-prefix", "https://ports.example"])
        guarded.setattr("iris.cli.connect.open_iris_client", reject)
        with AutomaticIrisBackend(IrisConfig("cached"), store, ports.parent) as automatic:
            run(ports, ["python/app"], store, state, automatic, offline=True)
    recipe_path.write_text(json.dumps({**recipe, "features": {"changed": True}}))
    _wheel_port(ports, source, "independent", [])
    process = subprocess.run(
        [sys.executable, "-m", "ports", "build", "--all", *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(process.stdout)["command"] == "build"
    assert "python/independent" in process.stderr
    changed = json.loads((state / "graph.json").read_text())
    assert {package["name"] for package in changed["packages"]} == {"base", "app", "independent"}
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


def test_check_dependencies_build_only_when_checks_are_requested(tmp_path):
    ports, state = tmp_path / "ports", tmp_path / "state"
    for name in ("app", "checker"):
        _wheel_port(ports, state / "execution/cache/ports", name, [])
    path = ports / "python/app/recipe.json"
    value = json.loads(path.read_text())
    value["test_dependencies"] = [{"port": "python/checker", "version": "1"}]
    value["tests"] = [{"kind": "python", "script": "probe.py"}]
    path.write_text(json.dumps(value))
    (path.parent / "probe.py").write_text("print(42)\n")
    store = StorageStore(str(tmp_path / "canonical"))
    backend = LocalBackend(store, state / "execution")
    built = run(ports, ["python/app"], store, state, backend, offline=True)
    checker = next(
        node for node in built.plan.nodes if node.port.name == "checker" and node.operation is Operation.BUILD
    )
    assert checker.reference not in built.records
    checked = run(ports, ["python/app"], store, state, backend, operation=Operation.CHECK, offline=True)
    assert checked.records[checker.reference]["reference"] == checker.reference
    check = next(node for node in checked.plan.nodes if node.operation is Operation.CHECK)
    assert checked.records[check.reference]["stdout"] == "42\n"
    assert {item["name"] for item in json.loads((state / "graph.json").read_text())["packages"]} == {"app"}


def test_progress_reports_active_work_and_remote_failure(tmp_path, monkeypatch, caplog):
    ports = tmp_path / "ports"
    _wheel_port(ports, tmp_path / "sources", "example", [])
    node = compile_plan(ports, ["python/example"]).nodes[0]
    outcomes = iter(
        [
            BuildResult("request", BuildState.RUNNING, (NodeResult(node.action_id, NodeState.RUNNING, 1),)),
            BuildResult("request", BuildState.RUNNING, (NodeResult(node.action_id, NodeState.RUNNING, 1),)),
            BuildResult(
                "request",
                BuildState.FAILED,
                (NodeResult(node.action_id, NodeState.FAILED, 1, error="compiler failed"),),
            ),
        ]
    )
    ticks = iter((0, 1, 11, 12))
    monkeypatch.setattr(workflow, "time", SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None))
    caplog.set_level(logging.INFO, logger=workflow.__name__)
    wait_for_build(SimpleNamespace(get=lambda _: next(outcomes)), "build", [node])
    assert "1 running" in caplog.text
    assert "1 failed" in caplog.text
    assert any(record.levelno == logging.ERROR and "python/example" in record.message for record in caplog.records)


def test_interrupt_cancels_the_local_submission(tmp_path, monkeypatch):
    ports, state = tmp_path / "ports", tmp_path / "state"
    _wheel_port(ports, state / "execution/cache/ports", "example", [])
    store = StorageStore(str(tmp_path / "canonical"))
    backend = LocalBackend(store, state / "execution")

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(backend, "get", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(ports, ["python/example"], store, state, backend, offline=True)
    coordinator = next(iter(backend.builds.values()))
    assert coordinator.tick().state is BuildState.CANCELLED
