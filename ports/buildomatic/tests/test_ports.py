"""Use real offline adapters and private core workers; no cloud or SDK is needed."""

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from types import ModuleType

import pytest

from ports._support import runner
from ports._support.graph import plan
from ports._support.store import identity
from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic.ports import code_files, collect_graph, prepare_graph, publish_manifest, run_graph


def test_local_cli_imports_no_iris():
    script = """
import importlib.abc, runpy, sys
class RejectCloud(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'iris', 'rigging', 'cw'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectCloud())
sys.argv = ['ports', '--help']
runpy.run_module('ports', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, check=True)
    assert "--backend" in result.stdout


@pytest.mark.parametrize("failure", ["key", "missing", "bytes", "mode"])
def test_admitted_predecessor_never_rebuilds(tmp_path, monkeypatch, failure):
    from ports import api

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "provider", [])
    _wheel_port(ports, store, "consumer", ["provider"])
    built = runner.build_graph(ports, ["python/provider"], None, store, offline=True)
    result = built.results["python/provider/recipe.json"]
    expected = result.name
    if failure == "key":
        expected = "0" * 64
    elif failure == "missing":
        shutil.rmtree(result)
    elif failure == "bytes":
        next((result / "wheels").iterdir()).write_bytes(b"changed")
    else:
        next((result / "wheels").iterdir()).chmod(0o755)
    calls = []
    monkeypatch.setattr(api, "build_port", lambda ctx: calls.append(ctx.port.reference))
    with pytest.raises(ValueError):
        runner.build_graph(
            ports,
            ["python/consumer"],
            None,
            store,
            offline=True,
            admitted_predecessors={"python/provider/recipe.json": expected},
        )
    assert calls == []


def test_code_closure_excludes_unselected_files(tmp_path):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "selected", [])
    _wheel_port(ports, store, "other", [])
    (ports / ".env").write_text("SECRET=hidden")
    (ports / "python/selected/cache").mkdir()
    (ports / "python/selected/cache/secret").write_text("hidden")
    files = code_files(ports, plan(ports, ["python/selected"]))
    assert "python/selected/recipe.json" in files
    assert "_support/runner.py" in files
    assert "_support/pure_wheel.py" in files
    assert "python/other/recipe.json" not in files
    assert not any("secret" in name or "cache/" in name or name == ".env" for name in files)
    assert not any(name.startswith("buildomatic/") for name in files)


def test_worker_builds_complete_dependency_results_and_manifest(tmp_path):
    ports, store = tmp_path / "ports", tmp_path / "store"
    for name, dependencies in (("base", []), ("provider", ["base"]), ("consumer", ["provider"])):
        _wheel_port(ports, store, name, dependencies)
    built = run_graph(ports, ["python/consumer"], None, store, offline=True, max_workers=2)
    assert len(built.results) == 3
    for name in ("base", "provider", "consumer"):
        assert (built.results[f"python/{name}/recipe.json"] / f"wheels/{name}-1-py3-none-any.whl").is_file()
    manifest = json.loads(next((store / "buildomatic/manifests").glob("*.json")).read_text())
    assert manifest["results"] == {reference: result.name for reference, result in built.results.items()}
    assert len(manifest["bundles"]) == 3
    assert not (store / "release.json").exists()


def test_source_tree_worker_transports_only_declared_sources(tmp_path):
    ports, store = tmp_path / "ports", tmp_path / "store"
    directory = ports / "toolchain/fixture"
    directory.mkdir(parents=True)
    source = directory / "source.txt"
    source.write_text("pinned source\n")
    from ports._support.store import file_hash

    entries = [{"path": "toolchain/fixture/source.txt", "destination": "source.txt", "sha256": file_hash(source)}]
    recipe = {
        "name": "fixture",
        "version": "1",
        "role": "host-tool",
        "build_system": "source-tree",
        "source": {"files": entries, "sha256": identity(entries)},
    }
    (directory / "recipe.json").write_text(json.dumps(recipe))
    (directory / "build.py").write_text(
        "import shutil\ndef build(ctx):\n    shutil.copyfile(ctx.source / 'source.txt', ctx.result / 'source.txt')\n"
    )
    (directory / "undeclared.txt").write_text("not transported")
    built = run_graph(ports, ["toolchain/fixture"], None, store, offline=True)
    result = built.results["toolchain/fixture/recipe.json"]
    assert (result / "source.txt").read_text() == "pinned source\n"
    assert not (result / "undeclared.txt").exists()


def test_manifest_publication_checks_cache_before_publishing(tmp_path, monkeypatch):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    built = run_graph(ports, ["python/example"], None, store, offline=True)
    manifest = next((store / "buildomatic/manifests").glob("*.json"))
    calls = []
    monkeypatch.setattr(
        runner, "publish_graph", lambda build, sdk, output: calls.append(build) or output / "release.json"
    )
    output = tmp_path / "release"
    assert publish_manifest(manifest, ports, None, store, output) == output / "release.json"
    assert calls[0].results == built.results
    shutil.rmtree(next(iter(built.results.values())))
    calls.clear()
    with pytest.raises(ValueError):
        publish_manifest(manifest, ports, None, store, output)
    assert calls == []


def test_preparation_missing_source_fails_offline(tmp_path):
    from ports.buildomatic import LocalStore

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    shutil.rmtree(store / "sources")
    with pytest.raises(ValueError):
        prepare_graph(ports, ["python/example"], None, store, LocalStore(tmp_path / "blobs"), offline=True)


@pytest.mark.parametrize("failure", ["request", "node", "duplicate"])
def test_collection_rejects_wrong_request_or_action_closure(tmp_path, failure):
    from ports.buildomatic import BuildResult, BuildState, LocalStore, NodeResult, NodeState
    from ports.buildomatic.contracts import request_id

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    blobs = LocalStore(tmp_path / "blobs")
    prepared = prepare_graph(ports, ["python/example"], None, store, blobs, offline=True)
    action_id = next(iter(prepared.action_ids.values()))
    node = NodeResult(action_id, NodeState.SUCCEEDED, 1)
    result = BuildResult(request_id(prepared.request), BuildState.SUCCEEDED, (node,))
    if failure == "request":
        result = replace(result, request_id="0" * 64)
    elif failure == "node":
        result = replace(result, nodes=(replace(node, action_id="different"),))
    else:
        result = replace(result, nodes=(node, node))
    with pytest.raises(ValueError):
        collect_graph(prepared, result, blobs, store)
    assert not (store / "results").exists()
    assert not (store / "buildomatic/manifests").exists()


def test_explicit_build_key_changes_request_but_not_action_inputs(tmp_path):
    from ports.buildomatic import LocalStore
    from ports.buildomatic.contracts import request_id

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    blobs = LocalStore(tmp_path / "blobs")
    first = prepare_graph(ports, ["python/example"], None, store, blobs, offline=True, build_key="attempt-1")
    second = prepare_graph(ports, ["python/example"], None, store, blobs, offline=True, build_key="attempt-2")
    assert first.request.actions == second.request.actions
    assert request_id(first.request) != request_id(second.request)


def test_port_actions_have_explicit_wall_budget_and_one_attempt(tmp_path):
    from ports.buildomatic import LocalStore

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    blobs = LocalStore(tmp_path / "blobs")
    prepared = prepare_graph(ports, ["python/example"], None, store, blobs, offline=True)
    assert prepared.request.actions[0].timeout_seconds == 4 * 3600
    assert prepared.request.actions[0].max_attempts == 1
    changed = prepare_graph(ports, ["python/example"], None, store, blobs, offline=True, port_timeout_seconds=7200)
    assert changed.request.actions[0].timeout_seconds == 7200
    assert changed.request.idempotency_key != prepared.request.idempotency_key
    for timeout in (0, -1, 86401, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            prepare_graph(ports, ["python/example"], None, store, blobs, offline=True, port_timeout_seconds=timeout)


def test_independent_ports_enter_ready_queue_concurrently(tmp_path, monkeypatch):
    import ports.buildomatic as core

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "first", [])
    _wheel_port(ports, store, "second", [])
    states = []
    original = core.Coordinator

    class ObservedCoordinator(original):
        def tick(self):
            result = super().tick()
            states.append(sum(node.state is core.NodeState.RUNNING for node in result.nodes))
            return result

    monkeypatch.setattr(core, "Coordinator", ObservedCoordinator)
    run_graph(ports, ["python/first", "python/second"], None, store, offline=True, max_workers=2)
    assert states[0] == 2


def test_iris_dispatch_uses_remote_service_only(tmp_path, monkeypatch):
    import ports.buildomatic as core
    import ports.buildomatic.ports as bridge

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    blobs = core.LocalStore(tmp_path / "blobs")
    service = core.Coordinator(blobs, "remote-service", {"worker": core.WorkerExecutor(blobs, tmp_path / "worker")})
    calls = []
    intervals = []
    sleep = bridge.time.sleep

    def observe_sleep(interval):
        intervals.append(interval)
        sleep(0.001)

    monkeypatch.setattr(bridge.time, "sleep", observe_sleep)

    class RemoteService:
        def submit(self, request):
            calls.append("submit")
            return service.submit(request)

        def get(self, build_id):
            calls.append("get")
            return service.tick()

        def acknowledge(self, build_id):
            calls.append("acknowledge")
            service.acknowledge()

    def reject_local_coordinator(*args, **kwargs):
        raise AssertionError("Iris bridge cannot create a local coordinator")

    monkeypatch.setattr(core, "Coordinator", reject_local_coordinator)
    build = run_graph(
        ports,
        ["python/example"],
        None,
        store,
        backend="iris",
        blob_store=blobs,
        remote_backend=RemoteService(),
        offline=True,
    )
    assert len(build.results) == 1
    assert calls[0] == "submit" and calls[-1] == "acknowledge"
    assert "get" in calls
    assert intervals and set(intervals) == {1.0}


def test_iris_service_descriptor_converts_wire_job_name(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import ports.buildomatic.ports as bridge

    calls = []
    sentinel = object()

    class JobName:
        @classmethod
        def from_wire(cls, value):
            assert value == "/power/service"
            return cls()

    class Namespace:
        @staticmethod
        def from_job_id(value):
            assert isinstance(value, JobName)
            return "/power/service"

    @contextmanager
    def connect(**kwargs):
        calls.append(kwargs)
        yield sentinel

    class Backend:
        store = sentinel

        def __init__(self, client, url, namespace, **kwargs):
            assert client is sentinel
            assert url == "https://iris.oa.dev" and namespace == "/power/service"
            assert kwargs == {"prefix": "durable", "cache_prefix": "cache"}

    for name, exports in {
        "iris.cli.connect": {"open_iris_client": connect},
        "iris.cluster.types": {"JobName": JobName, "Namespace": Namespace},
        "ports.buildomatic.backends.iris": {"IrisBackend": Backend},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, module)
    service = tmp_path / "service.json"
    connection = {
        "schema_version": 1,
        "job_id": "/power/service",
        "prefix": "durable",
        "cache_prefix": "cache",
        "controller_url": "https://iris.oa.dev",
        "cluster_name": "marin",
        "workspace": str(tmp_path),
        "config_sha256": "a" * 64,
        "task_image": "image@sha256:" + "b" * 64,
        "service_id": "service",
    }
    service.write_text(json.dumps(connection))
    actual_run = bridge.run_graph

    def capture_remote(*args, **kwargs):
        assert isinstance(kwargs["remote_backend"], Backend)
        assert kwargs["blob_store"] is sentinel
        assert kwargs["worker_identity"] == {
            name: connection[name] for name in ("config_sha256", "task_image", "service_id")
        }
        return sentinel

    monkeypatch.setattr(bridge, "run_graph", capture_remote)
    assert actual_run(tmp_path, ["python/example"], None, tmp_path, backend="iris", iris_service=service) is sentinel
    assert calls == [{"cluster_name": "marin", "workspace": tmp_path}]
    for invalid in (
        {**connection, "schema_version": 2},
        {**connection, "token": "unexpected"},
        {name: value for name, value in connection.items() if name != "task_image"},
    ):
        service.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            actual_run(tmp_path, ["python/example"], None, tmp_path, backend="iris", iris_service=service)
