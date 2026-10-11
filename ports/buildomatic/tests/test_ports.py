"""Use real offline adapters and private core workers; no cloud or SDK is needed."""

import json
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
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


@pytest.fixture
def remote_service(tmp_path, monkeypatch):
    from ports.buildomatic.transfers import download_bundles, upload_bundles

    import ports.buildomatic as core
    import ports.buildomatic.ports as bridge

    blobs = core.LocalStore(tmp_path / "remote-blobs")
    coordinator = core.Coordinator(blobs, "remote-service", {"worker": core.WorkerExecutor(blobs, tmp_path / "worker")})
    calls, transfers, intervals = [], [], []
    opened, closed = [], []
    lock, barrier = threading.Lock(), threading.Barrier(2)
    sleep = bridge.time.sleep

    def observe_sleep(interval):
        intervals.append(interval)
        sleep(0.001)

    monkeypatch.setattr(bridge.time, "sleep", observe_sleep)

    class RemoteService:
        fault = None
        corrupt = None
        missing = None
        limits = core.ResourceLimits()
        parallel_contexts = 0
        peak_contexts = 0

        @contextmanager
        def context(self):
            owner = threading.get_ident()
            parallel = threading.current_thread().name.startswith("buildomatic-transfer")
            with lock:
                opened.append(owner)
                self.parallel_contexts += int(parallel)
                self.peak_contexts = max(self.peak_contexts, self.parallel_contexts)
                wait = parallel and sum(thread != threading.main_thread().ident for thread in opened) <= 2

            class Client:
                def get_blob(client, key):
                    assert threading.get_ident() == owner
                    if key == self.missing:
                        raise FileNotFoundError(key)
                    return b"corrupt" if key == self.corrupt else blobs.get_blob(key)

                def put_blob(client, data):
                    assert threading.get_ident() == owner
                    return blobs.put_blob(data)

            try:
                if wait:
                    barrier.wait(timeout=10)
                yield Client()
            finally:
                with lock:
                    closed.append(owner)
                    self.parallel_contexts -= int(parallel)

        def upload_bundles(self, source, bundles, **kwargs):
            calls.append("upload")
            assert source.root == (tmp_path / "store/buildomatic/blobs").resolve()
            assert len(bundles) == len(set(bundles))
            assert kwargs == {"max_parallel": 8, "limits": self.limits}
            counts = upload_bundles(source, bundles, factory=self.context, **kwargs)
            transfers.append(("upload", counts))

        def submit(self, request):
            calls.append("submit")
            self.request = request
            return coordinator.submit(request)

        def get(self, build_id):
            calls.append("get")
            result = coordinator.tick()
            if result.state is not core.BuildState.SUCCEEDED or self.fault is None:
                return result
            node = result.nodes[0]
            if self.fault in {"failed", "cancelled"}:
                return replace(result, state=core.BuildState(self.fault))
            if self.fault == "request":
                return replace(result, request_id="0" * 64)
            if self.fault == "action":
                return replace(result, nodes=(replace(node, action_id="different"),))
            if self.fault == "duplicate":
                return replace(result, nodes=(node, node))
            if self.fault == "missing":
                return replace(result, nodes=(replace(node, bundle=None),))
            if self.fault == "state":
                return replace(result, nodes=(replace(node, state=core.NodeState.FAILED),))
            if self.fault in {"corrupt", "evicted"}:
                manifest = json.loads(blobs.get_blob(node.bundle.digest))
                receipt = next(entry for entry in manifest["entries"] if entry["path"] == "result/build-receipt.json")
                if self.fault == "corrupt":
                    self.corrupt = receipt["chunks"][0]
                else:
                    self.missing = receipt["chunks"][0]
                return result
            destination = tmp_path / "tampered-result"
            core.extract_tree(node.bundle, blobs, destination)
            if self.fault == "key":
                metadata = json.loads((destination / "node.json").read_text())
                (destination / "node.json").write_text(json.dumps({**metadata, "key": "0" * 64}))
            elif self.fault == "mode":
                next((destination / "result/wheels").iterdir()).chmod(0o755)
            else:
                next((destination / "result/wheels").iterdir()).write_bytes(b"changed")
            return replace(result, nodes=(replace(node, bundle=core.capture_tree(destination, blobs)),))

        def download_bundles(self, destination, bundles, **kwargs):
            calls.append("download")
            assert len(bundles) == len(set(bundles))
            assert kwargs == {"max_parallel": 8, "limits": self.limits}
            counts = download_bundles(destination, bundles, factory=self.context, **kwargs)
            transfers.append(("download", counts))

        def acknowledge(self, build_id):
            calls.append("acknowledge")
            assert list((tmp_path / "store/buildomatic/manifests").glob("*.json"))
            coordinator.acknowledge()

    def reject_local_coordinator(*args, **kwargs):
        raise AssertionError("Iris bridge cannot create a local coordinator")

    monkeypatch.setattr(core, "Coordinator", reject_local_coordinator)
    remote = RemoteService()
    yield remote, calls, transfers, intervals
    assert sorted(opened) == sorted(closed)
    assert remote.parallel_contexts == 0


def test_iris_dispatch_transfers_locally_prepared_bundles_and_reuses_cas(tmp_path, remote_service):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "first", [])
    _wheel_port(ports, store, "second", [])
    remote, calls, transfers, intervals = remote_service
    arguments = {"backend": "iris", "remote_backend": remote, "offline": True}
    build = run_graph(ports, ["python/first", "python/second"], None, store, **arguments)
    assert len(build.results) == 2
    assert calls[0:2] == ["upload", "submit"]
    assert calls[-2:] == ["download", "acknowledge"]
    assert transfers[0][1].bundles == 3  # One shared code bundle, two recipe bundles.
    assert remote.peak_contexts >= 2
    assert intervals and set(intervals) == {1.0}
    assert not (store / "release.json").exists()
    first_blobs = set((store / "buildomatic/blobs/blobs").iterdir())
    calls.clear()
    repeated = run_graph(ports, ["python/first", "python/second"], None, store, **arguments)
    assert repeated.results == build.results
    assert transfers[-1][0] == "download" and transfers[-1][1].bytes == 0
    assert set((store / "buildomatic/blobs/blobs").iterdir()) == first_blobs
    assert calls[-2:] == ["download", "acknowledge"]


@pytest.mark.parametrize("fault", ["request", "action", "duplicate", "missing", "state"])
def test_iris_rejects_result_identity_before_download_or_ack(tmp_path, remote_service, fault):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    remote, calls, _, _ = remote_service
    remote.fault = fault
    with pytest.raises(ValueError):
        run_graph(ports, ["python/example"], None, store, backend="iris", remote_backend=remote, offline=True)
    assert "download" not in calls and "acknowledge" not in calls
    assert not (store / "buildomatic/manifests").exists()


@pytest.mark.parametrize("fault", ["failed", "cancelled"])
def test_iris_terminal_failure_prevents_download_and_ack(tmp_path, remote_service, fault):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    remote, calls, _, _ = remote_service
    remote.fault = fault
    with pytest.raises(RuntimeError):
        run_graph(ports, ["python/example"], None, store, backend="iris", remote_backend=remote, offline=True)
    assert "download" not in calls and "acknowledge" not in calls


@pytest.mark.parametrize("fault", ["corrupt", "evicted", "key", "mode", "bytes"])
def test_iris_rejects_transferred_corruption_and_invalid_receipts_before_ack(tmp_path, remote_service, fault):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    remote, calls, _, _ = remote_service
    remote.fault = fault
    with pytest.raises(FileNotFoundError if fault == "evicted" else ValueError):
        run_graph(ports, ["python/example"], None, store, backend="iris", remote_backend=remote, offline=True)
    assert "download" in calls and "acknowledge" not in calls
    assert not (store / "buildomatic/manifests").exists()


def test_iris_passes_explicit_sdk_bundle_limits_in_both_directions(tmp_path, remote_service, monkeypatch):
    import ports.buildomatic as core
    import ports.buildomatic.ports as bridge

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    remote, _, transfers, _ = remote_service
    remote.limits = core.ResourceLimits(output_bytes=40 * 1024**3, max_files=400_000)
    prepare = bridge.prepare_graph

    def pure_fixture(ports, requests, sdk, *args, **kwargs):
        # Exercise SDK transport policy without admitting or materializing SDKs.
        return prepare(ports, requests, None, *args, **kwargs)

    monkeypatch.setattr(bridge, "prepare_graph", pure_fixture)
    run_graph(ports, ["python/example"], object(), store, backend="iris", remote_backend=remote, offline=True)
    assert [direction for direction, _ in transfers] == ["upload", "download"]


def test_iris_upload_failure_prevents_submission(tmp_path, remote_service, monkeypatch):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    remote, calls, _, _ = remote_service

    def fail(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(remote, "upload_bundles", fail)
    with pytest.raises(FileNotFoundError):
        run_graph(ports, ["python/example"], None, store, backend="iris", remote_backend=remote, offline=True)
    assert calls == []


def test_iris_rejects_caller_remote_blob_store(tmp_path):
    with pytest.raises(ValueError):
        run_graph(tmp_path, [], None, tmp_path, backend="iris", blob_store=object(), remote_backend=object())


@pytest.fixture
def iris_connection(tmp_path):
    return {
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


@pytest.mark.parametrize("task_image", [None, "image@sha256:" + "b" * 64])
def test_iris_service_descriptor_uses_backend_context_factory(tmp_path, monkeypatch, iris_connection, task_image):
    import ports.buildomatic.ports as bridge

    calls = []
    sentinel = object()

    @contextmanager
    def connect():
        calls.append("enter")
        try:
            yield sentinel
        finally:
            calls.append("close")

    def factory(**kwargs):
        calls.append(kwargs)
        return connect

    module = ModuleType("ports.buildomatic.backends.iris")
    module.backend_context_factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    service = tmp_path / "service.json"
    connection = {**iris_connection, "task_image": task_image}
    service.write_text(json.dumps(connection))
    actual_run = bridge.run_graph

    def capture_remote(*args, **kwargs):
        assert kwargs["remote_backend"] is sentinel
        assert "blob_store" not in kwargs
        assert kwargs["worker_identity"] == {
            name: connection[name]
            for name in ("config_sha256", "task_image", "service_id")
            if connection[name] is not None
        }
        calls.append("run")
        return sentinel

    monkeypatch.setattr(bridge, "run_graph", capture_remote)
    assert actual_run(tmp_path, ["python/example"], None, tmp_path, backend="iris", iris_service=service) is sentinel
    assert calls == [
        {
            "job_id": "/power/service",
            "cluster_name": "marin",
            "workspace": tmp_path,
            "controller_url": "https://iris.oa.dev",
            "prefix": "durable",
            "cache_prefix": "cache",
        },
        "enter",
        "run",
        "close",
    ]
    for invalid in (
        {**connection, "schema_version": 2},
        {**connection, "token": "unexpected"},
        {name: value for name, value in connection.items() if name != "task_image"},
    ):
        service.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            actual_run(tmp_path, ["python/example"], None, tmp_path, backend="iris", iris_service=service)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("job_id", 12),
        ("prefix", None),
        ("cache_prefix", []),
        ("controller_url", {}),
        ("cluster_name", False),
        ("workspace", None),
        ("config_sha256", 1),
        ("task_image", {}),
        ("service_id", []),
        ("controller_url", "https://user:password@iris.oa.dev"),
    ],
)
def test_iris_connection_rejects_wrong_types_before_cloud_imports(tmp_path, monkeypatch, iris_connection, field, value):
    import builtins

    original = builtins.__import__

    def reject_cloud(name, *args, **kwargs):
        if name.split(".")[0] == "iris":
            raise AssertionError("invalid descriptor imported Iris")
        return original(name, *args, **kwargs)

    service = tmp_path / "service.json"
    service.write_text(json.dumps({**iris_connection, field: value}))
    monkeypatch.setattr(builtins, "__import__", reject_cloud)
    with pytest.raises(ValueError):
        run_graph(tmp_path, ["python/example"], None, tmp_path, backend="iris", iris_service=service)
