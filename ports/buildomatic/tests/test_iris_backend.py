"""Exercise adapter failure semantics using core storage and explicit fake RPCs.

No Iris, Rigging, credentials or elapsed-time assertions are needed. Native CAS
and remote task execution are separately covered by the recorded live smoke.
"""

import hashlib
import io
import json
import sys
import threading
import zipfile
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from ports.buildomatic import (
    Action,
    Attempt,
    AttemptResult,
    AttemptState,
    BuildRequest,
    BuildState,
    ConditionalWriteError,
    IdempotencyConflict,
    InputMount,
    LocalStore,
    NodeState,
    TreeBundle,
    WorkerExecutor,
    WorkerReport,
    WorkerState,
    capture_tree,
    remote_store,
)
from ports.buildomatic.backends.iris import (
    CoordinatorService,
    IrisConfig,
    WorkerProxy,
    _CapabilityRPC,
    _deployed_files,
    _entrypoint,
    _WorkerActor,
    compiler_cache_environment,
)
from ports.buildomatic.contracts import request_id
from ports.buildomatic.remote_store import DEFAULT_PREFIX, RemoteStore


class NativeConflict(RuntimeError):
    pass


@pytest.fixture
def native_store(monkeypatch):
    objects = {}
    reads = []
    writes = []

    class Object:
        def __init__(self, path):
            self.path = path

        def read(self):
            return objects.get(self.path)

        def write(self, data, *, expected_version):
            previous = self.read()
            if (previous.version if previous else None) != expected_version:
                raise NativeConflict()
            version = f'"native-etag-{len(writes)}"'
            writes.append((self.path, expected_version))
            objects[self.path] = SimpleNamespace(data=data, version=version)
            return version

    class Handle(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    class Filesystem:
        def open(self, path, mode):
            assert mode == "rb"
            found = objects.get(path)
            if found is None:
                raise FileNotFoundError(path)
            return Handle(found.data)

    conditional = ModuleType("rigging.filesystem.conditional_object")
    conditional.ConditionalWriteError = NativeConflict
    buckets = ModuleType("rigging.filesystem.buckets")
    buckets.filesystem_for = lambda path: (Filesystem(), path)
    monkeypatch.setitem(sys.modules, conditional.__name__, conditional)
    monkeypatch.setitem(sys.modules, buckets.__name__, buckets)
    monkeypatch.setattr(remote_store, "_conditional", Object)
    return SimpleNamespace(objects=objects, reads=reads, writes=writes)


def test_remote_native_revisions_and_conflict_translation(native_store):
    store = RemoteStore()
    assert store.read_journal("build") is None
    version = store.write_journal("build", b"first", None)
    assert store.read_journal("build").version == version
    assert store.read_journal("build").data == b"first"
    next_version = store.write_journal("build", b"second", version)
    assert next_version != version
    with pytest.raises(ConditionalWriteError) as raised:
        store.write_journal("build", b"stale", version)
    assert isinstance(raised.value.__cause__, NativeConflict)
    assert store.read_journal("build").data == b"second"


def test_remote_blob_integrity_missing_and_repeated_upload(native_store):
    store = RemoteStore()
    digest = store.put_blob(b"hello")
    assert digest == hashlib.sha256(b"hello").hexdigest()
    assert store.put_blob(b"hello") == digest
    assert store.get_blob(digest) == b"hello"
    with pytest.raises(FileNotFoundError):
        store.get_blob("f" * 64)
    native_store.objects[f"{DEFAULT_PREFIX}/blobs/{digest}"].data = b"corruption"
    with pytest.raises(ValueError):
        store.get_blob(digest)
    with pytest.raises(ValueError):
        store.put_blob(b"hello")


def test_remote_blob_reads_are_bounded_before_validation(native_store, monkeypatch):
    monkeypatch.setattr(remote_store, "MAX_METADATA_BYTES", 8)
    digest = "a" * 64
    native_store.objects[f"{DEFAULT_PREFIX}/blobs/{digest}"] = SimpleNamespace(data=b"x" * 100, version="v")
    with pytest.raises(ValueError):
        RemoteStore().get_blob(digest)
    assert native_store.reads == [9]
    with pytest.raises(ValueError):
        RemoteStore().put_blob(b"x" * 9)


def test_cache_and_journal_namespaces_are_separate(native_store):
    cache = "s3://bucket/tmp/ttl=30d/outputs"
    store = RemoteStore(cache, journal_prefix=DEFAULT_PREFIX)
    store.put_blob(b"cached")
    store.write_journal("request", b"durable", None)
    assert any(path.startswith(cache + "/blobs/") for path in native_store.objects)
    assert DEFAULT_PREFIX + "/journals/request" in native_store.objects


@pytest.mark.parametrize(
    "prefix",
    [
        "http://bucket",
        "file:///path",
        "s3://",
        "gs://user:secret@bucket/key",
        "s3://bucket/a/../b",
        "s3://bucket/key?token=x",
    ],
)
def test_remote_prefix_rejects_unsupported_or_ambiguous_paths(prefix):
    with pytest.raises(ValueError):
        RemoteStore(prefix)


@pytest.mark.parametrize("key", ["", "/absolute", "../escape", "a//b", "a\\b", "a#b", "a?b"])
def test_journal_path_rejects_escape(key):
    with pytest.raises(ValueError):
        RemoteStore()._journal_path(key)


def test_storage_transport_errors_do_not_become_cas_conflicts(native_store, monkeypatch):
    def unavailable(path):
        raise PermissionError("denied")

    monkeypatch.setattr(remote_store, "_conditional", unavailable)
    with pytest.raises(PermissionError):
        RemoteStore().write_journal("build", b"data", None)


def test_worker_chunk_cache_reuses_verified_reads_and_republishes_outputs(native_store, tmp_path):
    remote = RemoteStore()
    digest = remote.put_blob(b"chunk")
    local = LocalStore(tmp_path / "cache")
    worker = RemoteStore(local_cache=local)
    assert worker.get_blob(digest) == b"chunk"
    reads = len(native_store.reads)
    del native_store.objects[f"{DEFAULT_PREFIX}/blobs/{digest}"]
    assert worker.get_blob(digest) == b"chunk"
    assert len(native_store.reads) == reads
    assert worker.put_blob(b"chunk") == digest
    assert f"{DEFAULT_PREFIX}/blobs/{digest}" in native_store.objects
    path = local.root / "blobs" / digest
    path.chmod(0o644)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        worker.get_blob(digest)


def test_remote_journal_nonce_prevents_identical_content_aba(native_store):
    store = RemoteStore()
    first = store.write_journal("state", b"same", None)
    first_bytes = native_store.objects[f"{DEFAULT_PREFIX}/journals/state"].data
    second = store.write_journal("state", b"different", first)
    third = store.write_journal("state", b"same", second)
    assert native_store.objects[f"{DEFAULT_PREFIX}/journals/state"].data != first_bytes
    assert store.read_journal("state").data == b"same"
    with pytest.raises(ConditionalWriteError):
        store.write_journal("state", b"stale", first)
    assert store.read_journal("state").version == third


class Worker:
    def __init__(self, bundle):
        self.bundle = bundle
        self.submitted = []
        self.reports = {}
        self.cancelled = []
        self.acknowledged = []

    def submit(self, attempt):
        self.submitted.append(attempt)
        self.reports[attempt.id] = WorkerReport(WorkerState.RUNNING)

    def poll(self, attempt_id):
        return self.reports.get(attempt_id, WorkerReport(WorkerState.UNKNOWN))

    def finish(self):
        for attempt_id in self.reports:
            self.reports[attempt_id] = WorkerReport(
                WorkerState.COMPLETED, AttemptResult(attempt_id, AttemptState.SUCCEEDED, self.bundle)
            )

    def cancel(self, attempt_id):
        self.cancelled.append(attempt_id)
        self.reports[attempt_id] = WorkerReport(
            WorkerState.COMPLETED, AttemptResult(attempt_id, AttemptState.CANCELLED)
        )

    def acknowledge(self, attempt_id):
        self.acknowledged.append(attempt_id)


@pytest.fixture
def service_parts(tmp_path):
    store = LocalStore(tmp_path / "store")
    empty = tmp_path / "empty"
    empty.mkdir()
    bundle = capture_tree(empty, store)
    workers = {f"worker-{i}": Worker(bundle) for i in range(2)}
    return store, workers


def request(key="build", *, max_workers=2):
    return BuildRequest(key, (Action("a", ("echo", "a")), Action("b", ("echo", "b"))), max_workers)


def test_service_accepts_durably_and_recovers_before_dispatch(service_parts):
    store, workers = service_parts
    original = CoordinatorService(store, "test", workers)
    build_id = original.submit(request())
    index = json.loads(store.read_journal("iris/test/index.json").data)
    assert index["builds"][0]["id"] == build_id
    assert store.read_journal(f"iris/test/requests/{build_id}.json") is not None
    recovered = CoordinatorService(store, "test", workers)
    assert recovered.submit(request()) == build_id
    assert recovered.get(build_id).state == BuildState.PENDING
    assert recovered.get(build_id).request_id == request_id(request())
    recovered.tick_once()
    assert recovered.get(build_id).state == BuildState.RUNNING
    assert [len(worker.submitted) for worker in workers.values()] == [1, 1]
    for worker in workers.values():
        worker.finish()
    recovered.tick_once()
    assert recovered.get(build_id).state == BuildState.SUCCEEDED


def test_service_serializes_builds_but_parallelizes_ready_nodes(service_parts):
    store, workers = service_parts
    service = CoordinatorService(store, "test", workers)
    first = service.submit(request("first"))
    second = service.submit(request("second"))
    service.tick_once()
    assert [len(worker.submitted) for worker in workers.values()] == [1, 1]
    assert service.get(first).state == BuildState.RUNNING
    assert service.get(second).state == BuildState.PENDING
    for worker in workers.values():
        worker.finish()
    service.tick_once()
    assert service.get(first).state == BuildState.SUCCEEDED
    assert service.get(second).state == BuildState.RUNNING
    assert [len(worker.submitted) for worker in workers.values()] == [2, 2]
    assert all(worker.acknowledged for worker in workers.values())


def test_service_duplicate_key_conflict_and_worker_limit(service_parts):
    store, workers = service_parts
    service = CoordinatorService(store, "test", workers)
    service.submit(request())
    with pytest.raises(IdempotencyConflict):
        service.submit(replace(request(), actions=(Action("different", ("true",)),)))
    with pytest.raises(ValueError):
        service.submit(request("too-many", max_workers=3))


def test_service_cancel_is_per_build_and_survives_restart(service_parts):
    store, workers = service_parts
    service = CoordinatorService(store, "test", workers)
    first = service.submit(request("first"))
    second = service.submit(request("second"))
    service.tick_once()
    assert service.cancel(second).state == BuildState.PENDING
    assert all(not worker.cancelled for worker in workers.values())
    assert service.get(first).state == BuildState.RUNNING
    recovered = CoordinatorService(store, "test", workers)
    assert recovered.get(second).state == BuildState.PENDING
    assert recovered.cancel(first).state == BuildState.RUNNING
    recovered.tick_once()
    assert recovered.get(second).state == BuildState.CANCELLED
    assert recovered.get(first).state == BuildState.CANCELLED
    assert all(worker.cancelled for worker in workers.values())


def test_short_rpcs_do_not_wait_for_worker_submit(service_parts, monkeypatch):
    store, workers = service_parts
    entered, release = threading.Event(), threading.Event()
    failures = []
    original = workers["worker-0"].submit

    def blocked(attempt):
        entered.set()
        assert release.wait(10)
        original(attempt)

    monkeypatch.setattr(workers["worker-0"], "submit", blocked)
    service = CoordinatorService(store, "test", workers)
    first = service.submit(request("first"))

    def drive():
        try:
            service.tick_once()
        except BaseException as error:
            failures.append(error)

    scheduler = threading.Thread(target=drive)
    scheduler.start()
    try:
        assert entered.wait(10)
        completed = threading.Event()

        def rpc():
            try:
                assert service.get(first).request_id == request_id(request("first"))
                second = service.submit(request("second"))
                service.cancel(first)
                service.acknowledge(first)
                assert service.get(second).state == BuildState.PENDING
            except BaseException as error:
                failures.append(error)
            finally:
                completed.set()

        caller = threading.Thread(target=rpc)
        caller.start()
        assert completed.wait(10)
        caller.join()
    finally:
        release.set()
        scheduler.join(10)
    assert not scheduler.is_alive()
    assert not failures
    service.tick_once()
    assert service.get(first).state == BuildState.CANCELLED


@pytest.mark.parametrize("preparation_failure", [True, False])
def test_completed_worker_without_log_remains_observable(tmp_path, preparation_failure):
    store = LocalStore(tmp_path / "store")
    executor = WorkerExecutor(store, tmp_path / "worker")
    actor = _WorkerActor(executor)
    proxy = WorkerProxy(actor, store=store, service_id="test", worker_id="worker-0")
    attempt = Attempt("attempt", "b" * 64, Action("a", ("true",)), (InputMount("missing", TreeBundle("a" * 64)),), 1)
    if preparation_failure:
        proxy.submit(attempt)
    else:
        proxy.cancel(attempt.id)
    report = proxy.poll(attempt.id)
    assert report.state == WorkerState.COMPLETED
    assert report.result.state == (AttemptState.FAILED if preparation_failure else AttemptState.CANCELLED)
    assert store.read_journal("iris/test/logs/attempt").data == b""


def test_worker_diagnostic_errors_propagate_and_bounds_match(monkeypatch):
    def denied(attempt_id, max_bytes):
        raise PermissionError()

    actor = _WorkerActor(SimpleNamespace(read_log=denied))
    with pytest.raises(PermissionError):
        actor.read_log("a")
    for bound in (0, 16385, True):
        with pytest.raises(ValueError):
            actor.read_log("a", max_bytes=bound)


def test_runtime_archive_preserves_nested_modules_for_child_workers(tmp_path, monkeypatch):
    module = ModuleType("iris.cluster.types")
    module.Entrypoint = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    files = {
        "ports/buildomatic/__init__.py": b"",
        "ports/buildomatic/remote_store.py": b"small module",
        "iris/client/__init__.py": b"",
        "pyproject.toml": b"[project]",
    }
    entrypoint = _entrypoint("coordinator", IrisConfig("test"), files=files)
    assert set(entrypoint.workdir_files) == {"_buildomatic_runtime.zip", "pyproject.toml"}
    archive = entrypoint.workdir_files["_buildomatic_runtime.zip"]
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert {name: bundle.read(name) for name in bundle.namelist()} == files
    (tmp_path / "_buildomatic_runtime.zip").write_bytes(archive)
    monkeypatch.setenv("IRIS_WORKDIR", str(tmp_path))
    assert _deployed_files() == files


def test_get_logs_retains_bounded_newest_tails(service_parts):
    store, workers = service_parts
    service = CoordinatorService(store, "test", workers)
    build_id = service.submit(request())
    records = [{"attempt_id": f"a{i}", "action_id": "a", "worker_id": "worker-0"} for i in range(70)]
    store.write_journal(f"iris/test/diagnostics/{request_id(request())}.json", json.dumps(records).encode(), None)
    for record in records:
        store.write_journal(f"iris/test/logs/{record['attempt_id']}", b"x" * 16384, None)
    logs = service.get_logs(build_id)
    assert len(logs) == 64
    assert logs[-1]["attempt_id"] == "a69"
    assert sum(len(log["tail"]) for log in logs) == 65536


def test_service_retries_index_cas_without_losing_request(service_parts, monkeypatch):
    store, workers = service_parts
    original = store.write_journal
    failures = []

    def contested(key, data, expected_version):
        if key.endswith("index.json") and not failures:
            failures.append(key)
            raise ConditionalWriteError()
        return original(key, data, expected_version)

    monkeypatch.setattr(store, "write_journal", contested)
    service = CoordinatorService(store, "test", workers)
    build_id = service.submit(request())
    assert len(failures) == 1
    assert service.get(build_id).nodes[0].state == NodeState.PENDING


def test_service_does_not_acknowledge_failed_index_write(service_parts, monkeypatch):
    store, workers = service_parts
    original = store.write_journal

    def unavailable(key, data, expected_version):
        if key.endswith("index.json"):
            raise OSError("unavailable")
        return original(key, data, expected_version)

    monkeypatch.setattr(store, "write_journal", unavailable)
    service = CoordinatorService(store, "test", workers)
    with pytest.raises(OSError):
        service.submit(request())
    assert store.read_journal("iris/test/index.json") is None
    assert all(not worker.submitted for worker in workers.values())
    monkeypatch.setattr(store, "write_journal", original)
    assert service.get(service.submit(request())).state == BuildState.PENDING


def test_worker_proxy_targets_chosen_actor_and_propagates_rpc_error():
    actor = Worker(TreeBundle("a" * 64))
    proxy = WorkerProxy(actor)
    assert proxy.poll("missing").state == WorkerState.UNKNOWN
    proxy.cancel("chosen")
    assert actor.cancelled == ["chosen"]

    def failed(attempt_id):
        raise ConnectionError("transport failure")

    actor.poll = failed
    with pytest.raises(ConnectionError):
        proxy.poll("chosen")


def test_compiler_cache_preserves_worker_region_endpoint_and_ttl():
    env = compiler_cache_environment(
        "s3://marin-us-east-02a/tmp/ttl=30d/shellsim/ports/sccache/v1",
        {"AWS_REGION": "US-EAST-02A", "AWS_ENDPOINT_URL": "http://lota.local", "AWS_SECRET_ACCESS_KEY": "secret"},
    )
    assert env == {
        "SCCACHE_BUCKET": "marin-us-east-02a",
        "SCCACHE_S3_KEY_PREFIX": "tmp/ttl=30d/shellsim/ports/sccache/v1",
        "SCCACHE_REGION": "US-EAST-02A",
        "SCCACHE_ENDPOINT": "http://lota.local",
        "SCCACHE_S3_USE_SSL": "false",
        "SCCACHE_S3_ENABLE_VIRTUAL_HOST_STYLE": "true",
    }
    assert compiler_cache_environment("gs://bucket/tmp/ttl=30d/cache", {}) == {
        "SCCACHE_GCS_BUCKET": "bucket",
        "SCCACHE_GCS_KEY_PREFIX": "tmp/ttl=30d/cache",
        "SCCACHE_GCS_RW_MODE": "READ_WRITE",
    }
    with pytest.raises(ValueError):
        compiler_cache_environment("file:///tmp/cache", {})


@pytest.mark.parametrize(
    "values",
    [{"workers": 0}, {"workers": 33}, {"worker_cpu": float("inf")}, {"tick_seconds": 0}, {"service_id": "../escape"}],
)
def test_iris_config_resource_and_name_bounds(values):
    with pytest.raises(ValueError):
        IrisConfig(**{"service_id": "test", **values})


def test_capability_transport_never_logs_url_and_sanitizes_errors(monkeypatch, caplog):
    address = "https://controller/proxy/t/sensitive-token/actor"
    modules = {}
    for name in (
        "iris.rpc.actor_connect",
        "iris.rpc.compression",
        "iris.actor.client",
        "iris.rpc.errors",
        "iris.rpc",
        "cloudpickle",
        "connectrpc.errors",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])

    class ConnectError(Exception):
        def __init__(self, code, message):
            self.code, self.message = code, message
            super().__init__(message)

    def unavailable(request):
        raise ConnectError("unavailable", f"failed {address}")

    modules["connectrpc.errors"].ConnectError = ConnectError
    modules["iris.rpc.actor_connect"].ActorServiceClientSync = lambda **kwargs: SimpleNamespace(call=unavailable)
    modules["iris.rpc.compression"].IRIS_RPC_COMPRESSIONS = ()
    modules["iris.rpc.compression"].IRIS_RPC_ZSTD = None
    modules["iris.actor.client"].unwrap_actor_response = lambda value: value
    modules["iris.rpc.errors"].call_with_retry = lambda name, invoke, **kwargs: invoke()
    modules["iris.rpc"].actor_pb2 = SimpleNamespace(ActorCall=lambda **kwargs: SimpleNamespace(**kwargs))
    modules["cloudpickle"].dumps = lambda value: b"serialized"
    actor = _CapabilityRPC(address, "sensitive-token", "user/service/coordinator", 30)
    assert "sensitive-token" not in repr(actor)
    with pytest.raises(ConnectError) as raised:
        actor.call("Get", "build")
    assert raised.value.code == "unavailable"
    assert "sensitive-token" not in str(raised.value)
    assert address not in str(raised.value)
    assert "sensitive-token" not in caplog.text
