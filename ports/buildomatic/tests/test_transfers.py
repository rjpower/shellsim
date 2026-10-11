"""Check bounded scheduling and owned connections using events and fake stores.

Real LocalStore/capture/extract checks cover transported bytes; no Iris packages,
network, task deployment or elapsed-time performance assertions are required.
"""

import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
from ports.buildomatic.contracts import encode

from ports.buildomatic import (
    ConditionalWriteError,
    LocalStore,
    ResourceLimits,
    TreeBundle,
    capture_tree,
    extract_tree,
    transfers,
)
from ports.buildomatic.backends import iris


@pytest.fixture
def source(tmp_path):
    return LocalStore(tmp_path / "source-store")


def tree(tmp_path, store, name, files):
    root = tmp_path / name
    root.mkdir()
    for path, data in files.items():
        (root / path).write_bytes(data)
    return capture_tree(root, store)


def manifest(store, entries):
    return TreeBundle(store.put_blob(encode({"version": 1, "entries": entries})))


def file_entry(data, *, path="file", size=None):
    return {
        "kind": "file",
        "mode": 0o644,
        "path": path,
        "size": len(data) if size is None else size,
        "chunks": [hashlib.sha256(data).hexdigest()] if data else [],
    }


@pytest.fixture
def remote():
    state = SimpleNamespace(blobs={}, puts=[], gets=[], opened=[], closed=[], active=0, maximum=0)
    lock = threading.Lock()

    @contextmanager
    def factory():
        owner = threading.get_ident()
        with lock:
            identity = len(state.opened)
            state.opened.append((identity, owner))
            state.active += 1
            state.maximum = max(state.maximum, state.active)

        class Store:
            def get_blob(self, key):
                assert owner == threading.get_ident()
                with lock:
                    state.gets.append(key)
                    if key not in state.blobs:
                        raise FileNotFoundError(key)
                    return state.blobs[key]

            def put_blob(self, data):
                assert owner == threading.get_ident()
                key = hashlib.sha256(data).hexdigest()
                with lock:
                    state.puts.append(key)
                    if key in state.blobs and state.blobs[key] != data:
                        raise ValueError("corrupt conditional winner")
                    state.blobs[key] = data
                return key

        try:
            yield Store()
        finally:
            assert owner == threading.get_ident()
            with lock:
                state.closed.append((identity, owner))
                state.active -= 1

    state.factory = factory
    return state


def test_upload_deduplicates_chunks_and_publishes_manifests_last(tmp_path, source, remote):
    left = tree(tmp_path, source, "left", {"left": b"shared"})
    right = tree(tmp_path, source, "right", {"right": b"shared"})
    shared = hashlib.sha256(b"shared").hexdigest()
    counts = transfers.upload_bundles(source, [left, right, left], factory=remote.factory)
    assert counts == transfers.TransferCounts(2, 3, sum(map(len, remote.blobs.values())))
    assert remote.puts == [shared, left.digest, right.digest]
    assert sorted(remote.opened) == sorted(remote.closed)
    assert remote.active == 0


def test_manifest_used_as_chunk_is_uploaded_once_in_dependency_order(tmp_path, source, remote):
    empty = manifest(source, [])
    parent = tree(tmp_path, source, "parent", {"manifest-file": source.get_blob(empty.digest)})
    counts = transfers.upload_bundles(source, [parent, empty], factory=remote.factory)
    assert counts.objects == 2
    assert remote.puts == [empty.digest, parent.digest]
    destination = LocalStore(tmp_path / "destination")
    transfers.download_bundles(destination, [parent, empty], factory=remote.factory)
    extract_tree(parent, destination, tmp_path / "extracted")
    assert (tmp_path / "extracted/manifest-file").read_bytes() == source.get_blob(empty.digest)


def test_limits_apply_to_each_distinct_tree_not_the_whole_closure(tmp_path, source, remote):
    left = tree(tmp_path, source, "left", {"a": b"left"})
    right = tree(tmp_path, source, "right", {"b": b"rite"})
    limits = ResourceLimits(output_bytes=4, max_files=1)
    counts = transfers.upload_bundles(source, [left, right], factory=remote.factory, limits=limits)
    assert counts.bundles == 2 and counts.objects == 4
    destination = LocalStore(tmp_path / "destination")
    transfers.download_bundles(destination, [left, right], factory=remote.factory, limits=limits)
    extract_tree(left, destination, tmp_path / "out-left", limits=limits)
    extract_tree(right, destination, tmp_path / "out-right", limits=limits)


def test_download_warms_verified_objects_and_reuses_local_chunks(tmp_path, source, remote):
    bundle = tree(tmp_path, source, "files", {"a": b"a", "b": b"b"})
    transfers.upload_bundles(source, [bundle], factory=remote.factory)
    destination = LocalStore(tmp_path / "destination")
    destination.put_blob(b"a")
    remote.gets.clear()
    counts = transfers.download_bundles(destination, [bundle, bundle], factory=remote.factory)
    assert counts.objects == 3 and counts.bundles == 1
    assert set(remote.gets) == {bundle.digest, hashlib.sha256(b"b").hexdigest()}
    assert counts.bytes == len(source.get_blob(bundle.digest)) + 1
    extract_tree(bundle, destination, tmp_path / "output")
    assert (tmp_path / "output/b").read_bytes() == b"b"
    remote.gets.clear()
    assert transfers.download_bundles(destination, [bundle], factory=remote.factory).bytes == 0
    assert remote.gets == []


@pytest.mark.parametrize("operation", [transfers.upload_bundles, transfers.download_bundles])
@pytest.mark.parametrize("parallel", [0, 9, True, 1.5])
def test_parallel_bound_is_checked_before_opening_a_provider(source, remote, operation, parallel):
    with pytest.raises(ValueError):
        operation(source, [], factory=remote.factory, max_parallel=parallel)
    assert remote.opened == []


def test_empty_transfer_opens_no_provider(source, remote):
    assert transfers.upload_bundles(source, [], factory=remote.factory) == transfers.TransferCounts()
    assert transfers.download_bundles(source, [], factory=remote.factory) == transfers.TransferCounts()
    assert remote.opened == []


def test_manifest_count_follows_core_action_input_protocol_bound(source, remote, monkeypatch):
    bundle = manifest(source, [])
    monkeypatch.setattr(transfers, "_MAX_BUNDLES", 2)
    with pytest.raises(ValueError):
        transfers.upload_bundles(source, [bundle] * 3, factory=remote.factory)
    assert remote.opened == []


@pytest.mark.parametrize(
    "entries,limits",
    [
        ([], ResourceLimits()),
        ([{"kind": "device"}], ResourceLimits()),
        ([{"kind": "directory", "chunks": ["a" * 64]}], ResourceLimits()),
        ([{"kind": "file", "size": True, "chunks": []}], ResourceLimits()),
        ([{"kind": "file", "size": -1, "chunks": []}], ResourceLimits()),
        ([{"kind": "file", "size": 1, "chunks": "a" * 64}], ResourceLimits()),
        ([{"kind": "file", "size": 1, "chunks": ["../blob"]}], ResourceLimits()),
        ([{"kind": "file", "size": 1, "chunks": []}], ResourceLimits()),
        ([file_entry(b"12345")], ResourceLimits(output_bytes=4)),
        ([{"kind": "directory"}] * 2, ResourceLimits(max_files=1)),
    ],
)
def test_invalid_or_oversized_manifests_do_not_publish(source, remote, entries, limits):
    value = {"version": 2 if entries == [] else 1, "entries": entries}
    bundle = TreeBundle(source.put_blob(encode(value)))
    with pytest.raises(ValueError):
        transfers.upload_bundles(source, [bundle], factory=remote.factory, limits=limits)
    assert remote.opened == [] and remote.puts == []


def test_manifest_byte_bound_applies_before_any_remote_publication(source, remote, monkeypatch):
    bundle = manifest(source, [{"kind": "directory", "path": "long-name"}])
    monkeypatch.setattr(transfers, "MAX_METADATA_BYTES", 16)
    with pytest.raises(ValueError):
        transfers.upload_bundles(source, [bundle], factory=remote.factory)
    remote.blobs[bundle.digest] = source.get_blob(bundle.digest)
    with pytest.raises(ValueError):
        transfers.download_bundles(source, [bundle], factory=remote.factory)
    assert remote.puts == []


def test_inconsistent_shared_chunk_lengths_fail_before_providers(source, remote):
    entry = file_entry(b"chunk")
    left = manifest(source, [entry])
    right = manifest(source, [{**entry, "size": 4}])
    with pytest.raises(ValueError):
        transfers.upload_bundles(source, [left, right], factory=remote.factory)
    assert remote.opened == []


@pytest.mark.parametrize("operation", ["upload", "download"])
def test_exact_chunk_length_is_checked(tmp_path, source, remote, operation):
    key = source.put_blob(b"12345")
    bundle = manifest(source, [file_entry(b"12345", size=4)])
    if operation == "upload":
        target, run = source, transfers.upload_bundles
    else:
        remote.blobs = {key: b"12345", bundle.digest: source.get_blob(bundle.digest)}
        destination = LocalStore(tmp_path / "destination")
        target, run = destination, transfers.download_bundles
    with pytest.raises(ValueError):
        run(target, [bundle], factory=remote.factory)
    assert bundle.digest not in remote.puts
    assert sorted(remote.opened) == sorted(remote.closed)


@pytest.mark.parametrize("where", ["local", "remote"])
def test_corrupt_download_chunk_is_never_treated_as_a_missing_object(tmp_path, source, remote, where):
    bundle = tree(tmp_path, source, "tree", {"file": b"content"})
    key = hashlib.sha256(b"content").hexdigest()
    transfers.upload_bundles(source, [bundle], factory=remote.factory)
    destination = LocalStore(tmp_path / "destination")
    if where == "local":
        destination.put_blob(b"content")
        path = destination.root / "blobs" / key
        path.chmod(0o644)
        path.write_bytes(b"corrupt")
    else:
        remote.blobs[key] = b"corrupt"
    remote.gets.clear()
    with pytest.raises(ValueError):
        transfers.download_bundles(destination, [bundle], factory=remote.factory)
    assert (key in remote.gets) == (where == "remote")
    assert sorted(remote.opened) == sorted(remote.closed)


@pytest.mark.parametrize("failure", [TimeoutError(), OSError(), ConditionalWriteError(), ValueError()])
def test_uncertain_and_conditional_errors_propagate_without_manifest_publication(tmp_path, source, failure):
    bundle = tree(tmp_path, source, "tree", {"a": b"a"})
    closed = []

    @contextmanager
    def factory():
        def put(data):
            raise failure

        try:
            yield SimpleNamespace(put_blob=put)
        finally:
            closed.append(threading.get_ident())

    with pytest.raises(type(failure)) as caught:
        transfers.upload_bundles(source, [bundle], factory=factory)
    assert caught.value is failure
    assert len(closed) == 1


def test_missing_upload_chunk_never_publishes_manifest(tmp_path, source, remote):
    bundle = tree(tmp_path, source, "tree", {"a": b"a"})
    (source.root / "blobs" / hashlib.sha256(b"a").hexdigest()).unlink()
    with pytest.raises(FileNotFoundError):
        transfers.upload_bundles(source, [bundle], factory=remote.factory)
    assert remote.puts == [] and sorted(remote.opened) == sorted(remote.closed)


def test_provider_initialization_failure_closes_owned_context(tmp_path, source):
    bundle = tree(tmp_path, source, "tree", {"a": b"a"})
    failure = PermissionError()
    closed = []

    @contextmanager
    def factory():
        try:
            raise failure
            yield  # pragma: no cover
        finally:
            closed.append(threading.get_ident())

    with pytest.raises(PermissionError) as caught:
        transfers.upload_bundles(source, [bundle], factory=factory)
    assert caught.value is failure and len(closed) == 1


def test_provider_close_failure_prevents_manifest_publication(tmp_path, source, remote):
    bundle = tree(tmp_path, source, "tree", {"a": b"a"})
    failure = OSError()

    @contextmanager
    def factory():
        with remote.factory() as store:
            try:
                yield store
            finally:
                raise failure

    with pytest.raises(OSError) as caught:
        transfers.upload_bundles(source, [bundle], factory=factory)
    assert caught.value is failure and bundle.digest not in remote.puts
    assert sorted(remote.opened) == sorted(remote.closed)


@pytest.mark.parametrize("problem", ["hash", "bound", "missing"])
def test_bad_remote_manifest_is_not_admitted(tmp_path, source, remote, monkeypatch, problem):
    bundle = manifest(source, [])
    destination = LocalStore(tmp_path / "destination")
    if problem == "hash":
        remote.blobs[bundle.digest] = b"corrupt"
        expected = ValueError
    elif problem == "bound":
        remote.blobs[bundle.digest] = source.get_blob(bundle.digest)
        monkeypatch.setattr(transfers, "MAX_METADATA_BYTES", 1)
        expected = ValueError
    else:
        expected = FileNotFoundError
    with pytest.raises(expected):
        transfers.download_bundles(destination, [bundle], factory=remote.factory)
    with pytest.raises(FileNotFoundError):
        destination.get_blob(bundle.digest)
    assert sorted(remote.opened) == sorted(remote.closed)


def test_put_digest_mismatch_fails_before_manifest_publication(tmp_path, source):
    bundle = tree(tmp_path, source, "tree", {"a": b"a"})

    @contextmanager
    def factory():
        yield SimpleNamespace(put_blob=lambda data: "0" * 64)

    with pytest.raises(ValueError):
        transfers.upload_bundles(source, [bundle], factory=factory)


def test_only_fixed_worker_futures_are_created(tmp_path, source, remote, monkeypatch):
    bundle = tree(tmp_path, source, "many", {f"file-{i}": str(i).encode() for i in range(257)})
    submissions = []
    barrier = threading.Barrier(8)
    original_factory = remote.factory

    @contextmanager
    def factory():
        with original_factory() as store:
            if len(remote.opened) <= 8:
                barrier.wait(timeout=10)
            yield store

    class Executor(ThreadPoolExecutor):
        def submit(self, function, *args, **kwargs):
            submissions.append((function, args, kwargs))
            return super().submit(function, *args, **kwargs)

    monkeypatch.setattr(transfers, "ThreadPoolExecutor", Executor)
    counts = transfers.upload_bundles(source, [bundle], factory=factory)
    assert counts.objects == 258
    assert len(submissions) == 8 and all(args == () and kwargs == {} for _, args, kwargs in submissions)
    assert remote.maximum == 8 and remote.active == 0
    assert len({owner for _, owner in remote.opened[:8]}) == 8
    assert sorted(remote.opened) == sorted(remote.closed)


def test_failure_stops_new_work_joins_blocked_peer_and_closes_providers(tmp_path, source, monkeypatch):
    bundle = tree(tmp_path, source, "many", {f"file-{i}": str(i).encode() for i in range(20)})
    barrier = threading.Barrier(2)
    failure_done, release, finished = threading.Event(), threading.Event(), threading.Event()
    opened, closed, calls, errors = [], [], [], []
    lock = threading.Lock()
    failure = OSError("uncertain transfer")

    @contextmanager
    def factory():
        with lock:
            identity = len(opened)
            opened.append(identity)

        def put(data):
            calls.append(identity)
            barrier.wait(timeout=10)
            if identity == 0:
                raise failure
            assert release.wait(timeout=10)
            return hashlib.sha256(data).hexdigest()

        try:
            yield SimpleNamespace(put_blob=put)
        finally:
            closed.append(identity)

    class Executor(ThreadPoolExecutor):
        def submit(self, function, *args, **kwargs):
            future = super().submit(function, *args, **kwargs)

            def done(result):
                if result.exception() is failure:
                    failure_done.set()

            future.add_done_callback(done)
            return future

    def run():
        try:
            transfers.upload_bundles(source, [bundle], factory=factory, max_parallel=2)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    monkeypatch.setattr(transfers, "ThreadPoolExecutor", Executor)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert failure_done.wait(timeout=10)
        assert not finished.is_set()
    finally:
        release.set()
        thread.join(timeout=10)
    assert finished.is_set() and errors == [failure]
    assert sorted(closed) == [0, 1] and sorted(calls) == [0, 1]


def test_failure_stops_peers_before_slow_provider_close(tmp_path, source):
    bundle = tree(tmp_path, source, "many", {f"file-{i}": str(i).encode() for i in range(20)})
    barrier = threading.Barrier(2)
    closing, release_close, release_peer, peer_closed = (threading.Event() for _ in range(4))
    opened, calls, errors = [], [], []
    lock = threading.Lock()
    failure = OSError()

    @contextmanager
    def factory():
        with lock:
            identity = len(opened)
            opened.append(identity)

        def put(data):
            calls.append(identity)
            barrier.wait(timeout=10)
            if identity == 0:
                raise failure
            assert release_peer.wait(timeout=10)
            return hashlib.sha256(data).hexdigest()

        try:
            yield SimpleNamespace(put_blob=put)
        finally:
            if identity == 0:
                closing.set()
                assert release_close.wait(timeout=10)
            else:
                peer_closed.set()

    def run():
        try:
            transfers.upload_bundles(source, [bundle], factory=factory, max_parallel=2)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert closing.wait(timeout=10)
        release_peer.set()
        assert peer_closed.wait(timeout=10)
        assert sorted(calls) == [0, 1]
    finally:
        release_peer.set()
        release_close.set()
        thread.join(timeout=10)
    assert not thread.is_alive() and errors == [failure]


def test_backend_requires_explicit_transfer_factory(source):
    backend = iris.IrisBackend.__new__(iris.IrisBackend)
    backend._transfer_factory = None
    with pytest.raises(ValueError):
        backend.upload_bundles(source, [])
    with pytest.raises(ValueError):
        backend.download_bundles(source, [])


def test_backend_passes_explicit_sdk_limits_to_each_bulk_operation(source, monkeypatch):
    limits = ResourceLimits(output_bytes=40 * 1024**3, max_files=400_000)
    received = []

    def transfer(store, bundles, **kwargs):
        received.append((store, bundles, kwargs))
        return transfers.TransferCounts()

    backend = iris.IrisBackend.__new__(iris.IrisBackend)
    backend._transfer_factory = lambda: None
    monkeypatch.setattr(iris, "_upload_bundles", transfer)
    monkeypatch.setattr(iris, "_download_bundles", transfer)
    backend.upload_bundles(source, [], max_parallel=3, limits=limits)
    backend.download_bundles(source, [], max_parallel=3, limits=limits)
    assert all(kwargs["limits"] is limits and kwargs["max_parallel"] == 3 for _, _, kwargs in received)


def test_context_factory_owns_auth_controller_and_capability_per_thread(tmp_path, source, monkeypatch):
    controllers, actors, closed_controllers, closed_actors, calls = [], [], [], [], []
    wire_jobs = []
    storage = LocalStore(tmp_path / "remote")
    lock = threading.Lock()

    @contextmanager
    def open_client(**kwargs):
        assert kwargs == {"cluster_name": "marin", "workspace": tmp_path}
        with lock:
            identity = len(controllers)
            owner = threading.get_ident()
            controllers.append((identity, owner))

        def mint(name):
            assert threading.get_ident() == owner
            assert name == "namespace/existing/coordinator"
            return SimpleNamespace(capability_url="https://controller/cap", token=str(identity))

        try:
            yield SimpleNamespace(mint_endpoint_token=mint)
        finally:
            assert threading.get_ident() == owner
            closed_controllers.append(identity)

    class Actor:
        def __init__(self, address, token, name, rpc_seconds):
            self.identity, self.owner = int(token), threading.get_ident()
            actors.append((self.identity, self.owner))

        def call(self, method, request):
            assert threading.get_ident() == self.owner and method == "SignBlob"
            calls.append(self.identity)

        def close(self):
            assert threading.get_ident() == self.owner
            closed_actors.append(self.identity)

    class Store:
        def __init__(self, signer):
            self.signer = signer

        def put_blob(self, data):
            self.signer(iris.BlobAccessRequest("put", hashlib.sha256(data).hexdigest(), len(data)))
            return storage.put_blob(data)

        def get_blob(self, key):
            self.signer(iris.BlobAccessRequest("get", key))
            return storage.get_blob(key)

    def parse(value):
        wire_jobs.append(value)
        assert value == "/owner/existing"
        return value

    connect, types, rigging = (
        ModuleType("iris.cli.connect"),
        ModuleType("iris.cluster.types"),
        ModuleType("rigging.connect"),
    )
    connect.open_iris_client = open_client
    types.JobName = SimpleNamespace(from_wire=parse)
    types.Namespace = SimpleNamespace(from_job_id=lambda job: "namespace/existing")
    rigging.capability_path = lambda *args: pytest.fail("minted capability URL is available")
    for name, module in (("iris.cli.connect", connect), ("iris.cluster.types", types), ("rigging.connect", rigging)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(iris, "_CapabilityRPC", Actor)
    monkeypatch.setattr(iris, "SignedBlobStore", Store)
    factory = iris.backend_context_factory(
        job_id="/owner/existing", cluster_name="marin", controller_url="https://controller", workspace=tmp_path
    )
    bundle = tree(tmp_path, source, "tree", {"a": b"a", "b": b"b"})
    with factory() as backend:
        backend.upload_bundles(source, [bundle], max_parallel=2)
        assert calls and 0 not in calls
    assert sorted(controllers) == sorted(actors)
    assert sorted(closed_controllers) == sorted(closed_actors) == list(range(len(controllers)))
    assert len(controllers) == 4
    assert wire_jobs == ["/owner/existing"] * 4


def test_context_factory_closes_auth_when_capability_discovery_fails(monkeypatch):
    closed = []
    failure = PermissionError()

    @contextmanager
    def open_client(**kwargs):
        def mint(name):
            raise failure

        try:
            yield SimpleNamespace(mint_endpoint_token=mint)
        finally:
            closed.append(threading.get_ident())

    connect, types, rigging = (
        ModuleType("iris.cli.connect"),
        ModuleType("iris.cluster.types"),
        ModuleType("rigging.connect"),
    )
    connect.open_iris_client = open_client
    types.JobName = SimpleNamespace(from_wire=lambda value: value)
    types.Namespace = SimpleNamespace(from_job_id=lambda job: "namespace/existing")
    rigging.capability_path = lambda *args: pytest.fail("failed mint must not construct a fallback URL")
    for name, module in (("iris.cli.connect", connect), ("iris.cluster.types", types), ("rigging.connect", rigging)):
        monkeypatch.setitem(sys.modules, name, module)
    factory = iris.backend_context_factory(
        job_id="/owner/existing", cluster_name="marin", controller_url="https://controller"
    )
    with pytest.raises(PermissionError) as caught, factory():
        pytest.fail("failed capability discovery must not yield a backend")
    assert caught.value is failure and len(closed) == 1
