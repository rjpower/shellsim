"""Deterministic journal transitions use fake workers; transport uses small trees."""

import hashlib
import json
from dataclasses import replace

import pytest

from ports.buildomatic import (
    Action,
    AttemptResult,
    AttemptState,
    BuildRequest,
    BuildState,
    ConditionalWriteError,
    Coordinator,
    CoordinatorFenced,
    IdempotencyConflict,
    InputMount,
    LocalStore,
    NodeState,
    ResourceLimits,
    TreeBundle,
    WorkerReport,
    WorkerState,
    capture_tree,
    extract_tree,
)
from ports.buildomatic.contracts import CHUNK_BYTES, encode


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "store")


@pytest.fixture
def bundle(tmp_path, store):
    tree = tmp_path / "source"
    tree.mkdir()
    return capture_tree(tree, store)


class FakeWorker:
    def __init__(self, store, key="build"):
        self.store, self.key = store, key
        self.attempts = {}
        self.reports = {}
        self.cancelled = set()
        self.acknowledged = set()
        self.unavailable = False
        self.uncertain_submit = False
        self.fail_ack = False

    def submit(self, attempt):
        journal = json.loads(self.store.read_journal(self.key).data)
        assert attempt.id in journal["attempts"]
        assert journal["nodes"][attempt.action.id]["state"] == "running"
        if attempt.id in self.cancelled:
            return
        if attempt.id in self.attempts:
            assert self.attempts[attempt.id] == attempt
        self.attempts[attempt.id] = attempt
        self.reports[attempt.id] = WorkerReport(WorkerState.RUNNING)
        if self.uncertain_submit:
            raise TimeoutError("response lost after accepted dispatch")

    def poll(self, attempt_id):
        if self.unavailable:
            raise TimeoutError("transient RPC timeout")
        return self.reports.get(attempt_id, WorkerReport(WorkerState.UNKNOWN))

    def cancel(self, attempt_id):
        self.cancelled.add(attempt_id)
        self.reports[attempt_id] = WorkerReport(
            WorkerState.COMPLETED,
            AttemptResult(attempt_id, AttemptState.CANCELLED),
        )

    def acknowledge(self, attempt_id):
        if self.fail_ack:
            raise TimeoutError("ack response unavailable")
        self.acknowledged.add(attempt_id)
        self.reports.pop(attempt_id, None)

    def complete(self, action_id, bundle=None, state=AttemptState.SUCCEEDED):
        attempt = next(a for a in reversed(list(self.attempts.values())) if a.action.id == action_id)
        self.reports[attempt.id] = WorkerReport(
            WorkerState.COMPLETED,
            AttemptResult(attempt.id, state, bundle, 0 if bundle else 1),
        )


def request(*actions, workers=1):
    return BuildRequest("test-key", tuple(actions), workers)


def action(key, *dependencies, **kwargs):
    return Action(key, ("trusted-tool",), tuple(dependencies), **kwargs)


@pytest.mark.parametrize("count", [0, 513])
def test_node_bound(count):
    with pytest.raises(ValueError):
        request(*(action(str(index)) for index in range(count)))


@pytest.mark.parametrize("workers", [0, 33, True])
def test_worker_bound(workers):
    with pytest.raises(ValueError):
        request(action("node"), workers=workers)


@pytest.mark.parametrize(
    "actions",
    [
        (action("a", "b"), action("b", "a")),
        (action("a", "absent"),),
        (action("a"), action("a")),
    ],
)
def test_invalid_dag(actions):
    with pytest.raises(ValueError):
        request(*actions)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": "../bad"},
        {"id": "native/zlib"},
        {"argv": ()},
        {"argv": ("a\0",)},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": 0},
        {"max_attempts": 11},
        {"env": (("BUILD_OUTPUT_DIR", "override"),)},
        {"dependencies": ("a", "a")},
        {"argv": ("tool", "x" * 65536)},
    ],
)
def test_action_validation(kwargs):
    values = {"id": "safe-hash-id", "argv": ("tool",)}
    values.update(kwargs)
    with pytest.raises(ValueError):
        Action(**values)


def test_journal_cas_and_identical_write_avoid_aba(store):
    version = store.write_journal("key", b"same", None)
    newer = store.write_journal("key", b"same", version)
    assert newer != version
    assert store.read_journal("key").data == b"same"
    for expected in (None, version):
        with pytest.raises(ConditionalWriteError):
            store.write_journal("key", b"stale", expected)


def test_immutable_blob_verifies_corruption(store):
    key = store.put_blob(b"value")
    assert store.put_blob(b"value") == key
    path = store.root / "blobs" / key
    path.chmod(0o644)
    path.write_bytes(b"bad")
    with pytest.raises(ValueError):
        store.get_blob(key)
    with pytest.raises(ValueError):
        store.put_blob(b"value")


def test_oversized_blob_read_is_bounded(store, monkeypatch):
    import ports.buildomatic.store as storage

    monkeypatch.setattr(storage, "MAX_METADATA_BYTES", 8)
    key = hashlib.sha256(b"tiny").hexdigest()
    (store.root / "blobs" / key).write_bytes(b"oversized")
    with pytest.raises(ValueError):
        store.get_blob(key)


def test_chunked_tree_roundtrip_preserves_links_modes_and_empty_dirs(tmp_path, store):
    root = tmp_path / "tree"
    (root / "bin").mkdir(parents=True)
    (root / "empty").mkdir()
    data = b"a" * CHUNK_BYTES + b"tail"
    (root / "bin" / "tool").write_bytes(data)
    (root / "bin" / "tool").chmod(0o751)
    (root / "tool").symlink_to("bin/tool")
    (root / "bin" / "alias").symlink_to("../tool")
    (root / "missing").symlink_to("empty/missing")
    (root / "zero").write_bytes(b"")
    sealed = capture_tree(root, store)
    manifest = json.loads(store.get_blob(sealed.digest))
    chunks = next(e["chunks"] for e in manifest["entries"] if e["path"] == "bin/tool")
    assert [len(store.get_blob(key)) for key in chunks] == [CHUNK_BYTES, 4]
    destination = tmp_path / "extracted"
    extract_tree(sealed, store, destination)
    assert (destination / "tool").read_bytes() == data
    assert (destination / "bin/tool").stat().st_mode & 0o777 == 0o751
    assert (destination / "bin/alias").readlink().as_posix() == "../tool"
    assert (destination / "empty").is_dir()
    assert capture_tree(destination, store) == sealed


@pytest.mark.parametrize("target", ["/etc/passwd", "../outside", "cycle"])
def test_capture_rejects_escaping_or_cyclic_symlinks(tmp_path, store, target):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "cycle").symlink_to(target)
    with pytest.raises(ValueError):
        capture_tree(root, store)


@pytest.mark.parametrize("limits", [ResourceLimits(output_bytes=3), ResourceLimits(max_files=1)])
def test_capture_resource_bounds(tmp_path, store, limits):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a").write_bytes(b"four")
    (root / "b").write_bytes(b"")
    with pytest.raises(ValueError):
        capture_tree(root, store, limits=limits)


@pytest.mark.parametrize(
    "entries",
    [
        [{"path": "../escape", "kind": "directory", "mode": 0o755}],
        [{"path": "/absolute", "kind": "directory", "mode": 0o755}],
        [{"path": "a/b", "kind": "file", "mode": 0o644, "size": 0, "chunks": []}],
        [{"path": "a", "kind": "directory", "mode": 0o4755}],
        [{"path": "a", "kind": "symlink", "mode": 0o777, "target": "../outside"}],
        [
            {"path": "a", "kind": "symlink", "mode": 0o777, "target": "b"},
            {"path": "a/file", "kind": "file", "mode": 0o644, "size": 0, "chunks": []},
        ],
    ],
)
def test_extract_rejects_unsafe_manifest_before_writes(tmp_path, store, entries):
    bundle = TreeBundle(store.put_blob(encode({"version": 1, "entries": entries})))
    destination = tmp_path / "output"
    with pytest.raises(ValueError):
        extract_tree(bundle, store, destination)
    assert not destination.exists()


def test_extract_rejects_bad_chunk_and_expired_cache(tmp_path, store):
    chunk = store.put_blob(b"one")
    manifest = {"version": 1, "entries": [{"path": "a", "kind": "file", "mode": 0o644, "size": 4, "chunks": [chunk]}]}
    bundle = TreeBundle(store.put_blob(encode(manifest)))
    with pytest.raises(ValueError):
        extract_tree(bundle, store, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    (store.root / "blobs" / chunk).unlink()
    with pytest.raises(FileNotFoundError):
        extract_tree(bundle, store, tmp_path / "expired")
    assert not (tmp_path / "expired").exists()


def test_scheduler_dispatches_dependant_while_independent_node_runs(store, bundle):
    left, right = FakeWorker(store), FakeWorker(store)
    coordinator = Coordinator(store, "build", {"left": left, "right": right})
    identity = coordinator.submit(request(action("a"), action("b"), action("c", "a"), workers=2))
    assert coordinator.submit(coordinator.request) == identity
    coordinator.tick()
    left.complete("a", bundle)
    result = coordinator.tick()
    assert {node.action_id: node.state for node in result.nodes} == {
        "a": NodeState.SUCCEEDED,
        "b": NodeState.RUNNING,
        "c": NodeState.RUNNING,
    }
    dependant = next(a for a in left.attempts.values() if a.action.id == "c")
    assert dependant.inputs == (InputMount("a", bundle),)
    assert not left.acknowledged


def test_restart_retains_attempt_and_fences_previous_coordinator(store, bundle):
    worker = FakeWorker(store)
    old = Coordinator(store, "build", {"worker": worker})
    old.submit(request(action("a")))
    old.tick()
    first_id = next(iter(worker.attempts))
    new = Coordinator(store, "build", {"worker": worker})
    assert new.tick().nodes[0].attempts == 1
    assert list(worker.attempts) == [first_id]
    with pytest.raises(CoordinatorFenced):
        old.tick()
    with pytest.raises(CoordinatorFenced):
        old.cancel()
    worker.complete("a", bundle)
    assert new.tick().state == BuildState.SUCCEEDED
    assert worker.poll(first_id).state == WorkerState.COMPLETED
    new.acknowledge()
    assert worker.poll(first_id).state == WorkerState.UNKNOWN
    retained = new.result()
    assert Coordinator(store, "build", {"worker": worker}).result() == retained


def test_idempotency_conflict_checks_entire_request(store):
    coordinator = Coordinator(store, "build", {"worker": FakeWorker(store)})
    coordinator.submit(request(action("a")))
    with pytest.raises(IdempotencyConflict):
        coordinator.submit(request(action("a", env=(("VALUE", "changed"),))))
    with pytest.raises(IdempotencyConflict):
        coordinator.submit(replace(coordinator.request, idempotency_key="different"))


def test_uncertain_dispatch_and_rpc_timeout_never_duplicate_work(store):
    worker, other = FakeWorker(store), FakeWorker(store)
    worker.uncertain_submit = True
    coordinator = Coordinator(store, "build", {"worker": worker, "other": other})
    coordinator.submit(request(action("a"), workers=2))
    coordinator.tick()
    worker.unavailable = True
    for _ in range(3):
        assert coordinator.tick().nodes[0].attempts == 1
    worker.unavailable = False
    assert coordinator.tick().nodes[0].state == NodeState.RUNNING
    assert len(worker.attempts) == 1
    assert not other.attempts


def test_explicit_worker_loss_and_failure_retry_are_bounded(store):
    worker = FakeWorker(store)
    coordinator = Coordinator(store, "build", {"worker": worker})
    coordinator.submit(request(action("a"), action("child", "a"), action("independent")))
    coordinator.tick()
    lost = next(iter(worker.attempts))
    worker.reports.clear()
    coordinator.tick()
    assert lost in worker.cancelled
    assert len(worker.attempts) == 2
    worker.complete("a", state=AttemptState.FAILED)
    result = coordinator.tick()
    assert [node.state for node in result.nodes] == [NodeState.FAILED, NodeState.BLOCKED, NodeState.RUNNING]
    assert result.nodes[0].attempts == 2


def test_cancellation_tombstones_unconfirmed_dispatch(store):
    worker = FakeWorker(store)
    worker.uncertain_submit = True
    coordinator = Coordinator(store, "build", {"worker": worker})
    coordinator.submit(request(action("a"), action("b")))
    coordinator.tick()
    result = coordinator.cancel()
    assert result.state == BuildState.CANCELLED
    assert all(node.state == NodeState.CANCELLED for node in result.nodes)
    assert len(worker.cancelled) == 1
    coordinator.acknowledge()


def test_acknowledgement_retries_after_restart(store, bundle):
    worker = FakeWorker(store)
    coordinator = Coordinator(store, "build", {"worker": worker})
    coordinator.submit(request(action("a")))
    coordinator.tick()
    worker.complete("a", bundle)
    coordinator.tick()
    worker.fail_ack = True
    coordinator.acknowledge()
    assert not worker.acknowledged
    worker.fail_ack = False
    new = Coordinator(store, "build", {"worker": worker})
    new.tick()
    assert worker.acknowledged == set(worker.attempts)


def test_independent_request_journals_remain_unaffected(store):
    first_worker = FakeWorker(store, "one")
    second_worker = FakeWorker(store, "two")
    first = Coordinator(store, "one", {"worker": first_worker})
    second = Coordinator(store, "two", {"worker": second_worker})
    first.submit(request(action("a")))
    second.submit(request(action("b")))
    first.tick()
    second.tick()
    first.cancel()
    assert second.tick().state == BuildState.RUNNING
    assert not second_worker.cancelled
