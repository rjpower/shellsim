"""Check publication traffic and failure semantics with bounded fake remote reads.

No Iris packages, credentials or live object-store requests are needed. Keep
these tests separate from the concurrently maintained Iris backend tests.
"""

import hashlib
import io
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ports.buildomatic import LocalStore, remote_store
from ports.buildomatic.remote_store import RemoteStore


class NativeConflict(RuntimeError):
    pass


@pytest.fixture
def backend(monkeypatch):
    objects = {}
    events = []
    state = SimpleNamespace(read_error=None, write_error=None, winner=None)

    class Handle(io.BytesIO):
        def read(self, size=-1):
            events.append(("read", size))
            return super().read(size)

    class Filesystem:
        def open(self, path, mode):
            assert mode == "rb"
            events.append(("get", path))
            if state.read_error is not None:
                raise state.read_error
            if path not in objects:
                raise FileNotFoundError(path)
            return Handle(objects[path])

    class Object:
        def __init__(self, path):
            self.path = path
            events.append(("conditional", path))

        def write(self, data, *, expected_version):
            assert expected_version is None
            events.append(("put", self.path, data))
            if state.write_error is not None:
                if state.winner is not None:
                    objects[self.path] = state.winner
                raise state.write_error
            assert self.path not in objects
            objects[self.path] = data
            return "opaque-revision"

    conditional = ModuleType("rigging.filesystem.conditional_object")
    conditional.ConditionalWriteError = NativeConflict
    buckets = ModuleType("rigging.filesystem.buckets")
    buckets.filesystem_for = lambda path: (Filesystem(), path)
    monkeypatch.setitem(sys.modules, conditional.__name__, conditional)
    monkeypatch.setitem(sys.modules, buckets.__name__, buckets)
    monkeypatch.setattr(remote_store, "_conditional", Object)
    return SimpleNamespace(objects=objects, events=events, state=state)


def blob(data=b"SDK chunk"):
    digest = hashlib.sha256(data).hexdigest()
    return data, digest, "s3://bucket/cache/blobs/" + digest


def test_existing_remote_bytes_are_verified_without_conditional_upload(backend, tmp_path):
    data, digest, path = blob()
    backend.objects[path] = data
    local = LocalStore(tmp_path / "local")
    local.put_blob(data)
    store = RemoteStore("s3://bucket/cache", local_cache=local)
    assert store.put_blob(data) == digest
    assert backend.events == [("get", path), ("read", remote_store.MAX_METADATA_BYTES + 1)]
    assert local.get_blob(digest) == data


def test_missing_remote_blob_is_conditionally_created_and_repeat_upload_is_avoided(backend):
    data, digest, path = blob()
    store = RemoteStore("s3://bucket/cache")
    assert store.put_blob(data) == digest
    assert backend.events == [("get", path), ("conditional", path), ("put", path, data)]
    assert store.put_blob(data) == digest
    assert backend.events[3:] == [("get", path), ("read", remote_store.MAX_METADATA_BYTES + 1)]
    assert backend.objects[path] == data


@pytest.mark.parametrize("status", [409, 412])
def test_conditional_create_conflict_reads_and_verifies_concurrent_winner(backend, status):
    data, digest, path = blob()
    backend.state.write_error = NativeConflict(status)
    backend.state.winner = data
    assert RemoteStore("s3://bucket/cache").put_blob(data) == digest
    assert backend.events == [
        ("get", path),
        ("conditional", path),
        ("put", path, data),
        ("get", path),
        ("read", remote_store.MAX_METADATA_BYTES + 1),
    ]


@pytest.mark.parametrize("concurrent", [False, True])
def test_corrupt_remote_bytes_are_errors_and_do_not_become_cache_misses(backend, tmp_path, concurrent):
    data, _digest, path = blob()
    if concurrent:
        backend.state.write_error = NativeConflict(412)
        backend.state.winner = b"corrupt winner"
    else:
        backend.objects[path] = b"corrupt existing object"
    local = LocalStore(tmp_path / "local")
    local.put_blob(data)
    with pytest.raises(ValueError):
        RemoteStore("s3://bucket/cache", local_cache=local).put_blob(data)
    assert sum(event[0] == "put" for event in backend.events) == int(concurrent)
    assert backend.objects[path] != data


def test_local_hit_revalidates_remote_and_recreates_evicted_blob(backend, tmp_path):
    data, digest, path = blob()
    local = LocalStore(tmp_path / "local")
    local.put_blob(data)
    store = RemoteStore("s3://bucket/cache", local_cache=local)
    assert store.put_blob(data) == digest
    del backend.objects[path]
    backend.events.clear()
    assert store.put_blob(data) == digest
    assert backend.events == [("get", path), ("conditional", path), ("put", path, data)]
    assert backend.objects[path] == data


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("error", [PermissionError("denied"), ConnectionError("unavailable"), OSError("I/O")])
def test_nonmissing_storage_errors_propagate(backend, operation, error):
    data, _digest, _path = blob()
    setattr(backend.state, operation + "_error", error)
    with pytest.raises(type(error)) as raised:
        RemoteStore("s3://bucket/cache").put_blob(data)
    assert raised.value is error
    assert sum(event[0] == "put" for event in backend.events) == int(operation == "write")


def test_missing_concurrent_winner_propagates_without_retry(backend):
    data, _digest, _path = blob()
    backend.state.write_error = NativeConflict(409)
    with pytest.raises(FileNotFoundError):
        RemoteStore("s3://bucket/cache").put_blob(data)
    assert sum(event[0] == "put" for event in backend.events) == 1


def test_existing_reads_are_bounded_and_oversize_data_cannot_trigger_upload(backend, monkeypatch):
    monkeypatch.setattr(remote_store, "MAX_METADATA_BYTES", 8)
    data, _digest, path = blob(b"bounded")
    backend.objects[path] = b"x" * 100
    with pytest.raises(ValueError):
        RemoteStore("s3://bucket/cache").put_blob(data)
    assert backend.events == [("get", path), ("read", 9)]
    backend.events.clear()
    with pytest.raises(ValueError):
        RemoteStore("s3://bucket/cache").put_blob(b"x" * 9)
    assert backend.events == []


def test_existing_bytes_must_equal_the_submitted_bytes(backend, monkeypatch):
    data, _digest, _path = blob()
    # Exercise the equality guard independently of the normal SHA check.
    monkeypatch.setattr(RemoteStore, "_get_remote_blob", lambda _self, _digest: b"different bytes")
    with pytest.raises(ValueError):
        RemoteStore("s3://bucket/cache").put_blob(data)
    assert backend.events == []
