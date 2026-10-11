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
            if self.path in objects:
                raise NativeConflict("exists")
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


def test_publication_uses_conditional_creation_without_reading_existing_bytes(backend):
    data, digest, path = blob()
    store = RemoteStore("s3://bucket/cache")
    assert store.put_blob(data) == store.put_blob(data) == digest
    assert backend.events == [("conditional", path), ("put", path, data)] * 2
    assert backend.objects[path] == data


def test_local_hit_recreates_an_evicted_remote_blob(backend, tmp_path):
    data, digest, path = blob()
    local = LocalStore(tmp_path / "local")
    local.put_blob(data)
    store = RemoteStore("s3://bucket/cache", local_cache=local)
    assert store.put_blob(data) == digest
    del backend.objects[path]
    backend.events.clear()
    assert store.put_blob(data) == digest
    assert backend.objects[path] == data
    assert all(event[0] not in {"get", "read"} for event in backend.events)


@pytest.mark.parametrize("error", [PermissionError("denied"), ConnectionError("unavailable"), OSError("I/O")])
def test_storage_write_errors_propagate(backend, error):
    data, _, _ = blob()
    backend.state.write_error = error
    with pytest.raises(type(error)) as raised:
        RemoteStore("s3://bucket/cache").put_blob(data)
    assert raised.value is error


def test_fetches_rely_on_store_integrity_and_remain_bounded(backend, monkeypatch):
    data, digest, path = blob()
    backend.objects[path] = data
    store = RemoteStore("s3://bucket/cache")
    assert store.get_blob(digest) == data
    monkeypatch.setattr(remote_store, "MAX_METADATA_BYTES", 8)
    backend.objects[path] = b"x" * 100
    with pytest.raises(ValueError):
        store.get_blob(digest)
    with pytest.raises(ValueError):
        store.put_blob(b"x" * 9)
