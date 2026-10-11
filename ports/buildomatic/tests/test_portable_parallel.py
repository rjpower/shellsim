"""Check bounded file concurrency and verified cache reuse with real admission."""

import hashlib
import json
import threading

import pytest

from ports.buildomatic import portable
from ports.buildomatic.tests import test_portable


@pytest.fixture
def sdk_fixture(tmp_path, monkeypatch):
    yield from test_portable.sdk_fixture.__wrapped__(tmp_path, monkeypatch)


def test_parallel_export_joins_workers_before_descriptor(tmp_path, sdk_fixture, monkeypatch):
    context = sdk_fixture(python=True, host_python=True)
    operation = portable._export_blob
    lock = threading.Lock()
    barrier = threading.Barrier(4)
    active = peak = entered = 0

    def observe(*args):
        nonlocal active, peak, entered
        with lock:
            entered += 1
            first = entered <= 4
            active += 1
            peak = max(peak, active)
        try:
            if first:
                barrier.wait(timeout=10)
            assert not (tmp_path / "export/sdk.json").exists()
            return operation(*args)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(portable, "_export_blob", observe)
    descriptor = portable.export_sdk(context, tmp_path / "export", max_workers=4)
    assert peak == 4 and active == 0
    assert portable.import_sdk(descriptor, tmp_path / "import").identity == context.identity


def test_interrupted_export_reuses_only_complete_verified_cache(tmp_path, sdk_fixture, monkeypatch):
    context = sdk_fixture(python=True)
    cache = tmp_path / "cache"
    operation = portable._publish_blob
    published = []

    def interrupt(source, directory, entry, *args):
        if published:
            (directory / "blobs/.interrupted.partial").write_bytes(b"partial")
            raise OSError("injected interruption")
        result = operation(source, directory, entry, *args)
        published.append(entry.sha256)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(portable, "_publish_blob", interrupt)
        with pytest.raises(OSError):
            portable.export_sdk(context, tmp_path / "interrupted", blob_cache=cache, max_workers=1)
    assert not (tmp_path / "interrupted/sdk.json").exists()
    calls = []

    def observe(source, directory, entry, *args):
        calls.append(entry.sha256)
        return operation(source, directory, entry, *args)

    monkeypatch.setattr(portable, "_publish_blob", observe)
    descriptor = portable.export_sdk(context, tmp_path / "complete", blob_cache=cache)
    assert published[0] not in calls
    assert portable.import_sdk(descriptor, tmp_path / "import").identity == context.identity
    assert len(list((cache / "blobs").glob(published[0]))) == 1
    assert (cache / "blobs/.interrupted.partial").read_bytes() == b"partial"


@pytest.mark.parametrize("damage", ["bytes", "size", "mode", "link"])
def test_cache_hit_is_fully_verified(tmp_path, sdk_fixture, damage):
    context = sdk_fixture()
    cache = tmp_path / "cache"
    portable.export_sdk(context, tmp_path / "first", blob_cache=cache)
    blob = next((cache / "blobs").iterdir())
    contents = blob.read_bytes()
    blob.chmod(0o644)
    if damage == "bytes":
        blob.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
        blob.chmod(0o444)
    elif damage == "size":
        blob.write_bytes(contents + b"x")
        blob.chmod(0o444)
    elif damage == "link":
        target = tmp_path / "linked"
        target.write_bytes(contents)
        blob.unlink()
        blob.symlink_to(target)
    with pytest.raises(ValueError):
        portable.export_sdk(context, tmp_path / "second", blob_cache=cache)
    assert not (tmp_path / "second/sdk.json").exists()


def test_failed_file_stops_and_joins_other_loops(tmp_path, monkeypatch):
    started = threading.Event()
    joined = threading.Event()

    def execute(sources, destination, cache, limits, stop):
        if sources == "fail":
            assert started.wait(timeout=10)
            raise ValueError("injected failure")
        started.set()
        assert stop.wait(timeout=10)
        joined.set()

    monkeypatch.setattr(portable, "_export_blob", execute)
    with pytest.raises(ValueError):
        portable._export_blobs({"a": "fail", "b": "wait"}, tmp_path, None, portable.PortableLimits(), 2)
    assert joined.is_set()


def test_interrupted_blob_never_publishes_cache_entry(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"payload")
    entry = portable.FileEntry(hashlib.sha256(b"payload").hexdigest(), 7, 0o644)
    cache = tmp_path / "cache"
    (cache / "blobs").mkdir(parents=True)

    def interrupt(source, destination, *args, **kwargs):
        destination.write_bytes(b"pay")
        raise OSError("injected partial write")

    monkeypatch.setattr(portable, "_stream", interrupt)
    with pytest.raises(OSError):
        portable._publish_blob(source, cache, entry, portable.PortableLimits(), threading.Event(), 0o644)
    assert list((cache / "blobs").iterdir()) == []


def test_export_never_calls_sdk_producers(tmp_path, sdk_fixture, monkeypatch):
    context = sdk_fixture(python=True)
    from ports._support import sdk

    def reject(*args, **kwargs):
        pytest.fail("portable export invoked a producer")

    for name in ("materialize", "_materialize", "_produce", "build_port", "_publish_json"):
        monkeypatch.setattr(sdk, name, reject)
    portable.export_sdk(context, tmp_path / "output", blob_cache=tmp_path / "cache")


@pytest.mark.parametrize("workers", [0, 33, True, 1.5])
def test_worker_bound_rejects_before_mutation(tmp_path, sdk_fixture, workers):
    with pytest.raises(ValueError):
        portable.export_sdk(sdk_fixture(), tmp_path / "output", max_workers=workers, blob_cache=tmp_path / "cache")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "cache").exists()


def test_inventory_and_free_space_bounds_precede_cache_mutation(tmp_path, sdk_fixture, monkeypatch):
    context = sdk_fixture()
    with pytest.raises(ValueError):
        portable.export_sdk(
            context, tmp_path / "output", blob_cache=tmp_path / "cache", limits=portable.PortableLimits(max_files=1)
        )
    assert not (tmp_path / "cache").exists()
    monkeypatch.setattr(portable.shutil, "disk_usage", lambda _p: type("Usage", (), {"free": 1})())
    with pytest.raises(OSError):
        portable.export_sdk(context, tmp_path / "output", blob_cache=tmp_path / "cache")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "cache").exists()


def test_cache_cannot_overlap_inputs_or_output(tmp_path, sdk_fixture):
    context = sdk_fixture()
    for cache in (context.sdk.root / "cache", tmp_path / "output/cache", tmp_path):
        with pytest.raises(ValueError):
            portable.export_sdk(context, tmp_path / "output", blob_cache=cache)
    assert not (tmp_path / "output").exists()


def test_dedup_writes_one_blob_per_digest_and_preserves_modes(tmp_path, sdk_fixture, monkeypatch):
    context = sdk_fixture(python=True)
    operation = portable._publish_blob
    writes = []

    def observe(source, directory, entry, *args):
        writes.append(entry.sha256)
        return operation(source, directory, entry, *args)

    monkeypatch.setattr(portable, "_publish_blob", observe)
    descriptor = portable.export_sdk(context, tmp_path / "output", blob_cache=tmp_path / "cache")
    value = json.loads(descriptor.read_text())
    expected = {item["sha256"] for root in value["roots"].values() for item in root["files"].values()}
    assert len(writes) == len(set(writes)) == len(expected)
    for digest in expected:
        blob = descriptor.parent / "blobs" / digest
        assert hashlib.sha256(blob.read_bytes()).hexdigest() == digest
        assert blob.stat().st_mode & 0o777 == 0o444
    assert portable.import_sdk(descriptor, tmp_path / "import").identity == context.identity
