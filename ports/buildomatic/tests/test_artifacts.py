"""Exercise catalog publication and disposable installations with real archives."""

import pytest

from ports.buildomatic import LocalStore
from ports.buildomatic.artifacts import install, register, resolve, upload


def test_canonical_store_survives_empty_client_and_worker_caches(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "value").write_text("a build product")
    (source / "alias").symlink_to("value")
    store = LocalStore(tmp_path / "canonical")
    artifact = upload(store, source, tmp_path / "scratch", workers=2)
    register(store, "port/example", {"artifact": artifact})
    fresh_client = LocalStore(tmp_path / "canonical")
    reference = resolve(fresh_client, "port/example")
    installed = install(fresh_client, reference["artifact"], tmp_path / "worker")
    assert (installed / "alias").read_text() == "a build product"

    def reject_download(*args):
        pytest.fail("warm installations must not fetch or hash product bytes")

    monkeypatch.setattr(fresh_client, "get_blob", reject_download)
    assert install(fresh_client, artifact, tmp_path / "worker") == installed


def test_failed_upload_does_not_publish_catalog_entry(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "value").write_bytes(b"input")
    store = LocalStore(tmp_path / "canonical")

    def fail_put(data):
        raise OSError("store unavailable")

    monkeypatch.setattr(store, "put_blob", fail_put)
    with pytest.raises(OSError):
        artifact = upload(store, source, tmp_path / "scratch")
        register(store, "port/example", {"artifact": artifact})
    assert resolve(store, "port/example") is None
