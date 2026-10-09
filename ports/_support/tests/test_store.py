"""Exercise cache identity, failed publication and hostile source handling."""

import hashlib
import io
import tarfile

import pytest

from ports._support.store import build_slot, extract, fetch, identity, verify


def test_cache_reuses_only_matching_inputs(tmp_path):
    inputs = {"recipe": "first", "dependencies": {"zlib": "one"}}
    with build_slot(tmp_path, inputs) as build:
        assert not build.cached
        (build.result / "extension.so").write_bytes(b"compiled")
    result = tmp_path / "results" / identity(inputs)
    assert verify(result, inputs)["files"]["extension.so"]["sha256"] == hashlib.sha256(b"compiled").hexdigest()
    with build_slot(tmp_path, inputs) as build:
        assert build.cached
        assert (build.result / "extension.so").read_bytes() == b"compiled"
    changed = {**inputs, "dependencies": {"zlib": "two"}}
    with build_slot(tmp_path, changed) as build:
        assert not build.cached
        (build.result / "extension.so").write_bytes(b"rebuilt")


@pytest.mark.parametrize("damage", ["changed", "extra", "missing", "symlink", "mode"])
def test_corrupt_results_fail_before_reuse(tmp_path, damage):
    inputs = {"recipe": "one"}
    with build_slot(tmp_path, inputs) as build:
        (build.result / "library.so").write_bytes(b"valid")
    result = tmp_path / "results" / identity(inputs)
    library = result / "library.so"
    if damage == "changed":
        library.write_bytes(b"changed")
    elif damage == "extra":
        (result / "undeclared").write_bytes(b"extra")
    elif damage == "missing":
        library.unlink()
    elif damage == "mode":
        library.chmod(0o755)
    else:
        library.unlink()
        library.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError), build_slot(tmp_path, inputs):
        pytest.fail("corrupt result reached the builder")


def test_failed_build_keeps_logs_and_never_publishes(tmp_path):
    inputs = {"recipe": "failure"}
    with pytest.raises(RuntimeError), build_slot(tmp_path, inputs) as build:
        (build.work / "build.log").write_text("compiler failed")
        (build.result / "partial.so").write_bytes(b"partial")
        raise RuntimeError("build failed")
    assert (build.work / "build.log").read_text() == "compiler failed"
    assert not (tmp_path / "results" / identity(inputs)).exists()
    with build_slot(tmp_path, inputs) as retry:
        assert not retry.cached
        assert not (retry.result / "partial.so").exists()
        (retry.result / "complete.so").write_bytes(b"complete")


def test_offline_fetch_verifies_bytes(tmp_path):
    data = b"pinned release"
    digest = hashlib.sha256(data).hexdigest()
    source = {"url": "https://example.invalid/release.tar.gz", "sha256": digest}
    with pytest.raises(ValueError):
        fetch(source, tmp_path, offline=True)
    path = tmp_path / digest / "release.tar.gz"
    path.parent.mkdir()
    path.write_bytes(data)
    assert fetch(source, tmp_path, offline=True) == path
    path.write_bytes(b"corruption")
    with pytest.raises(ValueError):
        fetch(source, tmp_path, offline=True)


def test_source_extract_preserves_executable_script(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        item = tarfile.TarInfo("release/configure")
        item.mode = 0o755
        item.size = 4
        output.addfile(item, io.BytesIO(b"echo"))
    source = extract(archive, tmp_path / "source", subdirectory="release")
    assert (source / "configure").read_bytes() == b"echo"
    assert (source / "configure").stat().st_mode & 0o111


@pytest.mark.parametrize("name,link", [("../outside", False), ("release/link", True)])
def test_source_rejects_escape_and_links(tmp_path, name, link):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as output:
        item = tarfile.TarInfo(name)
        if link:
            item.type = tarfile.SYMTYPE
            item.linkname = "../../outside"
        output.addfile(item)
    with pytest.raises(ValueError):
        extract(archive, tmp_path / "source", subdirectory="release")
    assert not (tmp_path / "outside").exists()
