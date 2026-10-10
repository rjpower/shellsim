"""Local platform sources must stay pinned and inside the declared ports tree."""

import hashlib

import pytest

from ports._support.local_sources import local_source_files, stage_local_sources
from ports._support.store import identity


def _manifest(path, data, destination="source.c"):
    files = [{"path": path, "destination": destination, "sha256": hashlib.sha256(data).hexdigest()}]
    return {"files": files, "sha256": identity(files)}


def test_local_sources_reject_changed_input_before_staging(tmp_path):
    root = tmp_path / "ports"
    root.mkdir()
    source = root / "source.c"
    source.write_bytes(b"original")
    manifest = _manifest("source.c", b"original")
    source.write_bytes(b"changed")
    destination = tmp_path / "build"
    with pytest.raises(ValueError):
        stage_local_sources(root, manifest, destination)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["../outside.c", "/outside.c", "redirect/source.c"])
def test_local_sources_reject_paths_outside_tree(tmp_path, name):
    root = tmp_path / "ports"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.c").write_bytes(b"secret")
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        local_source_files(root, _manifest(name, b"secret"))


def test_local_sources_do_not_follow_links_inside_tree(tmp_path):
    (tmp_path / "real.c").write_bytes(b"source")
    (tmp_path / "alias.c").symlink_to("real.c")
    with pytest.raises(ValueError):
        local_source_files(tmp_path, _manifest("alias.c", b"source"))
