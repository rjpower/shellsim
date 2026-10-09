"""Reject unsafe or changed compiler/sysroot bytes before provider compilation."""

import hashlib

import pytest
from ports.native.libffi.threaded.toolchain import MAX_FILE, validate_files


def test_changed_toolchain_file_is_rejected(tmp_path):
    target = tmp_path / "header.h"
    target.write_bytes(b"approved header")
    manifest = {"artifacts": {"header.h": hashlib.sha256(target.read_bytes()).hexdigest()}}
    validate_files(tmp_path, manifest)
    target.write_bytes(b"changed header")
    with pytest.raises(ValueError, match="artifact differs"):
        validate_files(tmp_path, manifest)


@pytest.mark.parametrize("name", ["../outside", "/outside", "header/../../outside"])
def test_artifact_path_cannot_escape_prefix(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe"):
        validate_files(tmp_path, {"artifacts": {name: "0" * 64}})


def test_large_file_is_rejected_before_hashing(tmp_path):
    target = tmp_path / "oversized"
    with target.open("wb") as stream:
        stream.truncate(MAX_FILE + 1)
    with pytest.raises(ValueError, match="oversized"):
        validate_files(tmp_path, {"artifacts": {"oversized": "0" * 64}})


def test_symlink_cannot_admit_external_file(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-external")
    outside.write_bytes(b"not inside provider")
    (tmp_path / "header.h").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        validate_files(tmp_path, {"artifacts": {"header.h": hashlib.sha256(outside.read_bytes()).hexdigest()}})


def test_unrecorded_header_cannot_change_include_search(tmp_path):
    from ports.native.libffi.threaded.toolchain import validate_tree

    directory = tmp_path / "sysroot/include"
    directory.mkdir(parents=True)
    target = directory / "stdio.h"
    target.write_bytes(b"approved header")
    artifacts = {"sysroot/include/stdio.h": hashlib.sha256(target.read_bytes()).hexdigest()}
    validate_tree(tmp_path, "sysroot", artifacts)
    (directory / "stddef.h").write_bytes(b"not recorded by producer")
    with pytest.raises(ValueError, match="undeclared input"):
        validate_tree(tmp_path, "sysroot", artifacts)
