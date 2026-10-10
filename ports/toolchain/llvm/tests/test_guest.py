"""Check guest build boundaries without launching target programs on the host."""

from pathlib import Path

import pytest

from ports._support.graph import plan
from ports.toolchain.llvm.guest import _inventory, _refresh_snapshot, _snapshot, _snapshot_tools

PORT = Path(__file__).resolve().parents[1]


def test_guest_compiler_graph_distinguishes_build_and_install_dependencies():
    graph = plan(PORT.parents[1], ["toolchain/llvm/guest-recipe.json"])
    clang = graph.ports[-1]
    assert clang.role == "guest-tool"
    assert {(edge.port, edge.kind) for edge in clang.dependencies} == {
        ("toolchain/llvm", "build"),
        ("toolchain/wasi_threads", "platform"),
        ("native/shellsim-posix", "target"),
        ("toolchain/wasi-development", "runtime"),
    }


def test_dependency_snapshot_rejects_modified_retained_bytes(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "retained"
    source.mkdir()
    (source / "header.h").write_text("original\n")
    inventory = _inventory(source)
    _snapshot(source, destination, inventory)
    _snapshot(source, destination, inventory)
    (destination / "header.h").write_text("changed\n")
    with pytest.raises(ValueError):
        _snapshot(source, destination, inventory)


def test_posix_snapshot_compatibility_uses_compilation_bytes(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "retained"
    (source / "include").mkdir(parents=True)
    (source / "include/process.h").write_text("original\n")
    (source / "artifact.json").write_text("first envelope\n")
    directories = ("include", "lib")
    inventory = _inventory(source, directories)
    _snapshot(source, destination, inventory, directories)
    (destination / "artifact.json").write_text("new envelope\n")
    _snapshot(source, destination, inventory, directories)
    (destination / "include/process.h").write_text("changed\n")
    with pytest.raises(ValueError):
        _snapshot(source, destination, inventory, directories)


def test_verified_library_update_preserves_snapshot_headers(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "retained"
    (source / "include").mkdir(parents=True)
    (source / "lib").mkdir()
    (source / "include/header.h").write_text("header\n")
    (source / "lib/libc.a").write_bytes(b"old archive")
    previous = _inventory(source)
    _snapshot(source, destination, previous)
    timestamp = (destination / "include/header.h").stat().st_mtime_ns
    (source / "lib/libc.a").write_bytes(b"new archive")
    _refresh_snapshot(source, destination, _inventory(source), previous)
    assert (destination / "lib/libc.a").read_bytes() == b"new archive"
    assert (destination / "include/header.h").stat().st_mtime_ns == timestamp


def test_linker_update_checks_previous_admitted_bytes(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "retained"
    source.mkdir()
    (source / "wasm-ld").write_bytes(b"old linker")
    previous = _inventory(source)
    _snapshot_tools(source, destination, previous, previous)
    (source / "wasm-ld").write_bytes(b"new linker")
    current = _inventory(source)
    _snapshot_tools(source, destination, current, previous)
    assert (destination / "wasm-ld").read_bytes() == b"new linker"
    (destination / "wasm-ld").write_bytes(b"corrupt linker")
    with pytest.raises(ValueError):
        _snapshot_tools(source, destination, current, previous)
