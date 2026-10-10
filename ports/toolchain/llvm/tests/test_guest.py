"""Check guest build boundaries without launching target programs on the host."""

from pathlib import Path

import pytest

from ports._support.graph import plan
from ports.toolchain.llvm.guest import _inventory, _refresh_snapshot, _snapshot, _snapshot_tools, validate_guest_abi

PORT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("defect", [None, "table-export", "fixed-table", "memory-bound", "tls-protocol", "tls-export"])
def test_stripped_main_abi_rejects_runtime_frontiers(tmp_path, defect):
    def leb(value):
        output = bytearray()
        while value > 127:
            output.append((value & 127) | 128)
            value >>= 7
        return bytes(output) + bytes([value])

    def text(value):
        encoded = value.encode()
        return leb(len(encoded)) + encoded

    def section(kind, payload):
        return bytes([kind]) + leb(len(payload)) + payload

    exports = {
        "_start": 0,
        "wasi_thread_start": 0,
        "__wasm_init_tls": 0,
        "__wasm_apply_global_tls_relocs": 0,
        "__indirect_function_table": 1,
        "memory": 2,
        "__stack_pointer": 3,
        "__tls_base": 3,
        "__tls_size": 3,
        "__tls_align": 3,
        "errno": 3,
    }
    if defect == "table-export":
        del exports["__indirect_function_table"]
    if defect == "tls-export":
        del exports["errno"]
    memory = (
        leb(1) + text("env") + text("memory") + b"\x02\x03" + leb(256) + leb(4097 if defect == "memory-bound" else 4096)
    )
    table = b"\x01\x70" + (b"\x01\x01\x01" if defect == "fixed-table" else b"\x00\x01")
    exported = leb(len(exports)) + b"".join(text(name) + bytes([kind]) + leb(0) for name, kind in exports.items())
    protocol = text("shellsim.main-tls") + leb(1)
    classified = leb(1) + text("errno") + leb(0x100)
    dylink = text("dylink.0") + section(3, classified)
    if defect != "tls-protocol":
        dylink += section(129, protocol)
    path = tmp_path / "main.wasm"
    path.write_bytes(
        b"\0asm\x01\0\0\0" + section(2, memory) + section(4, table) + section(7, exported) + section(0, dylink)
    )
    if defect is None:
        validate_guest_abi(path)
    else:
        with pytest.raises(ValueError):
            validate_guest_abi(path)


def test_guest_compiler_graph_distinguishes_build_and_install_dependencies():
    graph = plan(PORT.parents[1], ["toolchain/llvm:guest"])
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
