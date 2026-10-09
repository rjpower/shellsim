"""Exercise the emitted module contract with the pinned compiler artifact."""

import os
import subprocess
from pathlib import Path

import pytest

from ports._support.wasm_metadata import number, string


@pytest.fixture
def tools():
    sdk = os.environ.get("SHELLSIM_WASI_SDK34")
    linker = os.environ.get("SHELLSIM_LLVM_LINKER")
    if sdk is None or linker is None:
        pytest.skip("set SDK34 and LLVM_LINKER for actual compiler output checks")
    return Path(sdk) / "bin/clang", Path(linker) / "bin/wasm-ld"


@pytest.fixture
def object_file(tools, tmp_path):
    compiler, _ = tools
    source = tmp_path / "address.c"
    source.write_text(
        "int state = 3; int *address(void) { return &state; } int (*volatile callback)(void); int call(void) { return callback(); }\n"
    )
    output = tmp_path / "address.o"
    subprocess.run(
        [str(compiler), "--target=wasm32-wasip1-threads", "-pthread", "-fPIC", "-c", str(source), "-o", str(output)],
        check=True,
    )
    return output


def sections(path):
    data = path.read_bytes()
    assert data[:8] == b"\0asm\x01\0\0\0"
    offset = 8
    values = []
    while offset < len(data):
        kind = data[offset]
        size, start = number(data, offset + 1)
        offset = start + size
        assert offset <= len(data)
        values.append((kind, data[start:offset]))
    return values


def link(tools, object_file, output, extra=()):
    _, linker = tools
    return subprocess.run(
        [
            str(linker),
            "--shared",
            "--shared-memory",
            "--serial-memory-init",
            "--import-memory",
            "--export-all",
            *extra,
            str(object_file),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
    )


def test_default_library_emits_its_generated_start_section(tools, object_file, tmp_path):
    output = tmp_path / "default.so"
    result = link(tools, object_file, output)
    assert result.returncode == 0, result.stderr
    assert any(kind == 8 for kind, _ in sections(output))


def test_deferred_library_declares_protocol_and_exports_relocation_entry(tools, object_file, tmp_path):
    output = tmp_path / "deferred.so"
    result = link(tools, object_file, output, ["--defer-shared-init"])
    assert result.returncode == 0, result.stderr
    emitted = sections(output)
    assert not any(kind == 8 for kind, _ in emitted)
    exports = set()
    protocol = None
    for kind, payload in emitted:
        if kind == 7:
            count, offset = number(payload, 0)
            for _ in range(count):
                name, offset = string(payload, offset)
                offset += 1
                _, offset = number(payload, offset)
                exports.add(name)
        if kind == 0:
            name, offset = string(payload, 0)
            if name != "dylink.0":
                continue
            while offset < len(payload):
                subsection = payload[offset]
                size, start = number(payload, offset + 1)
                offset = start + size
                if subsection == 128:
                    vendor, cursor = string(payload, start)
                    version, cursor = number(payload, cursor)
                    assert cursor == offset
                    protocol = (vendor, version)
    assert protocol == ("shellsim.deferred-init", 1)
    assert "__wasm_init_memory" in exports


def test_deferred_memory_initializer_requires_shared_library_contract(tools, object_file, tmp_path):
    _, linker = tools
    result = subprocess.run(
        [str(linker), "--defer-shared-init", "--no-entry", str(object_file), "-o", str(tmp_path / "main.wasm")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "main.wasm").exists()
