"""Check executable TLS classification without inferring symbol names."""

import os
import subprocess
from pathlib import Path

import pytest

from ports._support.wasm_metadata import number, string


@pytest.fixture
def tools():
    if "SHELLSIM_WASI_SDK34" not in os.environ or "SHELLSIM_THREADED_LLVM" not in os.environ:
        pytest.skip("requires the normal threaded compiler and pinned SDK")
    sdk = Path(os.environ["SHELLSIM_WASI_SDK34"])
    linker = Path(os.environ["SHELLSIM_THREADED_LLVM"]) / "bin/wasm-ld"
    return sdk / "bin/clang", linker


def test_multiple_tls_exports_are_classified_and_ordinary_data_is_not(tools, tmp_path):
    compiler, linker = tools
    source = tmp_path / "storage.c"
    source.write_text(
        "__thread int first=1; __thread int second=2; int ordinary=3; int value(void){return first+second+ordinary;}\n"
    )
    obj = tmp_path / "storage.o"
    subprocess.run(
        [str(compiler), "--target=wasm32-wasip1-threads", "-pthread", "-c", str(source), "-o", str(obj)],
        check=True,
    )
    output = tmp_path / "main.wasm"
    subprocess.run(
        [
            str(linker),
            "--shared-memory",
            "--serial-memory-init",
            "--import-memory",
            "--no-entry",
            "--export-all",
            "--emit-main-tls-info",
            str(obj),
            "-o",
            str(output),
        ],
        check=True,
    )
    data = output.read_bytes()
    offset = 8
    tls = set()
    declared = False
    while offset < len(data):
        kind = data[offset]
        length, start = number(data, offset + 1)
        payload = data[start : start + length]
        offset = start + length
        if kind != 0:
            continue
        name, cursor = string(payload, 0)
        if name != "dylink.0":
            continue
        while cursor < len(payload):
            subsection = payload[cursor]
            size, begin = number(payload, cursor + 1)
            cursor = begin + size
            if subsection == 129:
                vendor, pos = string(payload, begin)
                version, pos = number(payload, pos)
                assert (vendor, version, pos) == ("shellsim.main-tls", 1, cursor)
                declared = True
            if subsection == 3:
                count, pos = number(payload, begin)
                for _ in range(count):
                    name, pos = string(payload, pos)
                    flags, pos = number(payload, pos)
                    if flags & 0x100:
                        tls.add(name)
    assert declared
    assert tls == {"first", "second"}


@pytest.mark.parametrize("flags", [[], ["--shared", "--shared-memory", "--serial-memory-init"]])
def test_main_tls_contract_rejects_unshared_main_and_side_library(tools, tmp_path, flags):
    compiler, linker = tools
    source = tmp_path / "storage.c"
    source.write_text("int value(void) { return 1; }\n")
    obj = tmp_path / "storage.o"
    subprocess.run(
        [str(compiler), "--target=wasm32-wasip1-threads", "-pthread", "-fPIC", "-c", str(source), "-o", str(obj)],
        check=True,
    )
    output = tmp_path / "invalid.wasm"
    result = subprocess.run(
        [str(linker), *flags, "--no-entry", "--emit-main-tls-info", str(obj), "-o", str(output)], capture_output=True
    )
    assert result.returncode != 0
    assert not output.exists()
