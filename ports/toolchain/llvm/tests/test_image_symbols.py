"""Link real shared objects without borrowing another image's absolute symbols."""

import os
import subprocess
from pathlib import Path

import pytest

from ports._support.wasm_metadata import number, string


def exported_global(data, expected_name):
    """Read a fixture's exported constant i32 address without executing host Wasm."""
    globals_ = []
    exports = {}
    offset = 8
    while offset < len(data):
        kind = data[offset]
        length, cursor = number(data, offset + 1)
        offset = cursor + length
        if kind == 2:
            count, cursor = number(data, cursor)
            for _ in range(count):
                _, cursor = string(data, cursor)
                _, cursor = string(data, cursor)
                category = data[cursor]
                cursor += 1
                if category == 0:
                    _, cursor = number(data, cursor)
                elif category in {1, 2}:
                    if category == 1:
                        cursor += 1
                    flags, cursor = number(data, cursor)
                    _, cursor = number(data, cursor)
                    if flags & 1:
                        _, cursor = number(data, cursor)
                elif category == 3:
                    cursor += 2
                    globals_.append(None)
                else:
                    raise AssertionError("unexpected fixture import")
        elif kind == 6:
            count, cursor = number(data, cursor)
            for _ in range(count):
                assert data[cursor] == 0x7F and data[cursor + 1] in {0, 1} and data[cursor + 2] == 0x41
                value, cursor = number(data, cursor + 3)
                assert data[cursor] == 0x0B
                cursor += 1
                globals_.append(value)
        elif kind == 7:
            count, cursor = number(data, cursor)
            for _ in range(count):
                name, cursor = string(data, cursor)
                category = data[cursor]
                index, cursor = number(data, cursor + 1)
                if category == 3:
                    exports[name] = index
    return globals_[exports[expected_name]]


@pytest.mark.parametrize("local_definition", [False, True])
def test_main_owns_optional_symbols_and_preserves_object_definitions(tmp_path, local_definition):
    prefix = os.environ.get("SHELLSIM_THREADED_CLANG")
    if prefix is None:
        pytest.skip("requires the sealed full threaded compiler product")
    compiler, linker = Path(prefix) / "bin/clang", Path(prefix) / "bin/wasm-ld"
    side = tmp_path / "side.s"
    side.write_text(".globl side_value\nside_value:\n.functype side_value () -> (i32)\ni32.const 3\nend_function\n")
    main = tmp_path / "main.s"
    main.write_text(
        ".globl page\npage:\n.functype page () -> (i32)\ni32.const __wasm_first_page_end\nend_function\n"
        + (
            '.globl __wasm_first_page_end\n.type __wasm_first_page_end,@object\n.section .data.page,"",@\n'
            "__wasm_first_page_end:\n.int32 7\n.size __wasm_first_page_end, 4\n"
            if local_definition
            else ""
        )
    )
    for name in ("side", "main"):
        subprocess.run(
            [
                str(compiler),
                "--target=wasm32-unknown-unknown",
                "-c",
                str(tmp_path / (name + ".s")),
                "-o",
                str(tmp_path / (name + ".o")),
            ],
            check=True,
        )
    shared = tmp_path / "side.so"
    subprocess.run([str(linker), "--shared", "--export-all", str(tmp_path / "side.o"), "-o", str(shared)], check=True)
    assert exported_global(shared.read_bytes(), "__wasm_first_page_end") == 65536
    output = tmp_path / "main.wasm"
    subprocess.run(
        [
            str(linker),
            "--no-entry",
            "--global-base=131072",
            "--export-all",
            str(tmp_path / "main.o"),
            "-Bdynamic",
            str(shared),
            "-o",
            str(output),
        ],
        check=True,
    )
    address = exported_global(output.read_bytes(), "__wasm_first_page_end")
    if local_definition:
        local_output = tmp_path / "local.wasm"
        subprocess.run(
            [
                str(linker),
                "--no-entry",
                "--global-base=131072",
                "--export-all",
                str(tmp_path / "main.o"),
                "-o",
                str(local_output),
            ],
            check=True,
        )
        assert address == exported_global(local_output.read_bytes(), "__wasm_first_page_end") == 131072
    else:
        assert address == 65536
