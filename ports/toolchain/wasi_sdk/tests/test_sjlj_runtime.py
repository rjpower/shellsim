"""Check canonical SDK SJLJ ownership in the actual package-free runtime."""

import os
from pathlib import Path

import pytest

from ports._support.wasm_metadata import number, string


def exports(path):
    data = path.read_bytes()
    position = 8
    result = {}
    while position < len(data):
        section = data[position]
        size, position = number(data, position + 1)
        end = position + size
        if section == 7:
            count, position = number(data, position)
            for _ in range(count):
                name, position = string(data, position)
                kind = data[position]
                _, position = number(data, position + 1)
                result[name] = kind
        position = end
    return result


def test_package_free_runtime_exports_longjmp_tag_and_functions():
    bundle = os.environ.get("SHELLSIM_DYNAMIC_V2_ARTIFACTS")
    if bundle is None:
        pytest.skip("set the rebuilt canonical runtime bundle")
    symbols = exports(Path(bundle) / "rootfs/usr/bin/python3.wasm")
    assert [symbols[name] for name in ("__wasm_setjmp", "__wasm_setjmp_test", "__wasm_longjmp", "__c_longjmp")] == [
        0,
        0,
        0,
        4,
    ]
