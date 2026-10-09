"""Run the real upstream libffi C backend through independent pthread Stores."""

import os
from pathlib import Path

import pytest


def test_real_threaded_libffi_callback_and_join():
    fixture = os.environ.get("SHELLSIM_THREADED_LIBFFI_FIXTURE")
    if not fixture:
        pytest.skip("set the pinned threaded libffi C fixture")
    from shellsim import Environment

    environment = Environment(cpu=10_000_000_000, memory=512 * 1024**2)
    environment.write_file("/usr/bin/libffi-threads.wasm", Path(fixture).read_bytes(), mode=0o755)
    result = environment.run("/usr/bin/libffi-threads.wasm")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"threaded libffi: scalar callback, nested call, TLS, errno, join passed\n"
    assert result.stderr == b""
