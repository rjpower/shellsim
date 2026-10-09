"""Exercise upstream ctypes with independent threaded libffi and C provider."""

import os
from pathlib import Path

import pytest


def test_threaded_ctypes_callbacks_and_store_local_state(guest_factory):
    consumer = os.environ.get("SHELLSIM_THREADED_CTYPES_CONSUMER")
    if not consumer:
        pytest.skip("set the independently built threaded ctypes consumer")
    guest = guest_factory(bundle_env="SHELLSIM_THREADED_CTYPES_BUNDLE", cpu=10_000_000_000)
    guest.environment.write_file("/lib/ctypes_threads_probe.so", Path(consumer).read_bytes())
    result = guest.run_script(Path(__file__).parent / "probes/threaded_ctypes_probe.py")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"threaded ctypes: callback, nested CDLL call, TLS, errno passed\n"
    assert result.stderr == b""
    guest.assert_interpreter_unchanged()
