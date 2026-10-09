"""Exercise upstream ctypes through the public process CPython runtime."""

from pathlib import Path

from ports._support.testing import stage_script


def test_shared_libffi_ctypes_zlib_and_child_python_in_guest(guest_factory) -> None:
    guest = guest_factory(bundle_env="SHELLSIM_CTYPES_CPYTHON_BUNDLE", cpu=2_000_000_000)
    path = stage_script(guest.environment, Path(__file__).parent / "probes/ctypes_probe.py")[0]
    result = guest.environment.run(f"python {path}")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"public ctypes zlib subprocess: ok\n"
    assert result.stderr == b""
    guest.assert_interpreter_unchanged()
