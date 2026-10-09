"""Numerical acceptance through the public independent-wheel installer."""

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest


def test_numpy_dynamic_array_linalg_fft_and_rng(guest_factory):
    guest = guest_factory(
        bundle_env="SHELLSIM_DYNAMIC_V2_ARTIFACTS",
        universe_env="SHELLSIM_NUMPY_UNIVERSE",
        requirements=["numpy==2.3.5"],
    )
    result = guest.run_script(Path(__file__).parent / "probes/operations.py")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"independent NumPy array/linalg/FFT/RNG passed\n"
    guest.assert_interpreter_unchanged()


def test_numpy_wheel_contains_verified_independent_extensions():
    wheel_path = os.environ.get("SHELLSIM_NUMPY_DYNAMIC_WHEEL")
    if wheel_path is None:
        pytest.skip("set the NumPy dynamic wheel artifact path")
    with zipfile.ZipFile(wheel_path) as wheel:
        manifest = json.loads(wheel.read("numpy-2.3.5.dist-info/shellsim-native.json"))
        assert manifest["artifacts"]
        for artifact in manifest["artifacts"]:
            assert hashlib.sha256(wheel.read(artifact["path"])).hexdigest() == artifact["sha256"]
