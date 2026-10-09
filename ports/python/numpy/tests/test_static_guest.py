"""Exercise the actual static NumPy port when its verified bundle is supplied.

These tests run upstream NumPy inside shellsim's WASI guest. They cover numerical
behavior and deliberate build frontiers rather than comparing random streams.
"""

from pathlib import Path

import pytest

from ports._support.testing import stage_script


@pytest.fixture
def numpy_runtime(guest_factory):
    guest = guest_factory(bundle_env="SHELLSIM_NUMPY_BUNDLE", cpu=5_000_000_000, memory=512 * 1024**2)
    return guest.runtime, guest.environment


def test_numpy_array_reduction_matrix_and_linalg(numpy_runtime):
    runtime, environment = numpy_runtime
    result = runtime.run(
        environment,
        stage_script(environment, Path(__file__).parent / "probes/numpy_array_reduction_matrix_and_linalg.py"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"numpy numerical operations passed\n"


def test_numpy_unique(numpy_runtime):
    runtime, environment = numpy_runtime
    result = runtime.run(
        environment,
        stage_script(environment, Path(__file__).parent / "probes/numpy_unique.py"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"unique sort fallback passed\n"


def test_numpy_random_legacy_and_modern_abis(numpy_runtime):
    runtime, environment = numpy_runtime
    result = runtime.run(
        environment,
        stage_script(environment, Path(__file__).parent / "probes/numpy_random_legacy_and_modern_abis.py"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"random integer ABIs passed\n"


def test_numpy_fft_round_trip_and_invalid_length(numpy_runtime):
    runtime, environment = numpy_runtime
    result = runtime.run(
        environment,
        stage_script(environment, Path(__file__).parent / "probes/numpy_fft_round_trip_and_invalid_length.py"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"FFT round trip and invalid length passed\n"


def test_numpy_floating_point_policy_frontier(numpy_runtime):
    runtime, environment = numpy_runtime
    port = next(port for port in runtime.manifest["native_ports"] if port["name"] == "numpy")
    assert port["features"]["floating_point_exceptions"] is False
    result = runtime.run(
        environment,
        stage_script(environment, Path(__file__).parent / "probes/numpy_floating_point_policy_frontier.py"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"floating-point policy gap reproduced\n"
