"""Numerical acceptance through the public independent-wheel installer."""

from pathlib import Path


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
