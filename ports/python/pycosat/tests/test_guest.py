"""Run the pinned static SAT provider inside the declared CPython bundle."""

from pathlib import Path


def test_sat_and_unsatisfiable_formula(guest_factory):
    guest = guest_factory(bundle_env="SHELLSIM_PYCOSAT_BUNDLE")
    result = guest.run_script(Path(__file__).parent / "probes/operations.py")
    assert result.returncode == 0, result.stderr
