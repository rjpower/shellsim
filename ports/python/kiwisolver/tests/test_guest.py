"""Exercise the independently installed upstream Kiwi extension in guest CPython."""

from pathlib import Path


def test_public_kiwisolver_install_and_constraint_exceptions(guest_factory):
    guest = guest_factory(
        bundle_env="SHELLSIM_KIWISOLVER_BUNDLE",
        universe_env="SHELLSIM_KIWISOLVER_UNIVERSE",
        requirements=["kiwisolver==1.5.1"],
    )
    result = guest.run_script(Path(__file__).parent / "probes/operations.py")
    assert result.returncode == 0, result.stderr
    assert b"Kiwi constraints and C++ exception translation passed" in result.stdout
    guest.assert_interpreter_unchanged()
