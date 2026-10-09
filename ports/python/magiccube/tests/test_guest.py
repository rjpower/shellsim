"""Run cube moves and NumPy object arrays through the public package installer."""

from pathlib import Path


def test_cube_moves_restore_solved_state(guest_factory):
    guest = guest_factory(
        bundle_env="SHELLSIM_MAGICCUBE_BUNDLE",
        universe_env="SHELLSIM_MAGICCUBE_UNIVERSE",
        requirements=["magiccube==0.3.0"],
    )
    result = guest.run_script(Path(__file__).parent / "probes/operations.py")
    assert result.returncode == 0, result.stderr
    assert b'"cube_restored": true' in result.stdout
    guest.assert_interpreter_unchanged()
