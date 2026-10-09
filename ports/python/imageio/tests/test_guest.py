"""Run the reviewed imageio NumPy/Pillow graph inside guest CPython."""

from pathlib import Path


def test_png_array_roundtrip_and_invalid_input(guest_factory):
    guest = guest_factory(
        bundle_env="SHELLSIM_IMAGEIO_BUNDLE", universe_env="SHELLSIM_IMAGEIO_UNIVERSE", requirements=["imageio==2.37.0"]
    )
    result = guest.run_script(Path(__file__).parent / "probes/operations.py")
    assert result.returncode == 0, result.stderr
    assert guest.environment.read_file("/tmp/graph.png").startswith(b"\x89PNG\r\n\x1a\n")
    guest.assert_interpreter_unchanged()


def test_static_png_graph_with_verified_adapted_wheel(guest_factory):
    guest = guest_factory(
        bundle_env="SHELLSIM_NATIVE_BUNDLE", wheel_envs=["SHELLSIM_IMAGEIO_WHEEL"], memory=256 * 1024**2
    )
    result = guest.run_script(Path(__file__).parent / "probes/operations.py")
    assert result.returncode == 0, result.stderr
    assert guest.environment.read_file("/tmp/graph.png").startswith(b"\x89PNG\r\n\x1a\n")
    guest.assert_interpreter_unchanged()
