"""Install genuine Pillow and shared codec/font providers into fixed CPython."""

from pathlib import Path


def test_public_pillow_install_and_imaging_graph(guest_factory):
    guest = guest_factory(
        bundle_env="SHELLSIM_PILLOW_BUNDLE", universe_env="SHELLSIM_PILLOW_UNIVERSE", requirements=["pillow==12.3.0"]
    )
    guest.environment.write_file(
        "/font.ttf", (Path(__file__).parents[4] / "tests/fixtures/fonts/DejaVuSans.ttf").read_bytes()
    )
    result = guest.run_script(Path(__file__).parent / "probes/dynamic_operations.py")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"Pillow PNG/JPEG recovery, font, math and morphology passed\n"
    assert guest.environment.read_file("/tmp/roundtrip.png").startswith(b"\x89PNG\r\n\x1a\n")
    assert guest.environment.read_file("/tmp/roundtrip.jpg").startswith(b"\xff\xd8")
    guest.assert_interpreter_unchanged()
