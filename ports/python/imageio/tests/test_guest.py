"""Install upstream ImageIO through public CPython and exercise its guest graph."""

import json
from pathlib import Path


def test_png_array_roundtrip_and_invalid_input(guest_factory):
    guest = guest_factory(
        bundle_env="SHELLSIM_IMAGEIO_BUNDLE", universe_env="SHELLSIM_IMAGEIO_UNIVERSE", requirements=["imageio==2.37.0"]
    )
    from shellsim import _native

    host_library = Path(_native.__file__).resolve()
    assert host_library.is_file()
    result = guest.run_script(Path(__file__).parent / "probes/operations.py", arguments=(str(host_library),))
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["shape"] == [3, 4, 3]
    assert outcome["pixel_sum"] == 630
    assert outcome["host_library_rejected"] and outcome["invalid_image_rejected"]
    assert guest.environment.read_file("/tmp/graph.png").startswith(b"\x89PNG\r\n\x1a\n")
    guest.assert_interpreter_unchanged()
