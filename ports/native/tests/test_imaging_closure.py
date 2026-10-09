"""Exercise real zlib and Pillow through a source-built WASI image and the VFS."""

from pathlib import Path

import pytest


@pytest.fixture
def native_runtime(guest_factory):
    guest = guest_factory(
        bundle_env="SHELLSIM_NATIVE_BUNDLE", cpu=2_000_000_000, memory=256 * 1024**2, disk=64 * 1024**2
    )
    guest.environment.write_file(
        "/font.ttf", (Path(__file__).parents[3] / "tests/fixtures/fonts/DejaVuSans.ttf").read_bytes()
    )
    return guest.runtime, guest.environment


def test_manifest_has_one_native_zlib_provider_for_both_consumers(native_runtime):
    runtime, _ = native_runtime
    libraries = runtime.manifest["native_libraries"]
    assert set(libraries) == {"native/zlib", "native/libjpeg-turbo", "native/freetype"}
    identity = libraries["native/zlib"]["artifact_sha256"]
    for consumer in ("cpython.zlib", "PIL._imaging"):
        assert runtime.manifest["link_consumers"][consumer]["dependency_artifacts"]["native/zlib"] == identity
    assert libraries["native/freetype"]["inputs"]["dependency_artifacts"] == {"native/zlib": identity}
    assert "zlib" in runtime.builtin_modules
    assert "PIL._imaging" in runtime.builtin_modules
    assert "PIL._imagingft" in runtime.builtin_modules
    assert all(port["name"] != "zlib" for port in runtime.manifest["native_ports"])
