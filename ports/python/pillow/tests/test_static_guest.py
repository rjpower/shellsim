"""Exercise real zlib and Pillow through a source-built WASI image and the VFS."""

from pathlib import Path

import pytest

from ports._support.testing import stage_script


@pytest.fixture
def native_runtime(guest_factory):
    guest = guest_factory(
        bundle_env="SHELLSIM_NATIVE_BUNDLE", cpu=2_000_000_000, memory=256 * 1024**2, disk=64 * 1024**2
    )
    guest.environment.write_file(
        "/font.ttf", (Path(__file__).parents[4] / "tests/fixtures/fonts/DejaVuSans.ttf").read_bytes()
    )
    return guest.runtime, guest.environment


def test_compression_and_png_round_trip_in_vfs(native_runtime):
    runtime, environment = native_runtime
    result = runtime.run(
        environment,
        stage_script(environment, Path(__file__).parent / "probes/compression_and_png_round_trip_in_vfs.py"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"shared zlib compression and PNG passed\n"
    assert environment.read_file("/tmp/shared-zlib.png").startswith(b"\x89PNG\r\n\x1a\n")
    assert result.usage.cpu_used > 0
    assert result.usage.memory_peak > 0


def test_invalid_stream_and_disabled_codecs(native_runtime):
    runtime, environment = native_runtime
    result = runtime.run(
        environment,
        stage_script(environment, Path(__file__).parent / "probes/invalid_stream_and_disabled_codecs.py"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"invalid data and optional codec frontiers passed\n"


def test_jpeg_recovery_and_freetype_rasterization_in_vfs(native_runtime):
    runtime, environment = native_runtime
    result = runtime.run(
        environment,
        stage_script(environment, Path(__file__).parent / "probes/jpeg_recovery_and_freetype_rasterization_in_vfs.py"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"JPEG recovery and FreeType rasterization passed\n"
    assert environment.read_file("/tmp/roundtrip.jpg").startswith(b"\xff\xd8")


def test_native_compression_obeys_guest_cpu_budget(native_runtime):
    runtime, environment = native_runtime
    result = runtime.run(
        environment,
        stage_script(environment, Path(__file__).parent / "probes/native_compression_obeys_guest_cpu_budget.py"),
    )
    assert result.stdout == b"compression loop entered\n"
    assert result.returncode != 0
    assert result.stop_reason == "cpu_exhausted"
