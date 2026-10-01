"""Check Doom package construction locally; external gameplay is an opt-in probe."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import shellsim

from examples.build_doom_package import LIMITS, build_package, doom_sources


@pytest.fixture
def sample_inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "doomgeneric"
    source.mkdir()
    (source / "Makefile.soso").write_text("SRC_DOOM = engine.o doomgeneric_soso.o\n")
    (source / "engine.c").write_text("int main(void) { return 0; }\n")
    (source / "i_system.c").write_text("!defined(__DJGPP__)\n#elif defined(__DJGPP__)\n")
    wad = tmp_path / "freedoom1.wad"
    wad.write_bytes(b"IWAD\0\0\0\0")
    return source, wad


def test_builds_one_guest_compile_and_launch_entrypoint(sample_inputs: tuple[Path, Path], tmp_path: Path) -> None:
    source, wad = sample_inputs
    package = build_package(source, wad)
    path = tmp_path / "doom.shl"
    path.write_bytes(package.to_bytes())
    loaded = shellsim.Package.from_path(path, expected_sha256=package.sha256)

    assert loaded.spec.entrypoint == ("sh", "/work/run.sh")
    assert loaded.spec.requested_clock == "real_time"
    container = loaded.instantiate(tools={}, limits=LIMITS, clock="virtual")
    script = container.read_file("/work/run.sh").decode()
    assert script.index("tar -xzf") < script.index("cc -DSHELLSIM_WASI") < script.index("/work/doom/doom.wasm -iwad")
    assert "cc -DSHELLSIM_WASI -c /work/doom/engine.c" in script
    assert "doomgeneric_soso.c" not in script
    assert container.read_file("/work/doom/freedoom1.wad") == wad.read_bytes()
    assert b"defined(SHELLSIM_WASI)" in container.read_file("/work/doom/i_system.c")
    assert container.read_file("/work/doom/engine.c") == (source / "engine.c").read_bytes()


def test_rejects_invalid_inputs(sample_inputs: tuple[Path, Path]) -> None:
    source, wad = sample_inputs
    wad.write_bytes(b"PWAD")
    with pytest.raises(ValueError, match="IWAD"):
        build_package(source, wad)
    wad.write_bytes(b"IWAD")
    (source / "Makefile.soso").write_text("SRC_DOOM = ../engine.o\n")
    with pytest.raises(ValueError, match="invalid object name"):
        doom_sources(source)


@pytest.mark.skipif(
    not (os.environ.get("SHELLSIM_DOOM_SOURCE") and os.environ.get("SHELLSIM_DOOM_WAD")),
    reason="requires separately obtained Doomgeneric source and Freedoom IWAD",
)
def test_external_doom_package_yields_frames_and_accepts_input() -> None:
    package = build_package(Path(os.environ["SHELLSIM_DOOM_SOURCE"]), Path(os.environ["SHELLSIM_DOOM_WAD"]))
    container = shellsim.Package.from_bytes(package.to_bytes()).instantiate(tools={}, limits=LIMITS, clock="virtual")
    with container.start_entrypoint() as action:
        first = None
        for _ in range(20_000):
            event = action.poll()
            if event.frame is not None:
                first = event.frame
                break
            if event.state == "complete":
                pytest.fail(f"Doom exited before its first frame: {event.returncode}")
        assert first is not None
        assert (first.width, first.height) == (640, 400)
        action.inject_key(27, True)
        changed = False
        for _ in range(20_000):
            event = action.poll()
            if event.frame is not None and event.frame.pixels != first.pixels:
                changed = True
                break
            if event.state == "complete":
                pytest.fail(f"Doom exited before handling input: {event.returncode}")
        assert changed
        action.stop()
