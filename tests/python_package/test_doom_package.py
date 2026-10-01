"""Check Doom package construction locally; external gameplay is an opt-in probe."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
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
    (tmp_path / "LICENSE").write_text("Doom GPL notice\n")
    release = tmp_path / "freedoom-0.13.0.zip"
    _write_release(release, b"IWAD\0\0\0\0")
    return source, release


def _write_release(path: Path, wad: bytes) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("freedoom-0.13.0/freedoom1.wad", wad)
        for name in ("COPYING.txt", "CREDITS.txt", "CREDITS-MUSIC.txt"):
            archive.writestr(f"freedoom-0.13.0/{name}", f"Freedoom {name}\n")


def test_builds_one_guest_compile_and_launch_entrypoint(sample_inputs: tuple[Path, Path], tmp_path: Path) -> None:
    source, release = sample_inputs
    wad = b"IWAD\0\0\0\0"
    package = build_package(
        source,
        release,
        doom_revision="d" * 40,
        expected_wad_sha256=hashlib.sha256(wad).hexdigest(),
    )
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
    assert container.read_file("/work/doom/freedoom1.wad") == wad
    assert b"defined(SHELLSIM_WASI)" in container.read_file("/work/doom/i_system.c")
    assert container.read_file("/work/doom/engine.c") == (source / "engine.c").read_bytes()
    assert container.read_file("/work/LICENSES/doomgeneric-GPL-2.0.txt") == b"Doom GPL notice\n"
    assert container.read_file("/work/LICENSES/freedoom-COPYING.txt") == b"Freedoom COPYING.txt\n"
    assert container.read_file("/work/LICENSES/freedoom-CREDITS.txt") == b"Freedoom CREDITS.txt\n"
    assert container.read_file("/work/LICENSES/tinycc-LGPL-2.1.txt")
    assert container.read_file("/work/LICENSES/tinycc-shellsim-libc-LICENSE")
    assert container.read_file("/work/LICENSES/wasi-libc-LICENSE")
    assert container.read_file("/work/LICENSES/shellsim-Apache-2.0.txt")
    provenance = json.loads(container.read_file("/work/PROVENANCE.json"))
    assert provenance["doomgeneric"]["revision"] == "d" * 40
    assert provenance["freedoom"]["wad_sha256"] == hashlib.sha256(wad).hexdigest()
    assert provenance["freedoom"]["release"] is None
    assert provenance["tinycc"]["archive_sha256"]
    assert provenance["wasi_libc"]["archive_sha256"]
    assert provenance["shellsim_adapter"]["license"] == "Apache-2.0 OR GPL-2.0-or-later"


def test_rejects_invalid_inputs(sample_inputs: tuple[Path, Path]) -> None:
    source, release = sample_inputs
    _write_release(release, b"PWAD")
    with pytest.raises(ValueError, match="IWAD"):
        build_package(source, release)
    _write_release(release, b"IWAD")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_package(source, release, expected_wad_sha256="0" * 64)
    (source / "Makefile.soso").write_text("SRC_DOOM = ../engine.o\n")
    with pytest.raises(ValueError, match="invalid object name"):
        doom_sources(source)


def test_rejects_missing_or_oversized_notices(sample_inputs: tuple[Path, Path]) -> None:
    source, release = sample_inputs
    (source.parent / "LICENSE").unlink()
    with pytest.raises(ValueError, match="missing its parent LICENSE"):
        build_package(source, release)
    (source.parent / "LICENSE").write_text("Doom GPL notice\n")
    with zipfile.ZipFile(release, "w") as archive:
        archive.writestr("freedoom-0.13.0/freedoom1.wad", b"IWAD")
        archive.writestr("freedoom-0.13.0/COPYING.txt", b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="oversized Freedoom release member"):
        build_package(source, release)


@pytest.mark.skipif(
    not (os.environ.get("SHELLSIM_DOOM_SOURCE") and os.environ.get("SHELLSIM_FREEDOOM_ARCHIVE")),
    reason="requires separately obtained Doomgeneric source and Freedoom release",
)
def test_external_doom_package_yields_frames_and_accepts_input() -> None:
    package = build_package(Path(os.environ["SHELLSIM_DOOM_SOURCE"]), Path(os.environ["SHELLSIM_FREEDOOM_ARCHIVE"]))
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
