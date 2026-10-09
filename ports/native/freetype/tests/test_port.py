"""Check the explicit FreeType source profile without downloading or compiling."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ports.native.freetype.build import pkg_config

ROOT = Path(__file__).resolve().parents[4]


def test_freetype_declares_only_pinned_zlib_dependency():
    recipe = json.loads((ROOT / "ports/native/freetype/recipe.json").read_text())
    assert recipe["target_profile"] == "wasi-cpython-v2"
    assert recipe["sdk"]["version"] == "34.0"
    assert recipe["target_dependencies"] == [{"port": "native/zlib", "version": "1.3.1"}]
    for optional in ("png", "bzip2", "brotli", "harfbuzz", "host_discovery"):
        assert recipe["features"][optional] is False


@pytest.fixture
def upstream_metadata(tmp_path):
    source = os.environ.get("SHELLSIM_FREETYPE_SOURCE")
    if not source:
        pytest.skip("set the pinned FreeType source directory")
    for name in ("include/freetype/freetype.h", "builds/unix/configure.raw", "builds/unix/freetype2.in"):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / name, destination)
    return tmp_path


def test_pkg_config_uses_upstream_libtool_version(upstream_metadata):
    recipe = json.loads((ROOT / "ports/native/freetype/recipe.json").read_text())
    rendered = pkg_config(upstream_metadata, recipe)
    assert "Version: 26.2.20\n" in rendered
    assert "Cflags: -I${includedir}/freetype2\n" in rendered
    assert "Requires.private: zlib\n" in rendered
    assert recipe["version"] == "2.13.3"


@pytest.mark.parametrize("field", ["version", "template", "release"])
def test_pkg_config_rejects_changed_upstream_contract(upstream_metadata, field):
    recipe = json.loads((ROOT / "ports/native/freetype/recipe.json").read_text())
    if field == "version":
        path = upstream_metadata / "builds/unix/configure.raw"
        path.write_text(path.read_text().replace("version_info='26:2:20'", "version_info='unknown'"))
    elif field == "template":
        path = upstream_metadata / "builds/unix/freetype2.in"
        path.write_text(path.read_text() + "Unsupported: %NEW_FIELD%\n")
    else:
        recipe["version"] = "2.14.3"
    with pytest.raises(ValueError):
        pkg_config(upstream_metadata, recipe)


def test_actual_pkg_config_accepts_upstream_version_requirement():
    prefix = os.environ.get("SHELLSIM_FREETYPE_PREFIX")
    zlib = os.environ.get("SHELLSIM_FREETYPE_ZLIB_PREFIX")
    if not prefix or not zlib:
        pytest.skip("set sealed FreeType and zlib artifact prefixes")
    pkgconf = os.environ.get("SHELLSIM_PKG_CONFIG", "pkg-config")
    environment = {"PATH": "/usr/bin:/bin", "PKG_CONFIG_LIBDIR": prefix + "/lib/pkgconfig:" + zlib + "/lib/pkgconfig"}
    assert subprocess.check_output([pkgconf, "--modversion", "freetype2"], env=environment).strip() == b"26.2.20"
    subprocess.run([pkgconf, "--exists", "freetype2 >= 9.11.3"], env=environment, check=True)
    flags = subprocess.check_output([pkgconf, "--static", "--libs", "freetype2"], env=environment).decode().split()
    assert "-lfreetype" in flags and "-lz" in flags and "-lm" in flags


def test_sealed_provider_rasterizes_real_font_in_guest(tmp_path):
    prefix = os.environ.get("SHELLSIM_FREETYPE_PREFIX")
    zlib = os.environ.get("SHELLSIM_FREETYPE_ZLIB_PREFIX")
    sdk = os.environ.get("SHELLSIM_FREETYPE_SDK")
    if not prefix or not zlib or not sdk:
        pytest.skip("set sealed provider prefixes and the pinned SDK")
    pkgconf = os.environ.get("SHELLSIM_PKG_CONFIG", "pkg-config")
    import shellsim

    from ports.native.dependencies import target_environment

    environment = {"PATH": "/usr/bin:/bin", "PKG_CONFIG_LIBDIR": prefix + "/lib/pkgconfig:" + zlib + "/lib/pkgconfig"}
    flags = (
        subprocess.check_output([pkgconf, "--static", "--cflags", "--libs", "freetype2"], env=environment)
        .decode()
        .split()
    )
    executable = tmp_path / "font.wasm"
    subprocess.run(
        [
            str(Path(sdk) / "bin/clang"),
            "-O2",
            "-fwasm-exceptions",
            "-mllvm",
            "-wasm-enable-sjlj",
            "-mllvm",
            "-wasm-use-legacy-eh=false",
            str(ROOT / "tests/fixtures/freetype/rasterize.c"),
            *flags,
            "-lsetjmp",
            "-o",
            str(executable),
        ],
        env=target_environment(Path(sdk)),
        check=True,
    )
    guest = shellsim.Environment(cpu=1_000_000_000, memory=256 * 1024**2)
    guest.write_file("/font.wasm", executable.read_bytes(), mode=0o755)
    guest.write_file("/font.ttf", (ROOT / "tests/fixtures/fonts/DejaVuSans.ttf").read_bytes())
    result = guest.run("/font.wasm")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"FreeType 2.13.3 rasterization and invalid font rejection passed\n"
