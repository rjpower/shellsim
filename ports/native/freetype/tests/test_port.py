"""Check the explicit FreeType source profile without downloading or compiling."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from ports._support.sdk_products import Receipt, file_hash, verify_product
from ports.native.dependencies import recipe_identity

ROOT = Path(__file__).resolve().parents[4]


def test_freetype_pins_production_sources():
    recipe = json.loads((ROOT / "ports/native/freetype/recipe.json").read_text())
    recipe_identity(recipe, ROOT / "ports/native/freetype")


@pytest.mark.parametrize("field", ["header", "pkg-config", "receipt"])
def test_development_export_drift_is_rejected_before_build(tmp_path, field):
    paths = {
        "header": "include/freetype2/freetype.h",
        "pkg-config": "lib/pkgconfig/freetype2.pc",
        "receipt": "licenses/FTL.txt",
    }
    for name in paths.values():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    hashes = {name: file_hash(tmp_path / name) for name in paths.values()}
    receipt = Receipt(
        tmp_path, tmp_path / "manifest.json", "", {"identity": {"recipe": {"name": "freetype"}}, "artifacts": hashes}
    )
    verify_product(receipt)
    (tmp_path / paths[field]).write_text("changed after admission")
    with pytest.raises(ValueError):
        verify_product(receipt)


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
