"""Check the explicit FreeType source profile without downloading or compiling."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_freetype_declares_only_pinned_zlib_dependency():
    recipe = json.loads((ROOT / "ports/native/freetype/recipe.json").read_text())
    assert recipe["target_profile"] == "wasi-cpython-v2"
    assert recipe["sdk"]["version"] == "34.0"
    assert recipe["target_dependencies"] == [{"port": "native/zlib", "version": "1.3.1"}]
    for optional in ("png", "bzip2", "brotli", "harfbuzz", "host_discovery"):
        assert recipe["features"][optional] is False
