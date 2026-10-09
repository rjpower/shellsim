"""Check the scalar JPEG scope and immutable build inputs without downloads."""

import importlib.util
import json
from pathlib import Path


def test_libjpeg_recipe_has_explicit_scalar_scope(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root))
    from ports.native.dependencies import recipe_identity

    directory = root / "ports/native/libjpeg-turbo"
    recipe = json.loads((directory / "recipe.json").read_text())
    recipe_identity(recipe, directory)
    assert recipe["target_dependencies"] == []
    assert recipe["transitive_link_flags"] == []
    assert recipe["features"]["sample_bits"] == 8
    assert all(recipe["features"][name] is False for name in ("simd", "assembly", "threads", "turbojpeg", "shared"))
    spec = importlib.util.spec_from_file_location("jpeg_build", directory / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "jsimd_none" in module.SOURCES
    assert "jmemnobs" in module.SOURCES
    assert "#define WITH_SIMD" not in module.CONFIG
    assert "#define C_ARITH_CODING_SUPPORTED" not in module.CONFIG
