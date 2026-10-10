"""Pin integrity plus optional full builds verify unchanged pure-wheel output."""

import json
from pathlib import Path

import pytest

from ports._support.graph import plan
from ports._support.python_adapters import _wheel
from ports._support.python_pep517 import package_snapshot, verify_preserved_package
from ports._support.store import fetch
from ports._support.store import file_hash as sha256
from ports._support.store import relative_path as _relative
from ports.api import implementation


def analyze(roots):
    from pathlib import Path

    return plan(Path(__file__).resolve().parents[4] / "ports", roots)


ROOT = Path(__file__).resolve().parents[4]


def test_pure_sdist_recipe_pins_builder():
    port = ROOT / "ports/python/cellpylib"
    selected = next(item for item in analyze(["python/cellpylib"]).ports if item.name == "cellpylib")
    assert implementation(selected)["builder"] == sha256(port / "build.py")


@pytest.mark.parametrize("path", ["/absolute", "../escape", "a/../escape", "a\\escape", "a\0escape"])
def test_build_archive_path_rejects_escape(path):
    with pytest.raises(ValueError):
        _relative(path)


def test_corrupt_source_rejected_before_build(tmp_path):
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        recipe = json.loads((ROOT / "ports/python/cellpylib/recipe.json").read_text())
        cache = tmp_path / "cache" / recipe["source"]["sha256"]
        cache.mkdir(parents=True)
        (cache / "cellpylib-2.4.0.tar.gz").write_bytes(source.read_bytes())
        fetch(recipe["source"], tmp_path / "cache", offline=True)
    assert not (tmp_path / "output").exists()


def test_unchanged_pure_wheel_is_reproducible(tmp_path):
    import zipfile

    stage = tmp_path / "stage"
    package = stage / "cellpylib"
    package.mkdir(parents=True)
    original = b"def apply_rule(state): return state\n"
    (package / "__init__.py").write_bytes(original)
    admitted = package_snapshot(stage, "cellpylib")
    verify_preserved_package(stage, "cellpylib", admitted)
    dist = stage / "cellpylib-2.4.0.dist-info"
    dist.mkdir()
    first, second = tmp_path / "first.whl", tmp_path / "second.whl"
    _wheel(stage, first, "cellpylib-2.4.0.dist-info/RECORD")
    (dist / "RECORD").unlink()
    _wheel(stage, second, "cellpylib-2.4.0.dist-info/RECORD")
    assert sha256(first) == sha256(second)
    with zipfile.ZipFile(first) as archive:
        assert archive.read("cellpylib/__init__.py") == original
    (package / "__init__.py").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_preserved_package(stage, "cellpylib", admitted)
    (dist / "RECORD").unlink()
    _wheel(stage, second, "cellpylib-2.4.0.dist-info/RECORD")
    assert sha256(first) != sha256(second)
