"""Check build-cache refusal and native-provider metadata without a compiler run."""

import importlib.util
from pathlib import Path

import pytest


def load_tool(name, relative):
    path = Path(__file__).parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CPYTHON_BUILD = load_tool("cpython_port_build", "ports/cpython/build.py")
NUMPY_BUILD = load_tool("numpy_port_build", "ports/numpy/build.py")


def test_profile_accepts_same_identity_and_refuses_changed_inputs(tmp_path):
    recipe = {"source": {"sha256": "source"}, "build_scripts": [{"sha256": "builder"}]}
    NUMPY_BUILD.check_profile(recipe, tmp_path)
    NUMPY_BUILD.check_profile(recipe, tmp_path)
    with pytest.raises(ValueError):
        NUMPY_BUILD.check_profile({**recipe, "build_scripts": [{"sha256": "changed"}]}, tmp_path)


def test_profile_refuses_unidentified_meson_cache(tmp_path):
    (tmp_path / "numpy-build").mkdir()
    with pytest.raises(ValueError):
        NUMPY_BUILD.check_profile({"version": "2.3.5"}, tmp_path)
    assert not (tmp_path / "numpy-profile.sha256").exists()


def test_native_metadata_preserves_upstream_dependencies(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    metadata = "Metadata-Version: 2.1\nName: sample\nVersion: 1.0\nRequires-Dist: pure>=2\n\nUpstream description\n"
    (source / "PKG-INFO").write_text(metadata)
    root = tmp_path / "root"
    port = {"dist_info": "/usr/lib/python3.13/site-packages/sample-1.0.dist-info"}
    CPYTHON_BUILD.install_metadata(port, source, root)
    destination = root / port["dist_info"].lstrip("/")
    assert (destination / "METADATA").read_text() == metadata
    assert "Tag: py313-none-any\n" in (destination / "WHEEL").read_text()
    assert "sample-1.0.dist-info/METADATA,sha256=" in (destination / "RECORD").read_text()
