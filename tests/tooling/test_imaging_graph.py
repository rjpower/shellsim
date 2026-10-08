"""Reject native metadata drift before selecting executable graph providers."""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "ports/spike_images.py"
_SPEC = importlib.util.spec_from_file_location("spike_images", _PATH)
IMAGES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(IMAGES)


@pytest.fixture
def metadata_source(tmp_path):
    path = tmp_path / "downloads/native.tar.gz"
    path.parent.mkdir()
    data = b"Name: pillow\nVersion: 12.3.0\nRequires-Dist: optional; extra == 'test'\n\nDescription\n"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("pillow-12.3.0/PKG-INFO")
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    recipe = {
        "name": "pillow",
        "version": "12.3.0",
        "requires_dist": ["optional; extra == 'test'"],
        "source": {
            "url": "https://example.invalid/native.tar.gz",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }
    return recipe, tmp_path, path


def test_native_metadata_preserves_real_dependencies(metadata_source):
    recipe, bundle, _ = metadata_source
    assert IMAGES.verified_metadata(recipe, bundle).endswith(b"Requires-Dist: optional; extra == 'test'\n\n")


@pytest.mark.parametrize("mutation", ["archive", "version", "dependencies"])
def test_native_metadata_drift_is_rejected(metadata_source, mutation):
    recipe, bundle, archive = metadata_source
    if mutation == "archive":
        archive.write_bytes(b"changed archive")
    elif mutation == "version":
        recipe["version"] = "1.0"
    else:
        recipe["requires_dist"] = []
    with pytest.raises(ValueError):
        IMAGES.verified_metadata(recipe, bundle)


def test_adaptation_rejects_dynamic_loading_before_ctypes_import(tmp_path):
    import types
    import zipfile

    wheel = tmp_path / "imageio-2.37.0-py3-none-any.whl"
    source = (
        "import sys\nimport ctypes\n"
        "def load_lib(exact_lib_names, lib_names, lib_dirs=None):\n"
        "    # Checks\n"
        '    raise AssertionError("discovery reached")\n'
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("imageio/core/findlib.py", source)
        archive.writestr(
            "imageio/plugins/pillow.py",
            "from PIL import ExifTags, GifImagePlugin, Image, ImageSequence, UnidentifiedImageError\n"
            'def read(self):\n        if self._image.format == "GIF":\n            # Converting GIF\n            pass\n',
        )
        archive.writestr("imageio-2.37.0.dist-info/RECORD", "")
    derived, evidence = IMAGES.adapted_wheel(wheel, tmp_path)
    with zipfile.ZipFile(derived) as archive:
        adapted = archive.read("imageio/core/findlib.py").decode()
        plugin = archive.read("imageio/plugins/pillow.py").decode()
        record = archive.read("imageio-2.37.0.dist-info/RECORD").decode()
    namespace = {}
    exec(adapted, namespace)
    namespace["sys"] = types.SimpleNamespace(platform="wasi")
    with pytest.raises(NotImplementedError):
        namespace["load_lib"]([], [])
    assert "ctypes" not in namespace
    assert "ExifTags, GifImagePlugin" not in plugin
    assert 'if self._image.format == "GIF":\n            from PIL import GifImagePlugin' in plugin
    assert "imageio/core/findlib.py,sha256=" in record
    assert evidence["disabled"] == ["dynamic-library-loading"]
    assert evidence["wheel_sha256"] == hashlib.sha256(derived.read_bytes()).hexdigest()
