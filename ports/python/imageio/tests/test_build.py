"""Verify pure-wheel input admission and reviewed import adaptations."""

import hashlib
import zipfile

import pytest

from ports._support.pure_wheel import verified_files
from ports.python.imageio import build as IMAGES


@pytest.fixture
def metadata_source(tmp_path):
    path = tmp_path / "imageio-2.37.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "imageio-2.37.0.dist-info/METADATA", "Name: imageio\nVersion: 2.37.0\nRequires-Dist: numpy\n\n"
        )
        archive.writestr("imageio-2.37.0.dist-info/WHEEL", "Root-Is-Purelib: true\nTag: py3-none-any\n")
    recipe = {
        "name": "imageio",
        "version": "2.37.0",
        "requires_dist": ["numpy"],
        "source": {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    return recipe, path


def test_source_metadata_preserves_real_dependencies(metadata_source):
    recipe, archive = metadata_source
    assert b"Requires-Dist: numpy" in verified_files(archive, recipe)["imageio-2.37.0.dist-info/METADATA"]


@pytest.mark.parametrize("mutation", ["archive", "version", "dependencies"])
def test_source_metadata_drift_is_rejected(metadata_source, mutation):
    recipe, archive = metadata_source
    if mutation == "archive":
        archive.write_bytes(b"changed archive")
    elif mutation == "version":
        recipe["version"] = "1.0"
    else:
        recipe["requires_dist"] = []
    with pytest.raises((ValueError, KeyError)):
        verified_files(archive, recipe)


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
    with zipfile.ZipFile(wheel) as original:
        files = {entry.filename: original.read(entry) for entry in original.infolist()}
    wheel.unlink()
    derived, evidence = IMAGES.adapted_wheel(wheel, tmp_path, files)
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


def test_pinned_upstream_adaptation_preserves_metadata_and_other_files(tmp_path):
    import os
    from pathlib import Path

    source = os.environ.get("SHELLSIM_IMAGEIO_SOURCE_WHEEL")
    if source is None:
        pytest.skip("set the pinned upstream Imageio wheel")
    source = Path(source)
    destination = IMAGES.build(source, tmp_path / "output")
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(destination) as after:
        assert before.namelist() and set(before.namelist()) == set(after.namelist())
        changed = {name for name in before.namelist() if before.read(name) != after.read(name)}
    assert changed == {"imageio/core/findlib.py", "imageio/plugins/pillow.py", "imageio-2.37.0.dist-info/RECORD"}


@pytest.mark.parametrize("member", ["../escape", "/absolute", "package/native.so", "package/hidden.py"])
def test_pure_wheel_rejects_unsafe_members_and_native_bytes(tmp_path, metadata_source, member):
    recipe, wheel = metadata_source
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, b"\0asm" if member.endswith("hidden.py") else b"x")
    recipe["source"]["sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        verified_files(wheel, recipe)
