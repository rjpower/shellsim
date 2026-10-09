"""Verify pinned pure-wheel admission and unchanged output bytes."""

import hashlib
import json
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


def test_build_copies_verified_upstream_bytes(metadata_source, tmp_path, monkeypatch):
    recipe, wheel = metadata_source
    recipe["build_scripts"] = []
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    monkeypatch.setattr(IMAGES, "PORT", tmp_path)
    original = wheel.read_bytes()
    copied = IMAGES.build(wheel, tmp_path / "output")
    assert copied.read_bytes() == original
    assert json.loads((copied.parent / "manifest.json").read_text())["sha256"] == hashlib.sha256(original).hexdigest()


def test_build_rejects_changed_source_before_copy(metadata_source, tmp_path, monkeypatch):
    recipe, wheel = metadata_source
    recipe["build_scripts"] = []
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    monkeypatch.setattr(IMAGES, "PORT", tmp_path)
    wheel.write_bytes(b"changed")
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        IMAGES.build(wheel, output)
    assert not output.exists()


@pytest.mark.parametrize("member", ["../escape", "/absolute", "package/native.so", "package/hidden.py"])
def test_pure_wheel_rejects_unsafe_members_and_native_bytes(tmp_path, metadata_source, member):
    recipe, wheel = metadata_source
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, b"\0asm" if member.endswith("hidden.py") else b"x")
    recipe["source"]["sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        verified_files(wheel, recipe)
