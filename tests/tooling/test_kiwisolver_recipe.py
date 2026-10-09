"""Validate the port's archive boundary, source ABI correction and wheel contract."""

import csv
import hashlib
import io
import json
import os
import re
import tarfile
import zipfile
from pathlib import Path

import pytest

from ports.kiwisolver.build import PORT, unpack_source, version_header, write_wheel
from ports.numpy.build import apply_patch, check_build_scripts


def test_pinned_build_inputs():
    recipe = json.loads((PORT / "recipe.json").read_text())
    check_build_scripts(recipe, PORT)
    patch = recipe["patch"]
    assert hashlib.sha256((PORT / patch["file"]).read_bytes()).hexdigest() == patch["sha256"]


@pytest.mark.parametrize("member", ["../escape", "kiwisolver-1.5.1/../../escape", "/absolute"])
def test_source_archive_rejects_unsafe_paths(tmp_path, member):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as output:
        info = tarfile.TarInfo(member)
        info.size = 1
        output.addfile(info, io.BytesIO(b"x"))
    recipe = {"version": "1.5.1", "source": {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}}
    with pytest.raises(ValueError):
        unpack_source(archive, tmp_path / "output", recipe)
    assert not (tmp_path / "escape").exists()


def test_source_archive_hash_checked_before_extraction(tmp_path):
    archive = tmp_path / "source.tar"
    archive.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="source archive differs"):
        unpack_source(archive, tmp_path / "output", {"source": {"sha256": "0" * 64}})
    assert not (tmp_path / "output").exists()


def test_upstream_noargs_patch_and_version_template(tmp_path):
    configured = os.environ.get("SHELLSIM_KIWISOLVER_SOURCE_ARCHIVE")
    if not configured:
        pytest.skip("set the pinned upstream Kiwi source archive")
    recipe = json.loads((PORT / "recipe.json").read_text())
    source = unpack_source(Path(configured), tmp_path, recipe)
    metadata = (source / "PKG-INFO").read_bytes()
    digest = version_header(source, recipe)
    assert digest == hashlib.sha256((source / "py/src/version.h").read_bytes()).hexdigest()
    assert '#define PY_KIWI_VERSION "1.5.1"' in (source / "py/src/version.h").read_text()
    patch = recipe["patch"]
    apply_patch(source, PORT / patch["file"], patch["sha256"])
    count = 0
    getters = 0
    for file in (source / "py/src").glob("*.cpp"):
        text = file.read_text()
        names = re.findall(r"\(\s*PyCFunction\s*\)\s*(\w+)\s*,\s*METH_NOARGS", text)
        for name in names:
            assert re.search(r"\b" + name + r"\(\s*\w+\s*\*\s*self\s*,\s*PyObject\*\)", text)
            count += 1
        for name in re.findall(r"\(\s*getter\s*\)\s*(\w+)", text):
            assert re.search(r"\b" + name + r"\(\s*\w+\s*\*\s*self\s*,\s*void\*\)", text)
            getters += 1
    assert count == 17
    assert getters == 4
    assert (source / "PKG-INFO").read_bytes() == metadata


def test_wheel_record_and_output_are_deterministic(tmp_path):
    stage = tmp_path / "stage"
    dist = stage / "kiwisolver-1.5.1.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Name: kiwisolver\nVersion: 1.5.1\n")
    first, second = tmp_path / "first.whl", tmp_path / "second.whl"
    write_wheel(stage, first)
    (dist / "RECORD").unlink()
    write_wheel(stage, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as wheel:
        rows = list(csv.reader(io.StringIO(wheel.read("kiwisolver-1.5.1.dist-info/RECORD").decode())))
        assert {row[0] for row in rows} == set(wheel.namelist())
        assert rows[-1] == ["kiwisolver-1.5.1.dist-info/RECORD", "", ""]
