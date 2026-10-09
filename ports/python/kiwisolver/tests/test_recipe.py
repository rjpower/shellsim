"""Validate the port's archive boundary and wheel integrity contract."""

import base64
import csv
import hashlib
import io
import json
import tarfile
import zipfile

import pytest

from ports._support.build import check_build_scripts
from ports.python.kiwisolver.build import PORT, unpack_source, write_wheel


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


def test_wheel_record_and_output_are_deterministic(tmp_path):
    stage = tmp_path / "stage"
    dist = stage / "kiwisolver-1.5.1.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Name: kiwisolver\nVersion: 1.5.1\n")
    (stage / "native.so").write_bytes(b"native payload")
    first, second = tmp_path / "first.whl", tmp_path / "second.whl"
    write_wheel(stage, first)
    (dist / "RECORD").unlink()
    write_wheel(stage, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as wheel:
        rows = list(csv.reader(io.StringIO(wheel.read("kiwisolver-1.5.1.dist-info/RECORD").decode())))
        assert {row[0] for row in rows} == set(wheel.namelist())
        for path, digest, size in rows:
            if path == "kiwisolver-1.5.1.dist-info/RECORD":
                assert (digest, size) == ("", "")
                continue
            content = wheel.read(path)
            actual = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            assert digest == "sha256=" + actual
            assert int(size) == len(content)
