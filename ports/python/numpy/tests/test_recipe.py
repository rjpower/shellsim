"""Verify package hashes and refusal of mismatched reusable archive providers."""

import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from ports._support.build import check_build_scripts
from ports.python.numpy.dynamic import provider_inputs, write_wheel


def test_dynamic_recipe_pins_builder():
    directory = Path(__file__).parents[1]
    recipe = json.loads((directory / "dynamic-recipe.json").read_text())
    check_build_scripts(recipe, directory)


def test_provider_refuses_changed_cohort_before_using_archives(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"native_ports": [{"name": "numpy"}]}))
    with pytest.raises(ValueError):
        provider_inputs(tmp_path, {"archive_provider_sha256": "different"})


def test_wheel_record_hashes_every_payload_and_is_reproducible(tmp_path):
    stage = tmp_path / "stage"
    dist = stage / "numpy-2.3.5.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Name: numpy\nVersion: 2.3.5\n")
    (stage / "native.so").write_bytes(b"native payload")
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    write_wheel(stage, first)
    write_wheel(stage, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as wheel:
        record = "numpy-2.3.5.dist-info/RECORD"
        rows = list(csv.reader(io.StringIO(wheel.read(record).decode())))
        assert {row[0] for row in rows} == set(wheel.namelist())
        for path, digest, size in rows:
            if path == record:
                assert (digest, size) == ("", "")
                continue
            content = wheel.read(path)
            actual = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            assert digest == "sha256=" + actual
            assert int(size) == len(content)
