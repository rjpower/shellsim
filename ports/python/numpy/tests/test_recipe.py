"""Verify package hashes and refusal of mismatched reusable archive providers."""

import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from ports._support.graph import plan
from ports._support.native_artifacts import NativeArtifact, NativeTarget, merge_dependency_sysroot
from ports._support.python_adapters import _wheel
from ports._support.runner import _admit_recipe


def analyze(roots):
    return plan(Path(__file__).resolve().parents[4] / "ports", roots)


def write_wheel(stage, path):
    (stage / "numpy-2.3.5.dist-info/RECORD").unlink(missing_ok=True)
    _wheel(stage, path, "numpy-2.3.5.dist-info/RECORD")


def test_dynamic_recipe_pins_builder():
    _admit_recipe(analyze(["python/numpy"]).ports[-1])


def test_provider_refuses_changed_cohort_before_using_archives(tmp_path):
    artifact = NativeArtifact(
        tmp_path,
        {
            "inputs": {
                "recipe": {"name": "openblas", "target": "wrong", "target_profile": "wrong", "abi": "wrong"},
                "toolchain": {},
            }
        },
    )
    with pytest.raises(ValueError):
        merge_dependency_sysroot(
            {"native/openblas": artifact}, {}, tmp_path / "output", NativeTarget("wasm32", "approved", "abi", {})
        )
    assert not (tmp_path / "output").exists()


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
