"""Verify that relinking admits sealed compile inputs rather than loose objects."""

import json
from pathlib import Path

import pytest

from ports.native.dependencies import digest
from ports.python.cpython.threaded import compile_receipt, relink


def test_compile_receipt_changes_for_objects_sources_and_generated_config(tmp_path):
    work = tmp_path / "work"
    source = work / "Python-3.13.7"
    source.mkdir(parents=True)
    (source / "module.c").write_bytes(b"source")
    (work / "process-source").mkdir()
    guest = work / "wasi-build"
    guest.mkdir()
    (guest / "module.o").write_bytes(b"object")
    (guest / "pyconfig.h").write_bytes(b"config")
    (tmp_path / "sysroot/include").mkdir(parents=True)
    receipt = compile_receipt(work, ["module.o"], tmp_path / "sysroot")
    for path in (source / "module.c", guest / "module.o", guest / "pyconfig.h"):
        original = path.read_bytes()
        path.write_bytes(b"changed")
        assert compile_receipt(work, ["module.o"], tmp_path / "sysroot") != receipt
        path.write_bytes(original)


def test_historical_runtime_without_compile_receipt_cannot_relink(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    profile = {"headers": {}}
    (previous / "manifest.json").write_text(
        json.dumps({"build_profile": profile, "build_profile_sha256": digest(profile)})
    )
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="sealed compile-input receipt"):
        relink(previous, {}, {}, {}, Path("sdk"), Path("sysroot"), Path("llvm"), output, {})
    assert not output.exists()
