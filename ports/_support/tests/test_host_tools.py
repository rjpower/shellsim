"""Exercise host-source and interpreter admission against explicit real tools."""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ports._support.cohort import verify_host_files
from ports._support.host_tools import make_read_only, meson_receipt, python_closure, verify_python_closure


@pytest.fixture
def receipts():
    root = os.environ.get("SHELLSIM_HOST_RECEIPTS")
    if root is None:
        pytest.skip("requires explicit produced scientific host receipts")
    return Path(root)


@pytest.fixture
def source_cache():
    root = os.environ.get("SHELLSIM_HOST_SOURCE_CACHE")
    if root is None:
        pytest.skip("requires explicit pinned host source cache")
    return Path(root)


def test_meson_materializes_from_pinned_source(receipts, source_cache, tmp_path):
    original = json.loads((receipts / "meson-receipt.json").read_text())
    interpreter = original["source"]["interpreter"]
    produced = meson_receipt(
        tmp_path / "meson",
        Path(interpreter["root"]),
        Path(interpreter["base_python"]["root"]),
        source_cache,
        offline=True,
        materialize=True,
    )
    verify_host_files(receipts / "meson-receipt.json", produced, "meson", tmp_path / "meson/bin/meson")
    assert (tmp_path / "meson/mesonbuild/dependencies/blas_lapack.py").is_file()
    code = tmp_path / "meson/mesonbuild/mesonmain.py"
    code.write_bytes(code.read_bytes() + b"\n# altered imported code\n")
    with pytest.raises(ValueError):
        verify_host_files(receipts / "meson-receipt.json", produced, "meson", tmp_path / "meson/bin/meson")


def test_meson_rejects_unpatched_compiler_code(receipts, tmp_path):
    original = json.loads((receipts / "meson-receipt.json").read_text())
    root = tmp_path / "meson"
    shutil.copytree(original["root"], root)
    producer = copy.deepcopy(original)
    producer["root"] = str(root)
    code = root / "mesonbuild/compilers/mixins/clike.py"
    code.write_text(code.read_text().replace("self.compiler.info.system != 'wasi' and ", ""))
    with pytest.raises(ValueError):
        verify_host_files(receipts / "meson-receipt.json", producer, "meson", root / "bin/meson")


def test_generator_rejects_changed_base_stdlib(receipts, tmp_path):
    receipt = json.loads((receipts / "python-receipt.json").read_text())
    environment, base = tmp_path / "environment", tmp_path / "base"
    shutil.copytree(receipt["source"]["base_python"]["root"], base, symlinks=True)
    environment.mkdir()
    (environment / "bin").mkdir()
    shutil.copy2(base / "bin/python3.13", environment / "bin/python")
    (environment / "pyvenv.cfg").write_text(f"home = {base}/bin\ninclude-system-site-packages = false\n")
    make_read_only(base)
    closure = python_closure(environment, base)
    verify_python_closure(environment, closure)
    code = base / "lib/python3.13/collections/__init__.py"
    mode = code.stat().st_mode
    code.chmod(mode | 0o200)
    code.write_bytes(code.read_bytes() + b"\n# altered stdlib code\n")
    code.chmod(mode)
    with pytest.raises(ValueError):
        verify_python_closure(environment, closure)


def test_generator_rejects_unpinned_source_definition(receipts):
    path = receipts / "cython-receipt.json"
    producer = json.loads(path.read_text())
    producer["source"]["definition"]["packages"]["Cython"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_host_files(path, producer, "cython", Path(producer["root"]) / producer["executable"])


def test_normal_python_use_preserves_private_closure(receipts):
    path = receipts / "python-receipt.json"
    producer = json.loads(path.read_text())
    executable = Path(producer["root"]) / producer["executable"]
    environment = dict(os.environ)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    subprocess.run(
        [executable, "-I", "-c", "import collections, typing, numpy.f2py, Cython"], env=environment, check=True
    )
    verify_host_files(path, producer, "python", executable)


def test_admission_does_not_execute_python(receipts, monkeypatch):
    path = receipts / "python-receipt.json"
    producer = json.loads(path.read_text())

    def unexpected_command(*args, **kwargs):
        raise AssertionError("admission executed an unverified program")

    monkeypatch.setattr(subprocess, "run", unexpected_command)
    verify_host_files(path, producer, "python", Path(producer["root"]) / producer["executable"])
