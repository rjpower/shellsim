"""Keep host publication imports independent of the native execution adapter."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_release_metadata_imports_without_native_adapter():
    source = Path(__file__).resolve().parents[2] / "python"
    script = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])

class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "shellsim._native":
            raise AssertionError("metadata import loaded the execution adapter")

sys.meta_path.insert(0, RejectNative())
from shellsim import CPythonRuntime, PackageInstallError
from shellsim._cpython_release import _descriptor
from shellsim._native_release import _validate_entry
from shellsim.native_packages import verify_release_catalog
assert "shellsim._native" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(source)], check=True)


def test_lazy_public_exports_preserve_execution_api():
    import shellsim
    from shellsim import _api

    assert shellsim.Environment is _api.Environment
    assert shellsim.Limits is _api.Limits
    assert set(shellsim.__all__) <= set(dir(shellsim))
    with pytest.raises(AttributeError):
        shellsim.missing_public_export
