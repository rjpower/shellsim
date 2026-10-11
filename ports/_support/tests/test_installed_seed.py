"""Host generators follow the target cohort even when transport uses another Python."""

from pathlib import Path

import pytest

from ports._support.sdk import PrerequisiteError, installed_seed


def test_target_python_generator_is_selected_from_installed_tools(monkeypatch):
    paths = {"python": "/transport/python", "python3.13": "/tools/python3.13"}
    monkeypatch.setattr("ports._support.sdk.shutil.which", paths.get)
    monkeypatch.setattr("ports._support.sdk.sys.executable", "/transport/python")
    seed = installed_seed(python_version="3.13.7")
    assert seed.tools["python"].path == Path("/tools/python3.13")
    assert seed.python_helper == seed.tools["python"]


def test_missing_target_python_generator_reports_a_prerequisite(monkeypatch):
    monkeypatch.setattr("ports._support.sdk.shutil.which", lambda name: None)
    with pytest.raises(PrerequisiteError):
        installed_seed(python_version="3.13.7")
