"""Check source admission and the sealed resolver's executable identity."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from ports.toolchain.uv import producer as build


def test_changed_driver_is_rejected_before_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "build.py").write_text("changed")
    monkeypatch.setattr(build, "HERE", tmp_path)
    monkeypatch.setattr(build, "RECIPE", {"build_scripts": [{"file": "build.py", "sha256": "0" * 64}]})

    def no_checkout(*args, **kwargs):
        raise AssertionError("source checkout must not start after a pin mismatch")

    monkeypatch.setattr(build.subprocess, "run", no_checkout)
    with pytest.raises(ValueError, match="build input changed"):
        build.build(tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_verified_distributable_matches_measured_binary() -> None:
    configured = os.environ.get("SHELLSIM_UV_RELEASE_ARTIFACT")
    if configured is None:
        pytest.skip("set SHELLSIM_UV_RELEASE_ARTIFACT to a built resolver directory")
    root = Path(configured)
    manifest = json.loads((root / "artifact.json").read_text())
    binary = root / manifest["executable"]["file"]
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifest["executable"]["sha256"]
    assert binary.stat().st_size == manifest["executable"]["size"]
