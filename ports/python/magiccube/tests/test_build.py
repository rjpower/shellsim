"""Reject changed upstream wheels before creating build output."""

import pytest

from ports.python.magiccube.build import build


def test_changed_wheel_rejected_before_copy(tmp_path):
    source = tmp_path / "magiccube-0.3.0-py3-none-any.whl"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError):
        build(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_upstream_wheel_is_copied_unchanged(tmp_path, monkeypatch):
    import hashlib
    import json
    import os
    from pathlib import Path

    source = os.environ.get("SHELLSIM_MAGICCUBE_SOURCE_WHEEL")
    if source is None:
        pytest.skip("set the pinned upstream Magiccube wheel")
    original = Path(source).read_bytes()
    source = tmp_path / Path(source).name
    source.write_bytes(original)
    from ports.python.magiccube import build as builder

    verify = builder.verified_files

    def mutate_after_admission(wheel, recipe, *, source):
        files = verify(wheel, recipe, source=source)
        wheel.write_bytes(b"changed after admission")
        return files

    monkeypatch.setattr(builder, "verified_files", mutate_after_admission)
    destination = build(source, tmp_path / "output")
    assert destination.read_bytes() == original
    manifest = json.loads((destination.parent / "manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(original).hexdigest()
