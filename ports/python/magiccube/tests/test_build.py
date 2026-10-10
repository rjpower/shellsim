"""Reject changed upstream wheels before creating build output."""

import pytest

from ports._support import python_adapters
from ports._support.graph import plan
from ports._support.python_adapters import PureWheelBuildRequest, build_pure_wheel


def analyze(roots):
    from pathlib import Path

    return plan(Path(__file__).resolve().parents[4] / "ports", roots)


def build(source, output):
    recipe = next(port.recipe for port in analyze(["python/magiccube"]).ports if port.name == "magiccube")
    result = build_pure_wheel(PureWheelBuildRequest(source, output, recipe))
    return result.staging_prefix / "wheels" / source.name


def test_changed_wheel_rejected_before_copy(tmp_path):
    source = tmp_path / "magiccube-0.3.0-py3-none-any.whl"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError):
        build(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_upstream_wheel_is_copied_unchanged(tmp_path, monkeypatch):
    import hashlib
    import os
    from pathlib import Path

    source = os.environ.get("SHELLSIM_MAGICCUBE_SOURCE_WHEEL")
    if source is None:
        pytest.skip("set the pinned upstream Magiccube wheel")
    original = Path(source).read_bytes()
    source = tmp_path / Path(source).name
    source.write_bytes(original)
    builder = python_adapters

    verify = builder.verified_files

    def mutate_after_admission(wheel, recipe, *, source):
        files = verify(wheel, recipe, source=source)
        wheel.write_bytes(b"changed after admission")
        return files

    monkeypatch.setattr(builder, "verified_files", mutate_after_admission)
    destination = build(source, tmp_path / "output")
    assert destination.read_bytes() == original
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == hashlib.sha256(original).hexdigest()
