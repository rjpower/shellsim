"""Check cache refusal and preserve upstream distribution metadata."""

import pytest

from ports._support.meson_workspace import retained_meson
from ports._support.python_adapters import _wheel
from ports._support.tests import test_meson_workspace as workspace_tests


@pytest.fixture
def meson_project(tmp_path):
    return workspace_tests.meson_project.__wrapped__(tmp_path)


def test_profile_accepts_same_identity_and_refuses_changed_inputs(meson_project):
    context, ninja, products, tools = meson_project
    with retained_meson(context, ninja, {}, products, tools):
        pass
    with retained_meson(context, ninja, {}, products, tools):
        pass
    with pytest.raises(ValueError):
        with retained_meson(context, ninja, {"changed": True}, products, tools):
            pytest.fail("changed compilation identity was reused")


def test_profile_refuses_unidentified_meson_cache(meson_project):
    context, ninja, products, tools = meson_project
    ninja.mkdir(parents=True)
    with pytest.raises(ValueError):
        with retained_meson(context, ninja, {}, products, tools):
            pytest.fail("unidentified tree was reused")
    assert not (ninja.parent.parent / ".meson-workspace.json").exists()


def test_native_metadata_preserves_upstream_dependencies(tmp_path):
    import zipfile

    metadata = "Metadata-Version: 2.1\nName: sample\nVersion: 1.0\nRequires-Dist: pure>=2\n\nUpstream description\n"
    stage = tmp_path / "stage"
    dist = stage / "sample-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(metadata)
    destination = tmp_path / "sample.whl"
    _wheel(stage, destination, "sample-1.0.dist-info/RECORD")
    with zipfile.ZipFile(destination) as wheel:
        assert wheel.read("sample-1.0.dist-info/METADATA").decode() == metadata
        assert b"sample-1.0.dist-info/METADATA,sha256=" in wheel.read("sample-1.0.dist-info/RECORD")
