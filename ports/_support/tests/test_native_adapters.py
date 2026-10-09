"""Reject invalid build requests before creating an unpublished staging tree."""

from dataclasses import replace
from pathlib import PurePosixPath

import pytest

from ports._support.native_adapters import NativeAdapter, NativeBuildContext, NativeBuildRequest, build_native


@pytest.fixture
def build_request(tmp_path):
    context = NativeBuildContext(
        source=tmp_path / "source",
        build=tmp_path / "build",
        staging_prefix=tmp_path / "stage",
        sdk=tmp_path / "sdk",
        compiler_prefix=tmp_path / "compiler",
        sysroot=tmp_path / "sysroot",
        target="wasm32-wasip1",
        compiler_flags=(),
        linker_flags=(),
        dependencies={},
        host_tools={},
        target_tools={},
        dependency_sysroot=tmp_path / "dependencies",
    )
    return NativeBuildRequest(NativeAdapter.CMAKE, context)


@pytest.mark.parametrize("jobs", [0, 17])
def test_invalid_parallelism_leaves_staging_absent(build_request, jobs):
    with pytest.raises(ValueError):
        build_native(replace(build_request, jobs=jobs))
    assert not build_request.context.build.exists()
    assert not build_request.context.staging_prefix.exists()


def test_unknown_target_leaves_staging_absent(build_request):
    context = replace(build_request.context, target="wasm32-wasip1-unadmitted")
    with pytest.raises(ValueError):
        build_native(replace(build_request, context=context))
    assert not context.build.exists()


def test_missing_tool_leaves_staging_absent(build_request):
    with pytest.raises(ValueError):
        build_native(build_request)
    assert not build_request.context.staging_prefix.exists()


def test_unsupported_meson_install_target_fails_before_build(build_request):
    with pytest.raises(ValueError):
        build_native(replace(build_request, adapter=NativeAdapter.MESON, install_targets=("custom",)))
    assert not build_request.context.build.exists()


def test_unapproved_install_prefix_fails_before_build(build_request):
    with pytest.raises(ValueError):
        build_native(replace(build_request, install_prefix=PurePosixPath("/usr")))
    assert not build_request.context.staging_prefix.exists()
