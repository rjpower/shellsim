"""Use a real Meson/Ninja compilation to distinguish reuse from fresh builds."""

import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ports._support.meson_workspace import retained_meson
from ports._support.native_adapters import NativeAdapter, NativeBuildContext, NativeBuildRequest, build_native
from ports._support.store import file_hash


@pytest.fixture
def meson_project(tmp_path):
    meson = os.environ.get("SHELLSIM_TEST_MESON") or shutil.which("meson")
    ninja = os.environ.get("SHELLSIM_TEST_NINJA") or shutil.which("ninja")
    if not meson or not ninja:
        pytest.skip("real Meson and Ninja tools are required")
    source = tmp_path / "input"
    source.mkdir()
    (source / "meson.build").write_text(
        "project('retained', 'c')\n"
        "static_library('value', 'value.c', install: true)\n"
        "install_data('one.txt', install_dir: 'share/value', install_tag: 'one')\n"
        "install_data('two.txt', install_dir: 'share/value', install_tag: 'two')\n"
    )
    (source / "value.c").write_text("int value(void) { return 19; }\n")
    (source / "one.txt").write_text("first packaging selection")
    (source / "two.txt").write_text("second packaging selection")
    dependencies = tmp_path / "deps"
    dependencies.mkdir()
    host = {
        "python": Path(sys.executable),
        "meson": Path(meson),
        "ninja": Path(ninja),
        "sh": Path("/usr/bin/sh"),
        "rm": Path("/usr/bin/rm"),
        "pkg-config": Path(os.environ.get("SHELLSIM_TEST_PKG_CONFIG") or "/usr/bin/true"),
    }
    target = {
        name: Path("/usr/bin/" + binary)
        for name, binary in {"cc": "gcc", "cxx": "g++", "ar": "ar", "ranlib": "ranlib", "strip": "strip"}.items()
    }
    context = NativeBuildContext(
        source,
        tmp_path / "unused",
        tmp_path / "stage-one",
        tmp_path / "sdk",
        tmp_path / "compiler",
        tmp_path / "sysroot",
        "wasm32-wasip1",
        (),
        (),
        {},
        host,
        target,
        dependencies,
    )
    tools = {name: {"path": str(path), "sha256": file_hash(path.resolve())} for name, path in host.items()}
    products = {name: file_hash(path.resolve()) for name, path in target.items()}
    return context, tmp_path / "retained/build/meson-build", products, tools


def test_real_build_reuses_objects_when_only_packaging_changes(meson_project):
    context, ninja, products, tools = meson_project
    with retained_meson(context, ninja, {}, products, tools) as admitted:
        build_native(NativeBuildRequest(NativeAdapter.MESON, admitted, meson_install_tags=("devel", "one")))
    object_file = next(ninja.rglob("*.o"))
    before = object_file.read_bytes(), object_file.stat().st_mtime_ns
    actions = [line.split("\t")[3] for line in (ninja / ".ninja_log").read_text().splitlines()[1:]]
    assert (context.staging_prefix / "usr/local/share/value/one.txt").exists()

    second = replace(context, staging_prefix=context.staging_prefix.parent / "stage-two")
    with retained_meson(second, ninja, {}, products, tools) as admitted:
        build_native(NativeBuildRequest(NativeAdapter.MESON, admitted, meson_install_tags=("devel", "two")))
    assert (object_file.read_bytes(), object_file.stat().st_mtime_ns) == before
    assert [line.split("\t")[3] for line in (ninja / ".ninja_log").read_text().splitlines()[1:]] == actions
    assert (second.staging_prefix / "usr/local/share/value/two.txt").exists()
    assert not (second.staging_prefix / "usr/local/share/value/one.txt").exists()

    (context.source / "value.c").write_text("int value(void) { return 23; }\n")
    with pytest.raises(ValueError, match="compilation inputs"):
        with retained_meson(second, ninja, {}, products, tools):
            pytest.fail("changed compilation input was reused")
    assert (object_file.read_bytes(), object_file.stat().st_mtime_ns) == before


def test_unrecorded_existing_tree_is_not_adopted(meson_project):
    context, ninja, products, tools = meson_project
    ninja.mkdir(parents=True)
    with pytest.raises(ValueError, match="no verified workspace receipt"):
        with retained_meson(context, ninja, {}, products, tools):
            pytest.fail("unverified build tree was adopted")
