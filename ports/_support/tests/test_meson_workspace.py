"""Use a real Meson/Ninja compilation to distinguish reuse from fresh builds."""

import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ports._support import meson_adapter, native_adapters
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
        "run_command(find_program('python', native: true), '-c', "
        "'import pathlib, sys; pathlib.Path(sys.argv[1]).write_text(\"enum { VALUE=19 };\")', "
        "meson.current_build_dir() / 'generated.h', check: true)\n"
        "static_library('value', 'value.c', include_directories: include_directories('.'), install: true)\n"
        "install_data('one.txt', install_dir: 'share/value', install_tag: 'one')\n"
        "install_data('two.txt', install_dir: 'share/value', install_tag: 'two')\n"
    )
    (source / "value.c").write_text('#include "generated.h"\nint value(void) { return VALUE; }\n')
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


@pytest.mark.parametrize("change", ["cc-wrapper", "cxx-wrapper", "environment", "meson-setup"])
def test_changed_driver_rejects_stale_objects_and_changes_fresh_output(meson_project, monkeypatch, change):
    context, ninja, products, tools = meson_project
    cxx = change == "cxx-wrapper"
    filename = "value.cpp" if cxx else "value.c"
    if cxx:
        project = context.source / "meson.build"
        project.write_text(project.read_text().replace("'c'", "'cpp'").replace("value.c", filename))
    (context.source / filename).write_text(
        '#include "generated.h"\n#ifndef BIAS\n#define BIAS 0\n#endif\nint value(void) { return VALUE + BIAS; }\n'
    )

    def observed_value(staging):
        main = staging.parent / ("main.cpp" if cxx else "main.c")
        main.write_text("int value(void); int main(void) { return value(); }\n")
        executable = staging.parent / (staging.name + "-value")
        subprocess.run(
            [
                context.target_tools["cxx" if cxx else "cc"],
                main,
                staging / "usr/local/lib/libvalue.a",
                "-o",
                executable,
            ],
            check=True,
        )
        return subprocess.run([executable], check=False).returncode

    with retained_meson(context, ninja, {}, products, tools) as admitted:
        build_native(NativeBuildRequest(NativeAdapter.MESON, admitted, meson_install_tags=("devel",)))
    assert observed_value(context.staging_prefix) == 19
    object_file = next(ninja.rglob("*.o"))
    before = object_file.read_bytes(), object_file.stat().st_mtime_ns
    if change == "meson-setup":
        original = meson_adapter.meson_configuration

        def changed_configuration(request):
            configuration = original(request)
            configuration["setup_command"].append("-Dc_args=-DBIAS=7")
            return configuration

        monkeypatch.setattr(meson_adapter, "meson_configuration", changed_configuration)
    elif change == "environment":
        original = native_adapters.build_environment

        def changed_environment(context, bindings):
            return {**original(context, bindings), "CFLAGS": "-DBIAS=7"}

        monkeypatch.setattr(native_adapters, "build_environment", changed_environment)
    else:
        original = native_adapters.compiler_wrapper_text

        def changed_wrapper(context, role, response_source):
            if role == ("cxx" if cxx else "cc"):
                context = replace(context, compiler_flags=(*context.compiler_flags, "-DBIAS=7"))
            return original(context, role, response_source)

        monkeypatch.setattr(native_adapters, "compiler_wrapper_text", changed_wrapper)
    second = replace(context, staging_prefix=context.staging_prefix.parent / "changed-stage")
    with pytest.raises(ValueError, match="compilation inputs"):
        with retained_meson(second, ninja, {}, products, tools):
            pytest.fail("changed compiler behavior reused old objects")
    assert (object_file.read_bytes(), object_file.stat().st_mtime_ns) == before
    fresh = ninja.parents[2] / "fresh/build/meson-build"
    with retained_meson(second, fresh, {}, products, tools) as admitted:
        build_native(NativeBuildRequest(NativeAdapter.MESON, admitted, meson_install_tags=("devel",)))
    assert observed_value(second.staging_prefix) == 26


def test_unrecorded_existing_tree_is_not_adopted(meson_project):
    context, ninja, products, tools = meson_project
    ninja.mkdir(parents=True)
    with pytest.raises(ValueError, match="no verified workspace receipt"):
        with retained_meson(context, ninja, {}, products, tools):
            pytest.fail("unverified build tree was adopted")


def test_shared_link_resolves_trailing_archive(meson_project):
    context, _, _, _ = meson_project
    source = context.source
    (source / "meson.build").write_text(
        "project('trailing', 'c')\n"
        "shared_library('value', 'value.c', link_args: ['-Wl,--no-undefined'], install: true)\n"
    )
    (source / "value.c").write_text("extern int helper(void); int value(void) { return helper(); }\n")
    helper = source.parent / "helper.c"
    helper.write_text("int helper(void) { return 19; }\n")
    obj, archive = helper.with_suffix(".o"), helper.with_suffix(".a")
    subprocess.run([context.target_tools["cc"], "-fPIC", "-c", helper, "-o", obj], check=True)
    subprocess.run([context.target_tools["ar"], "rcs", archive, obj], check=True)
    context = replace(context, shared_library_inputs=(archive,))
    build_native(NativeBuildRequest(NativeAdapter.MESON, context))
    assert next(context.staging_prefix.rglob("libvalue.so")).read_bytes().startswith(b"\x7fELF")
