"""Exercise LLVM response syntax and classification work bounds."""

import os
from pathlib import Path

import pytest

from ports._support.compiler_response import _MAX_RESPONSE_BYTES, response_arguments
from ports._support.import_sdk import load_legacy_cohort
from ports._support.native_adapters import NativeAdapter, NativeBuildContext, NativeBuildRequest, build_native
from ports._support.wasm_metadata import function_signatures


def test_nested_response_options_preserve_llvm_quoting(tmp_path):
    (tmp_path / "outer.rsp").write_text("'a\\ b.c' @nested.rsp \"-shared\" -o 'result.so'")
    (tmp_path / "nested.rsp").write_bytes('\ufeff-DNAME="a b" ""'.encode())
    assert response_arguments(["@outer.rsp", "last.o"], tmp_path) == [
        "a b.c",
        "-DNAME=a b",
        "",
        "-shared",
        "-o",
        "result.so",
        "last.o",
    ]


def test_response_nested_paths_follow_compiler_working_directory(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/outer.rsp").write_text("@nested.rsp")
    (tmp_path / "nested.rsp").write_bytes('-c "a b.c"'.encode("utf-16"))
    assert response_arguments(["@sub/outer.rsp"], tmp_path) == ["-c", "a b.c"]


@pytest.mark.parametrize("contents", ["@cycle.rsp", "@missing.rsp", "x" * (_MAX_RESPONSE_BYTES + 1)])
def test_unreadable_cyclic_or_oversized_response_fails(tmp_path, contents):
    (tmp_path / "cycle.rsp").write_text(contents)
    with pytest.raises(ValueError):
        response_arguments(["@cycle.rsp"], tmp_path)


def test_repeated_noncyclic_response_is_allowed(tmp_path):
    (tmp_path / "one.rsp").write_text("-shared")
    assert response_arguments(["@one.rsp", "@one.rsp"], tmp_path) == ["-shared", "-shared"]


def test_response_depth_is_bounded(tmp_path):
    for index in range(18):
        (tmp_path / f"{index}.rsp").write_text(f"@{index + 1}.rsp")
    with pytest.raises(ValueError):
        response_arguments(["@0.rsp"], tmp_path)


def test_actual_wrapper_compiles_and_links_nested_response_files(tmp_path):
    descriptor = os.environ.get("SHELLSIM_BUILD_COHORT")
    if descriptor is None:
        pytest.skip("requires an explicitly admitted build cohort")
    cohort = load_legacy_cohort(Path(descriptor))
    source = tmp_path / "source"
    source.mkdir()
    (source / "module source.c").write_text("int operation(int x) { return x + 1; }\n")
    (source / "compile.rsp").write_text('-c "module source.c" -o module.o -Werror\n')
    (source / "combo.rsp").write_text('-c -shared "module source.c" -o combo.o\n')
    (source / "inner.rsp").write_text("-shared module.o\n")
    (source / "link.rsp").write_text("@inner.rsp -o module.so\n")
    (source / "Makefile").write_text("all:\n\t$(CC) @compile.rsp\n\t$(CC) @combo.rsp\n\t$(CC) @link.rsp\n")
    context = NativeBuildContext(
        source=source,
        build=tmp_path / "build",
        staging_prefix=tmp_path / "staging",
        sdk=cohort.sdk.root,
        compiler_prefix=cohort.llvm.root,
        sysroot=cohort.sysroot.root / "sysroot",
        target=cohort.target,
        abi=cohort.dynamic_abi,
        compiler_resource_directory=cohort.compiler_resource_directory,
        linker=cohort.linker,
        compiler_flags=cohort.compiler_flags,
        linker_flags=cohort.linker_flags,
        dependencies={},
        host_tools={name: tool.path for name, tool in cohort.host_tools.items()},
        target_tools={name: tool.path for name, tool in cohort.target_tools.items()},
        dependency_sysroot=tmp_path / "dependencies",
        shared_library_flags=cohort.shared_library_flags,
        executable_flags=cohort.executable_flags,
        shared_library_inputs=(cohort.compiler_runtime_archive,),
    )
    build_native(NativeBuildRequest(NativeAdapter.PLAIN_MAKE, context, install_targets=()))
    assert (source / "module.o").read_bytes().startswith(b"\0asm\x01\0\0\0")
    assert (source / "combo.o").read_bytes().startswith(b"\0asm\x01\0\0\0")
    _, exports = function_signatures(source / "module.so")
    assert exports["operation"] == ((0x7F,), (0x7F,))
