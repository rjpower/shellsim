"""Exercise wheel bytes and rejection boundaries without a network source."""

from __future__ import annotations

import dataclasses
import hashlib
import zipfile
from pathlib import Path

import pytest

from ports._support.native_adapters import NativeBuildContext
from ports._support.python_adapters import (
    CPythonBuildContext,
    ExtensionBuildRequest,
    PureWheelBuildRequest,
    build_extension,
    build_pure_wheel,
)


def _wheel(path: Path, *, native: bool = False) -> dict:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("example/__init__.py", "VALUE = 42\n")
        archive.writestr("example-1.0.dist-info/METADATA", "Name: example\nVersion: 1.0\n")
        archive.writestr(
            "example-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        if native:
            archive.writestr("example/extension.so", b"\0asm\x01\0\0\0")
    return {
        "name": "example",
        "version": "1.0",
        "source": {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }


def test_pure_wheel_stages_exact_verified_source(tmp_path: Path) -> None:
    wheel = tmp_path / "example-1.0-py3-none-any.whl"
    recipe = _wheel(wheel)
    stage = tmp_path / "stage"
    output = build_pure_wheel(PureWheelBuildRequest(wheel, stage, recipe))
    assert (stage / "wheels" / wheel.name).read_bytes() == wheel.read_bytes()
    assert output.staging_prefix == stage
    assert output.commands == ()
    with pytest.raises(ValueError, match="already exists"):
        build_pure_wheel(PureWheelBuildRequest(wheel, stage, recipe))


def test_pure_wheel_rejects_mutation_and_native_content(tmp_path: Path) -> None:
    wheel = tmp_path / "example-1.0-py3-none-any.whl"
    recipe = _wheel(wheel)
    wheel.write_bytes(wheel.read_bytes() + b"changed")
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="identity"):
        build_pure_wheel(PureWheelBuildRequest(wheel, stage, recipe))
    assert not stage.exists()

    recipe = _wheel(wheel, native=True)
    with pytest.raises(ValueError, match="native code"):
        build_pure_wheel(PureWheelBuildRequest(wheel, stage, recipe))
    assert not stage.exists()


def _extension_request(tmp_path: Path) -> ExtensionBuildRequest:
    source = tmp_path / "source"
    source.mkdir()
    (source / "example.c").write_text("/* compile only after admission */\n")
    (source / "PKG-INFO").write_text("Name: example\nVersion: 1.0\n")
    native = NativeBuildContext(
        source=source,
        build=tmp_path / "build",
        staging_prefix=tmp_path / "stage",
        sdk=tmp_path / "sdk",
        compiler_prefix=tmp_path / "compiler",
        sysroot=tmp_path / "sysroot",
        target="wasm32-wasip1-threads",
        compiler_flags=(),
        linker_flags=(),
        dependencies={},
        host_tools={},
        target_tools={"cc": tmp_path / "cc", "cxx": tmp_path / "cxx"},
        dependency_sysroot=tmp_path / "dependency-sysroot",
        shared_library_flags=("-shared",),
    )
    cpython = CPythonBuildContext(
        include_dir=tmp_path / "include",
        generated_config_dir=tmp_path / "config",
        version="3.13.7",
        abi="shellsim-wasi-sdk34-cpython3137-threads-v3",
        runtime_target="wasm32-wasip1-threads",
        wheel_platform="wasm32_wasip1",
        receipt={},
    )
    recipe = {
        "name": "example",
        "version": "1.0",
        "abi": cpython.abi,
        "target": cpython.runtime_target,
        "requires_dist": [],
        "build": {"adapter": "python-extension", "module": "example", "sources": ["example.c"], "metadata": "PKG-INFO"},
    }
    return ExtensionBuildRequest(native, cpython, recipe)


def test_extension_rejects_wrong_abi_and_source_escape_before_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _extension_request(tmp_path)
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("compiler ran before admission"))
    wrong_abi = {**request.recipe, "abi": "shellsim-wasi-sdk34-cpython3137-v2"}
    with pytest.raises(ValueError, match="target or ABI"):
        build_extension(ExtensionBuildRequest(request.context, request.cpython, wrong_abi))
    assert not request.context.build.exists()
    assert not request.context.staging_prefix.exists()

    bad_build = {**request.recipe["build"], "sources": ["../outside.c"]}
    with pytest.raises(ValueError, match="escapes source root"):
        build_extension(ExtensionBuildRequest(request.context, request.cpython, {**request.recipe, "build": bad_build}))
    assert not request.context.build.exists()
    assert not request.context.staging_prefix.exists()


def test_extension_requires_exact_declared_native_link_inputs_before_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _extension_request(tmp_path)
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("compiler ran before admission"))
    context = dataclasses.replace(request.context, dependencies={"native/zlib": tmp_path / "zlib-artifact"})
    recipe = {**request.recipe, "target_dependencies": [{"port": "native/zlib", "version": "1.3.1"}]}
    with pytest.raises(ValueError, match="missing or oversized"):
        build_extension(
            ExtensionBuildRequest(
                context, request.cpython, {**recipe, "build": {**recipe["build"], "link_inputs": ["lib/libz.a"]}}
            )
        )
    with pytest.raises(ValueError, match="escapes merged sysroot"):
        build_extension(
            ExtensionBuildRequest(
                context, request.cpython, {**recipe, "build": {**recipe["build"], "link_inputs": ["../host.a"]}}
            )
        )
    with pytest.raises(ValueError, match="admitted graph"):
        build_extension(ExtensionBuildRequest(request.context, request.cpython, recipe))
    assert not context.build.exists()
    assert not context.staging_prefix.exists()
