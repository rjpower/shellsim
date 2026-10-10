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


@pytest.mark.parametrize(
    "change", [{"output": "unknown"}, {"output": "stdlib", "cpython_include_directories": ["../host"]}]
)
def test_stdlib_extension_rejects_unsupported_output_and_header_escape(tmp_path, monkeypatch, change):
    request = _extension_request(tmp_path)
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("compiler ran before admission"))
    recipe = {**request.recipe, "version": request.cpython.version, "build": {**request.recipe["build"], **change}}
    with pytest.raises(ValueError):
        build_extension(ExtensionBuildRequest(request.context, request.cpython, recipe))
    assert not request.context.build.exists()
    assert not request.context.staging_prefix.exists()


def test_host_backend_wheel_preserves_launchers_without_guest_admission(tmp_path: Path) -> None:
    from ports._support.graph import plan
    from ports._support.python_adapters import build_host_wheel

    wheel = tmp_path / "example-1.0-py3-none-any.whl"
    recipe = _wheel(wheel)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("example/launcher.exe", b"MZ\0pinned launcher")
    recipe["source"]["sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    recipe.update({"role": "host-tool", "build": {"adapter": "host-wheel"}})
    stage = tmp_path / "host"
    build_host_wheel(PureWheelBuildRequest(wheel, stage, recipe))
    assert (stage / "wheels" / wheel.name).read_bytes() == wheel.read_bytes()
    with pytest.raises(ValueError):
        build_pure_wheel(PureWheelBuildRequest(wheel, tmp_path / "guest", recipe))
    root = tmp_path / "ports"
    provider = root / "python" / "example"
    provider.mkdir(parents=True)
    import json

    (provider / "recipe.json").write_text(json.dumps(recipe))
    with pytest.raises(ValueError):
        plan(root, ["python/example"])


@pytest.mark.parametrize("invalid", ["dependency", "data"])
def test_backend_output_rejects_changed_runtime_dependencies_and_relocation(tmp_path, invalid):
    from ports._support.python_pep517 import PEP517BuildRequest, _stage_wheel

    extension = _extension_request(tmp_path)
    request = PEP517BuildRequest(extension.context, extension.cpython, tmp_path / "sysconfig.py", extension.recipe, ())
    wheel = tmp_path / "example-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("example.py", "VALUE = 42\n")
        metadata = "Name: example\nVersion: 1.0\n"
        if invalid == "dependency":
            metadata += "Requires-Dist: unadmitted-runtime==1\n"
        archive.writestr("example-1.0.dist-info/METADATA", metadata)
        archive.writestr(
            "example-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        archive.writestr("example-1.0.dist-info/RECORD", "")
        if invalid == "data":
            archive.writestr("example-1.0.data/scripts/launch", "#!/bin/sh\n")
    stage = tmp_path / "wheel-root"
    stage.mkdir()
    with pytest.raises(ValueError, match="metadata differs|relocation is unsupported"):
        _stage_wheel(wheel, stage, request)
    assert not (stage / "example-1.0.dist-info/shellsim-native.json").exists()


@pytest.mark.parametrize(
    "table",
    ["build-system = []", "[build-system]\nbuild-backend = 42", "[build-system]\nrequires = []\nbuild-backend = 42"],
)
def test_malformed_backend_declaration_fails_before_host_execution(tmp_path, table, monkeypatch):
    from ports._support.python_pep517 import PEP517BuildRequest, build_pep517

    extension = _extension_request(tmp_path)
    (extension.context.source / "pyproject.toml").write_text(table)
    request = PEP517BuildRequest(extension.context, extension.cpython, tmp_path / "sysconfig.py", extension.recipe, ())
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("host backend ran before admission"))
    with pytest.raises(ValueError, match="build-system|build-backend"):
        build_pep517(request)
    assert not extension.context.build.exists()
