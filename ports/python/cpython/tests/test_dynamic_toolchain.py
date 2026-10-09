"""Check reviewed build inputs and the separation of main and side runtimes."""

import json
from pathlib import Path

import pytest

from ports._support.build import check_build_scripts
from ports.native.dependencies import file_hash
from ports.python.cpython.dynamic import build


def test_dynamic_toolchain_pins_inputs_and_imports_canonical_runtime():
    directory = Path(__file__).parents[4] / "ports/toolchain/wasi_sdk"
    recipe = json.loads((directory / "recipe.json").read_text())
    check_build_scripts(recipe, directory)
    for notice in recipe["notices"]:
        assert file_hash(directory / notice["file"]) == notice["sha256"]
        assert (
            recipe["runtime_sources"]["llvm"]["commit"] in notice["url"]
            or recipe["runtime_sources"]["wasi-libc"]["commit"] in notice["url"]
        )
    assert recipe["abi"] == "shellsim-wasi-sdk34-cpython3137-v2"
    assert recipe["loader_namespace"] == "shellsim_dylink_v2"
    assert "-nostdlib" in recipe["side_link_flags"]
    assert "-fPIC" in recipe["side_link_flags"]
    assert any("import-dynamic" in flag for flag in recipe["side_link_flags"])
    assert set(recipe["runtime_archives"]) == {
        "../libc.a",
        "../libc-printscan-long-double.a",
        "libc++.a",
        "libc++abi.a",
        "libunwind.a",
    }


@pytest.mark.parametrize(
    ("profile", "providers"),
    [("wasi-cpython-v1", []), ("wasi-cpython-v2", [{"name": "numpy"}])],
)
def test_dynamic_builder_requires_fixed_bare_sdk34_interpreter(tmp_path, profile, providers):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"recipe": {"target_profile": profile}, "native_ports": providers})
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        build(bundle, output)
    assert not output.exists()


def test_production_builder_emits_runtime_without_fixture_artifacts(tmp_path, monkeypatch):
    from ports.python.cpython import dynamic

    bundle = tmp_path / "base"
    guest = bundle / "wasi-build"
    guest.mkdir(parents=True)
    source = bundle / "Python-3.13.7/Modules"
    source.mkdir(parents=True)
    (source / "posixmodule.c").write_text("upstream source")
    (guest / "Modules").mkdir()
    (guest / "Modules/posixmodule.o").write_bytes(b"base object")
    (bundle / "rootfs/usr/bin").mkdir(parents=True)
    (bundle / "rootfs/usr/bin/python3.wasm").write_bytes(b"static interpreter")
    (bundle / "manifest.json").write_text(
        json.dumps({"recipe": {"target_profile": "wasi-cpython-v2"}, "native_ports": []})
    )
    libc = tmp_path / "libc.a"
    libc.write_bytes(b"canonical libc")
    recipe = {"abi": "test-abi", "loader_namespace": "test-loader", "main_link_flags": [], "notices": []}
    monkeypatch.setattr(dynamic, "dynamic_toolchain", lambda sdk: (recipe, {}, [libc]))
    monkeypatch.setattr(dynamic, "target_environment", lambda sdk: {})
    monkeypatch.setattr(
        dynamic, "target_profile", lambda recipe: {"compiler_flags": [], "cpp_flags": [], "link_flags": []}
    )
    commands = []

    def run(command, cwd=None, env=None):
        commands.append(command)
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"compiled runtime")

    def output(command, **kwargs):
        if "llvm-nm" in command[0]:
            return "malloc T 0 1\n"
        if command[-1] == "shellsim_posix_compile":
            return "clang -c posixmodule.c"
        return "clang Programs/python.o Modules/posixmodule.o"

    monkeypatch.setattr(dynamic, "run_command", run)
    monkeypatch.setattr(dynamic.subprocess, "check_output", output)
    destination = tmp_path / "runtime"
    build(bundle, destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert set(manifest["runtime_sources"]) == {"dynamic.c"}
    assert "proof_artifacts" not in manifest
    assert "fixture_sources" not in manifest
    assert not list(destination.glob("*.so"))
    assert manifest["files"]["/usr/bin/python3.wasm"] == file_hash(destination / "python3.wasm")
    assert all("tests/fixtures" not in str(argument) for command in commands for argument in command)
