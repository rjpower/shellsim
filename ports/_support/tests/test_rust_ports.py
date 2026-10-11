"""Exercise the real link adapter and graph resource wiring without a compiler."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ports._support import runner
from ports._support.build_context import BuildContext
from ports._support.native_adapters import CompilerCacheLauncher, NativeBuildContext
from ports._support.python_adapters import CPythonBuildContext
from ports._support.python_rust import build_rust_extension, linker_script
from ports._support.rust_tools import run_cargo, rust_environment
from ports._support.store import file_hash, identity


def test_rust_wheel_contains_extension_metadata_licenses_and_record(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    metadata = "Metadata-Version: 2.1\nName: example\nVersion: 1.0\n\nRust example\n"
    (source / "PKG-INFO").write_text(metadata)
    (source / "LICENSE").write_text("upstream license\n")
    (source / "example.pyi").write_text("def value() -> int: ...\n")
    library = tmp_path / "example.wasm"
    library.write_bytes(b"\0asm\x01\0\0\0")
    stdlib = tmp_path / "stdlib"
    (stdlib / "licenses").mkdir(parents=True)
    (stdlib / "licenses/LICENSE-MIT").write_text("Rust license\n")
    scheduler = tmp_path / "scheduler"
    (scheduler / "bin").mkdir(parents=True)
    tool = scheduler / "bin/shellsim-wasm-scheduler"
    tool.write_text("#!" + sys.executable + "\nimport shutil, sys\nshutil.copyfile(sys.argv[1], sys.argv[2])\n")
    tool.chmod(0o755)
    native = NativeBuildContext(
        source=source,
        build=tmp_path / "build",
        staging_prefix=tmp_path / "install",
        sdk=tmp_path / "sdk",
        compiler_prefix=tmp_path / "compiler",
        sysroot=tmp_path / "sysroot",
        target="wasm32-wasip1-threads",
        compiler_flags=(),
        linker_flags=(),
        dependencies={},
        host_tools={},
        target_tools={},
        dependency_sysroot=tmp_path / "dependency-sysroot",
        shared_library_flags=(),
        abi="test-abi",
        linker=tmp_path / "wasm-ld",
        compiler_resource_directory=tmp_path / "compiler/lib/clang/23",
    )
    python = CPythonBuildContext(
        include_dir=tmp_path / "include",
        generated_config_dir=tmp_path / "config",
        version="3.13.7",
        abi="test-abi",
        runtime_target=native.target,
        wheel_platform="wasm32_wasip1",
        receipt={"version": "3.13.7"},
    )
    recipe = {"name": "example", "version": "1.0", "requires_dist": []}
    context = BuildContext(
        port=SimpleNamespace(recipe=recipe),
        source=source,
        result=native.staging_prefix,
        native=native,
        cpython=python,
        offline=True,
        compiler_cache=CompilerCacheLauncher(Path("/usr/bin/sccache"), "", (("SCCACHE_BUCKET", "cache"),)),
        build_artifacts={"toolchain/rust-stdlib": stdlib, "toolchain/wasm-scheduler": scheduler},
    )
    monkeypatch.setattr(
        "ports._support.python_rust.rust_tools", lambda *args: (Path("/rustc"), Path("/cargo"), None, None)
    )

    def compile(command, directory, environment, log):
        assert "--locked" in command and "--offline" in command
        assert directory == source
        assert str(stdlib) in environment["RUSTFLAGS"]
        assert environment["RUSTC_WRAPPER"] == "/usr/bin/sccache"
        assert environment["SCCACHE_BUCKET"] == "cache"
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {"crate_types": ["cdylib"]},
                    "filenames": [str(library)],
                }
            )
        )

    monkeypatch.setattr("ports._support.python_rust.run_cargo", compile)
    output = build_rust_extension(context, module="example", licenses=("LICENSE",), stubs=("example.pyi",))
    wheel = output.staging_prefix / "wheels/example-1.0-cp313-cp313-wasm32_wasip1.whl"
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read("example.so").startswith(b"\0asm")
        assert archive.read("example-1.0.dist-info/METADATA").decode() == metadata
        assert archive.read("example.pyi").decode() == "def value() -> int: ...\n"
        assert archive.read("example-1.0.dist-info/licenses/Rust-LICENSE-MIT") == b"Rust license\n"
        manifest = json.loads(archive.read("example-1.0.dist-info/shellsim-native.json"))
        assert manifest["cpython_receipt"] == python.receipt
        assert manifest["artifacts"][0]["sha256"] == file_hash(native.build / "wheel/example.so")
        record = archive.read("example-1.0.dist-info/RECORD").decode()
        assert "example.so,sha256=" in record and "example.pyi,sha256=" in record


def test_shared_link_uses_process_libc_and_discards_unused_exports(tmp_path):
    record = tmp_path / "args.json"
    linker = tmp_path / "linker"
    linker.write_text(
        "#!"
        + sys.executable
        + "\nimport json, sys\nfrom pathlib import Path\n"
        + "Path("
        + repr(str(record))
        + ").write_text(json.dumps(sys.argv[1:]))\n"
    )
    linker.chmod(0o755)
    wrapper = tmp_path / "wrapper"
    builtin = tmp_path / "builtins.a"
    wrapper.write_text(linker_script(linker, (builtin,)))
    subprocess.run(
        [
            sys.executable,
            str(wrapper),
            "-flavor",
            "wasm",
            "--export",
            "PyInit_example",
            "--export-memory",
            "--stack-first",
            "--max-memory=1073741824",
            "module.o",
            "-l",
            "c",
            "-lc",
            "-o",
            "example.so",
        ],
        check=True,
    )
    arguments = json.loads(record.read_text())
    assert arguments[-5:] == ["--export", "PyInit_example", "module.o", "-o", "example.so"]
    assert "--no-export-dynamic" in arguments
    assert "--defer-shared-init" in arguments
    assert "--serial-memory-init" in arguments
    assert str(builtin) in arguments
    assert not {
        "-flavor",
        "wasm",
        "--export-memory",
        "--stack-first",
        "--max-memory=1073741824",
        "-l",
        "c",
        "-lc",
    } & set(arguments)


def test_rust_environment_preserves_shared_caches_and_requested_jobs(monkeypatch):
    monkeypatch.setenv("CARGO_TARGET_DIR", "/shared/cargo")
    monkeypatch.setenv("RUSTUP_HOME", "/installed/rust")
    context = SimpleNamespace(
        jobs=64,
        compiler_cache=SimpleNamespace(path=Path("/usr/bin/sccache"), environment=(("SCCACHE_BUCKET", "cache"),)),
    )
    environment = rust_environment(context)
    assert environment["CARGO_TARGET_DIR"] == "/shared/cargo"
    assert environment["RUSTUP_HOME"] == "/installed/rust"
    assert environment["CARGO_BUILD_JOBS"] == "64"
    assert environment["RUSTC_WRAPPER"] == "/usr/bin/sccache"
    assert environment["SCCACHE_BUCKET"] == "cache"


def test_cargo_failure_keeps_its_diagnostic_tail(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_cargo(
            [sys.executable, "-c", "import sys; sys.stderr.write('x'*10000+'link failure');sys.exit(1)"],
            tmp_path,
            {},
            tmp_path / "cargo.log",
        )
    assert caught.value.returncode == 1
    assert caught.value.__notes__[0].endswith("link failure")
    assert len(caught.value.__notes__[0]) == 8192
    assert (tmp_path / "cargo.log").read_text().endswith("link failure")


def test_source_tree_builder_receives_only_declared_build_products(tmp_path, monkeypatch):
    ports = tmp_path / "ports"
    store = tmp_path / "store"
    for name in ("producer", "consumer"):
        directory = ports / "toolchain" / name
        directory.mkdir(parents=True)
        source = directory / "source.txt"
        source.write_text("source\n")
        entries = [{"path": f"toolchain/{name}/source.txt", "destination": "source.txt", "sha256": file_hash(source)}]
        recipe = {
            "name": name,
            "version": "1",
            "role": "host-tool",
            "build_system": "source-tree",
            "source": {"files": entries, "sha256": identity(entries)},
        }
        if name == "consumer":
            recipe["build_dependencies"] = [{"port": "toolchain/producer", "version": "1"}]
        (directory / "recipe.json").write_text(json.dumps(recipe))
    calls = []

    def build(context):
        calls.append(context.port.name)
        assert context.offline
        if context.port.name == "consumer":
            assert set(context.build_artifacts) == {"toolchain/producer"}
            data = (context.build_artifacts["toolchain/producer"] / "product").read_bytes()
            (context.result / "product").write_bytes(data + b" consumed")
        else:
            assert not context.build_artifacts
            (context.result / "product").write_bytes(b"compiled")

    monkeypatch.setattr("ports.api.build_port", build)
    first = runner.build_graph(ports, ["toolchain/consumer"], None, store, offline=True)
    assert (first.results["toolchain/consumer/recipe.json"] / "product").read_bytes() == b"compiled consumed"
    assert calls == ["producer", "consumer"]
    calls.clear()
    second = runner.build_graph(ports, ["toolchain/consumer"], None, store, offline=True)
    assert second.results == first.results
    assert not calls
