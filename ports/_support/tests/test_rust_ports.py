"""Exercise the real link adapter and graph resource wiring without a compiler."""

import json
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ports._support import runner
from ports._support.build_context import BuildContext
from ports._support.native_adapters import CompilerCacheLauncher, NativeBuildContext
from ports._support.python_adapters import CPythonBuildContext
from ports._support.python_rust import build_rust_extension, generated_file, linker_script
from ports._support.rust_tools import run_cargo, rust_environment
from ports._support.store import file_hash, identity


def test_shared_generated_file_is_complete_and_executable_when_published(tmp_path, monkeypatch):
    from ports._support import python_rust

    path = tmp_path / "tools/linker"
    replace_file = python_rust.os.replace
    seen = []

    def publish(source, destination):
        assert destination == path
        assert not destination.exists()
        assert source.read_text() == "complete\n"
        assert source.stat().st_mode & 0o111
        seen.append(source)
        replace_file(source, destination)

    monkeypatch.setattr(python_rust.os, "replace", publish)
    generated_file(path, "complete\n", executable=True)
    generated_file(path, "complete\n", executable=True)
    assert len(seen) == 1 and path.read_text() == "complete\n"
    assert not seen[0].exists()


@pytest.mark.parametrize("packaged", [False, True])
def test_rust_wheel_contains_extension_metadata_licenses_and_record(tmp_path, monkeypatch, packaged):
    # The command is mocked; this directory holds only generated text fixtures.
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "generated"))
    source = tmp_path / "source"
    source.mkdir()
    name = "example-project" if packaged else "example"
    dist = name.replace("-", "_") + "-1.0.dist-info"
    module = "example._native" if packaged else "example"
    extension_path = module.replace(".", "/") + ".so"
    metadata = "Metadata-Version: 2.1\nName: " + name.replace("-", "_") + "\nVersion: 1.0\n\nRust example\n"
    (source / "PKG-INFO").write_text(metadata)
    (source / "LICENSE").write_text("upstream license\n")
    (source / "example.pyi").write_text("def value() -> int: ...\n")
    if packaged:
        package = source / "python/example"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("from ._native import value\n")
        (package / "py.typed").touch()
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
    recipe = {"name": name, "version": "1.0", "requires_dist": []}
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

    configurations = []

    def compile(command, directory, environment, log):
        assert "--locked" in command and "--offline" in command
        assert command[command.index("--manifest-path") + 1] == str(
            source / ("src/rust/Cargo.toml" if packaged else "Cargo.toml")
        )
        assert environment["OPENSSL_STATIC"] == "1"
        assert directory == source
        assert str(stdlib) in environment["RUSTFLAGS"]
        assert environment["RUSTC_WRAPPER"] == "/usr/bin/sccache"
        assert environment["SCCACHE_BUCKET"] == "cache"
        configurations.append((environment["PYO3_CONFIG_FILE"], environment["RUSTFLAGS"]))
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

    def build(selected):
        return build_rust_extension(
            selected,
            module=module,
            python_sources=(("python/example", "example"),) if packaged else (),
            licenses=("LICENSE",),
            stubs=("example.pyi",),
            manifest_path="src/rust/Cargo.toml" if packaged else "Cargo.toml",
            configure_environment={"OPENSSL_STATIC": "1"},
        )

    output = build(context)
    wheel = output.staging_prefix / ("wheels/" + name.replace("-", "_") + "-1.0-cp313-cp313-wasm32_wasip1.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read(extension_path).startswith(b"\0asm")
        assert archive.read(dist + "/METADATA").decode() == metadata
        assert archive.read("example.pyi").decode() == "def value() -> int: ...\n"
        if packaged:
            assert archive.read("example/__init__.py").decode() == "from ._native import value\n"
            assert archive.read("example/py.typed") == b""
        assert archive.read(dist + "/licenses/Rust-LICENSE-MIT") == b"Rust license\n"
        manifest = json.loads(archive.read(dist + "/shellsim-native.json"))
        assert manifest["cpython_receipt"] == python.receipt
        assert manifest["artifacts"][0]["path"] == extension_path
        assert manifest["artifacts"][0]["sha256"] == file_hash(native.build / "wheel" / extension_path)
        record = archive.read(dist + "/RECORD").decode()
        assert extension_path + ",sha256=" in record and "example.pyi,sha256=" in record
    build(replace(context, native=replace(native, build=tmp_path / "retry", staging_prefix=tmp_path / "retry-install")))
    assert configurations[0] == configurations[1]


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
        (directory / "build.py").write_text(
            "def build(ctx):\n    assert ctx.offline\n    if ctx.port.name == 'consumer':\n        assert set(ctx.build_artifacts) == {'toolchain/producer'}\n        body=(ctx.build_artifacts['toolchain/producer']/'product').read_bytes()+b' consumed'\n    else:\n        assert not ctx.build_artifacts\n        body=b'compiled'\n    (ctx.result/'product').write_bytes(body)\n"
        )
    first = runner.build_graph(ports, ["toolchain/consumer"], None, store, offline=True)
    assert (first.results["toolchain/consumer/recipe.json"] / "product").read_bytes() == b"compiled consumed"
    second = runner.build_graph(ports, ["toolchain/consumer"], None, store, offline=True)
    assert second.results == first.results
