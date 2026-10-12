"""Exercise build wiring on a small native project; qualify installed Wasm opt-in."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ports._support.graph import plan
from ports._support.native_adapters import CompilerCacheLauncher
from ports._support.producer_tools import digest
from ports.native.dependencies import seal_artifact
from ports.toolchain.llvm.compiler import configure_and_build
from ports.toolchain.llvm.guest import (
    HOST_TOOLS,
    _inventory,
    _prepare_guest_locked,
    _refresh_snapshot,
    _snapshot,
    _snapshot_tools,
    prepare_guest,
    validate_guest_abi,
)

PORT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("defect", [None, "table-export", "fixed-table", "memory-bound", "tls-protocol", "tls-export"])
def test_stripped_main_abi_rejects_runtime_frontiers(tmp_path, defect):
    def leb(value):
        output = bytearray()
        while value > 127:
            output.append((value & 127) | 128)
            value >>= 7
        return bytes(output) + bytes([value])

    def text(value):
        encoded = value.encode()
        return leb(len(encoded)) + encoded

    def section(kind, payload):
        return bytes([kind]) + leb(len(payload)) + payload

    exports = {
        "_start": 0,
        "wasi_thread_start": 0,
        "__wasm_init_tls": 0,
        "__wasm_apply_global_tls_relocs": 0,
        "__indirect_function_table": 1,
        "memory": 2,
        "__stack_pointer": 3,
        "__tls_base": 3,
        "__tls_size": 3,
        "__tls_align": 3,
        "errno": 3,
    }
    if defect == "table-export":
        del exports["__indirect_function_table"]
    if defect == "tls-export":
        del exports["errno"]
    memory = (
        leb(1) + text("env") + text("memory") + b"\x02\x03" + leb(256) + leb(4097 if defect == "memory-bound" else 4096)
    )
    table = b"\x01\x70" + (b"\x01\x01\x01" if defect == "fixed-table" else b"\x00\x01")
    exported = leb(len(exports)) + b"".join(text(name) + bytes([kind]) + leb(0) for name, kind in exports.items())
    protocol = text("shellsim.main-tls") + leb(1)
    classified = leb(1) + text("errno") + leb(0x100)
    dylink = text("dylink.0") + section(3, classified)
    if defect != "tls-protocol":
        dylink += section(129, protocol)
    path = tmp_path / "main.wasm"
    path.write_bytes(
        b"\0asm\x01\0\0\0" + section(2, memory) + section(4, table) + section(7, exported) + section(0, dylink)
    )
    if defect is None:
        validate_guest_abi(path)
    else:
        with pytest.raises(ValueError):
            validate_guest_abi(path)


def test_guest_compiler_graph_distinguishes_build_and_install_dependencies():
    graph = plan(PORT.parents[1], ["toolchain/llvm:guest"])
    clang = graph.ports[-1]
    assert clang.role == "guest-tool"
    assert {(edge.port, edge.kind) for edge in clang.dependencies} == {
        ("toolchain/llvm", "build"),
        ("toolchain/wasi_threads", "platform"),
        ("native/shellsim-posix", "target"),
        ("toolchain/wasi-development", "runtime"),
    }


def test_dependency_snapshot_rejects_modified_retained_bytes(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "retained"
    source.mkdir()
    (source / "header.h").write_text("original\n")
    inventory = _inventory(source)
    _snapshot(source, destination, inventory)
    _snapshot(source, destination, inventory)
    (destination / "header.h").write_text("changed\n")
    with pytest.raises(ValueError):
        _snapshot(source, destination, inventory)


def test_posix_snapshot_compatibility_uses_compilation_bytes(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "retained"
    (source / "include").mkdir(parents=True)
    (source / "include/process.h").write_text("original\n")
    (source / "artifact.json").write_text("first envelope\n")
    directories = ("include", "lib")
    inventory = _inventory(source, directories)
    _snapshot(source, destination, inventory, directories)
    (destination / "artifact.json").write_text("new envelope\n")
    _snapshot(source, destination, inventory, directories)
    (destination / "include/process.h").write_text("changed\n")
    with pytest.raises(ValueError):
        _snapshot(source, destination, inventory, directories)


def test_verified_library_update_preserves_snapshot_headers(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "retained"
    (source / "include").mkdir(parents=True)
    (source / "lib").mkdir()
    (source / "include/header.h").write_text("header\n")
    (source / "lib/libc.a").write_bytes(b"old archive")
    previous = _inventory(source)
    _snapshot(source, destination, previous)
    timestamp = (destination / "include/header.h").stat().st_mtime_ns
    (source / "lib/libc.a").write_bytes(b"new archive")
    _refresh_snapshot(source, destination, _inventory(source), previous)
    assert (destination / "lib/libc.a").read_bytes() == b"new archive"
    assert (destination / "include/header.h").stat().st_mtime_ns == timestamp


def test_admitted_linker_update_preserves_other_compiler_tools(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "retained"
    source.mkdir()
    (source / "wasm-ld").write_bytes(b"old linker")
    (source / "clang").write_bytes(b"same compiler")
    previous = _inventory(source)
    _snapshot_tools(source, destination, previous, previous)
    timestamp = (destination / "clang").stat().st_mtime_ns
    (source / "wasm-ld").write_bytes(b"new linker")
    current = _inventory(source)
    _snapshot_tools(source, destination, current, previous)
    assert (destination / "wasm-ld").read_bytes() == b"new linker"
    assert (destination / "clang").stat().st_mtime_ns == timestamp


def test_guest_build_uses_cache_for_c_and_cxx_with_requested_parallelism(tmp_path):
    cmake, ninja, cc, cxx = (shutil.which(name) for name in ("cmake", "ninja", "cc", "c++"))
    if not all((cmake, ninja, cc, cxx)):
        pytest.skip("requires CMake, Ninja and native C/C++ compilers")
    source = tmp_path / "upstream/llvm"
    source.mkdir(parents=True)
    (source / "main.c").write_text("int main(void) { return 17; }\n")
    (source / "main.cpp").write_text("int main() { return 23; }\n")
    # Use real CMake launcher semantics without building LLVM or a host SDK.
    # The fixture changes only the target platform and project size.
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(cache_probe NONE)\n"
        "set(CMAKE_SYSTEM_NAME Linux)\nset(CMAKE_SYSTEM_PROCESSOR native)\n"
        'set(CMAKE_SYSROOT "")\nset(CMAKE_C_COMPILER_TARGET "")\nset(CMAKE_CXX_COMPILER_TARGET "")\n'
        'set(CMAKE_C_FLAGS "")\nset(CMAKE_CXX_FLAGS "")\nset(CMAKE_EXE_LINKER_FLAGS "")\n'
        "enable_language(C CXX)\nadd_executable(clang main.c)\nadd_executable(lld main.cpp)\n"
        "add_custom_target(llvm-ar DEPENDS clang)\nadd_custom_target(llvm-nm DEPENDS lld)\n"
        "add_custom_target(llvm-objcopy DEPENDS clang lld)\n"
    )
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as output:
        output.add(source.parent, arcname="upstream")
    host = tmp_path / "host"
    (host / "bin").mkdir(parents=True)
    for name in (*HOST_TOOLS, "llvm-strip"):
        path = host / "bin" / name
        command = {"clang": cc, "clang++": cxx}.get(name, "/usr/bin/true")
        path.write_text(f'#!/bin/sh\nexec {shlex.quote(command)} "$@"\n')
        path.chmod(0o755)
    (host / "bin/clang").rename(host / "bin/clang-23")
    (host / "bin/clang").symlink_to("clang-23")
    compiler = SimpleNamespace(root=host, contents={"artifacts": _inventory(host)}, sha256="admitted compiler")
    posix = tmp_path / "posix"
    posix.mkdir()
    seal_artifact(posix, {"recipe": {"exports": {}}})
    sysroot, sdk = tmp_path / "sysroot", tmp_path / "sdk"
    sysroot.mkdir()
    (sdk / "lib/clang/23").mkdir(parents=True)
    launcher = tmp_path / "cache-launcher"
    launcher.write_text(
        f"#!{sys.executable}\nimport json, os, subprocess, sys\n"
        "with open(os.environ['SCCACHE_DIR'], 'a') as output:\n"
        "    output.write(json.dumps({'args': sys.argv[1:], 'environment': "
        "{k: v for k, v in os.environ.items() if k.startswith('SCCACHE_')}}) + '\\n')\n"
        "sys.exit(subprocess.call(sys.argv[1:]))\n"
    )
    launcher.chmod(0o755)
    cache_log = tmp_path / "cache.log"
    work = tmp_path / "work"
    context = SimpleNamespace(
        source=archive,
        build=work / "build",
        target="wasm32-wasip1-threads",
        dependencies={"native/shellsim-posix": posix},
        sysroot=sysroot,
        sdk=sdk,
        host_tools={"cmake": Path(cmake), "ninja": Path(ninja)},
        compiler_cache=CompilerCacheLauncher(
            launcher,
            digest(launcher),
            (("SCCACHE_DIR", str(cache_log)), ("SCCACHE_BUCKET", "shared"), ("SCCACHE_BASEDIRS", "/shared/inputs")),
        ),
    )
    recipe = dict(plan(PORT.parents[1], ["toolchain/llvm:guest"]).ports[-1].recipe)
    recipe.update(source={"sha256": digest(archive)}, patches=[])
    prepared = prepare_guest(context, recipe, compiler, jobs=128)
    assert not (context.build / "clang").exists()
    toolchain = work / "guest-toolchain.cmake"
    timestamp = toolchain.stat().st_mtime_ns
    prepared = _prepare_guest_locked(context, recipe, compiler, work, 128)
    assert toolchain.stat().st_mtime_ns == timestamp
    configure_and_build(
        prepared.commands,
        context.build,
        prepared.state,
        work / "workspace.json",
        work / "attempts" / prepared.prefix.name,
        prepared.environment,
    )
    assert subprocess.run([str(context.build / "clang")], check=False).returncode == 17
    assert subprocess.run([str(context.build / "lld")], check=False).returncode == 23
    calls = [json.loads(line) for line in cache_log.read_text().splitlines()]
    assert {Path(call["args"][0]).name for call in calls} >= {"clang", "clang++"}
    for call in calls:
        assert call["environment"]["SCCACHE_BUCKET"] == "shared"
        assert call["environment"]["SCCACHE_CLIENT_SIDE"] == "1"
        assert call["environment"]["SCCACHE_BASEDIRS"].split(os.pathsep) == [
            str(work / "source"),
            str(context.build),
            str(work / "inputs"),
            "/shared/inputs",
        ]
    # Full LLVM compilation is impractical here; Ninja's argv is the external
    # parallelism contract, independent of the fixture's number of source files.
    assert "-j128" in prepared.commands[1]
    alternate = tmp_path / "alternate-launcher"
    alternate.symlink_to(launcher)
    context.compiler_cache = CompilerCacheLauncher(alternate, digest(launcher), (("SCCACHE_BUCKET", "other"),))
    changed_cache = _prepare_guest_locked(context, recipe, compiler, work, 128)
    assert changed_cache.prefix == prepared.prefix
    context.compiler_cache = None
    uncached = _prepare_guest_locked(context, recipe, compiler, work, 128)
    assert uncached.prefix == prepared.prefix


def test_installed_guest_compiler_and_make_from_catalog():
    import shellsim

    catalog = os.environ.get("SHELLSIM_GUEST_CATALOG")
    if not catalog:
        pytest.skip("set SHELLSIM_GUEST_CATALOG to an exported guest LLVM and make catalog")
    environment = shellsim.Environment(
        tools=[
            "clang",
            "clang++",
            "lld",
            "wasm-ld",
            "cc",
            "c++",
            "ar",
            "ranlib",
            "llvm-ar",
            "llvm-ranlib",
            "llvm-nm",
            "make",
        ],
        pypi_catalog=catalog,
        cpu=500_000_000_000,
        memory=8 * 1024**3,
        disk=384 * 1024**2,
    )
    environment.write_file("/work/guest-compiler.sh", (PORT / "tests/guest-compiler.sh").read_bytes())
    for name in ("runtime_ctors_main.cpp", "runtime_ctors_side.cpp"):
        environment.write_file("/work/" + name, (PORT / "tests/fixtures" / name).read_bytes())
    compiled = environment.run("sh /work/guest-compiler.sh")
    assert compiled.returncode == 0, compiled.stderr
    assert compiled.stdout.endswith(b"guest compiler acceptance passed\n")
    environment.write_file(
        "/work/Makefile",
        b"all: make-c.wasm make-cpp.wasm\n"
        b"make-value.o: value.c\n\tcc -c $< -o $@\n"
        b"make-c.wasm: main.c make-value.o\n\tcc $^ -o $@\n"
        b"make-cpp.wasm: cpp.cpp\n\tc++ $< -o $@\n",
    )
    made = environment.run(
        "cd /work && make -j2 && make -q && chmod +x make-c.wasm make-cpp.wasm && ./make-c.wasm && ./make-cpp.wasm"
    )
    assert made.returncode == 0, made.stderr
    assert made.stdout.endswith(b"guest C 42\nguest C++ 42\n")
