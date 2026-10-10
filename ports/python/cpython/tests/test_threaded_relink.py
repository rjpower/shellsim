"""Verify that relinking admits sealed compile inputs rather than loose objects."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ports.native.dependencies import digest
from ports.python.cpython.threaded import (
    admit_process_refresh,
    compile_process_facades,
    compile_receipt,
    final_link,
    relink,
)


def test_compile_receipt_changes_for_objects_sources_and_generated_config(tmp_path):
    work = tmp_path / "work"
    source = work / "Python-3.13.7"
    source.mkdir(parents=True)
    (source / "module.c").write_bytes(b"source")
    (work / "process-source").mkdir()
    guest = work / "wasi-build"
    guest.mkdir()
    (guest / "module.o").write_bytes(b"object")
    (guest / "pyconfig.h").write_bytes(b"config")
    (tmp_path / "sysroot/include").mkdir(parents=True)
    receipt = compile_receipt(work, ["module.o"], tmp_path / "sysroot")
    for path in (source / "module.c", guest / "module.o", guest / "pyconfig.h"):
        original = path.read_bytes()
        path.write_bytes(b"changed")
        assert compile_receipt(work, ["module.o"], tmp_path / "sysroot") != receipt
        path.write_bytes(original)


def test_historical_runtime_without_compile_receipt_cannot_relink(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    profile = {"headers": {}}
    (previous / "manifest.json").write_text(
        json.dumps({"build_profile": profile, "build_profile_sha256": digest(profile)})
    )
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="sealed compile-input receipt"):
        relink(previous, {}, {}, {}, Path("sdk"), Path("sysroot"), Path("llvm"), output, {})
    assert not output.exists()


def test_corrected_platform_threads_errno_and_subprocess(guest_factory):
    guest = guest_factory(bundle_env="SHELLSIM_RELINKED_CPYTHON_BUNDLE", cpu=10_000_000_000)
    result = guest.run_script(Path(__file__).parent / "probes/threaded_process_errno.py")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"threaded CPython: parent/worker errno and subprocess passed\n"
    assert result.stderr == b""
    guest.assert_interpreter_unchanged()


def test_real_atfork_registry_order_and_spawn_separation(guest_factory):
    guest = guest_factory(bundle_env="SHELLSIM_RELINKED_CPYTHON_BUNDLE", cpu=10_000_000_000)
    result = guest.run_script(Path(__file__).parent / "probes/threaded_atfork.py")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"threaded CPython: real atfork registry ordering and spawn separation passed\n"
    assert result.stderr == b""
    guest.assert_interpreter_unchanged()


def test_final_link_normalizes_repeated_ceiling_and_preserves_other_options():
    command = ["clang", "-Wl,--shared-memory,--max-memory=67108864", "-Wl,--max-memory=67108864", "main.o"]
    assert final_link(command, 268435456) == ["clang", "-Wl,--shared-memory", "main.o", "-Wl,--max-memory=268435456"]
    with pytest.raises(ValueError, match="conflicting"):
        final_link([*command, "-Wl,--max-memory=131072"], 268435456)
    with pytest.raises(ValueError, match="malformed"):
        final_link(["clang", "-Wl,--max-memory=bad"], 268435456)


def test_threaded_main_allocates_and_touches_more_than_64_mib(guest_factory):
    guest = guest_factory(bundle_env="SHELLSIM_RELINKED_CPYTHON_BUNDLE", cpu=10_000_000_000)
    result = guest.run_script(Path(__file__).parent / "probes/threaded_memory.py")
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"threaded CPython: allocation beyond 64 MiB passed\n"
    assert result.stderr == b""
    guest.assert_interpreter_unchanged()


@pytest.mark.parametrize(
    "options",
    [
        ["-Wl,--max-memory,67108864"],
        ["-Xlinker", "--max-memory=67108864"],
        ["-Xlinker", "--max-memory", "-Xlinker", "67108864"],
    ],
)
def test_final_link_rejects_noncanonical_ceiling_spelling(options):
    with pytest.raises(ValueError):
        final_link(["clang", *options], 268435456)


def test_facade_refresh_recompiles_real_process_objects_and_preserves_core(tmp_path):
    cc, make = shutil.which("cc"), shutil.which("make")
    if cc is None or make is None:
        pytest.skip("requires a native C compiler and Make")
    directory = tmp_path / "ports/python/cpython"
    process = tmp_path / "ports/toolchain/wasi_process"
    sdk = tmp_path / "ports/toolchain/wasi_sdk"
    for path in (directory, process, sdk):
        path.mkdir(parents=True)
    header = process / "process_abi.h"
    header.write_text("#define LIMIT 1\n")
    (process / "process_port.h").write_text("")
    (sdk / "posix.h").write_text("")
    (directory / "threaded_posix.c").write_text("int posix_facade(void) { return 0; }\n")
    (process / "process.c").write_text('#include "process_abi.h"\nint facade(void) { return LIMIT; }\n')
    work = tmp_path / "work"
    guest = work / "wasi-build"
    guest.mkdir(parents=True)
    source = work / "Python-3.13.7/Modules"
    staged = work / "process-source/patched-source/Modules"
    source.mkdir(parents=True)
    staged.mkdir(parents=True)
    for name in ("posixmodule", "signalmodule", "faulthandler"):
        text = f"int {name}(void) {{ return 0; }}\n"
        (source / (name + ".c")).write_text(text)
        (staged / (name + ".c")).write_text(text)
    (guest / "Makefile").write_text(
        "CC=/obsolete/compiler\nsrcdir=/obsolete/original-source\n"
        "Makefile: missing-config-input\n\tfalse\n"
        "missing-config-input:\n\tfalse\n"
    )
    core = guest / "core.c"
    core.write_text("int facade(void); int main(void) { return facade(); }\n")
    core_object = guest / "core.o"
    subprocess.run([cc, "-c", str(core), "-o", str(core_object)], check=True)
    before = core_object.read_bytes(), core_object.stat().st_mtime_ns
    environment = dict(os.environ)
    objects = compile_process_facades(work, Path(make), [cc], environment, directory)
    program = work / "program"
    subprocess.run([cc, str(core_object), *(str(path) for path in objects.values()), "-o", str(program)], check=True)
    assert subprocess.run([str(program)]).returncode == 1
    old = {"../../toolchain/wasi_process/process_abi.h": "old", "dynamic.c": "unchanged"}
    new = {**old, "../../toolchain/wasi_process/process_abi.h": "new"}
    with pytest.raises(ValueError):
        admit_process_refresh(old, new, False)
    admit_process_refresh(old, new, True)
    header.write_text("#define LIMIT 2\n")
    objects = compile_process_facades(work, Path(make), [cc], environment, directory)
    subprocess.run([cc, str(core_object), *(str(path) for path in objects.values()), "-o", str(program)], check=True)
    assert subprocess.run([str(program)]).returncode == 2
    assert (core_object.read_bytes(), core_object.stat().st_mtime_ns) == before
    with pytest.raises(ValueError):
        admit_process_refresh(old, {**new, "dynamic.c": "changed"}, True)
