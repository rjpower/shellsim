"""Verify that relinking admits sealed compile inputs rather than loose objects."""

import json
from pathlib import Path

import pytest

from ports.native.dependencies import digest
from ports.python.cpython.threaded import compile_receipt, final_link, relink


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
