"""Installed-package contract tests for the native shellsim Python facade."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading

import pytest

import shellsim


def test_fresh_run_returns_bytes_and_structured_usage() -> None:
    result = shellsim.run("printf 'hello\\n'")

    assert result.returncode == 0
    assert result.stdout == b"hello\n"
    assert result.stderr == b""
    assert result.stdout_text == "hello\n"
    assert result.usage.cpu_used > 0
    assert result.cost_model_version == 1


def test_environment_preserves_state_and_vfs_bytes() -> None:
    environment = shellsim.Environment()
    environment.mkdir("/work/package")
    environment.write_file("/work/package/data", b"\x00\xffvalue")

    first = environment.run("value=41; cd /work/package")
    second = environment.run("printf '%s ' $((value + 1)); cat data")

    assert first.returncode == 0
    assert second.stdout == b"42 \x00\xffvalue"
    assert environment.read_file("data") == b"\x00\xffvalue"


def test_stdin_is_explicit_and_byte_preserving() -> None:
    assert shellsim.run("cat", b"\x00\xff\n").stdout == b"\x00\xff\n"


def test_fidelity_telemetry_exposes_noop_and_partial_commands() -> None:
    result = shellsim.run("pip install sample; sed -n '1p' /missing")

    assert "pip" in result.noop_commands
    assert "sed" in result.partial_commands
    assert {invocation.trust for invocation in result.invocations} >= {"no_op", "partial"}


def test_exhaustion_is_terminal_and_reuse_is_observable() -> None:
    environment = shellsim.Environment(cpu=200)
    exhausted = environment.run("while true; do :; done")
    reused = environment.run("printf unreachable")

    assert exhausted.returncode == 137
    assert exhausted.stop_reason == "cpu_exhausted"
    assert environment.terminated
    assert reused.returncode == exhausted.returncode
    assert reused.stop_reason == exhausted.stop_reason
    assert reused.stdout == b""
    assert reused.usage == exhausted.usage


def test_invalid_limits_are_rejected_before_native_conversion() -> None:
    with pytest.raises(TypeError, match="cpu must be int"):
        shellsim.Environment(cpu=True)
    with pytest.raises(ValueError, match="cpu must be between"):
        shellsim.Environment(cpu=-1)
    with pytest.raises(ValueError, match="cannot be combined"):
        shellsim.Environment(shellsim.Limits(), cpu=1)


def test_mount_reports_skips_and_runs_python_from_trusted_tree(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret")
    (tmp_path / "helper.py").write_text("answer = 42\n")
    (tmp_path / "main.py").write_text("from helper import answer\nprint(answer)\n")
    environment = shellsim.Environment()

    report = environment.mount(tmp_path)
    result = environment.run("cd /work; python3.14 main.py")

    assert report.files == 2
    assert report.skipped_directories == (".git",)
    assert result.stdout == b"42\n"
    with pytest.raises(shellsim.SimulationError, match="No such file"):
        environment.read_file("/work/.git/config")


def test_failed_mount_rolls_back_all_files(tmp_path) -> None:
    (tmp_path / "kept.txt").write_text("must roll back")
    try:
        (tmp_path / "link.txt").symlink_to("kept.txt")
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    environment = shellsim.Environment()

    with pytest.raises(shellsim.SimulationError, match="refusing host symlink"):
        environment.mount(tmp_path)
    with pytest.raises(shellsim.SimulationError, match="No such file"):
        environment.read_file("/work/kept.txt")


def test_deep_python_source_is_contained_by_the_worker_stack() -> None:
    environment = shellsim.Environment()
    source = "value = " + "(" * 300 + "1" + ")" * 300 + "\n"
    environment.write_file("/work/deep.py", source)

    result = environment.run("python3.14 /work/deep.py")

    assert result.returncode != 0
    assert result.stderr


def test_run_releases_the_python_interpreter_lock() -> None:
    started = threading.Event()
    stopped = threading.Event()
    counter = [0]

    def count() -> None:
        started.set()
        while not stopped.is_set():
            counter[0] += 1

    thread = threading.Thread(target=count)
    thread.start()
    assert started.wait(timeout=1)
    before = counter[0]
    try:
        result = shellsim.run("python3.14 -c 'for value in range(20000): value * value'")
    finally:
        stopped.set()
        thread.join(timeout=1)

    assert result.returncode == 0
    assert counter[0] > before


def test_environment_can_run_from_a_python_worker_thread() -> None:
    environment = shellsim.Environment()
    results = []

    thread = threading.Thread(target=lambda: results.append(environment.run("printf worker")))
    thread.start()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert results[0].stdout == b"worker"


def test_embedding_does_not_seccomp_the_host_process() -> None:
    assert shellsim.run("true").returncode == 0
    completed = subprocess.run(
        [sys.executable, "-c", "print('host subprocess')"],
        check=True,
        capture_output=True,
    )
    host_socket = socket.socket()
    host_socket.close()

    assert completed.stdout.splitlines() == [b"host subprocess"]


def test_check_returncode_raises_package_exception() -> None:
    result = shellsim.run("false")

    with pytest.raises(shellsim.SimulationError, match="status 1"):
        result.check_returncode()
