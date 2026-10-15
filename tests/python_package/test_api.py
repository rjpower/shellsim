"""Installed-package contract tests for the native shellsim Python facade."""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import shellsim


def test_cancellation_from_another_thread_preserves_the_machine() -> None:
    context = shellsim.Context()
    environment = context.Environment(cpu=2**64 - 1)
    runner = environment.create_shell()
    assert runner.run("value=kept").returncode == 0
    token = shellsim.CancellationToken()
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(runner.run, "while :; do :; done", cancellation=token)
        token.cancel()
        result = pending.result()
    assert result.interrupted
    assert result.returncode == 130
    assert result.stop_reason is None
    assert not environment.terminated
    assert runner.run("printf '%s' \"$value\"").stdout == b"kept"
    assert context.Environment().run("printf independent").stdout == b"independent"
    with pytest.raises(shellsim.SimulationError):
        runner.run("printf unexpected", cancellation=token)


def test_cancelled_token_does_not_execute_or_cancel_later_actions() -> None:
    environment = shellsim.Context().Environment()
    token = shellsim.CancellationToken()
    token.cancel()
    result = environment.run("printf unexpected > /marker", cancellation=token)
    assert result.interrupted
    assert result.usage.cpu_used == 0
    assert environment.run("test ! -e /marker").returncode == 0
    completed = shellsim.CancellationToken()
    assert environment.run("printf completed", cancellation=completed).stdout == b"completed"
    completed.cancel()
    assert not completed.cancelled
    assert environment.run("printf later").stdout == b"later"


def test_fresh_run_returns_bytes_and_structured_usage() -> None:
    result = shellsim.run("printf 'hello\\n'")

    assert result.returncode == 0
    assert result.stdout == b"hello\n"
    assert result.stderr == b""
    assert result.stdout_text == "hello\n"
    assert result.usage.cpu_used > 0
    assert result.cost_model_version == 2














def test_direct_python_validates_arguments() -> None:
    environment = shellsim.Context().Environment()

    with pytest.raises(TypeError, match="source must be str"):
        environment.run_python(b"print(1)")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="argv must be a sequence of str"):
        environment.run_python("print(1)", "not-an-argv")
    with pytest.raises(TypeError, match="argv must be a sequence of str"):
        environment.run_python("print(1)", ["ok", 1])  # type: ignore[list-item]
    with pytest.raises(shellsim.SimulationError):
        environment.run_python("open('/work/unexpected', 'w').write('host fallback')")
    assert environment.run("python3 -c 'print(1)'").returncode == 127
    assert environment.run("test -e /work/unexpected").returncode != 0


def test_environment_preserves_state_and_vfs_bytes() -> None:
    environment = shellsim.Context().Environment()
    environment.mkdir("/work/package")
    environment.write_file("/work/package/data", b"\x00\xffvalue")

    first = environment.run("value=41; cd /work/package")
    second = environment.run("printf '%s ' $((value + 1)); cat data")

    assert first.returncode == 0
    assert second.stdout == b"42 \x00\xffvalue"
    assert environment.read_file("data") == b"\x00\xffvalue"


@pytest.mark.parametrize("operation", ("new-file", "updated-file", "directory", "parents", "mount"))
def test_host_vfs_mutations_use_current_virtual_time(operation: str, tmp_path: Path) -> None:
    environment = shellsim.Context().Environment()
    environment.write_file("/reference", b"before")
    if operation == "updated-file":
        environment.write_file("/node", b"before")
    before = int(environment.run("stat -c %Y /reference").stdout)
    assert environment.run("sleep 1").returncode == 0

    target = "/node"
    if operation in ("new-file", "updated-file"):
        environment.write_file(target, b"after")
    elif operation == "directory":
        environment.mkdir(target)
    elif operation == "parents":
        target = "/node/child"
        environment.mkdir(target, parents=True)
    else:
        (tmp_path / "input").write_bytes(b"after")
        environment.mount(tmp_path, target)
        target += "/input"

    observed = environment.run(f"stat -c %Y {target}")
    assert observed.returncode == 0, observed.stderr
    assert int(observed.stdout) == before + 1


def test_multiple_shell_sessions_share_files_but_not_shell_state() -> None:
    environment = shellsim.Context().Environment()
    first = environment.create_shell()
    second = environment.create_shell()
    assert first.pid != second.pid

    assert first.run("cd /work; export LABEL=first; printf common > note").returncode == 0
    assert second.run("cd /tmp; export LABEL=second; cat /work/note").stdout == b"common"
    assert first.run('printf \'%s:%s\' "$PWD" "$LABEL"').stdout == b"/work:first"
    assert second.run('printf \'%s:%s\' "$PWD" "$LABEL"').stdout == b"/tmp:second"
    assert environment.run("printf '%s' \"${LABEL-unset}\"").stdout == b"unset"
    assert first.run("cat", stdin=b"session input").stdout == b"session input"

    assert first.run("exit 7").returncode == 7
    with pytest.raises(shellsim.SimulationError, match="not idle"):
        first.run("printf stale")


def test_environment_configures_static_http_routes() -> None:
    environment = shellsim.Context().Environment(http={
            "https://api.test/items": shellsim.HttpResponse(
                status=201,
                headers={"Content-Type": "application/json"},
                body='{"id":7}',
            )
        })

    result = environment.run("curl -i -X POST https://api.test/items")

    assert result.returncode == 0
    assert result.stdout == b'HTTP/1.1 201 Created\r\nContent-Type: application/json\r\n\r\n{"id":7}'
    assert result.network_requests == (
        shellsim.HttpRequest(
            method="POST",
            url="https://api.test/items",
            headers=(),
            dropped_headers=0,
            body_bytes=0,
            matched=True,
            response_status=201,
        ),
    )




def test_http_routes_support_methods_globs_and_validation() -> None:
    environment = shellsim.Context().Environment()
    environment.route_http(
        "https://api.test/items/*",
        shellsim.HttpResponse(body=b"created"),
        method="POST",
    )

    assert environment.run("curl -d value https://api.test/items/1").stdout == b"created"
    assert environment.run("curl https://api.test/items/1").returncode == 7
    with pytest.raises(TypeError, match="shellsim.HttpResponse"):
        environment.route_http("https://api.test", {"body": "wrong"})  # type: ignore[arg-type]
    with pytest.raises(shellsim.SimulationError, match="invalid HTTP status"):
        environment.route_http("https://api.test", shellsim.HttpResponse(status=99))


def test_stdin_is_explicit_and_byte_preserving() -> None:
    assert shellsim.run("cat", b"\x00\xff\n").stdout == b"\x00\xff\n"


def test_fidelity_telemetry_exposes_unsupported_and_partial_commands() -> None:
    result = shellsim.run("npm install; sed -n '1p' /missing")

    assert "npm" in result.unsupported_commands
    assert "sed" in result.partial_commands
    assert {invocation.trust for invocation in result.invocations} >= {"unsupported", "partial"}


def test_exhaustion_is_terminal_and_reuse_is_observable() -> None:
    environment = shellsim.Context().Environment(cpu=200)
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
        shellsim.Context().Environment(cpu=True)
    with pytest.raises(ValueError, match="cpu must be between"):
        shellsim.Context().Environment(cpu=-1)
    with pytest.raises(ValueError, match="cannot be combined"):
        shellsim.Context().Environment(shellsim.Limits(), cpu=1)


def test_mount_reports_skips_and_runs_shell_from_trusted_tree(tmp_path) -> None:
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "config").write_text("secret")
    (tmp_path / "helper.sh").write_text("answer=42\n")
    (tmp_path / "main.sh").write_text('source /work/helper.sh; printf "%s\\n" "$answer"\n')
    environment = shellsim.Context().Environment()

    report = environment.mount(tmp_path)
    result = environment.run("cd /work; bash main.sh")

    assert report.files == 2
    assert report.skipped_directories == (".venv",)
    assert result.stdout == b"42\n"
    with pytest.raises(shellsim.SimulationError, match="No such file"):
        environment.read_file("/work/.venv/config")


def test_failed_mount_rolls_back_all_files(tmp_path) -> None:
    (tmp_path / "kept.txt").write_text("must roll back")
    try:
        (tmp_path / "link.txt").symlink_to("kept.txt")
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    environment = shellsim.Context().Environment()

    with pytest.raises(shellsim.SimulationError, match="refusing host symlink"):
        environment.mount(tmp_path)
    with pytest.raises(shellsim.SimulationError, match="No such file"):
        environment.read_file("/work/kept.txt")




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
        result = shellsim.run("i=0; while [ $i -lt 20000 ]; do i=$((i + 1)); done", cpu=100_000_000)
    finally:
        stopped.set()
        thread.join(timeout=1)

    assert result.returncode == 0
    assert counter[0] > before


def test_environment_can_run_from_a_python_worker_thread() -> None:
    environment = shellsim.Context().Environment()
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


def test_container_host_can_stage_outside_workspace() -> None:
    guest = shellsim.Container(tools={})
    guest.mkdir("/opt/toolchain/include")
    guest.write_file("/opt/toolchain/include/example.h", b"#define VALUE 42\n")

    assert guest.read_file("/opt/toolchain/include/example.h") == b"#define VALUE 42\n"
    result = guest.run("cat /opt/toolchain/include/example.h")
    assert result.stdout == b"#define VALUE 42\n"
    assert not result.interrupted
    with pytest.raises(shellsim.SimulationError):
        guest.write_file("/", b"not a file")
    assert guest.run("pwd").stdout == b"/work\n"
    with pytest.raises(shellsim.SimulationError):
        guest.write_file("/proc/blocked", b"not a host write")
