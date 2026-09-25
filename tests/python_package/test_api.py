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


def test_direct_python_source_avoids_shell_quoting_and_preserves_argv_stdin() -> None:
    result = shellsim.python.run(
        'import sys\nprint(sys.argv)\nprint(input())\nprint(sys.stdin.read())\nprint("\'\\"$()")',
        argv=["one", "two words"],
        stdin=b"input\nrest",
    )

    assert result.returncode == 0
    assert result.stdout == b"['-c', 'one', 'two words']\ninput\nrest\n'\"$()\n"
    assert result.commands == ("python3.14",)
    assert result.partial_commands == ("python3.14",)
    assert result.invocations[0].argv[:2] == ("python3.14", "-c")


def test_environment_direct_python_reuses_vfs_and_cumulative_resources() -> None:
    environment = shellsim.Environment()
    environment.write_file("/work/helper.py", "answer = 42\n")
    environment.run("cd /work")

    first = environment.run_python("from helper import answer\nprint(answer)")
    second = environment.run_python("print('again')")

    assert first.stdout == b"42\n"
    assert second.stdout == b"again\n"
    assert second.usage.cpu_used > first.usage.cpu_used


def test_direct_python_uses_the_modeled_process_scheduler() -> None:
    result = shellsim.python.run(
        'import subprocess\nprint(subprocess.run(["printf", "child"], capture_output=True).stdout)'
    )

    assert result.returncode == 0
    assert result.stdout == b"b'child'\n"
    assert [invocation.argv[0] for invocation in result.invocations] == ["python3.14", "printf"]


def test_direct_python_input_raises_catchable_eof() -> None:
    result = shellsim.python.run('try:\n    input()\nexcept EOFError:\n    print("caught")')

    assert result.returncode == 0
    assert result.stdout == b"caught\n"


def test_direct_python_stdin_is_utf8_text_and_iterable() -> None:
    source = "import sys\nprint(sys.stdin.read(1))\nfor line in sys.stdin:\n    print(repr(line))"
    result = shellsim.python.run(source, stdin="éa\nb\n".encode())

    assert result.returncode == 0
    assert result.stdout == "é\n'a\\n'\n'b\\n'\n".encode()


def test_direct_python_rejects_binary_stdin_and_obeys_resource_limits() -> None:
    invalid = shellsim.python.run("import sys\nsys.stdin.read()", stdin=b"\xff")
    exhausted = shellsim.python.run("while True:\n    pass", cpu=100)

    assert invalid.returncode != 0
    assert b"standard input is not valid UTF-8" in invalid.stderr
    assert exhausted.returncode == 137
    assert exhausted.stop_reason == "cpu_exhausted"


def test_direct_python_validates_arguments() -> None:
    environment = shellsim.Environment()

    with pytest.raises(TypeError, match="source must be str"):
        environment.run_python(b"print(1)")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="argv must be a sequence of str"):
        environment.run_python("print(1)", "not-an-argv")
    with pytest.raises(TypeError, match="argv must be a sequence of str"):
        environment.run_python("print(1)", ["ok", 1])  # type: ignore[list-item]


def test_environment_preserves_state_and_vfs_bytes() -> None:
    environment = shellsim.Environment()
    environment.mkdir("/work/package")
    environment.write_file("/work/package/data", b"\x00\xffvalue")

    first = environment.run("value=41; cd /work/package")
    second = environment.run("printf '%s ' $((value + 1)); cat data")

    assert first.returncode == 0
    assert second.stdout == b"42 \x00\xffvalue"
    assert environment.read_file("data") == b"\x00\xffvalue"


def test_multiple_shell_sessions_share_files_but_not_shell_state() -> None:
    environment = shellsim.Environment()
    first = environment.create_shell()
    second = environment.create_shell()
    assert isinstance(first, shellsim.ShellSession)
    assert first.pid != second.pid

    assert first.run("cd /work; export LABEL=first; printf common > note").returncode == 0
    assert second.run("cd /tmp; export LABEL=second; cat /work/note").stdout == b"common"
    assert first.run("printf '%s:%s' \"$PWD\" \"$LABEL\"").stdout == b"/work:first"
    assert second.run("printf '%s:%s' \"$PWD\" \"$LABEL\"").stdout == b"/tmp:second"
    assert environment.run("printf '%s' \"${LABEL-unset}\"").stdout == b"unset"
    assert first.run("cat", stdin=b"session input").stdout == b"session input"

    assert first.run("exit 7").returncode == 7
    with pytest.raises(shellsim.SimulationError, match="not idle"):
        first.run("printf stale")


def test_environment_configures_static_http_routes() -> None:
    environment = shellsim.Environment(
        http={
            "https://api.test/items": shellsim.HttpResponse(
                status=201,
                headers={"Content-Type": "application/json"},
                body='{"id":7}',
            )
        }
    )

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


def test_static_http_routes_back_python_urllib() -> None:
    environment = shellsim.Environment(
        http={
            "https://api.test/items": shellsim.HttpResponse(
                headers={"Content-Type": "application/json"},
                body='{"items":[]}',
            )
        }
    )

    result = environment.run_python(
        "from urllib.request import urlopen\n"
        "with urlopen('https://api.test/items') as response:\n"
        "    print(response.status, response.getheader('content-type'), response.read())\n"
    )

    assert result.returncode == 0
    assert result.stdout == b"200 application/json b'{\"items\":[]}'\n"
    assert result.network_requests[0].matched


def test_http_routes_support_methods_globs_and_validation() -> None:
    environment = shellsim.Environment()
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
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "config").write_text("secret")
    (tmp_path / "helper.py").write_text("answer = 42\n")
    (tmp_path / "main.py").write_text("from helper import answer\nprint(answer)\n")
    environment = shellsim.Environment()

    report = environment.mount(tmp_path)
    result = environment.run("cd /work; python3.14 main.py")

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
