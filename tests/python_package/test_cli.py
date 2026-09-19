"""Installed-wheel tests for shellsim's console and module entry points."""

from __future__ import annotations

import io
import subprocess
import sys
import sysconfig
from importlib.metadata import distribution
from pathlib import Path

import pytest
from shellsim._cli import main


def run_cli(*arguments: str, stdin: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    """Run the module entry point through the interpreter that loaded the wheel."""

    return subprocess.run(
        [sys.executable, "-m", "shellsim", *arguments],
        input=stdin,
        capture_output=True,
        check=False,
    )


def test_distribution_exposes_shellsim_console_script() -> None:
    scripts = {
        entry.name: entry.value for entry in distribution("shellsim").entry_points if entry.group == "console_scripts"
    }

    assert scripts["shellsim"] == "shellsim._cli:main"
    executable = Path(sysconfig.get_path("scripts")) / ("shellsim.exe" if sys.platform == "win32" else "shellsim")
    assert executable.is_file()


def test_command_forwards_stdin_output_and_exit_status() -> None:
    completed = run_cli("-c", "cat; printf done >&2; exit 7", stdin=b"payload\x00\n")

    assert completed.returncode == 7
    assert completed.stdout == b"payload\x00\n"
    assert completed.stderr == b"done"


def test_piped_shell_source_runs_from_work() -> None:
    completed = run_cli(stdin=b"pwd; printf piped\n")

    assert completed.returncode == 0
    assert completed.stdout == b"/work\npiped"
    assert completed.stderr == b""


def test_terminal_without_arguments_opens_a_persistent_shell(monkeypatch) -> None:
    class TerminalInput(io.StringIO):
        def isatty(self) -> bool:
            return True

    stdin = TerminalInput("value=41\nprintf '%s\\n' $((value + 1))\n")
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    status = main([])

    assert status == 0
    assert "42\n" in stdout.getvalue()
    assert stdout.getvalue().count("shellsim$ ") == 3
    assert stderr.getvalue() == ""


def test_root_is_a_disposable_snapshot_at_work(tmp_path: Path) -> None:
    host_file = tmp_path / "value.txt"
    host_file.write_bytes(b"host\n")

    completed = run_cli(
        "--root",
        str(tmp_path),
        "-c",
        "cat value.txt; printf simulated > value.txt; cat value.txt",
    )

    assert completed.returncode == 0
    assert completed.stdout == b"host\nsimulated"
    assert completed.stderr == b""
    assert host_file.read_bytes() == b"host\n"


def test_trailing_separator_starts_the_default_shell(tmp_path: Path) -> None:
    (tmp_path / "value.txt").write_bytes(b"mounted\n")

    completed = run_cli("--root", str(tmp_path), "--", stdin=b"cat value.txt")

    assert completed.returncode == 0
    assert completed.stdout == b"mounted\n"
    assert completed.stderr == b""


def test_separator_does_not_accept_positional_arguments() -> None:
    completed = run_cli("--", "unexpected")

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert b"unrecognized arguments: -- unexpected" in completed.stderr


def test_root_rejects_host_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("host")
    link = tmp_path / "link"
    try:
        link.symlink_to(target.name)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    completed = run_cli("--root", str(tmp_path), "-c", "true")

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert b"refusing host symlink" in completed.stderr


def test_limit_suffixes_match_the_rust_cli() -> None:
    completed = run_cli("--cpu", "1k", "--memory", "1m", "-c", "printf bounded")

    assert completed.returncode == 0
    assert completed.stdout == b"bounded"
