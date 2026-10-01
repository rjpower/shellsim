"""Command-line interface for the installed shellsim Python package."""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
from collections.abc import Sequence
from typing import Any, Optional

from ._api import Environment, Limits, RunResult, SimulationError
from .display_host import DisplayHost
from .package import Package

_MAX_U64 = (1 << 64) - 1


def _quantity(value: str) -> int:
    """Parse a non-negative count with an optional binary k, m, or g suffix."""

    suffixes = {"k": 1024, "m": 1024**2, "g": 1024**3}
    suffix = value[-1:].lower()
    multiplier = suffixes.get(suffix, 1)
    digits = value[:-1] if suffix in suffixes else value
    try:
        number = int(digits)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a count, optionally suffixed with k, m, or g") from error
    result = number * multiplier
    if number < 0 or result > _MAX_U64:
        raise argparse.ArgumentTypeError(f"expected a count between 0 and {_MAX_U64}")
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser shared by the console and module entry points."""

    parser = argparse.ArgumentParser(
        prog="shellsim",
        description="Run shell commands in a deterministic in-memory simulation.",
    )
    parser.add_argument("-c", "--command", metavar="SOURCE", help="execute one shell action")
    parser.add_argument(
        "--root",
        metavar="DIR",
        help="copy this trusted host directory into the simulated /work tree",
    )
    parser.add_argument("--cpu", type=_quantity, help="cumulative CPU fuel limit")
    parser.add_argument("--memory", type=_quantity, help="modeled memory limit")
    parser.add_argument("--disk", type=_quantity, help="simulated filesystem limit")
    parser.add_argument("--output", type=_quantity, help="cumulative output limit")
    return parser


def _binary_stream(stream: Any) -> Any:
    """Return a stream that accepts bytes, including under in-process test doubles."""

    return getattr(stream, "buffer", stream)


def _write_bytes(stream: Any, data: bytes) -> None:
    """Write exact simulator bytes to a host console stream."""

    binary = _binary_stream(stream)
    try:
        binary.write(data)
    except TypeError:
        binary.write(data.decode("utf-8", errors="replace"))
    binary.flush()


def _emit_result(result: RunResult, stdout: Any, stderr: Any) -> None:
    """Forward one simulated action's byte-preserving output."""

    _write_bytes(stdout, result.stdout)
    _write_bytes(stderr, result.stderr)


def _execute_action(
    environment: Environment,
    source: str,
    stdin: bytes,
    stdout: Any,
    stderr: Any,
) -> Optional[RunResult]:
    """Execute and forward one action, reporting adapter failures without a traceback."""

    try:
        result = environment.run(source, stdin)
    except SimulationError as error:
        print(f"shellsim: {error}", file=stderr)
        return None
    _emit_result(result, stdout, stderr)
    return result


def _read_bytes(stream: Any) -> bytes:
    """Read all remaining input without depending on a text-stream encoding."""

    data = _binary_stream(stream).read()
    return data.encode() if isinstance(data, str) else data


def _run_package(arguments: Sequence[str]) -> int:
    """Fetch a remote package on the host and run its entrypoint in a fresh guest."""

    parser = argparse.ArgumentParser(prog="shellsim run")
    parser.add_argument("url", help="HTTPS URL or local path of a .shl package")
    parser.add_argument("--sha256", help="expected SHA-256 of the complete package blob")
    for name in ("cpu", "memory", "disk", "output"):
        parser.add_argument(f"--{name}", type=_quantity, help=f"host {name} limit")
    options = parser.parse_args(arguments)
    try:
        if options.url.startswith("https://"):
            package = Package.from_url(options.url, expected_sha256=options.sha256)
        elif "://" in options.url:
            raise ValueError("package URL must use HTTPS")
        else:
            package = Package.from_path(options.url, expected_sha256=options.sha256)
        defaults = Limits(
            cpu=400_000_000_000, memory=512 * 1024 * 1024, disk=512 * 1024 * 1024, output=32 * 1024 * 1024
        )
        limits = Limits(
            **{
                name: getattr(options, name) if getattr(options, name) is not None else getattr(defaults, name)
                for name in ("cpu", "memory", "disk", "output")
            }
        )
        container = package.instantiate(tools={}, limits=limits, clock=package.spec.requested_clock)
        with container.start_entrypoint() as action:
            display: Optional[DisplayHost] = None
            try:
                while True:
                    event = action.poll()
                    _write_bytes(sys.stdout, event.stdout)
                    _write_bytes(sys.stderr, event.stderr)
                    if event.frame is not None:
                        if display is None and event.state != "complete":
                            display = DisplayHost(action)
                            display.__enter__()
                            print(f"shellsim display: {display.url}", file=sys.stderr)
                        if display is not None:
                            display.publish(event.frame)
                    if display is not None:
                        for code, pressed in display.drain_keys():
                            action.inject_key(code, pressed)
                        if display.stop_requested() and event.state != "complete":
                            action.stop()
                    if event.state == "complete":
                        if display is not None:
                            display.mark_stopped()
                        return event.returncode if event.returncode is not None else 1
                    if event.state == "blocked" and event.reason is not None:
                        time.sleep(0.002)
            finally:
                if display is not None:
                    display.__exit__(None, None, None)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, SimulationError, urllib.error.URLError) as error:
        print(f"shellsim: {error}", file=sys.stderr)
        return 2


def _prepare_environment(options: argparse.Namespace, stdout: Any, stderr: Any) -> tuple[Optional[Environment], int]:
    """Create a simulated machine, import an optional host snapshot, and enter `/work`."""

    environment = Environment(
        cpu=options.cpu,
        memory=options.memory,
        disk=options.disk,
        output=options.output,
    )
    try:
        if options.root is not None:
            environment.mount(options.root)
        prepared = environment.run("cd /work")
    except SimulationError as error:
        print(f"shellsim: {error}", file=stderr)
        return None, 2

    if prepared.returncode != 0:
        _emit_result(prepared, stdout, stderr)
        return None, prepared.returncode
    return environment, 0


def _interactive(environment: Environment, stdin: Any, stdout: Any, stderr: Any) -> int:
    """Run one action per input line while retaining simulated state."""

    status = 0
    while not environment.terminated:
        stdout.write("shellsim$ ")
        stdout.flush()
        try:
            line = stdin.readline()
        except KeyboardInterrupt:
            stdout.write("\n")
            stdout.flush()
            continue
        if not line:
            stdout.write("\n")
            stdout.flush()
            break
        result = _execute_action(environment, line, b"", stdout, stderr)
        if result is None:
            return 1
        status = result.returncode
    return status


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Run the package CLI and return the simulated or usage exit status."""

    parsed_arguments = list(sys.argv[1:] if arguments is None else arguments)
    if parsed_arguments[:1] == ["run"]:
        return _run_package(parsed_arguments[1:])
    if parsed_arguments[-1:] == ["--"]:
        parsed_arguments.pop()
    options = _parser().parse_args(parsed_arguments)
    environment, preparation_status = _prepare_environment(options, sys.stdout, sys.stderr)
    if environment is None:
        return preparation_status

    if options.command is not None:
        stdin = b"" if sys.stdin.isatty() else _read_bytes(sys.stdin)
        result = _execute_action(environment, options.command, stdin, sys.stdout, sys.stderr)
        if result is None:
            return 1
        return result.returncode

    if sys.stdin.isatty():
        return _interactive(environment, sys.stdin, sys.stdout, sys.stderr)

    source_bytes = _read_bytes(sys.stdin)
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        print(f"shellsim: stdin is not valid UTF-8 shell source: {error}", file=sys.stderr)
        return 2
    result = _execute_action(environment, source, b"", sys.stdout, sys.stderr)
    if result is None:
        return 1
    return result.returncode
