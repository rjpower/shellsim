"""Public, typed facade over shellsim's narrow native adapter."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from . import _native

SimulationError = _native.SimulationError

_MAX_U64 = (1 << 64) - 1


@dataclass(frozen=True)
class Limits:
    """Cumulative resource limits for one simulated environment."""

    cpu: int = 10_000_000
    memory: int = 64 * 1024 * 1024
    disk: int = 64 * 1024 * 1024
    output: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("cpu", "memory", "disk", "output"):
            _validate_limit(name, getattr(self, name))


@dataclass(frozen=True)
class Usage:
    """Cumulative resource usage after an action."""

    cpu_used: int
    memory_current: int
    memory_peak: int
    disk_current: int
    disk_peak: int
    output_bytes: int


@dataclass(frozen=True)
class CommandUsage:
    """Cumulative resource delta recorded for one completed command."""

    command: str
    cpu: int
    disk_delta: int


@dataclass(frozen=True)
class Invocation:
    """One command occurrence observed during an action."""

    sequence: int
    pid: int
    argv: Tuple[str, ...]
    trust: str
    status: Optional[int]
    cpu: Optional[int]
    disk_delta: Optional[int]
    unsupported_reason: Optional[str] = None


@dataclass(frozen=True)
class HttpRequest:
    """One bounded request record emitted by the simulated HTTP broker."""

    method: str
    url: str
    headers: Tuple[Tuple[str, str], ...]
    dropped_headers: int
    body_bytes: int
    matched: bool
    response_status: Optional[int]


@dataclass(frozen=True)
class MountResult:
    """Result of copying an explicitly trusted host tree into the VFS."""

    files: int
    skipped_directories: Tuple[str, ...]


@dataclass(frozen=True)
class HttpResponse:
    """One static response returned by the simulated HTTP broker."""

    status: int = 200
    headers: Union[Mapping[str, str], Sequence[Tuple[str, str]]] = ()
    body: Union[bytes, bytearray, memoryview, str] = b""


@dataclass(frozen=True)
class RunResult:
    """Byte-preserving output and structured fidelity/resource telemetry for one action."""

    returncode: int
    stdout: bytes
    stderr: bytes
    stop_reason: Optional[str]
    limits: Limits
    usage: Usage
    command_usage: Tuple[CommandUsage, ...]
    cost_model_version: int
    unsupported: Tuple[str, ...]
    dropped_unsupported: int
    commands: Tuple[str, ...]
    dropped_commands: int
    unsupported_commands: Tuple[str, ...]
    partial_commands: Tuple[str, ...]
    invocations: Tuple[Invocation, ...]
    dropped_invocations: int
    network_requests: Tuple[HttpRequest, ...]
    dropped_network_requests: int

    @property
    def stdout_text(self) -> str:
        """Decode stdout as UTF-8, replacing malformed sequences."""

        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        """Decode stderr as UTF-8, replacing malformed sequences."""

        return self.stderr.decode("utf-8", errors="replace")

    def check_returncode(self) -> None:
        """Raise `SimulationError` when the simulated action did not succeed."""

        if self.returncode != 0:
            diagnostic = self.stderr_text.strip()
            suffix = f": {diagnostic}" if diagnostic else ""
            raise SimulationError(f"shellsim action exited with status {self.returncode}{suffix}")


class Environment:
    """A persistent deterministic machine with an isolated in-memory filesystem.

    Resource limits and usage are cumulative. CPU, memory, or output exhaustion permanently
    terminates the environment; subsequent calls return the same terminal outcome without work.
    """

    def __init__(
        self,
        limits: Optional[Limits] = None,
        *,
        cpu: Optional[int] = None,
        memory: Optional[int] = None,
        disk: Optional[int] = None,
        output: Optional[int] = None,
        http: Optional[Mapping[str, HttpResponse]] = None,
    ) -> None:
        overrides = {"cpu": cpu, "memory": memory, "disk": disk, "output": output}
        if limits is not None and any(value is not None for value in overrides.values()):
            raise ValueError("limits cannot be combined with per-resource overrides")
        if limits is not None and not isinstance(limits, Limits):
            raise TypeError("limits must be a shellsim.Limits instance")
        resolved = limits or Limits(
            **{name: _validate_limit(name, value) for name, value in overrides.items() if value is not None}
        )
        self._native = _native.NativeEnvironment(
            resolved.cpu,
            resolved.memory,
            resolved.disk,
            resolved.output,
        )
        if http is not None:
            if not isinstance(http, Mapping):
                raise TypeError("http must be a mapping from URL patterns to shellsim.HttpResponse")
            for pattern, response in http.items():
                self.route_http(pattern, response)

    @property
    def terminated(self) -> bool:
        """Whether a terminal resource limit has stopped this environment."""

        return bool(self._native.terminated)

    def run(self, source: str, stdin: Union[bytes, bytearray, memoryview] = b"") -> RunResult:
        """Execute one complete shell action with an explicit input byte stream."""

        if not isinstance(source, str):
            raise TypeError("source must be str")
        metadata, stdout, stderr = self._native.run(source, _as_bytes("stdin", stdin))
        return _decode_result(metadata, stdout, stderr)

    def run_python(
        self,
        source: str,
        argv: Sequence[str] = (),
        stdin: Union[bytes, bytearray, memoryview] = b"",
    ) -> RunResult:
        """Execute Python source directly in this environment.

        The Python interpreter is fresh for each call. The simulated filesystem, cwd, resource
        usage, and terminal exhaustion state belong to the persistent environment.
        """

        if not isinstance(source, str):
            raise TypeError("source must be str")
        if isinstance(argv, (str, bytes, bytearray, memoryview)) or not isinstance(argv, Sequence):
            raise TypeError("argv must be a sequence of str")
        arguments = list(argv)
        if not all(isinstance(argument, str) for argument in arguments):
            raise TypeError("argv must be a sequence of str")
        metadata, stdout, stderr = self._native.run_python(
            source,
            arguments,
            _as_bytes("stdin", stdin),
        )
        return _decode_result(metadata, stdout, stderr)

    def write_file(
        self,
        path: str,
        data: Union[bytes, bytearray, memoryview, str],
        *,
        mode: int = 0o644,
    ) -> None:
        """Write exact bytes, or UTF-8 text, to one VFS path."""

        if not isinstance(path, str):
            raise TypeError("path must be str")
        if isinstance(data, str):
            encoded = data.encode()
        else:
            encoded = _as_bytes("data", data)
        if isinstance(mode, bool) or not isinstance(mode, int):
            raise TypeError("mode must be int")
        if not 0 <= mode <= 0o7777:
            raise ValueError("mode must be between 0 and 0o7777")
        self._native.write_file(path, encoded, mode)

    def read_file(self, path: str) -> bytes:
        """Read one VFS file without decoding its contents."""

        if not isinstance(path, str):
            raise TypeError("path must be str")
        return self._native.read_file(path)

    def mkdir(self, path: str, *, parents: bool = False) -> None:
        """Create one VFS directory, optionally including missing parents."""

        if not isinstance(path, str):
            raise TypeError("path must be str")
        if not isinstance(parents, bool):
            raise TypeError("parents must be bool")
        self._native.mkdir(path, parents)

    def route_http(
        self,
        pattern: str,
        response: HttpResponse,
        *,
        method: Optional[str] = None,
    ) -> None:
        """Register a static HTTP response for an exact URL or ``*`` glob.

        Requests are handled inside the simulator. This does not grant the environment DNS,
        sockets, TLS, or access to the host network. When ``method`` is omitted, the route matches
        every HTTP method.
        """

        if not isinstance(pattern, str):
            raise TypeError("pattern must be str")
        if not isinstance(response, HttpResponse):
            raise TypeError("response must be a shellsim.HttpResponse instance")
        if method is not None and not isinstance(method, str):
            raise TypeError("method must be str or None")
        if isinstance(response.status, bool) or not isinstance(response.status, int):
            raise TypeError("response status must be int")
        if not 0 <= response.status <= 0xFFFF:
            raise ValueError("response status must fit in an unsigned 16-bit integer")
        headers = _http_headers(response.headers)
        body = response.body.encode() if isinstance(response.body, str) else _as_bytes("response body", response.body)
        self._native.route_http(pattern, method, response.status, headers, body)

    def mount(self, host_root: Union[str, os.PathLike[str]], destination: str = "/work") -> MountResult:
        """Copy an explicitly trusted host directory into the bounded VFS.

        The walk rejects symlinks and non-regular files, has a 10,000-file limit, and skips
        `.git`, `.venv`, `venv`, `target`, `node_modules`, and `__pycache__` directories. A failed
        import leaves the VFS unchanged. The trusted host tree must not be mutated concurrently.
        """

        root = os.fspath(host_root)
        if not isinstance(root, str):
            raise TypeError("host_root must resolve to a text path")
        if not isinstance(destination, str):
            raise TypeError("destination must be str")
        report = json.loads(self._native.mount(root, destination))
        return MountResult(
            files=report["files"],
            skipped_directories=tuple(report["skipped_directories"]),
        )


def run(
    source: str,
    stdin: Union[bytes, bytearray, memoryview] = b"",
    limits: Optional[Limits] = None,
    *,
    cpu: Optional[int] = None,
    memory: Optional[int] = None,
    disk: Optional[int] = None,
    output: Optional[int] = None,
    http: Optional[Mapping[str, HttpResponse]] = None,
) -> RunResult:
    """Execute one action in a fresh environment."""

    return Environment(
        limits,
        cpu=cpu,
        memory=memory,
        disk=disk,
        output=output,
        http=http,
    ).run(source, stdin)


def _validate_limit(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if not 0 <= value <= _MAX_U64:
        raise ValueError(f"{name} must be between 0 and {_MAX_U64}")
    return value


def _as_bytes(name: str, value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be bytes-like")
    return bytes(value)


def _http_headers(value: Any) -> list[Tuple[str, str]]:
    if isinstance(value, Mapping):
        headers = list(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        headers = list(value)
    else:
        raise TypeError("response headers must be a mapping or sequence of name/value pairs")
    normalized = []
    for header in headers:
        if not isinstance(header, Sequence) or isinstance(header, (str, bytes)) or len(header) != 2:
            raise TypeError("response headers must contain name/value pairs")
        name, item = header
        if not isinstance(name, str) or not isinstance(item, str):
            raise TypeError("response header names and values must be str")
        normalized.append((name, item))
    return normalized


def _decode_result(metadata_json: str, stdout: bytes, stderr: bytes) -> RunResult:
    metadata: Mapping[str, Any] = json.loads(metadata_json)
    outcome = metadata["outcome"]
    limits = Limits(**outcome["limits"])
    usage = Usage(**outcome["usage"])
    command_usage = tuple(CommandUsage(**item) for item in outcome["command_usage"])
    invocations = tuple(
        Invocation(
            sequence=item["sequence"],
            pid=item["pid"],
            argv=tuple(item["argv"]),
            trust=item["trust"],
            status=item["status"],
            cpu=item["cpu"],
            disk_delta=item["disk_delta"],
            unsupported_reason=item.get("unsupported_reason"),
        )
        for item in metadata["invocations"]
    )
    network_requests = tuple(
        HttpRequest(
            method=item["method"],
            url=item["url"],
            headers=tuple(tuple(header) for header in item["headers"]),
            dropped_headers=item["dropped_headers"],
            body_bytes=item["body_bytes"],
            matched=item["matched"],
            response_status=item["response_status"],
        )
        for item in metadata["network_requests"]
    )
    return RunResult(
        returncode=outcome["exit_status"],
        stdout=stdout,
        stderr=stderr,
        stop_reason=outcome["stop_reason"],
        limits=limits,
        usage=usage,
        command_usage=command_usage,
        cost_model_version=outcome["cost_model_version"],
        unsupported=tuple(metadata["unsupported"]),
        dropped_unsupported=metadata["dropped_unsupported"],
        commands=tuple(metadata["commands"]),
        dropped_commands=metadata["dropped_commands"],
        unsupported_commands=tuple(metadata["unsupported_commands"]),
        partial_commands=tuple(metadata["partial_commands"]),
        invocations=invocations,
        dropped_invocations=metadata["dropped_invocations"],
        network_requests=network_requests,
        dropped_network_requests=metadata["dropped_network_requests"],
    )
