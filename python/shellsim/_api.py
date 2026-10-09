"""Public, typed facade over shellsim's narrow native adapter."""

from __future__ import annotations

import base64
import json
import os
import shlex
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from . import _native

if TYPE_CHECKING:
    from .cpython import CPythonRuntime

SimulationError = _native.SimulationError

ToolHandler = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class ToolError(Exception):
    """An intentional tool failure whose message may be returned to the guest."""


@dataclass
class _NativeInstallation:
    """Immutable native export identities owned by one environment."""

    artifacts: dict[str, str]
    files: dict[str, tuple[str, int, int]]


_MAX_U64 = (1 << 64) - 1


@dataclass(frozen=True)
class Limits:
    """Cumulative resource limits for one simulated environment."""

    cpu: int = 100_000_000
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


@dataclass(frozen=True)
class DisplayFrame:
    """One complete RGBA frame copied from the virtual display."""

    generation: int
    width: int
    height: int
    pixels: bytes


@dataclass(frozen=True)
class ActionPoll:
    """One bounded foreground-action step and any newly presented frame or output."""

    state: str
    frame: Optional[DisplayFrame]
    stdout: bytes
    stderr: bytes
    returncode: Optional[int]
    reason: Optional[Mapping[str, Any]]


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
        self._native_installation = _NativeInstallation({}, {})
        self._native_install_lock = threading.RLock()
        self._cpython_runtime: Optional[CPythonRuntime] = None
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

    def create_shell(self) -> ShellSession:
        """Create a persistent shell with private state and a shared virtual filesystem."""

        return ShellSession(self, self._native.create_shell())

    def run_python(
        self,
        source: str,
        argv: Sequence[str] = (),
        stdin: Union[bytes, bytearray, memoryview] = b"",
    ) -> RunResult:
        """Execute Python source directly in this environment.

        An explicitly mounted CPython bundle runs this source through its real WASI interpreter.
        Otherwise the shellsim Python VM runs it. Filesystem and resource state persist.
        """

        if not isinstance(source, str):
            raise TypeError("source must be str")
        if isinstance(argv, (str, bytes, bytearray, memoryview)) or not isinstance(argv, Sequence):
            raise TypeError("argv must be a sequence of str")
        arguments = list(argv)
        if not all(isinstance(argument, str) for argument in arguments):
            raise TypeError("argv must be a sequence of str")
        input_bytes = _as_bytes("stdin", stdin)
        if self._cpython_runtime is not None:
            return self._cpython_runtime.run(self, ["-c", source, *arguments], stdin=input_bytes)
        metadata, stdout, stderr = self._native.run_python(
            source,
            arguments,
            input_bytes,
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

    def install_pypi(self, requirement: Union[str, Sequence[str]]) -> None:
        """Install packages for the mounted CPython or the default shellsim Python VM.

        The default VM path resolves with host uv and can build source distributions
        on the host. A mounted dynamic CPython uses its explicit WASI universe.
        """
        if self._cpython_runtime is not None:
            self._cpython_runtime.install_pypi(self, requirement)
            return
        from .pypi import install_pypi

        install_pypi(self, requirement)

    def install_lock(
        self,
        lock: Union[str, os.PathLike[str]],
        *,
        extras: Sequence[str] = (),
        groups: Sequence[str] = (),
        project_mounted: bool = False,
    ) -> None:
        """Install a uv lock into the explicitly mounted CPython environment."""
        if self._cpython_runtime is None:
            raise SimulationError("install_lock requires a mounted CPython runtime")
        self._cpython_runtime.install_lock(self, lock, extras=extras, groups=groups, project_mounted=project_mounted)


class Container:
    """A shellsim machine whose guest can call explicitly registered Python host tools.

    Tool handlers run in the constructing Python process. Guest HTTP reaches only the virtual
    ``http://host.shellsim/tools`` endpoint; no guest socket or host API handle is exposed.
    """

    def __init__(
        self, tools: Mapping[str, ToolHandler], limits: Optional[Limits] = None, *, clock: str = "virtual"
    ) -> None:
        if not isinstance(tools, Mapping):
            raise TypeError("tools must be a mapping from names to callable handlers")
        for name, handler in tools.items():
            if not isinstance(name, str) or not name or not callable(handler):
                raise TypeError("tool names must be nonempty strings with callable handlers")
        if limits is not None and not isinstance(limits, Limits):
            raise TypeError("limits must be a shellsim.Limits instance")
        if clock not in ("virtual", "real_time"):
            raise ValueError("clock must be 'virtual' or 'real_time'")
        self._tools = dict(tools)
        self._entrypoint: Optional[Tuple[str, ...]] = None
        self._working_directory = "/work"
        resolved = limits or Limits()
        self._native = _native.NativeContainer(
            resolved.cpu,
            resolved.memory,
            resolved.disk,
            resolved.output,
            clock == "real_time",
        )

    @property
    def clock(self) -> str:
        """Host-selected clock mode fixed at container construction."""

        return self._native.clock_mode()

    def write_file(self, path: str, data: Union[bytes, bytearray, memoryview, str], *, mode: int = 0o644) -> None:
        """Stage one file in the guest VFS without exposing its source host path."""

        if not isinstance(path, str):
            raise TypeError("path must be str")
        encoded = data.encode() if isinstance(data, str) else _as_bytes("data", data)
        if isinstance(mode, bool) or not isinstance(mode, int):
            raise TypeError("mode must be int")
        if not 0 <= mode <= 0o7777:
            raise ValueError("mode must be between 0 and 0o7777")
        self._native.write_file(path, encoded, mode)

    def read_file(self, path: str) -> bytes:
        """Read one file from this container's guest VFS."""

        if not isinstance(path, str):
            raise TypeError("path must be str")
        return self._native.read_file(path)

    def mkdir(self, path: str, *, mode: int = 0o755) -> None:
        """Create a directory in the guest VFS, including missing parents."""

        if not isinstance(path, str):
            raise TypeError("path must be str")
        if isinstance(mode, bool) or not isinstance(mode, int):
            raise TypeError("mode must be int")
        if not 0 <= mode <= 0o7777:
            raise ValueError("mode must be between 0 and 0o7777")
        self._native.mkdir(path, mode)

    def run_entrypoint(self) -> RunResult:
        """Run the entrypoint installed by a `.shl` package."""

        if self._entrypoint is None:
            raise SimulationError("container has no package entrypoint")
        return self.run(f"cd {shlex.quote(self._working_directory)} && {shlex.join(self._entrypoint)}")

    def start_entrypoint(self) -> Action:
        """Start a package entrypoint as a host-driven foreground action."""

        if self._entrypoint is None:
            raise SimulationError("container has no package entrypoint")
        return self.start(f"cd {shlex.quote(self._working_directory)} && {shlex.join(self._entrypoint)}")

    def start(self, source: str, stdin: Union[bytes, bytearray, memoryview] = b"") -> Action:
        """Start a shell action without hiding frame, input, or cancellation control."""

        if not isinstance(source, str):
            raise TypeError("source must be str")
        started = json.loads(self._native.start_execute(source, _as_bytes("stdin", stdin)))
        return Action(self, started)

    def run(self, source: str, stdin: Union[bytes, bytearray, memoryview] = b"") -> RunResult:
        """Run an action, servicing guest tool requests with the registered Python handlers."""

        if not isinstance(source, str):
            raise TypeError("source must be str")
        started = json.loads(self._native.start_execute(source, _as_bytes("stdin", stdin)))
        action_id = started["action_id"]
        view = started
        try:
            while view["state"]["state"] != "complete":
                view = json.loads(self._native.poll_action(action_id))
                state = view["state"]["state"]
                if state == "complete":
                    break
                for call in view["tool_calls"]:
                    self._dispatch_tool(call)
                if state == "blocked" and not view["tool_calls"]:
                    raise SimulationError(f"action blocked without a tool call: {view['state'].get('reason')}")
                if state == "stopped":
                    raise SimulationError("action stopped before completion")
            output = json.loads(self._native.read_action_output(action_id))
            invocations = view["invocations"]
            for trust, key in (("unsupported", "unsupported_commands"), ("partial", "partial_commands")):
                view[key] = sorted({item["argv"][0] for item in invocations if item["trust"] == trust and item["argv"]})
            return _decode_result(
                json.dumps(view),
                base64.b64decode(output["stdout_base64"]),
                base64.b64decode(output["stderr_base64"]),
            )
        finally:
            try:
                if view["state"]["state"] != "complete":
                    self._native.cancel_action(action_id)
            finally:
                self._native.drop_action(action_id)

    def _dispatch_tool(self, call: Mapping[str, Any]) -> None:
        name = call["tool"]
        handler = self._tools.get(name)
        if handler is None:
            self._native.respond_tool_call(call["request_id"], None, "unknown tool")
            return
        try:
            result = handler(call["arguments"])
            if not isinstance(result, Mapping):
                raise TypeError("tool result must be a JSON object")
            encoded = json.dumps(dict(result), allow_nan=False)
        except ToolError as error:
            self._native.respond_tool_call(call["request_id"], None, str(error))
        except Exception:
            self._native.respond_tool_call(call["request_id"], None, "tool failed")
        else:
            self._native.respond_tool_call(call["request_id"], encoded, None)


class Action:
    """One foreground action whose guest may present frames or call host tools."""

    def __init__(self, container: Container, view: dict[str, Any]) -> None:
        self._container = container
        self._view = view
        self._action_id = view["action_id"]
        self._seen_generation = view["display_generation"]
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._closed = False

    def __enter__(self) -> Action:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def poll(self) -> ActionPoll:
        """Advance bounded guest work, dispatch tools, and return new output and frame data."""

        if self._closed:
            raise SimulationError("action is closed")
        if self._view["state"]["state"] != "complete":
            self._view = json.loads(self._container._native.poll_action(self._action_id))
        for call in self._view["tool_calls"]:
            self._container._dispatch_tool(call)
        self._view["tool_calls"] = []
        output = json.loads(self._container._native.read_action_output(self._action_id))
        stdout = base64.b64decode(output["stdout_base64"])
        stderr = base64.b64decode(output["stderr_base64"])
        self._stdout.extend(stdout)
        self._stderr.extend(stderr)
        generation = self._view["display_generation"]
        frame = self.frame() if generation > self._seen_generation else None
        self._seen_generation = generation
        state = self._view["state"]
        return ActionPoll(
            state=state["state"],
            frame=frame,
            stdout=stdout,
            stderr=stderr,
            returncode=state.get("status"),
            reason=state.get("reason"),
        )

    def frame(self) -> Optional[DisplayFrame]:
        """Copy the most recent complete frame without exposing guest memory."""

        if self._closed:
            raise SimulationError("action is closed")
        frame = self._container._native.display_frame(self._action_id)
        if frame is None:
            return None
        generation, width, height, pixels = frame
        return DisplayFrame(generation, width, height, pixels)

    def inject_key(self, code: int, pressed: bool) -> None:
        """Queue a key transition for the guest's bounded virtual input device."""

        if self._closed:
            raise SimulationError("action is closed")
        if isinstance(code, bool) or not isinstance(code, int) or not 0 < code <= 0xFFFF:
            raise ValueError("key code must be between 1 and 65535")
        if not isinstance(pressed, bool):
            raise TypeError("pressed must be bool")
        self._container._native.inject_key(self._action_id, code, pressed)

    def stop(self) -> None:
        """Cancel the foreground action and its descendants."""

        if not self._closed and self._view["state"]["state"] != "complete":
            self._container._native.cancel_action(self._action_id)
            self._view = json.loads(self._container._native.poll_action(self._action_id))

    @property
    def result(self) -> RunResult:
        """Return the final action result after it completes and output has been collected."""

        if self._view["state"]["state"] != "complete":
            raise SimulationError("action is not complete")
        view = dict(self._view)
        for trust, key in (("unsupported", "unsupported_commands"), ("partial", "partial_commands")):
            view[key] = sorted(
                {item["argv"][0] for item in view["invocations"] if item["trust"] == trust and item["argv"]}
            )
        return _decode_result(json.dumps(view), bytes(self._stdout), bytes(self._stderr))

    def close(self) -> None:
        """Release the retained action, cancelling live work first."""

        if self._closed:
            return
        try:
            if self._view["state"]["state"] != "complete":
                self._container._native.cancel_action(self._action_id)
        finally:
            self._container._native.drop_action(self._action_id)
            self._closed = True


class ShellSession:
    """One persistent shell process in an :class:`Environment`.

    Actions are serialized through the owning environment. An exited shell cannot be reused.
    """

    def __init__(self, environment: Environment, pid: int) -> None:
        self._environment = environment
        self.pid = pid

    def run(self, source: str, stdin: Union[bytes, bytearray, memoryview] = b"") -> RunResult:
        """Execute one action in this shell, retaining its state for the next action."""

        if not isinstance(source, str):
            raise TypeError("source must be str")
        metadata, stdout, stderr = self._environment._native.run_shell(self.pid, source, _as_bytes("stdin", stdin))
        return _decode_result(metadata, stdout, stderr)


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
