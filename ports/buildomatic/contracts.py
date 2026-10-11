"""Typed contracts for trusted host builds, independent of ports and deployment.

Actions have host capabilities. These contracts must never be exposed directly to
simulated programs. Names are single path components; dependencies mount by ID.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Protocol

CHUNK_BYTES = 4 * 1024**2
MAX_METADATA_BYTES = 32 * 1024**2
MAX_REQUEST_BYTES = 4 * 1024**2
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")


def name(value: str) -> str:
    """Reject ambiguous or traversing mount, action and attempt names."""
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError("invalid build name")
    return value


def digest(value: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("invalid SHA256 digest")
    return value


def encode(value: object) -> bytes:
    """Use one canonical JSON encoding for identities and durable records."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class TreeBundle:
    """SHA256 of a versioned tree manifest, whose files reference bounded chunks."""

    digest: str

    def __post_init__(self):
        digest(self.digest)


@dataclass(frozen=True)
class InputMount:
    name: str
    bundle: TreeBundle

    def __post_init__(self):
        name(self.name)


@dataclass(frozen=True)
class ResourceLimits:
    """Per-attempt bounds; output_bytes also bounds each transported input tree.

    Increase output_bytes explicitly for large SDKs. Transport reads bounded
    chunks and metadata, independently of expanded tree size.
    """

    memory_bytes: int = 8 * 1024**3
    cpu_seconds: int = 3600
    output_bytes: int = 2 * 1024**3
    max_files: int = 100_000
    log_bytes: int = 8 * 1024**2

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("resource limits must be positive integers")


@dataclass(frozen=True)
class Action:
    """Trusted argv, with no implicit shell, and a bounded retry allowance."""

    id: str
    argv: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    inputs: tuple[InputMount, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    timeout_seconds: float = 3600
    max_attempts: int = 2

    def __post_init__(self):
        name(self.id)
        if not isinstance(self.argv, tuple) or not 1 <= len(self.argv) <= 1024:
            raise ValueError("argv must be a nonempty bounded tuple")
        if any(not isinstance(arg, str) or "\0" in arg for arg in self.argv) or not self.argv[0]:
            raise ValueError("invalid argv")
        if not all(isinstance(value, tuple) for value in (self.dependencies, self.inputs, self.env)):
            raise ValueError("action collections must be tuples")
        if len(self.dependencies) > 512 or len(self.inputs) > 512 or len(self.env) > 256:
            raise ValueError("action exceeds collection bounds")
        mounts = [name(dep) for dep in self.dependencies] + [mount.name for mount in self.inputs]
        if len(set(mounts)) != len(mounts):
            raise ValueError("duplicate dependency or input mount")
        keys = []
        for key, value in self.env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or "\0" in value:
                raise ValueError("invalid environment")
            if key == "BUILD_OUTPUT_DIR" or key.startswith("BUILD_INPUT_"):
                raise ValueError("reserved build environment")
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate environment key")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 86400:
            raise ValueError("timeout must be finite and at most one day")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 10:
            raise ValueError("attempt allowance must be between 1 and 10")
        if len(encode(asdict(self))) > 64 * 1024:
            raise ValueError("action exceeds byte bound")


@dataclass(frozen=True)
class BuildRequest:
    idempotency_key: str
    actions: tuple[Action, ...]
    max_workers: int = 1

    def __post_init__(self):
        if not isinstance(self.idempotency_key, str) or not 1 <= len(self.idempotency_key.encode()) <= 512:
            raise ValueError("invalid idempotency key")
        if not isinstance(self.actions, tuple) or not 1 <= len(self.actions) <= 512:
            raise ValueError("request needs 1..512 actions")
        if type(self.max_workers) is not int or not 1 <= self.max_workers <= 32:
            raise ValueError("request needs 1..32 workers")
        actions = {action.id: action for action in self.actions}
        if len(actions) != len(self.actions):
            raise ValueError("duplicate action id")
        resolved = set()
        while len(resolved) < len(actions):
            ready = {
                key for key, action in actions.items() if key not in resolved and set(action.dependencies) <= resolved
            }
            if not ready:
                raise ValueError("cyclic or missing dependency")
            resolved.update(ready)
        if len(encode(asdict(self))) > MAX_REQUEST_BYTES:
            raise ValueError("request exceeds byte bound")


class NodeState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class AttemptState(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerState(str, Enum):
    UNKNOWN = "unknown"
    RUNNING = "running"
    COMPLETED = "completed"


class BuildState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Attempt:
    id: str
    request_id: str
    action: Action
    inputs: tuple[InputMount, ...]
    generation: int

    def __post_init__(self):
        name(self.id)
        digest(self.request_id)
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("invalid coordinator generation")
        mounts = [mount.name for mount in self.inputs]
        if len(mounts) > 1024 or len(set(mounts)) != len(mounts):
            raise ValueError("invalid attempt mounts")


@dataclass(frozen=True)
class NodeResult:
    action_id: str
    state: NodeState
    attempts: int
    bundle: TreeBundle | None = None
    error: str | None = None


@dataclass(frozen=True)
class AttemptResult:
    attempt_id: str
    state: AttemptState
    bundle: TreeBundle | None = None
    returncode: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class WorkerReport:
    state: WorkerState
    result: AttemptResult | None = None


@dataclass(frozen=True)
class BuildResult:
    request_id: str
    state: BuildState
    nodes: tuple[NodeResult, ...]


@dataclass(frozen=True)
class VersionedBytes:
    data: bytes
    version: str


class ConditionalWriteError(RuntimeError):
    """The backend-native journal revision no longer matches the writer's claim."""


class Store(Protocol):
    """Immutable, verified bounded blobs and linearizable native journal CAS.

    Missing blobs raise FileNotFoundError. Blob methods accept at most
    MAX_METADATA_BYTES. Journal writes are durable before returning. None means
    create-if-absent, never unconditional overwrite. Revisions must resist ABA.
    """

    def put_blob(self, data: bytes) -> str: ...
    def get_blob(self, digest: str) -> bytes: ...
    def read_journal(self, key: str) -> VersionedBytes | None: ...
    def write_journal(self, key: str, data: bytes, expected_version: str | None) -> str: ...


BlobStore = Store


class Worker(Protocol):
    """Retry-safe durable attempts; UNKNOWN definitively means no active attempt.

    RPC failures must raise, never become UNKNOWN. submit is idempotent by the
    entire Attempt. Completion is retained until acknowledge. cancel creates a
    tombstone even for absent attempts, preventing delayed dispatch from running.
    """

    def submit(self, attempt: Attempt) -> None: ...
    def poll(self, attempt_id: str) -> WorkerReport: ...
    def cancel(self, attempt_id: str) -> None: ...
    def acknowledge(self, attempt_id: str) -> None: ...


def action_from_dict(value: Mapping) -> Action:
    return Action(
        id=value["id"],
        argv=tuple(value["argv"]),
        dependencies=tuple(value["dependencies"]),
        inputs=tuple(InputMount(m["name"], TreeBundle(m["bundle"]["digest"])) for m in value["inputs"]),
        env=tuple(tuple(pair) for pair in value["env"]),
        timeout_seconds=value["timeout_seconds"],
        max_attempts=value["max_attempts"],
    )


def request_from_dict(value: Mapping) -> BuildRequest:
    return BuildRequest(
        value["idempotency_key"], tuple(action_from_dict(a) for a in value["actions"]), value["max_workers"]
    )


def request_id(request: BuildRequest) -> str:
    return hashlib.sha256(encode(asdict(request))).hexdigest()
