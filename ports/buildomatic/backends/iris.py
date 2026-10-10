"""Persistent Iris actors around the generic build coordinator and executor.

One coordinator serves a named service. Builds run in durable admission order;
ready nodes within the active build use the worker pool concurrently. Submit,
Get and Cancel never wait for a build to finish. Worker actors are private and
external clients use the controller's authenticated scoped-capability proxy.
Iris actors deserialize Python objects, so capability holders are trusted peers.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from dataclasses import asdict, dataclass
from importlib.metadata import requires
from pathlib import Path
from typing import TYPE_CHECKING, Mapping
from urllib.parse import urlsplit

import tomllib

from ports.buildomatic.remote_store import DEFAULT_PREFIX, MAX_METADATA_BYTES, RemoteStore, validate_prefix

if TYPE_CHECKING:
    from ports.buildomatic import Attempt, BuildRequest, BuildResult, Store, Worker, WorkerReport

logger = logging.getLogger(__name__)
_SERVICE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")
_BUILD_ID = re.compile(r"[0-9a-f]{64}\Z")
_TERMINAL = frozenset(("succeeded", "failed", "cancelled"))
_MAX_BUILDS = 4096


@dataclass(frozen=True)
class IrisConfig:
    """Modest CPU service resources; all tasks use BATCH priority.

    ``setup_scripts`` installs only the caller's required runtime into the task
    image. Empty scripts require an image with Iris and Rigging already present.
    Pin ``task_image`` to a digest for reproducible compiler environments.
    """

    service_id: str
    prefix: str = DEFAULT_PREFIX
    cache_prefix: str | None = None
    target_cluster: str = "cw-us-east-02a"
    workers: int = 2
    worker_cpu: float = 1
    worker_memory_bytes: int = 8 * 1024**3
    worker_disk_bytes: int = 10 * 1024**3
    worker_output_bytes: int = 2 * 1024**3
    worker_max_files: int = 100000
    worker_cpu_seconds: int = 3600
    worker_log_bytes: int = 8 * 1024**2
    job_seconds: int = 3600
    task_image: str | None = None
    setup_scripts: tuple[str, ...] = ("uv sync --no-dev --no-install-project",)
    compiler_cache: bool = False
    tick_seconds: float = 0.5

    def __post_init__(self):
        if _SERVICE_NAME.fullmatch(self.service_id) is None:
            raise ValueError("invalid Iris service name")
        validate_prefix(self.prefix)
        if self.cache_prefix is not None:
            validate_prefix(self.cache_prefix)
        if type(self.workers) is not int or not 1 <= self.workers <= 32:
            raise ValueError("Iris needs 1..32 workers")
        if not math.isfinite(self.worker_cpu) or self.worker_cpu <= 0:
            raise ValueError("worker CPU must be positive and finite")
        if self.worker_memory_bytes <= 0 or self.worker_disk_bytes <= 0 or self.job_seconds <= 0:
            raise ValueError("Iris resource limits must be positive")
        if any(
            value <= 0
            for value in (
                self.worker_output_bytes,
                self.worker_max_files,
                self.worker_cpu_seconds,
                self.worker_log_bytes,
            )
        ):
            raise ValueError("worker execution limits must be positive")
        if not math.isfinite(self.tick_seconds) or self.tick_seconds < 0.1:
            raise ValueError("tick interval must be finite and at least 0.1 seconds")


def _encode(value) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > MAX_METADATA_BYTES:
        raise ValueError("service metadata exceeds byte bound")
    return data


def _decode_request(data: bytes) -> BuildRequest:
    from ports.buildomatic import Action, BuildRequest, InputMount, TreeBundle

    value = json.loads(data)
    actions = tuple(
        Action(
            id=action["id"],
            argv=tuple(action["argv"]),
            dependencies=tuple(action["dependencies"]),
            inputs=tuple(
                InputMount(mount["name"], TreeBundle(mount["bundle"]["digest"])) for mount in action["inputs"]
            ),
            env=tuple(tuple(pair) for pair in action["env"]),
            timeout_seconds=action["timeout_seconds"],
            max_attempts=action["max_attempts"],
        )
        for action in value["actions"]
    )
    return BuildRequest(value["idempotency_key"], actions, value["max_workers"])


class CoordinatorService:
    """Durable service index, with generic core owning each build journal.

    Accepted requests are immutable journals referenced by a CAS index before the
    RPC returns. A replacement coordinator reconstructs submit from that index.
    One Iris coordinator task must own this service; replicas are not supported.
    The first cut retains at most 4096 build records per named service.
    """

    def __init__(self, store: Store, service_id: str, workers: Mapping[str, Worker]):
        if _SERVICE_NAME.fullmatch(service_id) is None:
            raise ValueError("invalid Iris service name")
        self._store, self._workers = store, dict(workers)
        self._base = f"iris/{service_id}"
        self._index_key = f"{self._base}/index.json"
        self._coordinators = {}
        self._released = set()
        self._lock = threading.RLock()
        self._drive_lock = threading.Lock()

    def _index(self):
        found = self._store.read_journal(self._index_key)
        if found is None:
            return {"schema": 1, "builds": []}, None
        value = json.loads(found.data)
        if value["schema"] != 1 or len(value["builds"]) > _MAX_BUILDS:
            raise ValueError("unsupported or oversized service index")
        return value, found.version

    def _coordinator(self, build_id: str, request_digest: str):
        from ports.buildomatic import Coordinator

        if build_id not in self._coordinators:
            found = self._store.read_journal(f"{self._base}/requests/{build_id}.json")
            if found is None or hashlib.sha256(found.data).hexdigest() != request_digest:
                raise ValueError("accepted request journal is missing or corrupt")
            request = _decode_request(found.data)
            coordinator = Coordinator(self._store, f"{self._base}/builds/{build_id}.json", self._workers)
            coordinator.submit(request)
            self._coordinators[build_id] = coordinator
        return self._coordinators[build_id]

    def _lookup_record(self, build_id: str):
        if _BUILD_ID.fullmatch(build_id) is None:
            raise ValueError("invalid service build ID")
        index, _ = self._index()
        record = next((item for item in index["builds"] if item["id"] == build_id), None)
        if record is None:
            raise KeyError(build_id)
        return record

    def submit(self, request: BuildRequest) -> str:
        """Durably accept a request; reject reuse of a key for different content."""
        from ports.buildomatic import ConditionalWriteError, IdempotencyConflict

        if request.max_workers > len(self._workers):
            raise ValueError("build requests more workers than this service provides")
        data = _encode(asdict(request))
        digest = hashlib.sha256(data).hexdigest()
        build_id = hashlib.sha256(request.idempotency_key.encode()).hexdigest()
        with self._lock:
            request_key = f"{self._base}/requests/{build_id}.json"
            try:
                self._store.write_journal(request_key, data, None)
            except ConditionalWriteError:
                found = self._store.read_journal(request_key)
                if found is None or found.data != data:
                    raise IdempotencyConflict("service idempotency key owns another request") from None
            for _ in range(16):
                index, version = self._index()
                previous = next((item for item in index["builds"] if item["id"] == build_id), None)
                if previous is not None:
                    if previous["request"] != digest:
                        raise IdempotencyConflict("service idempotency key owns another request")
                    return build_id
                if len(index["builds"]) >= _MAX_BUILDS:
                    raise ValueError("service build index is full")
                index["builds"].append({"id": build_id, "request": digest})
                try:
                    self._store.write_journal(self._index_key, _encode(index), version)
                except ConditionalWriteError:
                    continue
                return build_id
        raise ConditionalWriteError("service index remained contended")

    def get(self, build_id: str) -> BuildResult:
        """Return durable state using the service ID returned from Submit."""
        from ports.buildomatic import BuildResult, BuildState, NodeResult, NodeState, read_build_result, request_id

        record = self._lookup_record(build_id)
        result = read_build_result(self._store, f"{self._base}/builds/{build_id}.json")
        if result is not None:
            return result
        found = self._store.read_journal(f"{self._base}/requests/{build_id}.json")
        if found is None or hashlib.sha256(found.data).hexdigest() != record["request"]:
            raise ValueError("accepted request journal is missing or corrupt")
        request = _decode_request(found.data)
        return BuildResult(
            request_id(request),
            BuildState.PENDING,
            tuple(NodeResult(action.id, NodeState.PENDING, 0) for action in request.actions),
        )

    def cancel(self, build_id: str) -> BuildResult:
        """Durably queue cancellation; the scheduler delivers worker RPCs."""
        self._control(build_id, "cancelled")
        return self.get(build_id)

    def acknowledge(self, build_id: str) -> None:
        self._control(build_id, "acknowledged")

    def _control(self, build_id: str, flag: str) -> None:
        from ports.buildomatic import ConditionalWriteError

        self._lookup_record(build_id)
        key = f"{self._base}/control/{build_id}.json"
        with self._lock:
            for _ in range(16):
                found = self._store.read_journal(key)
                intent = json.loads(found.data) if found else {}
                if intent.get(flag):
                    return
                intent[flag] = True
                try:
                    self._store.write_journal(key, _encode(intent), found.version if found else None)
                except ConditionalWriteError:
                    continue
                return
        raise ConditionalWriteError("control intent remained contended")

    def get_logs(self, build_id: str) -> tuple[dict[str, str], ...]:
        """Return newest 64 attempt tails, bounded to 64 KiB of log bytes."""
        request_id = self.get(build_id).request_id
        found = self._store.read_journal(f"{self._base}/diagnostics/{request_id}.json")
        if found is None:
            return ()
        result = []
        remaining = 65536
        for record in reversed(json.loads(found.data)[-64:]):
            tail = self._store.read_journal(f"{self._base}/logs/{record['attempt_id']}")
            data = b"" if tail is None else tail.data[-min(16384, remaining) :] if remaining else b""
            remaining -= len(data)
            result.append({**record, "tail": data.decode(errors="replace")})
        return tuple(reversed(result))

    def tick_once(self) -> None:
        """Advance the oldest unfinished build, skipping terminal builds without ack."""
        with self._drive_lock:
            index, _ = self._index()
            # Process control for queued builds too, without dispatching their
            # nodes while another build owns the worker pool.
            for record in index["builds"]:
                found = self._store.read_journal(f"{self._base}/control/{record['id']}.json")
                if found is None:
                    continue
                intent = json.loads(found.data)
                coordinator = self._coordinator(record["id"], record["request"])
                if intent.get("cancelled"):
                    coordinator.cancel()
                if intent.get("acknowledged") and coordinator.result().state.value in _TERMINAL:
                    self._release(record["id"], coordinator)
            for record in index["builds"]:
                coordinator = self._coordinator(record["id"], record["request"])
                if coordinator.result().state.value in _TERMINAL:
                    self._release(record["id"], coordinator)
                    continue
                result = coordinator.tick()
                if result.state.value not in _TERMINAL:
                    return
                self._release(record["id"], coordinator)

    def _release(self, build_id, coordinator):
        if build_id not in self._released:
            coordinator.acknowledge()
            self._released.add(build_id)
        else:
            # Core retries uncertain acknowledgements, keeping outputs retained
            # until the worker confirms cleanup without delaying the next build.
            coordinator.tick()


class _CoordinatorActor:
    """Expose short client operations on the guarded capability endpoint."""

    def __init__(self, service: CoordinatorService):
        self._service = service

    def Submit(self, request: BuildRequest) -> str:
        return self._service.submit(request)

    def Get(self, build_id: str) -> BuildResult:
        return self._service.get(build_id)

    def Cancel(self, build_id: str) -> BuildResult:
        return self._service.cancel(build_id)

    def Acknowledge(self, build_id: str) -> None:
        self._service.acknowledge(build_id)

    def GetLogs(self, build_id: str) -> tuple[dict[str, str], ...]:
        return self._service.get_logs(build_id)


class WorkerProxy:
    """Target one named worker; transport failures propagate rather than imply loss."""

    def __init__(
        self, actor, *, store: Store | None = None, service_id: str | None = None, worker_id: str | None = None
    ):
        self._actor = actor
        self._store = store
        self._base = f"iris/{service_id}"
        self._worker_id = worker_id

    def submit(self, attempt: Attempt) -> None:
        if self._store is not None:
            from ports.buildomatic import ConditionalWriteError

            key = f"{self._base}/diagnostics/{attempt.request_id}.json"
            record = {"attempt_id": attempt.id, "action_id": attempt.action.id, "worker_id": self._worker_id}
            for _ in range(16):
                found = self._store.read_journal(key)
                records = json.loads(found.data) if found else []
                if record in records:
                    break
                records.append(record)
                try:
                    self._store.write_journal(key, _encode(records), found.version if found else None)
                except ConditionalWriteError:
                    continue
                break
            else:
                raise ConditionalWriteError("diagnostic index remained contended")
        self._actor.submit(attempt)

    def poll(self, attempt_id: str) -> WorkerReport:
        report = self._actor.poll(attempt_id)
        if self._store is not None and report.state.value == "completed":
            from ports.buildomatic import ConditionalWriteError

            key = f"{self._base}/logs/{attempt_id}"
            if self._store.read_journal(key) is None:
                tail = self._actor.read_log(attempt_id, max_bytes=16384)
                if len(tail) > 16384:
                    raise ValueError("worker diagnostic exceeds byte bound")
                try:
                    self._store.write_journal(key, tail, None)
                except ConditionalWriteError:
                    pass
        return report

    def cancel(self, attempt_id: str) -> None:
        self._actor.cancel(attempt_id)

    def acknowledge(self, attempt_id: str) -> None:
        self._actor.acknowledge(attempt_id)


class _WorkerActor:
    """Expose the executor protocol and bounded logs only on a private endpoint."""

    def __init__(self, executor):
        self._executor = executor

    def submit(self, attempt: Attempt) -> None:
        self._executor.submit(attempt)

    def poll(self, attempt_id: str) -> WorkerReport:
        return self._executor.poll(attempt_id)

    def cancel(self, attempt_id: str) -> None:
        self._executor.cancel(attempt_id)

    def acknowledge(self, attempt_id: str) -> None:
        self._executor.acknowledge(attempt_id)

    def read_log(self, attempt_id: str, max_bytes: int = 16384) -> bytes:
        if type(max_bytes) is not int or not 1 <= max_bytes <= 16384:
            raise ValueError("diagnostics require a bounded read")
        try:
            return self._executor.read_log(attempt_id, max_bytes=max_bytes)
        except FileNotFoundError:
            return b""


def compiler_cache_environment(path: str, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Derive sccache settings from the resolved TTL path and ambient routing.

    Credentials remain in the worker's existing environment. No anonymous cache
    access is enabled. The S3 endpoint and signing region retain runtime values.
    """
    environ = os.environ if environ is None else environ
    parsed = urlsplit(path)
    if parsed.scheme not in ("s3", "gs") or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("compiler cache requires a routed S3 or GCS object prefix")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("invalid compiler cache prefix")
    prefix = parsed.path.strip("/")
    if parsed.scheme == "gs":
        return {
            "SCCACHE_GCS_BUCKET": parsed.netloc,
            "SCCACHE_GCS_KEY_PREFIX": prefix,
            "SCCACHE_GCS_RW_MODE": "READ_WRITE",
        }
    result = {"SCCACHE_BUCKET": parsed.netloc, "SCCACHE_S3_KEY_PREFIX": prefix}
    endpoint = environ.get("AWS_ENDPOINT_URL_S3") or environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        result["SCCACHE_ENDPOINT"] = endpoint
        result["SCCACHE_S3_USE_SSL"] = "true" if urlsplit(endpoint).scheme == "https" else "false"
        result["SCCACHE_S3_ENABLE_VIRTUAL_HOST_STYLE"] = "true"
    region = environ.get("AWS_REGION") or environ.get("AWS_DEFAULT_REGION")
    if region:
        result["SCCACHE_REGION"] = region
    return result


def _worker_environment(config: IrisConfig) -> dict[str, str]:
    if not config.compiler_cache:
        return {}
    from rigging.filesystem.cluster_config import marin_temp_bucket
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    configure_coreweave_s3()
    path = marin_temp_bucket(30, prefix="shellsim/ports/sccache/v1")
    if shutil.which("sccache") is None:
        raise RuntimeError("compiler cache requires sccache in the worker task image")
    return {**compiler_cache_environment(path), "SCCACHE_SERVER_PORT": "4226"}


def _source_files() -> dict[str, bytes]:
    root = Path(__file__).resolve().parents[1]
    return {
        f"ports/buildomatic/{path.relative_to(root).as_posix()}": path.read_bytes()
        for path in root.rglob("*.py")
        if "tests" not in path.relative_to(root).parts
    }


def _runtime_files() -> dict[str, bytes]:
    """Transport the installed Iris/Rigging source and a small dependency project.

    Task setup resolves only this project. It cannot discover shellsim's root
    build system or trigger a Rust extension build from the submitting checkout.
    Source overlays preserve the actor and native CAS APIs used by this adapter.
    """
    import iris.client.client
    import rigging.filesystem.factory
    from rigging.config_discovery import find_project_root, resolve_cluster_config
    from rigging.filesystem.cluster_config import MARIN_CLUSTER_CONFIG_DIRS

    result = {}
    dependencies = {"google-cloud-storage>=2.0"}
    for package, root in (
        ("iris", Path(iris.client.client.__file__).resolve().parents[1]),
        ("rigging", Path(rigging.filesystem.factory.__file__).resolve().parents[1]),
    ):
        project = root.parent.parent / "pyproject.toml"
        requirements = (
            tomllib.loads(project.read_text())["project"]["dependencies"]
            if project.is_file()
            else requires(f"marin-{package}") or []
        )
        # The transported package already supplies its Python modules. Resolve
        # its own runtime requirements, including checkout-only additions that
        # may not yet appear in the latest published package metadata.
        dependencies.update(
            requirement
            for requirement in requirements
            if not requirement.startswith(("marin-rigging", "marin-finelog-server", "marin-iris-native"))
        )
        for path in root.rglob("*.py"):
            result[f"{package}/{path.relative_to(root).as_posix()}"] = path.read_bytes()
    result["pyproject.toml"] = (
        '[project]\nname = "buildomatic-iris-runtime"\nversion = "0.0.0"\n'
        'requires-python = ">=3.12,<3.14"\n'
        f"dependencies = {json.dumps(sorted(dependencies))}\n"
        "[tool.uv]\nrequired-environments = [\"sys_platform == 'linux' and platform_machine == 'x86_64'\"]\n"
    ).encode()
    checkout = find_project_root(Path(rigging.filesystem.factory.__file__).parent)
    directories = tuple(
        checkout / entry if checkout and entry == "config" else entry for entry in MARIN_CLUSTER_CONFIG_DIRS
    )
    for name in ("marin", "coreweave"):
        path = Path(resolve_cluster_config(name, dirs=directories))
        result[f"rigging/clusters/{name}.yaml"] = path.read_bytes()
    return result


def _entrypoint(role: str, config: IrisConfig, *, worker_id: str | None = None, files=None):
    from iris.cluster.types import Entrypoint

    files = _source_files() if files is None else files
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in sorted(files.items()):
            bundle.writestr(name, data)
    # Kubernetes ConfigMap directory projections are symlinks. Some Iris init
    # images skip nested small files when walking that projection. A single
    # archive preserves package markers and all modules across these runtimes.
    bootstrap = (
        'import runpy,sys,zipfile;zipfile.ZipFile("_buildomatic_runtime.zip").extractall(".");'
        'sys.argv=["ports.buildomatic.backends.iris",*sys.argv[1:]];'
        'runpy.run_module("ports.buildomatic.backends.iris",run_name="__main__")'
    )
    command = ["python", "-c", bootstrap, role, json.dumps(asdict(config))]
    if worker_id is not None:
        command.append(worker_id)
    staged = {"_buildomatic_runtime.zip": archive.getvalue()}
    if "pyproject.toml" in files:
        staged["pyproject.toml"] = files["pyproject.toml"]
    return Entrypoint(command=command, workdir_files=staged)


def _deployed_files() -> dict[str, bytes]:
    """Reuse the admitted runtime payload verbatim for child worker tasks."""
    with zipfile.ZipFile(Path(os.environ["IRIS_WORKDIR"]) / "_buildomatic_runtime.zip") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def launch(client, config: IrisConfig, *, environment=None):
    """Submit one persistent coordinator through an authenticated Iris hub client.

    The returned Iris Job can be used to derive the actor namespace. Connect an
    external IrisBackend to the target controller after its endpoint registers.
    The client's workspace and environment own runtime dependency installation.
    """
    from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
    from iris.cluster.types import EnvironmentSpec, ResourceSpec
    from iris.rpc import job_pb2
    from rigging.timing import Duration

    files = {**_runtime_files(), **_source_files()}
    return client.submit(
        _entrypoint("coordinator", config, files=files),
        name=f"buildomatic-{config.service_id}",
        resources=ResourceSpec(cpu=0.5, memory="1GB", disk="5GB"),
        environment=environment or EnvironmentSpec(setup_scripts=list(config.setup_scripts)),
        ports=["actor"],
        constraints=[Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=config.target_cluster)],
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        timeout=Duration.from_seconds(config.job_seconds),
        task_image=config.task_image,
        max_retries_failure=2,
        max_task_failures=2,
    )


class _CapabilityRPC:
    """Call the capability route without ActorClient's endpoint URL logging.

    The public hub exposes the capability path, while its ordinary proxy path
    may require a separate edge identity. Capability material remains private,
    and transport errors are sanitized before Iris's retry logger sees them.
    """

    def __init__(self, address: str, token: str, actor_name: str, rpc_seconds: float):
        from iris.rpc.actor_connect import ActorServiceClientSync
        from iris.rpc.compression import IRIS_RPC_COMPRESSIONS, IRIS_RPC_ZSTD

        self._address, self._token, self._name = address, token, actor_name
        self._client = ActorServiceClientSync(
            address=address,
            timeout_ms=int(rpc_seconds * 1000),
            accept_compression=IRIS_RPC_COMPRESSIONS,
            send_compression=IRIS_RPC_ZSTD,
        )

    def call(self, method: str, *args):
        import cloudpickle
        from connectrpc.errors import ConnectError
        from iris.actor.client import unwrap_actor_response
        from iris.rpc import actor_pb2
        from iris.rpc.errors import call_with_retry

        request = actor_pb2.ActorCall(
            method_name=method,
            actor_name=self._name,
            serialized_args=cloudpickle.dumps(args),
            serialized_kwargs=cloudpickle.dumps({}),
        )

        def invoke():
            try:
                response = self._client.call(request)
            except ConnectError as error:
                message = error.message.replace(self._address, "[capability]").replace(self._token, "[token]")
                raise ConnectError(error.code, message) from None
            except Exception as error:
                raise RuntimeError(f"Iris actor transport failed: {type(error).__name__}") from None
            return unwrap_actor_response(response)

        return call_with_retry(f"{self._name}.{method}", invoke, max_attempts=3)


class IrisBackend:
    """External client for durable short RPCs through a scoped bearer proxy.

    ``controller_client`` is authenticated to the target controller to mint the
    endpoint capability. Worker endpoints remain PRIVATE. Closing this client
    does not cancel builds or stop the persistent service.
    """

    def __init__(
        self,
        controller_client,
        controller_url: str,
        namespace: str,
        *,
        rpc_seconds: float = 30,
        prefix: str = DEFAULT_PREFIX,
        cache_prefix: str | None = None,
    ):
        self.store = RemoteStore(cache_prefix or prefix, journal_prefix=prefix)
        self._controller = controller_client
        self._name = f"{namespace.rstrip('/')}/coordinator"
        self._rpc_seconds = rpc_seconds
        self._controller_url = controller_url
        self.refresh_capability()

    def refresh_capability(self) -> None:
        """Renew with authenticated controller access; never expose the token."""
        from rigging.connect import capability_path

        capability = self._controller.mint_endpoint_token(self._name)
        address = (
            capability.capability_url
            or f"{self._controller_url.rstrip('/')}{capability_path(self._name, capability.token)}"
        )
        self._actor = _CapabilityRPC(address, capability.token, self._name, self._rpc_seconds)

    def submit(self, request: BuildRequest) -> str:
        return self._actor.call("Submit", request)

    def get(self, build_id: str) -> BuildResult:
        return self._actor.call("Get", build_id)

    def cancel(self, build_id: str) -> BuildResult:
        return self._actor.call("Cancel", build_id)

    def acknowledge(self, build_id: str) -> None:
        self._actor.call("Acknowledge", build_id)

    def get_logs(self, build_id: str) -> tuple[dict[str, str], ...]:
        return self._actor.call("GetLogs", build_id)


def discover(client, controller_url: str, job, config: IrisConfig, *, rpc_seconds: float = 30) -> IrisBackend:
    """Attach after coordinator registry admission; caller chooses a wait policy.

    The authenticated hub may mint and route the capability for a federated
    endpoint. Missing registration raises a transport error for caller retry.
    """
    from iris.cluster.types import Namespace

    namespace = str(Namespace.from_job_id(job.job_id))
    return IrisBackend(
        client,
        controller_url,
        namespace,
        rpc_seconds=rpc_seconds,
        prefix=config.prefix,
        cache_prefix=config.cache_prefix,
    )


def connection_descriptor(
    config: IrisConfig,
    job,
    *,
    cluster_name: str = "marin",
    controller_url: str = "https://iris.oa.dev",
    workspace: Path | None = None,
) -> dict[str, str | int | None]:
    """Project public connection fields accepted by the ports Iris CLI.

    Persist this descriptor after launch. Consumers parse ``job_id`` with Iris
    ``JobName.from_wire`` before deriving its Namespace, then authenticate to
    ``cluster_name`` and mint a fresh scoped capability. Tokens and full runtime
    configuration are deliberately absent from this connection document.
    """
    origin = urlsplit(controller_url)
    if (
        origin.scheme not in ("https", "http")
        or not origin.netloc
        or origin.username
        or origin.password
        or origin.query
        or origin.fragment
        or origin.path not in ("", "/")
    ):
        raise ValueError("controller URL must be a public origin")
    return {
        "schema_version": 1,
        "job_id": str(job.job_id),
        "cluster_name": cluster_name,
        "controller_url": controller_url.rstrip("/"),
        "prefix": config.prefix,
        "cache_prefix": config.cache_prefix,
        "service_id": config.service_id,
        "task_image": config.task_image,
        "config_sha256": hashlib.sha256(_encode(asdict(config))).hexdigest(),
        "workspace": str((workspace or Path.cwd()).resolve()),
    }


def _runtime_identity(config: IrisConfig, files: Mapping[str, bytes]) -> dict[str, str]:
    from iris.version import client_revision_date

    source_digest = hashlib.sha256()
    for name, data in sorted(files.items()):
        source_digest.update(name.encode() + b"\0" + data)
    return {
        "task_image": config.task_image or "cluster-default",
        "iris_revision_date": client_revision_date(),
        "image_git_hash": os.environ.get("IRIS_GIT_HASH", "unknown"),
        "source_sha256": source_digest.hexdigest(),
        "python": sys.version.split()[0],
    }


def _serve_actor(actor, name: str, access: int):
    from iris.actor.server import ActorServer
    from iris.client.client import iris_ctx
    from iris.cluster.client import get_job_info

    ctx, info = iris_ctx(), get_job_info()
    if info is None:
        raise RuntimeError("Iris task context is required")
    server = ActorServer(host="0.0.0.0", port=ctx.get_port("actor"))
    full_name = f"{ctx.namespace}/{name}"
    server.register(full_name, actor)
    # Namespaced worker callers send the short actor name, while capability
    # callers send the full registry name.
    server.register(name, actor)
    port = server.serve_background()
    endpoint_id = ctx.registry.register(name, f"http://{info.advertise_host}:{port}", access=access)
    return server, endpoint_id


def _run_worker(config: IrisConfig, worker_id: str) -> None:
    from iris.client.client import iris_ctx
    from iris.cluster.types import EndpointAccess

    from ports.buildomatic import LocalStore, ResourceLimits, WorkerExecutor

    env = _worker_environment(config)
    os.environ.update(env)
    # sccache uses one worker-local server, keeping cloud credentials outside
    # the sanitized action environment owned by WorkerExecutor.
    if env:
        subprocess.run(["sccache", "--start-server"], check=True, stdout=subprocess.DEVNULL)
    root = Path(os.environ["IRIS_WORKDIR"]) / ".buildomatic" / worker_id
    store = RemoteStore(
        config.cache_prefix or config.prefix,
        journal_prefix=config.prefix,
        local_cache=LocalStore(root / "object-cache"),
    )
    identity = _runtime_identity(config, _source_files())
    identity["cache_enabled"] = str(config.compiler_cache).lower()
    provenance_key = f"iris/{config.service_id}/provenance/{worker_id}.json"
    before = store.read_journal(provenance_key)
    store.write_journal(provenance_key, _encode(identity), None if before is None else before.version)
    executor = WorkerExecutor(
        store,
        root,
        limits=ResourceLimits(
            memory_bytes=config.worker_memory_bytes,
            output_bytes=config.worker_output_bytes,
            max_files=config.worker_max_files,
            cpu_seconds=config.worker_cpu_seconds,
            log_bytes=config.worker_log_bytes,
        ),
        max_running=1,
    )
    server, endpoint_id = _serve_actor(_WorkerActor(executor), worker_id, EndpointAccess.ENDPOINT_ACCESS_PRIVATE)
    try:
        server.wait()
    finally:
        iris_ctx().registry.unregister(endpoint_id)
        server.stop()


def _run_coordinator(config: IrisConfig) -> None:
    from iris.actor.client import ActorClient
    from iris.client.client import iris_ctx
    from iris.cluster.types import EndpointAccess, EnvironmentSpec, ResourceSpec
    from iris.rpc import job_pb2
    from rigging.timing import Duration

    ctx = iris_ctx()
    files = _deployed_files()
    store = RemoteStore(config.cache_prefix or config.prefix, journal_prefix=config.prefix)
    workers = {}
    for index in range(config.workers):
        name = f"worker-{index}"
        ctx.client.submit(
            _entrypoint("worker", config, worker_id=name, files=files),
            name=name,
            resources=ResourceSpec(
                cpu=config.worker_cpu, memory=config.worker_memory_bytes, disk=config.worker_disk_bytes
            ),
            environment=EnvironmentSpec(setup_scripts=list(config.setup_scripts)) if config.setup_scripts else None,
            ports=["actor"],
            priority_band=job_pb2.PRIORITY_BAND_BATCH,
            timeout=Duration.from_seconds(config.job_seconds),
            task_image=config.task_image,
            existing_job_policy=job_pb2.EXISTING_JOB_POLICY_KEEP,
            max_retries_failure=2,
            max_task_failures=2,
        )
        workers[name] = WorkerProxy(
            ActorClient(ctx.resolver, name, call_timeout=5, max_call_attempts=1),
            store=store,
            service_id=config.service_id,
            worker_id=name,
        )
    service = CoordinatorService(store, config.service_id, workers)
    # Current Iris calls scoped-capability access LINK; no anonymous proxy
    # access is enabled. The external transport keeps its minted URL private.
    server, endpoint_id = _serve_actor(_CoordinatorActor(service), "coordinator", EndpointAccess.ENDPOINT_ACCESS_LINK)
    stop = threading.Event()
    errors = []

    def drive():
        while not stop.wait(config.tick_seconds):
            try:
                service.tick_once()
            except Exception as error:
                # Fail the task on storage or fencing errors. Iris then retries
                # from the durable index; silence would leave builds stuck.
                logger.exception("Iris build coordinator stopped")
                errors.append(error)
                server.stop()
                return

    thread = threading.Thread(target=drive, name="buildomatic-coordinator", daemon=True)
    thread.start()
    try:
        server.wait()
    finally:
        stop.set()
        thread.join(timeout=30)
        ctx.registry.unregister(endpoint_id)
        server.stop()
    if errors:
        raise RuntimeError("Iris coordinator requires recovery") from errors[0]


def main() -> None:
    """Run only explicitly selected in-cluster actor roles."""
    role, serialized = sys.argv[1:3]
    values = json.loads(serialized)
    values["setup_scripts"] = tuple(values["setup_scripts"])
    config = IrisConfig(**values)
    if role == "coordinator":
        _run_coordinator(config)
    elif role == "worker":
        _run_worker(config, sys.argv[3])
    else:
        raise ValueError("unknown Iris backend role")


if __name__ == "__main__":
    main()
