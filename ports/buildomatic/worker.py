"""Durable local trusted-host worker with detached subprocess supervision.

Workers may be reconstructed with the same store and root after client restart.
Supervisor status, output publication and cancellation tombstones live on disk;
completed results remain until acknowledgement. Linux process start tokens and
advisory locks prevent restart from guessing whether an action is still running.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, replace
from pathlib import Path

from ._process import kill_group, process_token, tree_usage, wait_group
from .bundles import capture_tree, extract_tree
from .contracts import (
    Attempt,
    AttemptResult,
    AttemptState,
    InputMount,
    ResourceLimits,
    Store,
    TreeBundle,
    VersionedBytes,
    WorkerReport,
    WorkerState,
    encode,
    name,
)
from .store import atomic_write, locked, read_bounded

_DEFAULT_LIMITS = ResourceLimits()


class _TransferCancelled(RuntimeError):
    pass


class _TransferStore:
    """Observe durable tombstones between bounded transport chunks."""

    def __init__(self, store: Store, directory: Path):
        self.store, self.directory = store, directory

    def check(self) -> None:
        if (self.directory / "cancelled").exists() or (self.directory / "acknowledged").exists():
            raise _TransferCancelled()

    def get_blob(self, digest: str) -> bytes:
        self.check()
        data = self.store.get_blob(digest)
        self.check()
        return data

    def put_blob(self, data: bytes) -> str:
        self.check()
        digest = self.store.put_blob(data)
        self.check()
        return digest

    def read_journal(self, key: str) -> VersionedBytes | None:
        return self.store.read_journal(key)

    def write_journal(self, key: str, data: bytes, expected_version: str | None) -> str:
        return self.store.write_journal(key, data, expected_version)


class WorkerBusy(RuntimeError):
    """No configured execution slot is available; submission can be retried."""


def _read(path: Path) -> dict | None:
    try:
        return json.loads(read_bounded(path))
    except FileNotFoundError:
        return None


def _running(directory: Path) -> bool:
    with (directory / "run.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def _remove_workspace(root: Path) -> None:
    """Remove private copies even when the input tree preserved read-only modes."""
    if root.is_symlink():
        root.unlink()
        return
    directories = [root]
    while directories:
        directory = directories.pop()
        directory.chmod(directory.stat().st_mode | 0o700)
        with os.scandir(directory) as children:
            for child in children:
                if child.is_dir(follow_symlinks=False):
                    directories.append(Path(child.path))
    shutil.rmtree(root)


def _startup_log(stream, path: Path, maximum: int) -> None:
    """Drain launcher diagnostics with constant memory and bounded retained bytes."""
    with stream, path.open("wb") as log:
        remaining = maximum
        while data := stream.read(65536):
            if remaining:
                retained = data[:remaining]
                log.write(retained)
                log.flush()
                remaining -= len(retained)


class WorkerExecutor:
    """Execute trusted host argv on Linux, with 1..32 durable concurrent slots.

    Resource rlimits bound each process; the supervisor monitors aggregate disk
    and log growth. Polling resumes durable preparation/publication work. Input mounts
    share a total expanded byte/entry allowance. There is no shell by default.
    The root is private to this worker identity and must survive client restart.
    """

    def __init__(self, store: Store, root: Path, limits: ResourceLimits = _DEFAULT_LIMITS, max_running: int = 1):
        if type(max_running) is not int or not 1 <= max_running <= 32:
            raise ValueError("worker needs 1..32 execution slots")
        if sys.platform != "linux":
            raise ValueError("persistent subprocess workers require Linux")
        self.store, self.root, self.limits, self.max_running = store, Path(root).resolve(), limits, max_running
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "attempts").mkdir(exist_ok=True)
        self._children = []
        self._transfers: dict[str, threading.Thread] = {}
        with locked(self.root / "worker.lock"):
            for directory in (self.root / "attempts").iterdir():
                self._schedule(directory)

    def _directory(self, attempt_id: str) -> Path:
        return self.root / "attempts" / name(attempt_id)

    def _launch(self, directory: Path) -> None:
        if (directory / "status.json").exists() or (directory / "acknowledged").exists():
            return
        launch = _read(directory / "launch.json")
        if launch is not None:
            # A persisted launch is an attempt, even if the supervisor exited
            # before recording status. Reconciliation must not launch it again.
            return
        self._children = [child for child in self._children if child.poll() is None]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        atomic_write(directory / "launch.json", encode({"pid": None, "token": None}))
        gate_read, gate_write = os.pipe()
        try:
            child = subprocess.Popen(
                [sys.executable, "-m", "ports.buildomatic._runner", "--supervisor", str(directory), str(gate_read)],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(gate_read,),
            )
            self._children.append(child)
            threading.Thread(
                target=_startup_log, args=(child.stdout, directory / "startup.log", self.limits.log_bytes), daemon=True
            ).start()
            atomic_write(directory / "launch.json", encode({"pid": child.pid, "token": process_token(child.pid)}))
            os.write(gate_write, b"1")
        except Exception as error:
            atomic_write(
                directory / "status.json",
                encode(
                    {
                        "phase": "completed",
                        "returncode": None,
                        "cancelled": False,
                        "error": f"worker supervisor failed to start: {type(error).__name__}: {error}"[:4096],
                    }
                ),
            )
        finally:
            os.close(gate_read)
            os.close(gate_write)

    def submit(self, attempt: Attempt) -> None:
        directory = self._directory(attempt.id)
        plan = {"attempt": asdict(attempt), "limits": asdict(self.limits)}
        with locked(self.root / "worker.lock"):
            existing = _read(directory / "plan.json")
            if existing is not None:
                if encode(existing["attempt"]) != encode(plan["attempt"]):
                    raise ValueError("attempt id reused with different content")
                self._schedule(directory)
                return
            if (directory / "cancelled").exists() or (directory / "acknowledged").exists():
                return
            active = 0
            for previous in (self.root / "attempts").iterdir():
                if _read(previous / "plan.json") is None or (previous / "acknowledged").exists():
                    continue
                if not (previous / "result.json").exists():
                    active += 1
            if active >= self.max_running:
                raise WorkerBusy("worker has no available slot")
            directory.mkdir(exist_ok=True)
            atomic_write(directory / "plan.json", encode(plan))
            self._schedule(directory)

    def _schedule(self, directory: Path) -> None:
        """Under worker.lock, enqueue only work recoverable from durable records."""
        transfer = self._transfers.get(directory.name)
        if transfer is not None and transfer.is_alive():
            return
        if (directory / "acknowledged").exists():
            if not any((directory / key).exists() for key in ("work", "log", "startup.log", "result.json")):
                return
        else:
            if not (directory / "plan.json").exists() or (directory / "result.json").exists():
                return
            status = _read(directory / "status.json")
            if status is None:
                launch = _read(directory / "launch.json")
                if launch is not None and launch["pid"] is not None:
                    token = process_token(launch["pid"])
                    if token is not None and token == launch["token"]:
                        return
            if status is not None and status["phase"] != "completed" and _running(directory):
                return
        # Another executor may own the transfer across this client's restart.
        with (directory / "transfer.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
        transfer = threading.Thread(target=self._run_transfer, args=(directory,), daemon=True)
        self._transfers[directory.name] = transfer
        transfer.start()

    def _run_transfer(self, directory: Path) -> None:
        try:
            self._drive(directory)
        finally:
            with locked(self.root / "worker.lock"):
                self._transfers.pop(directory.name, None)
                # An acknowledgement can arrive while publication is returning.
                # Its durable cleanup intent must not depend on another RPC.
                if (directory / "acknowledged").exists():
                    self._schedule(directory)

    def _prepare(self, directory: Path, plan: dict, transport: _TransferStore) -> None:
        transport.check()
        workspace = directory / "work"
        # Preparation may be repeated only before the durable launch intent.
        if workspace.exists():
            _remove_workspace(workspace)
        for key in ("inputs", "output", "tmp"):
            (workspace / key).mkdir(parents=True)
        limits = ResourceLimits(**plan["limits"])
        files = size = 0
        for value in plan["attempt"]["inputs"]:
            transport.check()
            mount = InputMount(value["name"], TreeBundle(value["bundle"]["digest"]))
            if files >= limits.max_files or size >= limits.output_bytes:
                raise ValueError("inputs exceed aggregate transport bounds")
            remaining = replace(limits, max_files=limits.max_files - files, output_bytes=limits.output_bytes - size)
            target = workspace / "inputs" / mount.name
            extract_tree(mount.bundle, transport, target, limits=remaining)
            count, expanded = tree_usage(target, remaining.max_files, remaining.output_bytes)
            files += count
            size += expanded
        transport.check()
        atomic_write(directory / "prepared", b"1")

    def _drive(self, directory: Path) -> None:
        """Resume one durable lifecycle without holding the shared RPC lock."""
        with (directory / "transfer.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            if (directory / "acknowledged").exists():
                self._cleanup(directory)
                return
            if (directory / "result.json").exists():
                return
            plan = _read(directory / "plan.json")
            transport = _TransferStore(self.store, directory)
            if not (directory / "launch.json").exists() and not (directory / "status.json").exists():
                try:
                    if not (directory / "prepared").exists():
                        self._prepare(directory, plan, transport)
                    with locked(self.root / "worker.lock"):
                        transport.check()
                        self._launch(directory)
                except Exception as error:
                    atomic_write(
                        directory / "status.json",
                        encode(
                            {
                                "phase": "completed",
                                "returncode": None,
                                "cancelled": isinstance(error, _TransferCancelled),
                                "error": None
                                if isinstance(error, _TransferCancelled)
                                else f"input preparation failed: {type(error).__name__}: {error}"[:4096],
                            }
                        ),
                    )
            status = self._status(directory)
            if status is None or status["phase"] != "completed":
                return
            bundle = None
            state = AttemptState.FAILED
            error = status["error"]
            if status["cancelled"]:
                state = AttemptState.CANCELLED
            elif status["returncode"] == 0 and error is None:
                try:
                    bundle = capture_tree(directory / "work/output", transport, limits=ResourceLimits(**plan["limits"]))
                    state = AttemptState.SUCCEEDED
                except _TransferCancelled:
                    state = AttemptState.CANCELLED
                except Exception as failure:
                    error = f"output publication failed: {type(failure).__name__}: {failure}"[:4096]
            with locked(self.root / "worker.lock"):
                if (directory / "acknowledged").exists():
                    return
                if (directory / "cancelled").exists() and state == AttemptState.SUCCEEDED:
                    state, bundle = AttemptState.CANCELLED, None
                atomic_write(
                    directory / "result.json",
                    encode(asdict(AttemptResult(directory.name, state, bundle, status["returncode"], error))),
                )

    def _cleanup(self, directory: Path) -> None:
        if (directory / "work").exists():
            _remove_workspace(directory / "work")
        for key in ("log", "startup.log", "result.json"):
            (directory / key).unlink(missing_ok=True)

    def _status(self, directory: Path) -> dict | None:
        status = _read(directory / "status.json")
        if status is None:
            launch = _read(directory / "launch.json")
            if launch is None:
                self._launch(directory)
                return None
            pid = launch["pid"]
            if pid is not None and (token := process_token(pid)) is not None and token == launch["token"]:
                return None
        elif status["phase"] == "completed":
            return status
        if _running(directory):
            return status
        # Completion is persisted before the supervisor releases run.lock. The
        # first read may predate that write, so an unlocked attempt needs a fresh
        # status read before worker-loss reconciliation can overwrite it.
        status = _read(directory / "status.json")
        if status is not None and status["phase"] == "completed":
            return status
        # The supervisor may have died while the action was still running. Kill
        # that owned group before reporting failure; never launch it again.
        child = status.get("child") if status is not None else None
        if child is not None:
            token = process_token(child)
            # A changed start token proves the owned action is gone. Never kill
            # the unrelated process which now happens to use its numeric PID.
            if token is None or token == status.get("child_token"):
                kill_group(child)
                wait_group(child)
        status = {
            "phase": "completed",
            "returncode": None,
            "cancelled": (directory / "cancelled").exists(),
            "error": "worker supervisor lost"
            if status is not None
            else "worker supervisor exited before recording status",
        }
        atomic_write(directory / "status.json", encode(status))
        return status

    def poll(self, attempt_id: str) -> WorkerReport:
        directory = self._directory(attempt_id)
        with locked(self.root / "worker.lock"):
            if (directory / "acknowledged").exists():
                return WorkerReport(WorkerState.UNKNOWN)
            plan = _read(directory / "plan.json")
            if plan is None:
                if (directory / "cancelled").exists():
                    return WorkerReport(WorkerState.COMPLETED, AttemptResult(attempt_id, AttemptState.CANCELLED))
                return WorkerReport(WorkerState.UNKNOWN)
            result = _read(directory / "result.json")
            if result is None:
                self._schedule(directory)
                return WorkerReport(WorkerState.RUNNING)
            return WorkerReport(
                WorkerState.COMPLETED,
                AttemptResult(
                    result["attempt_id"],
                    AttemptState(result["state"]),
                    TreeBundle(result["bundle"]["digest"]) if result["bundle"] else None,
                    result["returncode"],
                    result["error"],
                ),
            )

    def cancel(self, attempt_id: str) -> None:
        directory = self._directory(attempt_id)
        with locked(self.root / "worker.lock"):
            directory.mkdir(exist_ok=True)
            atomic_write(directory / "cancelled", b"1")
            self._schedule(directory)

    def read_log(self, attempt_id: str, max_bytes: int = 16384) -> bytes:
        """Read at most 64 KiB of the combined stdout/stderr tail on demand.

        Diagnostics are local to the worker and retained until acknowledgement.
        Missing or acknowledged logs raise FileNotFoundError. This method reads
        at most min(max_bytes, log_bytes) bytes, even while the action is writing.
        """
        if type(max_bytes) is not int or not 1 <= max_bytes <= 65536:
            raise ValueError("log reads need 1..65536 bytes")
        directory = self._directory(attempt_id)
        if (directory / "acknowledged").exists():
            raise FileNotFoundError("attempt log was acknowledged")
        plan = _read(directory / "plan.json")
        if plan is None:
            raise FileNotFoundError("attempt has no log")
        maximum = min(max_bytes, plan["limits"]["log_bytes"])
        path = directory / "log"
        if not path.exists():
            path = directory / "startup.log"
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - maximum))
            return stream.read(maximum)

    def acknowledge(self, attempt_id: str) -> None:
        directory = self._directory(attempt_id)
        with locked(self.root / "worker.lock"):
            if not directory.exists():
                directory.mkdir()
            if (directory / "plan.json").exists() and not (directory / "result.json").exists():
                if not (directory / "acknowledged").exists():
                    raise ValueError("cannot acknowledge an unpublished attempt")
            atomic_write(directory / "acknowledged", b"1")
            self._schedule(directory)
