"""Detached POSIX attempt supervisor, independent of stores and RPC clients.

The supervisor holds an attempt lock through execution and writes terminal status
durably. A replacement executor can inspect that lock and PID start tokens. The
action runs in its own process group, killed on timeout, cancellation, completion
or a resource breach. This is a trusted host executor, not a sandbox.
"""

from __future__ import annotations

import fcntl
import json
import os
import resource
import selectors
import subprocess
import sys
import time
from pathlib import Path

from ports.buildomatic._process import kill_group, process_token, tree_usage, wait_group
from ports.buildomatic.contracts import encode
from ports.buildomatic.store import atomic_write, read_bounded


def set_limits(limits: dict) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_bytes"], limits["memory_bytes"]))
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits["output_bytes"], limits["output_bytes"]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def action_child(directory: Path, gate: int) -> None:
    """Wait for durable child identity before granting permission to exec argv.

    EOF means the supervisor died before dispatch was committed. The child then
    exits without executing the action, closing the Popen-to-journal crash gap.
    """
    if os.read(gate, 1) != b"1":
        raise SystemExit(125)
    os.close(gate)
    plan = json.loads(read_bounded(directory / "plan.json"))
    workspace = directory / "work"
    action = plan["attempt"]["action"]
    environment = {"PATH": os.defpath, "HOME": str(workspace), "TMPDIR": str(workspace / "tmp")}
    environment.update(action["env"])
    for mount in plan["attempt"]["inputs"]:
        environment["BUILD_INPUT_" + mount["name"]] = str(workspace / "inputs" / mount["name"])
    environment["BUILD_OUTPUT_DIR"] = str(workspace / "output")
    argv = action["argv"][:]
    if argv[0] in ("python", "python3"):
        argv[0] = sys.executable
    set_limits(plan["limits"])
    os.execvpe(argv[0], argv, environment)


def run(directory: Path) -> None:
    """Run at most once: the lock and durable launch state reject later runners."""
    with (directory / "run.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        status_path = directory / "status.json"
        if status_path.exists():
            # Once launching is persisted, an uncertain crash is a failed
            # attempt, never permission to execute the same attempt a second time.
            return
        plan = json.loads(read_bounded(directory / "plan.json"))
        action = plan["attempt"]["action"]
        limits = plan["limits"]
        status = {"phase": "launching", "supervisor": os.getpid(), "token": process_token(os.getpid())}
        atomic_write(status_path, encode(status))
        result = {"phase": "completed", "returncode": None, "error": None, "cancelled": False}
        process = None
        try:
            if (directory / "cancelled").exists():
                result["cancelled"] = True
                return
            workspace = directory / "work"
            gate_read, gate_write = os.pipe()
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "ports.buildomatic._runner", "--action", str(directory), str(gate_read)],
                    cwd=workspace,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(gate_read,),
                )
                status.update(phase="running", child=process.pid, child_token=process_token(process.pid))
                atomic_write(status_path, encode(status))
                os.write(gate_write, b"1")
            finally:
                os.close(gate_read)
                os.close(gate_write)
            started = time.monotonic()
            next_usage_check = started
            log_size = 0
            selector = selectors.DefaultSelector()
            with selector, (directory / "log").open("wb") as log:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    if (directory / "cancelled").exists():
                        result["cancelled"] = True
                        break
                    now = time.monotonic()
                    if now - started >= action["timeout_seconds"]:
                        result["error"] = "action timed out"
                        break
                    if now >= next_usage_check:
                        try:
                            tree_usage(workspace, limits["max_files"], limits["output_bytes"], skip_inputs=True)
                        except ValueError as error:
                            result["error"] = str(error)
                            break
                        # Recursive scans must not throttle log draining. File
                        # rlimits remain immediate; final accounting is mandatory.
                        next_usage_check = now + 1.0
                    for key, _ in selector.select(timeout=0.05):
                        data = os.read(key.fileobj.fileno(), min(65536, limits["log_bytes"] - log_size + 1))
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        if log_size + len(data) > limits["log_bytes"]:
                            result["error"] = "action log exceeds byte bound"
                            break
                        log.write(data)
                        log.flush()
                        log_size += len(data)
                    if result["error"]:
                        break
                    if process.poll() is not None:
                        # Background descendants cannot retain pipes or continue
                        # writing after their designated action exits.
                        kill_group(process.pid)
                        if not selector.get_map():
                            break
            kill_group(process.pid)
            result["returncode"] = process.wait()
            wait_group(process.pid)
            if result["error"] is None and not result["cancelled"]:
                try:
                    tree_usage(workspace, limits["max_files"], limits["output_bytes"], skip_inputs=True)
                except ValueError as error:
                    result["error"] = str(error)
            if result["returncode"] and not result["error"] and not result["cancelled"]:
                result["error"] = "action exited unsuccessfully"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"[:4096]
        finally:
            if process is not None:
                kill_group(process.pid)
                process.wait()
                wait_group(process.pid)
                process.stdout.close()
            atomic_write(status_path, encode(result))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--action":
        action_child(Path(sys.argv[2]).resolve(), int(sys.argv[3]))
    elif len(sys.argv) == 4 and sys.argv[1] == "--supervisor":
        gate = int(sys.argv[3])
        allowed = os.read(gate, 1) == b"1"
        os.close(gate)
        if allowed:
            run(Path(sys.argv[2]).resolve())
    elif len(sys.argv) == 2:
        run(Path(sys.argv[1]).resolve())
    else:
        raise SystemExit("expected one private attempt directory")
