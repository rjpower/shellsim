"""Hold a real cache-miss Clang on stdin to check action containment on Linux."""

from __future__ import annotations

import json
import os
import resource
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .. import Action, Attempt, AttemptState, LocalStore, ResourceLimits, WorkerExecutor, WorkerState
from .._process import process_token


@dataclass(frozen=True)
class ContainmentResult:
    compiler_in_action_group: bool
    memory_bytes: int
    cpu_seconds: int
    file_bytes: int
    cancelled: bool
    independent_compile_succeeded: bool


def _await(read, accept):
    for _ in range(1000):
        value = read()
        if accept(value):
            return value
        time.sleep(0.005)
    raise RuntimeError("compiler containment handshake did not complete")


def probe_compiler_containment(
    sccache: Path, compiler: Path, root: Path, *, endpoint: Mapping[str, str]
) -> ContainmentResult:
    """Require a real compiler in the worker action group with inherited limits.

    A compiler shim delegates detection/preprocessing unchanged, then execs the
    real Clang on a FIFO at the cache-miss compilation stage. The FIFO prevents
    completion while /proc identity, group and rlimits are inspected. Cancelling
    that action must kill Clang and leave an independent compile successful.
    No backend data or persistent daemon process is altered.
    """
    from .compiler_cache import _client_environment

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    sccache, compiler = Path(sccache).resolve(), Path(compiler).absolute()
    fifo, ready = root / "stdin.fifo", root / "compiler.json"
    os.mkfifo(fifo)
    wrapper = root / "clang"
    wrapper.write_text(
        f"#!{sys.executable}\nimport json,os,pathlib,sys\n"
        "args=sys.argv[1:]\n"
        "if '-c' in args and '-E' not in args:\n"
        f"    fd=os.open({str(fifo)!r},os.O_RDONLY)\n"
        "    os.dup2(fd,0); os.close(fd)\n"
        f"    pathlib.Path({str(ready)!r}).write_text(json.dumps({{'pid':os.getpid()}}))\n"
        f"    os.execv({str(compiler)!r},[{str(compiler)!r},'--target=wasm32-wasip1','-x','c','-','-c','-o','held.o'])\n"
        f"os.execv({str(compiler)!r},[{str(compiler)!r},*args])\n"
    )
    wrapper.chmod(0o755)
    limits = ResourceLimits(cpu_seconds=13, output_bytes=16 * 1024**2)
    worker = WorkerExecutor(LocalStore(root / "store"), root / "worker", limits, max_running=2)
    environment = tuple(_client_environment(endpoint, root).items())
    nonce = uuid.uuid4().hex

    def item(key, driver):
        code = (
            "import os,pathlib\npathlib.Path('source').mkdir(); pathlib.Path('build').mkdir()\n"
            f"pathlib.Path('source/probe.c').write_text('int probe_{nonce}(void) {{return 42;}}')\n"
            "os.chdir('build')\n"
            f"os.execv({str(sccache)!r},[{str(sccache)!r},{str(driver)!r},'--target=wasm32-wasip1',"
            "'-c','../source/probe.c','-o','../output/probe.o'])\n"
        )
        return Attempt(key, "a" * 64, Action(key, ("python", "-c", code), env=environment), (), 1)

    held, independent = item("held", wrapper), item("independent", compiler)
    pid = None
    gate = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
    try:
        worker.submit(held)
        _await(lambda: ready.exists(), bool)
        pid = json.loads(ready.read_text())["pid"]
        _await(lambda: Path(f"/proc/{pid}/exe").resolve(), lambda path: path == compiler.resolve())
        status = json.loads((worker.root / "attempts/held/status.json").read_bytes())
        if os.getpgid(pid) != status["child"]:
            raise ValueError("cache-miss compiler escaped the action process group")
        for kind, value in (
            (resource.RLIMIT_AS, limits.memory_bytes),
            (resource.RLIMIT_CPU, limits.cpu_seconds),
            (resource.RLIMIT_FSIZE, limits.output_bytes),
        ):
            if resource.prlimit(pid, kind) != (value, value):
                raise ValueError("cache-miss compiler escaped action resource limits")
        worker.submit(independent)
        worker.cancel(held.id)
        cancelled = _await(lambda: worker.poll(held.id), lambda report: report.state == WorkerState.COMPLETED)
        if cancelled.result.state != AttemptState.CANCELLED or process_token(pid) is not None:
            raise ValueError("action cancellation left cache-miss compiler alive")
        other = _await(lambda: worker.poll(independent.id), lambda report: report.state == WorkerState.COMPLETED)
        if other.result.state != AttemptState.SUCCEEDED:
            raise ValueError("compiler cancellation affected independent work")
        return ContainmentResult(True, limits.memory_bytes, limits.cpu_seconds, limits.output_bytes, True, True)
    finally:
        os.close(gate)
        # If execution escaped containment, stop only this known compiler; never
        # signal its group, which might belong to the persistent cache daemon.
        if pid is not None and process_token(pid) is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for attempt in (held, independent):
            worker.cancel(attempt.id)
            _await(lambda key=attempt.id: worker.poll(key), lambda report: report.state != WorkerState.RUNNING)
        for child in worker._children:
            child.wait(timeout=10)
