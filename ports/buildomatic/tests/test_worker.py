"""Exercise real subprocesses using file handshakes, never timing assertions.

Polling has a bounded test-harness wait solely to detect hangs. Scheduling, loss
and retry semantics are covered without clocks in test_core.py.
"""

import fcntl
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from ports.buildomatic import (
    Action,
    Attempt,
    AttemptState,
    BuildRequest,
    BuildState,
    Coordinator,
    InputMount,
    LocalStore,
    ResourceLimits,
    WorkerBusy,
    WorkerExecutor,
    WorkerState,
    capture_tree,
    extract_tree,
)
from ports.buildomatic._process import process_token
from ports.buildomatic.contracts import encode
from ports.buildomatic.store import atomic_write


def eventually(read, accept):
    for _ in range(2000):
        value = read()
        if accept(value):
            return value
        time.sleep(0.005)
    raise AssertionError("subprocess handshake did not complete")


def complete(worker, attempt):
    return eventually(lambda: worker.poll(attempt.id), lambda report: report.state == WorkerState.COMPLETED).result


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "store")


@pytest.fixture
def worker(tmp_path, store):
    executor = WorkerExecutor(store, tmp_path / "worker", limits=ResourceLimits(output_bytes=16 * 1024**2))
    yield executor
    for directory in (executor.root / "attempts").iterdir():
        executor.cancel(directory.name)
        eventually(lambda key=directory.name: executor.poll(key), lambda report: report.state != WorkerState.RUNNING)
    for child in executor._children:
        child.wait(timeout=10)


def attempt(code, *, inputs=(), key="test", **kwargs):
    action = Action("node", ("python", "-c", code), inputs=inputs, **kwargs)
    return Attempt(key, "a" * 64, action, inputs, 1)


def test_named_inputs_output_modes_symlinks_and_literal_argv(worker, store, tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "data").write_text("sdk and code")
    bundle = capture_tree(source, store)
    mounts = (InputMount("sdk", bundle), InputMount("code", bundle))
    code = """
import os, pathlib, sys
out = pathlib.Path(os.environ['BUILD_OUTPUT_DIR'])
assert pathlib.Path('inputs/code/data').read_text() == 'sdk and code'
assert pathlib.Path(os.environ['BUILD_INPUT_sdk'], 'data').read_text() == 'sdk and code'
(out / 'tool').write_text(sys.argv[1])
(out / 'tool').chmod(0o751)
(out / 'alias').symlink_to('tool')
"""
    action = Action("node", ("python", "-c", code, "$(touch shell-executed);*"), inputs=mounts)
    item = Attempt("test", "a" * 64, action, mounts, 1)
    worker.submit(item)
    result = complete(worker, item)
    assert result.state == AttemptState.SUCCEEDED, result.error
    output = tmp_path / "retrieved"
    extract_tree(result.bundle, store, output)
    assert (output / "alias").read_text() == "$(touch shell-executed);*"
    assert (output / "tool").stat().st_mode & 0o777 == 0o751
    assert not (worker.root / "attempts/test/work/shell-executed").exists()


def test_duplicate_submit_and_client_restart_preserve_running_attempt(worker, store):
    item = attempt("""
import pathlib, time, os
pathlib.Path('ready').write_text(str(os.getpid()))
while not pathlib.Path('release').exists():
    time.sleep(0.01)
pathlib.Path(os.environ['BUILD_OUTPUT_DIR'], 'value').write_text('done')
""")
    worker.submit(item)
    work = worker.root / "attempts" / item.id / "work"
    ready = eventually(lambda: (work / "ready").exists(), bool)
    assert ready
    pid = int((work / "ready").read_text())
    worker.submit(item)
    restarted = WorkerExecutor(store, worker.root, worker.limits)
    restarted.submit(item)
    for _ in range(5):
        assert restarted.poll(item.id).state == WorkerState.RUNNING
    assert int((work / "ready").read_text()) == pid
    (work / "release").touch()
    first = complete(restarted, item)
    assert first.state == AttemptState.SUCCEEDED, first.error
    again = WorkerExecutor(store, worker.root, worker.limits)
    assert again.poll(item.id).result == first
    again.submit(item)
    assert again.poll(item.id).result == first
    with pytest.raises(ValueError):
        again.submit(replace(item, generation=2))
    again.acknowledge(item.id)
    assert again.poll(item.id).state == WorkerState.UNKNOWN
    again.submit(item)
    assert again.poll(item.id).state == WorkerState.UNKNOWN


def test_pending_dispatch_launch_is_bounded_under_rapid_poll(worker, monkeypatch):
    launches = []

    class Child:
        pid = os.getpid()
        stdout = io.BytesIO()

        def poll(self):
            return None

    def spawn(*args, **kwargs):
        launches.append(args)
        return Child()

    monkeypatch.setattr("ports.buildomatic.worker.subprocess.Popen", spawn)
    item = attempt("pass")
    worker.submit(item)
    eventually(lambda: len(launches), bool)
    for _ in range(100):
        assert worker.poll(item.id).state == WorkerState.RUNNING
        worker.submit(item)
    assert len(launches) == 1
    # Persist a terminal status for the synthetic supervisor before teardown.
    monkeypatch.undo()
    atomic_write(
        worker.root / "attempts/test/status.json",
        encode(
            {
                "phase": "completed",
                "returncode": None,
                "error": "synthetic launch",
                "cancelled": False,
            }
        ),
    )
    worker._children.clear()


def test_module_launch_with_bridge_ports_module_alongside_core(worker, tmp_path, monkeypatch):
    import ports.buildomatic.worker as implementation

    package = tmp_path / "package/ports/buildomatic"
    package.mkdir(parents=True)
    for path in Path(implementation.__file__).parent.glob("*.py"):
        shutil.copyfile(path, package / path.name)
    bridge = os.environ.get("BUILDOMATIC_BRIDGE_MODULE")
    if bridge:
        shutil.copyfile(bridge, package / "ports.py")
    else:
        (package / "ports.py").write_text("from ports._support import native_adapters\n")
    monkeypatch.setattr(implementation, "__file__", str(package / "worker.py"))
    item = attempt("import os,pathlib; pathlib.Path(os.environ['BUILD_OUTPUT_DIR'],'ok').write_text('executed')")
    worker.submit(item)
    assert complete(worker, item).state == AttemptState.SUCCEEDED


def test_pre_status_supervisor_exit_is_terminal_and_never_relaunched(worker, store, monkeypatch):
    import ports.buildomatic.worker as implementation

    spawn = implementation.subprocess.Popen
    launches = []

    def fail_startup(argv, **kwargs):
        launches.append(argv)
        return spawn(
            [sys.executable, "-c", "import sys; print('bootstrap diagnostic', file=sys.stderr); sys.exit(7)"], **kwargs
        )

    monkeypatch.setattr(implementation.subprocess, "Popen", fail_startup)
    item = attempt("raise AssertionError('must never execute')")
    worker.submit(item)
    result = complete(worker, item)
    assert result.state == AttemptState.FAILED
    eventually(lambda: worker.read_log(item.id), lambda data: b"bootstrap diagnostic" in data)
    restarted = WorkerExecutor(store, worker.root, worker.limits)
    for _ in range(5):
        restarted.submit(item)
        assert restarted.poll(item.id).result == result
    assert len(launches) == 1


@pytest.mark.parametrize("initial_phase", [None, "launching", "running"])
@pytest.mark.parametrize("state", [AttemptState.SUCCEEDED, AttemptState.FAILED, AttemptState.CANCELLED])
def test_terminal_status_published_between_read_and_lock_check(
    worker, store, tmp_path, monkeypatch, initial_phase, state
):
    import ports.buildomatic.worker as implementation

    item = attempt("raise AssertionError('must never execute')")
    directory = worker.root / "attempts" / item.id
    output = directory / "work/output"
    output.mkdir(parents=True)
    (output / "value").write_text("completed output")
    atomic_write(directory / "plan.json", encode({"attempt": asdict(item), "limits": asdict(worker.limits)}))
    atomic_write(directory / "launch.json", encode({"pid": None, "token": None}))
    if initial_phase is not None:
        atomic_write(directory / "status.json", encode({"phase": initial_phase, "child": 999999}))
    completed = {
        "phase": "completed",
        "returncode": 7 if state == AttemptState.FAILED else 0,
        "error": "action diagnostic" if state == AttemptState.FAILED else None,
        "cancelled": state == AttemptState.CANCELLED,
    }
    running = implementation._running

    def unexpected_kill(pid):
        raise AssertionError("completed supervisor was mistaken for worker loss")

    monkeypatch.setattr(implementation, "kill_group", unexpected_kill)
    with (directory / "run.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert running(directory)

        def publish_then_unlock(path):
            # Force the supervisor's final write and release after poll read the
            # old status, but before its advisory-lock check can observe liveness.
            atomic_write(path / "status.json", encode(completed))
            fcntl.flock(lock, fcntl.LOCK_UN)
            return running(path)

        monkeypatch.setattr(implementation, "_running", publish_then_unlock)
        assert worker._status(directory) == completed
    result = complete(worker, item)
    assert result.state == state
    assert result.returncode == completed["returncode"]
    assert result.error == completed["error"]
    assert json.loads((directory / "status.json").read_bytes()) == completed
    assert worker.poll(item.id).result == result
    if state == AttemptState.SUCCEEDED:
        extract_tree(result.bundle, store, tmp_path / "recovered")
        assert (tmp_path / "recovered/value").read_text() == "completed output"
    else:
        assert result.bundle is None


@pytest.mark.parametrize(
    "code,limits",
    [
        ("print('x' * 1000)", ResourceLimits(log_bytes=32)),
        ("import pathlib; pathlib.Path('huge').write_bytes(b'x' * 4096)", ResourceLimits(output_bytes=1024)),
        ("import pathlib; [pathlib.Path(str(i)).touch() for i in range(10)]", ResourceLimits(max_files=5)),
    ],
)
def test_log_disk_and_entry_bounds(worker, store, code, limits):
    bounded = WorkerExecutor(store, worker.root, limits)
    item = attempt(code)
    bounded.submit(item)
    assert complete(bounded, item).state == AttemptState.FAILED


def test_supervisor_drains_log_without_scanning_each_poll(tmp_path, monkeypatch):
    import ports.buildomatic._runner as runner

    directory = tmp_path / "attempt"
    work = directory / "work"
    for name in ("inputs", "output", "tmp"):
        (work / name).mkdir(parents=True)
    item = attempt("import sys; sys.stdout.write('x'*300000)")
    atomic_write(directory / "plan.json", encode({"attempt": asdict(item), "limits": asdict(ResourceLimits())}))
    monkeypatch.setenv("PYTHONPATH", str(Path(runner.__file__).resolve().parents[2]))
    # Frozen supervisor time makes the scan cadence independent of host load.
    # Multiple bounded log reads still require multiple selector iterations.
    monkeypatch.setattr(runner.time, "monotonic", lambda: 0.0)
    usage = runner.tree_usage
    scans = []

    def count_scan(*args, **kwargs):
        scans.append(args)
        return usage(*args, **kwargs)

    monkeypatch.setattr(runner, "tree_usage", count_scan)
    runner.run(directory)
    status = json.loads((directory / "status.json").read_bytes())
    assert status["returncode"] == 0
    assert status["error"] is None
    assert (directory / "log").stat().st_size == 300000
    assert len(scans) == 2  # Initial accounting and mandatory final accounting.


def test_timeout_is_terminal_failure_and_retained(worker):
    item = attempt("import signal; signal.pause()", timeout_seconds=0.05)
    worker.submit(item)
    result = complete(worker, item)
    assert result.state == AttemptState.FAILED
    assert worker.poll(item.id).result == result


def test_failure_diagnostics_are_bounded_and_retained_until_acknowledgement(worker):
    item = attempt("import sys; print('compiler diagnostic', file=sys.stderr); sys.exit(3)")
    worker.submit(item)
    result = complete(worker, item)
    assert result.state == AttemptState.FAILED
    assert worker.read_log(item.id) == b"compiler diagnostic\n"
    assert worker.read_log(item.id, max_bytes=5) == b"stic\n"
    for invalid in (0, -1, 65537, True):
        with pytest.raises(ValueError):
            worker.read_log(item.id, invalid)
    worker.acknowledge(item.id)
    with pytest.raises(FileNotFoundError):
        worker.read_log(item.id)


def test_cancel_kills_action_process_group_and_tombstones_future_dispatch(worker):
    item = attempt("""
import os, pathlib, signal, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import signal; signal.pause()'])
pathlib.Path('ready').write_text(str(os.getpid()) + ',' + str(child.pid))
signal.pause()
""")
    worker.submit(item)
    work = worker.root / "attempts" / item.id / "work"
    eventually(lambda: (work / "ready").exists(), bool)
    pids = [int(value) for value in (work / "ready").read_text().split(",")]
    worker.cancel(item.id)
    result = complete(worker, item)
    assert result.state == AttemptState.CANCELLED
    assert all(process_token(pid) is None for pid in pids)
    late = replace(item, id="late")
    worker.cancel(late.id)
    worker.submit(late)
    assert worker.poll(late.id).result.state == AttemptState.CANCELLED


def test_supervisor_loss_reconciliation_never_reexecutes_action(worker, store):
    item = attempt(
        """
import pathlib, signal
with pathlib.Path('count').open('a') as stream:
    stream.write('run\n')
pathlib.Path('ready').touch()
signal.pause()
""".replace("run\n", "run\\n")
    )
    worker.submit(item)
    directory = worker.root / "attempts" / item.id
    eventually(lambda: (directory / "work/ready").exists(), bool)
    status = json.loads((directory / "status.json").read_bytes())
    os.kill(status["supervisor"], signal.SIGKILL)
    for child in worker._children:
        child.wait(timeout=10)
    restarted = WorkerExecutor(store, worker.root, worker.limits)
    result = complete(restarted, item)
    assert result.state == AttemptState.FAILED
    assert process_token(status["child"]) is None
    restarted.submit(item)
    assert restarted.poll(item.id).result == result
    assert (directory / "work/count").read_text() == "run\n"


def test_worker_capacity_preserves_other_attempt(worker):
    first = attempt("import pathlib, signal; pathlib.Path('ready').touch(); signal.pause()")
    worker.submit(first)
    eventually(lambda: (worker.root / "attempts/test/work/ready").exists(), bool)
    with pytest.raises(WorkerBusy):
        worker.submit(replace(first, id="other"))
    assert worker.poll(first.id).state == WorkerState.RUNNING
    worker.cancel("other")
    assert worker.poll(first.id).state == WorkerState.RUNNING


def test_invalid_output_and_missing_input_do_not_publish_success(worker, store, tmp_path):
    item = attempt("import pathlib, os; pathlib.Path(os.environ['BUILD_OUTPUT_DIR'], 'bad').symlink_to('/etc/passwd')")
    worker.submit(item)
    assert complete(worker, item).state == AttemptState.FAILED
    root = tmp_path / "input"
    root.mkdir()
    tree = capture_tree(root, store)
    (store.root / "blobs" / tree.digest).unlink()
    missing = attempt("pass", inputs=(InputMount("gone", tree),), key="missing")
    worker.submit(missing)
    assert complete(worker, missing).state == AttemptState.FAILED


def test_acknowledge_removes_read_only_input_copies(worker, store, tmp_path):
    root = tmp_path / "input"
    (root / "readonly").mkdir(parents=True)
    (root / "readonly/data").write_text("data")
    (root / "readonly").chmod(0o555)
    bundle = capture_tree(root, store)
    item = attempt("pass", inputs=(InputMount("readonly", bundle),))
    worker.submit(item)
    assert complete(worker, item).state == AttemptState.SUCCEEDED
    worker.acknowledge(item.id)
    worker.acknowledge(item.id)
    eventually(lambda: not (worker.root / "attempts/test/work").exists(), bool)
    (root / "readonly").chmod(0o755)


def test_launch_crash_window_becomes_failure_without_reexecution(worker):
    item = attempt("raise AssertionError('must not execute')")
    directory = worker.root / "attempts/test"
    directory.mkdir()
    atomic_write(
        directory / "plan.json",
        encode(
            {
                "attempt": {
                    "id": item.id,
                    "action": {"argv": list(item.action.argv)},
                },
                "limits": {},
            }
        ),
    )
    atomic_write(directory / "status.json", encode({"phase": "launching", "supervisor": 999999, "token": None}))
    result = complete(worker, item)
    assert result.state == AttemptState.FAILED


def test_reused_action_pid_terminalizes_without_signalling_unrelated_group(worker, monkeypatch):
    import ports.buildomatic.worker as implementation

    item = attempt("must not execute")
    directory = worker.root / "attempts/test"
    directory.mkdir()
    atomic_write(directory / "plan.json", encode({"attempt": asdict(item), "limits": asdict(worker.limits)}))
    atomic_write(directory / "status.json", encode({"phase": "running", "child": 999999, "child_token": "original"}))
    monkeypatch.setattr(implementation, "process_token", lambda pid: "different")

    def unexpected_signal(pid):
        raise AssertionError("reused process must never be signalled")

    monkeypatch.setattr(implementation, "kill_group", unexpected_signal)
    result = complete(worker, item)
    assert result.state == AttemptState.FAILED
    assert result.error == "worker supervisor lost"
    assert worker.poll(item.id).result == result


@pytest.mark.parametrize("phase", ["prepare", "publish"])
def test_blocked_transport_does_not_block_rpcs_or_independent_actions(tmp_path, phase):
    entered, release = threading.Event(), threading.Event()

    class BlockingStore(LocalStore):
        blocked_digest = None

        def get_blob(self, digest):
            if digest == self.blocked_digest:
                entered.set()
                assert release.wait(10), "test transport was never released"
            return super().get_blob(digest)

        def put_blob(self, data):
            if phase == "publish" and data == b"held-output":
                entered.set()
                assert release.wait(10), "test transport was never released"
            return super().put_blob(data)

    store = BlockingStore(tmp_path / "store")
    source = tmp_path / "sdk"
    source.mkdir()
    (source / "input").write_text("shared sdk")
    bundle = capture_tree(source, store)
    mounts = (InputMount("sdk", bundle),) if phase == "prepare" else ()
    if mounts:
        store.blocked_digest = bundle.digest
    worker = WorkerExecutor(store, tmp_path / "worker", max_running=2)
    held = attempt(
        "import pathlib,os; pathlib.Path('ran').touch(); "
        "pathlib.Path(os.environ['BUILD_OUTPUT_DIR'],'result').write_bytes(b'held-output')",
        inputs=mounts,
    )
    other = attempt("pass", key="independent")
    with ThreadPoolExecutor(max_workers=2) as rpc:
        try:
            submitted = rpc.submit(worker.submit, held)
            if phase == "publish":
                polling = rpc.submit(lambda: eventually(lambda: (worker.poll(held.id), entered.is_set())[1], bool))
            assert entered.wait(5), "transport handshake did not start"
            # No timeout/performance assertion: neither RPC may depend on the
            # deliberately unreleased transport operation completing.
            submitted.result(timeout=5)
            if phase == "publish":
                polling.result(timeout=5)
            assert rpc.submit(worker.poll, held.id).result(timeout=5).state == WorkerState.RUNNING
            rpc.submit(worker.submit, other).result(timeout=5)
            assert complete(worker, other).state == AttemptState.SUCCEEDED
            rpc.submit(worker.cancel, held.id).result(timeout=5)
            assert worker.poll(other.id).result.state == AttemptState.SUCCEEDED
        finally:
            release.set()
    result = complete(worker, held)
    assert result.state == AttemptState.CANCELLED
    if phase == "prepare":
        assert not (worker.root / "attempts/test/work/ran").exists()
    for child in worker._children:
        child.wait(timeout=10)


@pytest.mark.parametrize("phase", ["prepare", "publish"])
def test_process_crash_during_transport_recovers_without_reexecuting_argv(tmp_path, phase):
    store = LocalStore(tmp_path / "store")
    source = tmp_path / "sdk"
    source.mkdir()
    (source / "input").write_text("shared sdk")
    bundle = capture_tree(source, store)
    counter = tmp_path / "executions"
    code = (
        f"import os,pathlib; pathlib.Path({str(counter)!r}).open('a').write('run\\n'); "
        "pathlib.Path(os.environ['BUILD_OUTPUT_DIR'],'result').write_bytes(b'recovered-output')"
    )
    item = attempt(code, inputs=(InputMount("sdk", bundle),))
    plan = asdict(item)
    script = f"""
import json,os,time
from pathlib import Path
from ports.buildomatic import Attempt,InputMount,TreeBundle,LocalStore,WorkerExecutor
from ports.buildomatic.contracts import action_from_dict
class CrashStore(LocalStore):
    def get_blob(self,digest):
        if {phase!r} == 'prepare':
            os._exit(0)
        return super().get_blob(digest)
    def put_blob(self,data):
        if {phase!r} == 'publish' and data == b'recovered-output':
            os._exit(0)
        return super().put_blob(data)
value=json.loads({json.dumps(plan)!r})
item=Attempt(value['id'],value['request_id'],action_from_dict(value['action']),
    tuple(InputMount(m['name'],TreeBundle(m['bundle']['digest'])) for m in value['inputs']),value['generation'])
worker=WorkerExecutor(CrashStore(Path({str(store.root)!r})),Path({str(tmp_path / "worker")!r}))
worker.submit(item)
while True:
    worker.poll(item.id)
    time.sleep(0.005)
"""
    environment = {"PATH": os.defpath, "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
    child = subprocess.Popen([sys.executable, "-c", script], env=environment)
    try:
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
    directory = tmp_path / "worker/attempts/test"
    assert (directory / "plan.json").exists()
    assert not (directory / "result.json").exists()
    if phase == "prepare":
        assert not (directory / "launch.json").exists()
        assert not counter.exists()
    else:
        assert json.loads((directory / "status.json").read_bytes())["phase"] == "completed"
        assert counter.read_text() == "run\n"
    restarted = WorkerExecutor(store, tmp_path / "worker")
    result = complete(restarted, item)
    assert result.state == AttemptState.SUCCEEDED, result.error
    restarted.submit(item)
    assert restarted.poll(item.id).result == result
    assert counter.read_text() == "run\n"
    extract_tree(result.bundle, store, tmp_path / "recovered")
    assert (tmp_path / "recovered/result").read_bytes() == b"recovered-output"
    for process in restarted._children:
        process.wait(timeout=10)


def test_real_dag_dependency_bundle_and_independent_progress(worker, store, tmp_path):
    other = WorkerExecutor(store, tmp_path / "other-worker", worker.limits)
    first = Action(
        "first",
        (
            "python",
            "-c",
            "import os,pathlib; pathlib.Path(os.environ['BUILD_OUTPUT_DIR'],'data').write_text('provider')",
        ),
    )
    second = Action(
        "second",
        (
            "python",
            "-c",
            "import os,pathlib; pathlib.Path(os.environ['BUILD_OUTPUT_DIR'],'data').write_text(pathlib.Path('inputs/first/data').read_text()+' consumer')",
        ),
        ("first",),
    )
    coordinator = Coordinator(store, "dag", {"first": worker, "other": other})
    coordinator.submit(BuildRequest("dag", (first, second), 2))
    result = eventually(coordinator.tick, lambda value: value.state == BuildState.SUCCEEDED)
    output = tmp_path / "result"
    extract_tree(result.nodes[1].bundle, store, output)
    assert (output / "data").read_text() == "provider consumer"
    coordinator.acknowledge()
    for child in other._children:
        child.wait(timeout=10)
