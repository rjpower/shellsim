"""Exercise real subprocesses using file handshakes, never timing assertions.

Polling has a bounded test-harness wait solely to detect hangs. Scheduling, loss
and retry semantics are covered without clocks in test_core.py.
"""

import json
import os
import signal
import time
from dataclasses import replace

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
from ports.buildomatic._runner import process_token
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

        def poll(self):
            return None

    def spawn(*args, **kwargs):
        launches.append(args)
        return Child()

    monkeypatch.setattr("ports.buildomatic.worker.subprocess.Popen", spawn)
    item = attempt("pass")
    worker.submit(item)
    for _ in range(100):
        assert worker.poll(item.id).state == WorkerState.RUNNING
        worker.submit(item)
    assert len(launches) == 1
    # Remove the synthetic live marker so fixture teardown can reconcile it.
    monkeypatch.undo()
    (worker.root / "attempts/test/launch.json").unlink()
    worker._children.clear()


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
    assert not (worker.root / "attempts/test/work").exists()
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
