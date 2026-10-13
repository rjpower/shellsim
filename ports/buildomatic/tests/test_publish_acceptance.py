"""Run real guest probes, retaining completed results when a later probe fails."""

import json

import pytest

from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic.artifacts import resolve
from ports.buildomatic.backends.local import LocalBackend
from ports.buildomatic.nodes import Operation
from ports.buildomatic.remote_store import StorageStore
from ports.buildomatic.workflow import run


@pytest.fixture
def acceptance_graph(tmp_path):
    ports, state = tmp_path / "ports", tmp_path / "state"
    for name, source in [("good", "cat /work/nested/input.txt; printf '\\n'\n"), ("bad", "false\n")]:
        path, recipe = _wheel_port(ports, state / "execution/cache/ports", name, [])
        recipe["tests"] = [
            {
                "kind": "shell",
                "script": "probe.sh",
                "files": [{"source": "input.txt", "destination": "nested/input.txt"}],
            }
        ]
        path.write_text(json.dumps(recipe))
        (path.parent / "probe.sh").write_text(source)
        (path.parent / "input.txt").write_text("42")
    store = StorageStore(str(tmp_path / "canonical"))
    backend = LocalBackend(store, state / "execution")
    built = run(ports, ["python/good", "python/bad"], store, state, backend, offline=True)
    return ports, state, store, backend, built


def test_failed_guest_retains_completed_sibling(acceptance_graph):
    ports, state, store, backend, built = acceptance_graph
    with pytest.raises(RuntimeError):
        run(ports, ["python/good", "python/bad"], store, state, backend, operation=Operation.CHECK)
    good = next(node for node in built.plan.nodes if node.operation is Operation.CHECK and node.port.name == "good")
    assert resolve(store, good.record_key)["stdout"] == "42\n"


def test_selected_guest_rechecks_changed_source_without_building(acceptance_graph):
    ports, state, store, backend, built = acceptance_graph
    first = run(ports, ["python/good"], store, state, backend, operation=Operation.CHECK)
    (ports / "python/good/probe.sh").write_text("printf '43\\n'\n")
    repeat = run(ports, ["python/good"], store, state, backend, operation=Operation.CHECK)
    check = next(node for node in repeat.plan.nodes if node.operation is Operation.CHECK)
    assert repeat.records[check.reference]["stdout"] == "43\n"
    first_build = next(node for node in first.plan.nodes if node.operation is Operation.BUILD)
    assert repeat.records[first_build.reference] == first.records[first_build.reference]
    outside = ports.parent / "outside.txt"
    outside.write_text("host-only input")
    fixture = ports / "python/good/input.txt"
    fixture.unlink()
    fixture.symlink_to(outside)
    with pytest.raises(ValueError):
        run(ports, ["python/good"], store, state, backend, operation=Operation.CHECK)
