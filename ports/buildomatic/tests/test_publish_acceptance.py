"""Guest failures retain successful sibling receipts and requested-port selection."""

import json

import pytest

from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic import publish


@pytest.fixture
def acceptance_graph(tmp_path, monkeypatch):
    pytest.importorskip("shellsim._native")
    ports = tmp_path / "ports"
    for name in ("good", "bad"):
        path, recipe = _wheel_port(ports, tmp_path / "sources", name, [])
        recipe["tests"] = [{"kind": "python", "script": "probe.py"}]
        path.write_text(json.dumps(recipe))
        (path.parent / "probe.py").write_text("pass\n")
    monkeypatch.setattr(publish, "__file__", str(ports / "buildomatic/publish.py"))
    return {"roots": ["python/good", "python/bad"], "sdk_name": "fixture"}


@pytest.mark.parametrize("max_workers", [1, 4])
def test_failed_guest_retains_successful_sibling_receipt(tmp_path, monkeypatch, acceptance_graph, max_workers):
    calls = {}

    def accept(request):
        calls[request.port.name] = request.output
        if request.port.name == "bad":
            raise RuntimeError("guest probe failed")
        request.output.mkdir(parents=True)
        (request.output / "receipt.json").write_text('{"passed": true}\n')

    monkeypatch.setattr("ports._support.acceptance.accept_port", accept)
    output = tmp_path / "proofs"
    with pytest.raises(RuntimeError):
        publish.accept_release(
            acceptance_graph, tmp_path / "release.json", output, cpu=123, memory=456, max_workers=max_workers
        )
    assert set(calls) == {"good", "bad"}
    assert calls["good"] != calls["bad"]
    assert json.loads((output / "good/receipt.json").read_text())["passed"]


def test_only_selected_guest_receives_requested_limits(tmp_path, monkeypatch, acceptance_graph):
    calls = []
    monkeypatch.setattr("ports._support.acceptance.accept_port", calls.append)
    publish.accept_release(
        acceptance_graph, tmp_path / "release.json", tmp_path / "proofs", cpu=123, memory=456, only=["good"]
    )
    assert [request.port.name for request in calls] == ["good"]
    assert calls[0].limits.cpu == 123
    assert calls[0].limits.memory == 456


@pytest.mark.parametrize("max_workers", [0, 33])
def test_invalid_guest_concurrency_fails_before_execution(tmp_path, acceptance_graph, max_workers):
    with pytest.raises(ValueError):
        publish.accept_release(
            acceptance_graph,
            tmp_path / "release.json",
            tmp_path / "proofs",
            cpu=123,
            memory=456,
            max_workers=max_workers,
        )
