"""Keep service loss controls independent of provisioning."""

import json
from types import SimpleNamespace

import pytest

from ports.buildomatic import Action, BuildRequest, BuildState, LocalStore, NodeState, TreeBundle
from ports.buildomatic.backends.iris import CoordinatorService, WorkerPool
from ports.buildomatic.tests.test_iris_backend import IrisTaskState, Worker


@pytest.mark.parametrize("failure", ["submit", "status", "identity"])
@pytest.mark.parametrize("cancel", [False, True])
def test_provisioning_uncertainty_preserves_loss_controls_and_independent_work(tmp_path, failure, cancel):
    store = LocalStore(tmp_path / "store")
    statuses, proxies, submissions = {}, {}, []
    unavailable = False

    def submit(name):
        # An uncertain acknowledgement can hide a successful admission. Every
        # retry must recover this exact name, including after reconstruction.
        journal = json.loads(store.read_journal("iris/checkpoint/workers.json").data)
        assert any(record is not None and record["id"] == name for record in journal["slots"])
        submissions.append(name)
        statuses.setdefault(
            name,
            SimpleNamespace(attempt_number=0, attempt_uid="uid-" + name, state=IrisTaskState.RUNNING, finished_at=None),
        )
        if unavailable and failure == "submit":
            raise RuntimeError("uncertain submission")
        return name

    def task_status(name):
        if unavailable and name not in proxies:
            if failure == "status":
                raise RuntimeError("admission lookup unavailable")
            if failure == "identity":
                return SimpleNamespace(current_attempt_number=0, attempts=())
        return SimpleNamespace(current_attempt_number=0, attempts=(statuses[name],))

    def make_worker(name):
        return proxies.setdefault(name, Worker(TreeBundle("c" * 64)))

    def make_pool():
        return WorkerPool(
            store,
            "checkpoint",
            2,
            submit_worker=submit,
            task_status=task_status,
            attempt_status=lambda name, number: statuses[name],
            make_worker=make_worker,
            terminal_states=frozenset({IrisTaskState.FAILED}),
        )

    pool = make_pool()
    available, retired = pool.reconcile()
    old, survivor = tuple(available)
    service = CoordinatorService(store, "checkpoint", available, capacity=2)
    build = service.submit(
        BuildRequest("checkpoint", tuple(Action(name, ("unused",), max_attempts=1) for name in ("a", "b", "c")), 2)
    )
    service.tick_once()
    assert service.get(build).state == BuildState.RUNNING
    statuses[old].state = IrisTaskState.FAILED
    statuses[old].finished_at = "confirmed-terminal"
    unavailable = True
    if cancel:
        service.cancel(build)
    else:
        proxies[survivor].finish()

    replacement = None
    for _ in range(3):
        # Use the production driver's order on every reconstruction.
        pool = make_pool()
        service = CoordinatorService(store, "checkpoint", {}, capacity=2)
        available, retired = pool.reconcile()
        assert set(available) == {survivor}
        assert retired == (old,)
        service.update_workers(available, retired)
        service.tick_once()
        journal = json.loads(store.read_journal(f"iris/checkpoint/builds/{build}.json").data)
        assert journal["lost_workers"] == [old]
        assert journal["cancelled"] is cancel
        membership = json.loads(store.read_journal("iris/checkpoint/workers.json").data)
        assert membership["retired"] == [
            {
                "id": old,
                "task_id": old,
                "attempt_number": 0,
                "attempt_uid": "uid-" + old,
                "terminal_state": "failed",
                "finished_at": "confirmed-terminal",
            }
        ]
        name = membership["slots"][0]["id"]
        assert name != old
        if replacement is None:
            replacement = name
        assert name == replacement
        assert set(statuses) == {old, survivor, replacement}
        assert replacement not in proxies
        proxies[survivor].finish()

    result = service.get(build)
    if cancel:
        assert result.state == BuildState.CANCELLED
        assert all(node.state == NodeState.CANCELLED for node in result.nodes)
        assert proxies[survivor].cancelled
        assert len(proxies[survivor].submitted) == 1
    else:
        assert result.state == BuildState.FAILED
        assert [node.state for node in result.nodes] == [NodeState.FAILED, NodeState.SUCCEEDED, NodeState.SUCCEEDED]
        assert [attempt.action.id for attempt in proxies[survivor].submitted] == ["b", "c"]
    assert submissions[:2] == [old, survivor]
    assert submissions[2:] == [replacement] * (3 if failure == "submit" else 1)
    unavailable = False
    available, retired = pool.reconcile()
    assert set(available) == {survivor, replacement}
    assert retired == (old,)
    assert set(submissions) == {old, survivor, replacement}


@pytest.mark.parametrize("count", [0, 33])
def test_cli_worker_count_precedes_storage(tmp_path, count):
    from ports.__main__ import main

    with pytest.raises(SystemExit):
        main(["build", "python/six", "--state", str(tmp_path / "unused"), "--max-workers", str(count)])
    assert not (tmp_path / "unused").exists()
