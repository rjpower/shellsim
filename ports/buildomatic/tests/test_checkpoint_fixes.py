"""Keep service controls independent of provisioning, using only fake workers.

Pool and service reconstruction exercise durable retirement and uncertain
same-name admission. Bridge spies reject setup before invalid count admission.
"""

import json
import sys
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


@pytest.mark.parametrize("count", [0, 33, -1, True, False, 1.0, "1", None, 10**100])
@pytest.mark.parametrize("entry", ["local", "connected", "descriptor", "prepare"])
def test_bad_worker_count_precedes_all_preparation(tmp_path, monkeypatch, count, entry):
    from ports.buildomatic import ports as bridge
    from ports.buildomatic.backends import iris

    calls = []

    def reject(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid count reached setup")

    remote = SimpleNamespace(upload_bundles=reject, submit=reject, get=reject, download_bundles=reject)
    with monkeypatch.context() as guarded:
        guarded.setattr("ports.buildomatic.LocalStore", reject)
        guarded.setattr("ports.buildomatic.WorkerExecutor", reject)
        guarded.setattr(iris, "backend_context_factory", reject)
        guarded.setattr(bridge, "plan", reject)
        guarded.setattr(bridge, "fetch", reject)
        guarded.setattr(bridge, "_product_results", reject)
        guarded.setattr(type(tmp_path), "mkdir", reject)
        guarded.setattr(type(tmp_path), "stat", reject)
        guarded.setattr(type(tmp_path), "read_text", reject)
        with pytest.raises(ValueError):
            if entry == "prepare":
                bridge.prepare_graph(tmp_path, [], object(), tmp_path, object(), max_workers=count)
            else:
                guarded.setattr(bridge, "prepare_graph", reject)
                bridge.run_graph(
                    tmp_path,
                    [],
                    object(),
                    tmp_path,
                    max_workers=count,
                    backend="buildomatic" if entry == "local" else "iris",
                    remote_backend=remote if entry == "connected" else None,
                    iris_service=tmp_path / "service.json" if entry == "descriptor" else None,
                )
    assert calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("count", [None, 1, 32])
def test_worker_count_bounds_preserve_local_default(tmp_path, monkeypatch, count):
    from ports.buildomatic import ports as bridge

    expected = 1 if count is None else count
    roots, accepted = [], []

    def worker(store, root, **kwargs):
        roots.append(root)
        assert kwargs["max_running"] == 1
        return object()

    def prepare(*args, max_workers, **kwargs):
        request = BuildRequest("bounds", (Action("a", ("unused",)),), max_workers)
        return SimpleNamespace(request=request)

    class Coordinator:
        def __init__(self, store, key, workers):
            assert len(workers) == expected

        def submit(self, request):
            accepted.append(request.max_workers)

        def tick(self):
            return SimpleNamespace(state=BuildState.SUCCEEDED)

        def acknowledge(self):
            pass

    monkeypatch.setattr("ports.buildomatic.WorkerExecutor", worker)
    monkeypatch.setattr("ports.buildomatic.Coordinator", Coordinator)
    monkeypatch.setattr(bridge, "prepare_graph", prepare)
    result = object()
    monkeypatch.setattr(bridge, "collect_graph", lambda *args: result)
    kwargs = {} if count is None else {"max_workers": count}
    assert bridge.run_graph(tmp_path, [], None, tmp_path, **kwargs) is result
    assert accepted == [expected]
    assert roots == [tmp_path / f"buildomatic/worker-{index}" for index in range(expected)]


@pytest.mark.parametrize("backend", ["local", "buildomatic", "iris"])
@pytest.mark.parametrize("count", ["0", "33", "-1", str(10**100), "True", "1.0", "1", "32"])
def test_cli_worker_count_preflight_precedes_graph_and_sdk_setup(tmp_path, monkeypatch, backend, count):
    from ports._support import runner, sdk
    from ports.buildomatic import portable
    from ports.buildomatic import ports as bridge
    from ports.buildomatic.backends import iris

    calls = []
    context = object()
    graph = SimpleNamespace(ports=(SimpleNamespace(recipe={"build_system": "python-extension"}),))

    def spy(name, result):
        def invoke(*args, **kwargs):
            calls.append(name)
            return result

        return invoke

    def execute(ports, requests, sdk_context, store, **kwargs):
        calls.append("execute")
        assert sdk_context is context
        if backend == "local":
            assert "max_workers" not in kwargs
        else:
            assert kwargs["max_workers"] == int(count)
        return SimpleNamespace(results={})

    def reject_setup(*args, **kwargs):
        calls.append("unexpected setup")
        raise AssertionError("CLI spy reached preparation, connection or writes")

    monkeypatch.setattr(runner, "plan", spy("graph", graph))
    monkeypatch.setattr(sdk, "materialize", spy("materialize", context))
    monkeypatch.setattr(portable, "original_root_bindings", spy("bindings", {}))
    monkeypatch.setattr(portable, "import_sdk", spy("import", context))
    monkeypatch.setattr(runner, "build_graph", execute)
    monkeypatch.setattr(bridge, "run_graph", execute)
    monkeypatch.setattr(bridge, "prepare_graph", reject_setup)
    monkeypatch.setattr(iris, "backend_context_factory", reject_setup)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ports",
            "native/example",
            "--store",
            str(tmp_path / "store"),
            "--backend",
            backend,
            "--max-workers",
            count,
            "--sdk-descriptor",
            str(tmp_path / "sdk.json"),
            "--iris-service",
            str(tmp_path / "iris.json"),
        ],
    )
    accepted = count in {"1", "32"} or (backend == "local" and count not in {"True", "1.0"})
    with monkeypatch.context() as guarded:
        for method in ("mkdir", "open", "write_text", "write_bytes"):
            guarded.setattr(type(tmp_path), method, reject_setup)
        if accepted:
            runner.main()
        else:
            with pytest.raises(SystemExit) as error:
                runner.main()
            assert error.value.code == 2
    expected = []
    if accepted:
        expected = (
            ["graph", "materialize", "execute"] if backend == "local" else ["graph", "bindings", "import", "execute"]
        )
    assert calls == expected
    assert list(tmp_path.iterdir()) == []
