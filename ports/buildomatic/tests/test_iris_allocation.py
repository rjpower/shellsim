"""Fake Iris admission to exercise ownership across uncertain startup failures."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.types import JobName
from iris.resources.state import JobState

from ports.buildomatic import Action, BuildRequest, LocalStore
from ports.buildomatic.backends import iris


@pytest.fixture
def controller(monkeypatch):
    jobs = {}

    class Job:
        def __init__(self, identity):
            self.job_id, self.status = identity, JobState.RUNNING

        @property
        def state(self):
            return self.status

        def cancel(self):
            self.status = JobState.KILLED

    def job(identity):
        return jobs.setdefault(str(identity), Job(identity))

    client = SimpleNamespace(job=job)
    monkeypatch.setattr("iris.cli.connect.open_iris_client", lambda **_: nullcontext(client))
    monkeypatch.setattr("iris.client.client.get_iris_ctx", lambda: None)
    monkeypatch.setattr("iris.cluster.client.resolve_job_user", lambda _: "test")
    return client, job(JobName.root("test", "buildomatic-owned"))


@pytest.mark.parametrize("failure", ["launch-acknowledgement", "registration", "build"])
def test_owned_pool_stops_on_startup_or_build_failure(tmp_path, monkeypatch, controller, failure):
    client, job = controller
    closed = []

    def launch(*_):
        if failure == "launch-acknowledgement":
            raise ConnectError(Code.UNAVAILABLE, "uncertain admission")
        if failure == "registration":
            job.status = JobState.FAILED
        return job

    def discover(*_):
        if failure == "registration":
            raise ConnectError(Code.NOT_FOUND, "not registered")
        return SimpleNamespace(submit=lambda _: "build", close=lambda: closed.append(True))

    monkeypatch.setattr(iris, "launch", launch)
    monkeypatch.setattr(iris, "discover", discover)
    expected = ConnectError if failure == "launch-acknowledgement" else RuntimeError
    with pytest.raises(expected):
        with iris.AutomaticIrisBackend(iris.IrisConfig("owned"), LocalStore(tmp_path / "store"), tmp_path) as backend:
            backend.submit(BuildRequest("request", (Action("action", ("true",)),)))
            raise RuntimeError("build failed")
    assert job.state is JobState.KILLED
    if failure == "build":
        assert closed


def test_owned_pool_stops_when_startup_is_interrupted(tmp_path, monkeypatch, controller):
    _, job = controller
    monkeypatch.setattr(iris, "launch", lambda *_: job)

    def interrupt(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(iris, "discover", interrupt)
    with pytest.raises(KeyboardInterrupt):
        with iris.AutomaticIrisBackend(iris.IrisConfig("owned"), LocalStore(tmp_path / "store"), tmp_path) as backend:
            backend.submit(BuildRequest("request", (Action("action", ("true",)),)))
    assert job.state is JobState.KILLED
