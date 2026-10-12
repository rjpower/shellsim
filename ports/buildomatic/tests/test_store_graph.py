"""Keep canonical product storage independent of service journals."""

from ports.buildomatic import LocalStore
from ports.buildomatic.artifacts import register
from ports.buildomatic.backends.iris import CoordinatorService


def test_artifact_catalog_is_independent_of_service_journals(tmp_path):
    canonical = LocalStore(tmp_path / "canonical")
    register(canonical, "sdk/example", {"identity": "sdk"})
    service = CoordinatorService(LocalStore(tmp_path / "service"), "test", {}, artifact_store=canonical)
    assert service.resolve_artifact("sdk/example") == {"identity": "sdk"}
