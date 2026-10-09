"""Exercise dependency rebuild propagation and cache rejection through real adapters."""

import hashlib
import json
import zipfile
from dataclasses import dataclass

import pytest

from ports._support import runner


@dataclass(frozen=True)
class PureCohort:
    target: str = "wasm32-wasip1-threads"
    dynamic_abi: str = "test-pure-cohort"
    identity: str = "pinned-cohort"


def _wheel_port(ports, store, name, dependencies):
    directory = ports / "python" / name
    directory.mkdir(parents=True)
    wheel = directory / f"{name}-1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{name}.py", "VALUE = 42\n")
        archive.writestr(f"{name}-1.dist-info/METADATA", f"Name: {name}\nVersion: 1\n")
        archive.writestr(f"{name}-1.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    source = store / "sources" / digest / wheel.name
    source.parent.mkdir(parents=True)
    source.write_bytes(wheel.read_bytes())
    recipe = {
        "name": name,
        "version": "1",
        "build": {"adapter": "pure-wheel"},
        "source": {"filename": wheel.name, "url": "https://example.invalid/" + wheel.name, "sha256": digest},
        "target_dependencies": [{"port": "python/" + dependency, "version": "1"} for dependency in dependencies],
    }
    path = directory / "recipe.json"
    path.write_text(json.dumps(recipe))
    return path, recipe


def test_graph_reuses_verified_builds_and_rebuilds_dependents(tmp_path, monkeypatch):
    ports, store = tmp_path / "ports", tmp_path / "store"
    provider, recipe = _wheel_port(ports, store, "provider", [])
    _wheel_port(ports, store, "consumer", ["provider"])
    calls = []
    real_build = runner.build_pure_wheel

    def observed_build(request):
        calls.append(request.recipe["name"])
        return real_build(request)

    monkeypatch.setattr(runner, "build_pure_wheel", observed_build)
    first = runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert calls == ["provider", "consumer"]
    calls.clear()
    second = runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert not calls
    assert second.results == first.results

    provider.write_text(json.dumps({**recipe, "features": {"new-build-input": True}}))
    third = runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert calls == ["provider", "consumer"]
    assert third.results["python/consumer/recipe.json"] != first.results["python/consumer/recipe.json"]

    artifact = third.results["python/provider/recipe.json"] / "wheels/provider-1-py3-none-any.whl"
    artifact.write_bytes(b"corrupt")
    calls.clear()
    with pytest.raises(ValueError):
        runner.build_graph(ports, ["python/consumer"], PureCohort(), store, offline=True)
    assert not calls
