"""Use real offline adapters and private core workers; no cloud or SDK is needed."""

import json
import shutil
import subprocess
import sys

import pytest

from ports._support import runner
from ports._support.graph import plan
from ports._support.store import identity
from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic.ports import code_files, prepare_graph, publish_manifest, run_graph


def test_local_cli_imports_no_iris():
    script = """
import importlib.abc, runpy, sys
class RejectCloud(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'iris', 'rigging', 'cw'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectCloud())
sys.argv = ['ports', '--help']
runpy.run_module('ports', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, check=True)
    assert "--backend" in result.stdout


@pytest.mark.parametrize("failure", ["key", "missing", "bytes", "mode"])
def test_admitted_predecessor_never_rebuilds(tmp_path, monkeypatch, failure):
    from ports import api

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "provider", [])
    _wheel_port(ports, store, "consumer", ["provider"])
    built = runner.build_graph(ports, ["python/provider"], None, store, offline=True)
    result = built.results["python/provider/recipe.json"]
    expected = result.name
    if failure == "key":
        expected = "0" * 64
    elif failure == "missing":
        shutil.rmtree(result)
    elif failure == "bytes":
        next((result / "wheels").iterdir()).write_bytes(b"changed")
    else:
        next((result / "wheels").iterdir()).chmod(0o755)
    calls = []
    monkeypatch.setattr(api, "build_port", lambda ctx: calls.append(ctx.port.reference))
    with pytest.raises(ValueError):
        runner.build_graph(
            ports,
            ["python/consumer"],
            None,
            store,
            offline=True,
            admitted_predecessors={"python/provider/recipe.json": expected},
        )
    assert calls == []


def test_code_closure_excludes_unselected_files(tmp_path):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "selected", [])
    _wheel_port(ports, store, "other", [])
    (ports / ".env").write_text("SECRET=hidden")
    (ports / "python/selected/cache").mkdir()
    (ports / "python/selected/cache/secret").write_text("hidden")
    files = code_files(ports, plan(ports, ["python/selected"]))
    assert "python/selected/recipe.json" in files
    assert "_support/runner.py" in files
    assert "_support/pure_wheel.py" in files
    assert "python/other/recipe.json" not in files
    assert not any("secret" in name or "cache/" in name or name == ".env" for name in files)
    assert not any(name.startswith("buildomatic/") for name in files)


def test_worker_builds_complete_dependency_results_and_manifest(tmp_path):
    ports, store = tmp_path / "ports", tmp_path / "store"
    for name, dependencies in (("base", []), ("provider", ["base"]), ("consumer", ["provider"])):
        _wheel_port(ports, store, name, dependencies)
    built = run_graph(ports, ["python/consumer"], None, store, offline=True, max_workers=2)
    assert len(built.results) == 3
    for name in ("base", "provider", "consumer"):
        assert (built.results[f"python/{name}/recipe.json"] / f"wheels/{name}-1-py3-none-any.whl").is_file()
    manifest = json.loads(next((store / "buildomatic/manifests").glob("*.json")).read_text())
    assert manifest["results"] == {reference: result.name for reference, result in built.results.items()}
    assert len(manifest["bundles"]) == 3
    assert not (store / "release.json").exists()


def test_source_tree_worker_transports_only_declared_sources(tmp_path):
    ports, store = tmp_path / "ports", tmp_path / "store"
    directory = ports / "toolchain/fixture"
    directory.mkdir(parents=True)
    source = directory / "source.txt"
    source.write_text("pinned source\n")
    from ports._support.store import file_hash

    entries = [{"path": "toolchain/fixture/source.txt", "destination": "source.txt", "sha256": file_hash(source)}]
    recipe = {
        "name": "fixture",
        "version": "1",
        "role": "host-tool",
        "build_system": "source-tree",
        "source": {"files": entries, "sha256": identity(entries)},
    }
    (directory / "recipe.json").write_text(json.dumps(recipe))
    (directory / "build.py").write_text(
        "import shutil\ndef build(ctx):\n    shutil.copyfile(ctx.source / 'source.txt', ctx.result / 'source.txt')\n"
    )
    (directory / "undeclared.txt").write_text("not transported")
    built = run_graph(ports, ["toolchain/fixture"], None, store, offline=True)
    result = built.results["toolchain/fixture/recipe.json"]
    assert (result / "source.txt").read_text() == "pinned source\n"
    assert not (result / "undeclared.txt").exists()


def test_manifest_publication_checks_cache_before_publishing(tmp_path, monkeypatch):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    built = run_graph(ports, ["python/example"], None, store, offline=True)
    manifest = next((store / "buildomatic/manifests").glob("*.json"))
    calls = []
    monkeypatch.setattr(
        runner, "publish_graph", lambda build, sdk, output: calls.append(build) or output / "release.json"
    )
    output = tmp_path / "release"
    assert publish_manifest(manifest, ports, None, store, output) == output / "release.json"
    assert calls[0].results == built.results
    shutil.rmtree(next(iter(built.results.values())))
    calls.clear()
    with pytest.raises(ValueError):
        publish_manifest(manifest, ports, None, store, output)
    assert calls == []


def test_preparation_missing_source_fails_offline(tmp_path):
    from ports.buildomatic import LocalStore

    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "example", [])
    shutil.rmtree(store / "sources")
    with pytest.raises(ValueError):
        prepare_graph(ports, ["python/example"], None, store, LocalStore(tmp_path / "blobs"), offline=True)
