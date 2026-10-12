"""Exercise selected-source transport and worker-side source failures."""

import json

import pytest

from ports._support import runner
from ports._support.graph import plan
from ports._support.store import identity
from ports._support.tests.test_runner import _wheel_port
from ports.buildomatic.transport import code_files


def test_code_closure_excludes_unselected_files(tmp_path):
    ports, store = tmp_path / "ports", tmp_path / "store"
    _wheel_port(ports, store, "selected", [])
    _wheel_port(ports, store, "other", [])
    (ports / ".env").write_text("SECRET=hidden")
    (ports / "python/selected/cache").mkdir()
    (ports / "python/selected/cache/secret").write_text("hidden")
    files = code_files(ports, plan(ports, ["python/selected"]))
    assert "python/selected/recipe.json" in files
    assert "_support/node_build.py" in files
    assert "_support/pure_wheel.py" in files
    assert "python/other/recipe.json" not in files
    assert not any("secret" in name or "cache/" in name or name == ".env" for name in files)
    assert not any(name.startswith("buildomatic/") for name in files)


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
    built = runner.build_graph(ports, ["toolchain/fixture"], None, store, offline=True)
    result = built.results["toolchain/fixture/recipe.json"]
    assert (result / "source.txt").read_text() == "pinned source\n"
    assert not (result / "undeclared.txt").exists()


def test_missing_offline_source_fails_in_worker(tmp_path):
    ports, state = tmp_path / "ports", tmp_path / "state"
    _wheel_port(ports, tmp_path / "unavailable", "example", [])
    with pytest.raises(RuntimeError):
        runner.build_graph(ports, ["python/example"], None, state, offline=True)
