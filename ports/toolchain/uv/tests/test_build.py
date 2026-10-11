"""Check source admission and the sealed resolver's executable identity."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from ports import api
from ports._support.graph import Port
from ports._support.producer_policy import load_policy, metadata
from ports._support.sdk import HostSeed, SDKProduct, node_inputs
from ports._support.sdk_products import json_hash
from ports.toolchain.uv import producer as build


@pytest.fixture
def producer_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Port]:
    directory, recipe = metadata("toolchain/uv")
    original = directory.parents[1]
    port = Port("toolchain/uv/recipe.json", directory, recipe["name"], recipe["version"], "", (), recipe)
    files = {"toolchain/uv/build.py" if name == "builder" else name for name in api.implementation(port)} | {
        "api.py",
        "toolchain/uv/recipe.json",
        "toolchain/uv/tests/verify_target.py",
        "toolchain/uv/" + recipe["patch"]["file"],
    }
    root = tmp_path / "ports"
    for name in sorted(files):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, target)
    monkeypatch.setattr(api, "__file__", str(root / "api.py"))
    directory = root / "toolchain/uv"
    monkeypatch.setattr(build, "HERE", directory)
    monkeypatch.setattr(build, "RECIPE", load_policy("toolchain/uv", root))
    return root, Port(port.reference, directory, port.name, port.version, "", (), recipe)


def test_verifier_is_a_producer_input_and_changes_product_identity(producer_tree):
    root, port = producer_tree
    verifier = port.directory / "tests/verify_target.py"
    expected = build.file_hash(verifier)
    assert api.implementation(port)["toolchain/uv/tests/verify_target.py"] == expected
    assert {item["file"]: item["sha256"] for item in build.RECIPE["build_scripts"]}[
        "tests/verify_target.py"
    ] == expected
    assert build._inputs()["tests/verify_target.py"] == expected
    node = SDKProduct("resolver", "uv-host", port.directory / "recipe.json", build.RECIPE, (), port)
    before = node_inputs(node, {}, HostSeed({}, {}, None))

    verifier.write_text(verifier.read_text() + "\n# Changed verifier implementation.\n")

    after = node_inputs(node, {}, HostSeed({}, {}, None))
    assert json_hash(after) != json_hash(before)
    current = load_policy("toolchain/uv", root)
    assert {item["file"]: item["sha256"] for item in current["build_scripts"]}[
        "tests/verify_target.py"
    ] == build.file_hash(verifier)


@pytest.mark.parametrize(("change", "error"), [("missing", FileNotFoundError), ("tampered", ValueError)])
def test_missing_or_tampered_verifier_fails_before_build_tools(producer_tree, tmp_path, monkeypatch, change, error):
    _, port = producer_tree
    verifier = port.directory / "tests/verify_target.py"
    if change == "missing":
        verifier.unlink()
    else:
        verifier.write_text(verifier.read_text() + "\n# Tampered after input admission.\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("host tools must not run after a verifier input mismatch")

    monkeypatch.setattr(build, "_tool", forbidden)
    output = tmp_path / "release"
    with pytest.raises(error):
        build.build(output)
    assert not output.exists()


def test_changed_driver_is_rejected_before_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "build.py").write_text("changed")
    monkeypatch.setattr(build, "HERE", tmp_path)
    monkeypatch.setattr(build, "RECIPE", {"build_scripts": [{"file": "build.py", "sha256": "0" * 64}]})

    def no_checkout(*args, **kwargs):
        raise AssertionError("source checkout must not start after a pin mismatch")

    monkeypatch.setattr(build.subprocess, "run", no_checkout)
    with pytest.raises(ValueError, match="build input changed"):
        build.build(tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_verified_distributable_matches_measured_binary() -> None:
    configured = os.environ.get("SHELLSIM_UV_RELEASE_ARTIFACT")
    if configured is None:
        pytest.skip("set SHELLSIM_UV_RELEASE_ARTIFACT to a built resolver directory")
    root = Path(configured)
    manifest = json.loads((root / "artifact.json").read_text())
    binary = root / manifest["executable"]["file"]
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == manifest["executable"]["sha256"]
    assert binary.stat().st_size == manifest["executable"]["size"]
