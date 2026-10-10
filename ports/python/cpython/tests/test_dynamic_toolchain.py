"""Check reviewed build inputs and reject incompatible interpreter profiles."""

import json
from pathlib import Path

import pytest

from ports._support.graph import plan
from ports._support.producer_policy import verify_policy
from ports._support.runner import _admit_recipe
from ports.native.dependencies import file_hash


def analyze(roots):
    from pathlib import Path

    return plan(Path(__file__).resolve().parents[4] / "ports", roots)


def test_dynamic_toolchain_pins_inputs():
    directory = Path(__file__).parents[4] / "ports/toolchain/wasi_sdk"
    recipe = json.loads((directory / "recipe.json").read_text())
    _admit_recipe(analyze(["toolchain/wasi_sdk"]).ports[0])
    for notice in recipe["source"]["files"]:
        assert file_hash(directory.parents[1] / notice["path"]) == notice["sha256"]


@pytest.mark.parametrize(
    ("profile", "providers"),
    [("wasi-cpython-v1", []), ("wasi-cpython-v2", [{"name": "numpy"}])],
)
def test_dynamic_builder_requires_fixed_bare_sdk34_interpreter(tmp_path, profile, providers):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"recipe": {"target_profile": profile}, "native_ports": providers})
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        verify_policy({"target_profile": profile, "native_ports": providers}, "python/cpython:runtime")
    assert not output.exists()
