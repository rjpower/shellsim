"""Check reviewed build inputs and reject incompatible interpreter profiles."""

import json
from pathlib import Path

import pytest

from ports._support.build import check_build_scripts
from ports.native.dependencies import file_hash
from ports.python.cpython.dynamic import build


def test_dynamic_toolchain_pins_inputs():
    directory = Path(__file__).parents[4] / "ports/toolchain/wasi_sdk"
    recipe = json.loads((directory / "recipe.json").read_text())
    check_build_scripts(recipe, directory)
    for notice in recipe["notices"]:
        assert file_hash(directory / notice["file"]) == notice["sha256"]


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
        build(bundle, output)
    assert not output.exists()
