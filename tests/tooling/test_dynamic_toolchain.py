"""Check reviewed build inputs and the separation of main and side runtimes."""

import json
from pathlib import Path

import pytest

from ports.dynamic.build_v2 import build_v2
from ports.native.dependencies import file_hash
from ports.numpy.build import check_build_scripts


def test_dynamic_toolchain_pins_inputs_and_imports_canonical_runtime():
    directory = Path(__file__).parents[2] / "ports/toolchain/wasi_sdk"
    recipe = json.loads((directory / "recipe.json").read_text())
    check_build_scripts(recipe, directory)
    for notice in recipe["notices"]:
        assert file_hash(directory / notice["file"]) == notice["sha256"]
        assert (
            recipe["runtime_sources"]["llvm"]["commit"] in notice["url"]
            or recipe["runtime_sources"]["wasi-libc"]["commit"] in notice["url"]
        )
    assert recipe["abi"] == "shellsim-wasi-sdk34-cpython3137-v2"
    assert recipe["loader_namespace"] == "shellsim_dylink_v2"
    assert "-nostdlib" in recipe["side_link_flags"]
    assert "-fPIC" in recipe["side_link_flags"]
    assert any("import-dynamic" in flag for flag in recipe["side_link_flags"])
    assert set(recipe["runtime_archives"]) == {
        "../libc.a",
        "../libc-printscan-long-double.a",
        "libc++.a",
        "libc++abi.a",
        "libunwind.a",
    }


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
        build_v2(bundle, output)
    assert not output.exists()
