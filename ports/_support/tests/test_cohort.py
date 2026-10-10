"""Check admission failures against real producer receipts, not mock inventories."""

import json
import os
import shutil
from pathlib import Path

import pytest

from ports._support.import_sdk import load_legacy_cohort
from ports._support.sdk_products import verify_host_files


@pytest.fixture
def descriptor(tmp_path):
    original = os.environ.get("SHELLSIM_BUILD_COHORT")
    if original is None:
        pytest.skip("requires an explicitly admitted build cohort")
    original = Path(original).resolve()
    value = json.loads(original.read_text())
    for name in ("sdk", "llvm", "sysroot", "cpython", "runtime"):
        if value[name] is not None:
            value[name]["root"] = str((original.parent / value[name]["root"]).resolve())
            value[name]["manifest"] = str((original.parent / value[name]["manifest"]).resolve())
    for item in value["host_tools"].values():
        item["path"] = str((original.parent / item["path"]).resolve())
        if item["receipt"] is not None:
            item["receipt"]["path"] = str((original.parent / item["receipt"]["path"]).resolve())
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(value))
    return path, value


def test_verified_python_receipt_binds_headers_to_assembled_runtime(descriptor):
    path, _ = descriptor
    cohort = load_legacy_cohort(path)
    assert cohort.python.dynamic_abi == cohort.dynamic_abi
    assert cohort.python.runtime_bundle == cohort.runtime.root
    assert (cohort.python.source_root / "Include/Python.h").is_file()
    assert (cohort.python.generated_config_dir / "pyconfig.h").is_file()


def test_receipt_digest_mismatch_is_rejected_before_admission(descriptor):
    path, value = descriptor
    value["sysroot"]["sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_legacy_cohort(path)


def test_target_executable_cannot_be_rebound_to_a_different_admitted_tool(descriptor):
    path, value = descriptor
    value["target_tools"]["cc"] = value["target_tools"]["ar"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_legacy_cohort(path)


def test_patched_uv_requires_its_producer_receipt(descriptor):
    path, value = descriptor
    value["host_tools"]["uv"]["receipt"] = None
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_legacy_cohort(path)


def test_llc_only_product_cannot_be_used_as_a_standard_frontend(descriptor):
    path, _ = descriptor
    cohort = load_legacy_cohort(path)
    if cohort.has_frontend:
        pytest.skip("this behavior requires the independently sealed llc-only product")
    with pytest.raises(ValueError):
        cohort.compiler()
    with pytest.raises(ValueError):
        _ = cohort.compiler_flags


def test_package_tool_rejects_changed_imported_code(tmp_path):
    receipt_path = os.environ.get("SHELLSIM_HOST_TOOL_RECEIPT")
    if receipt_path is None:
        pytest.skip("requires an admitted Meson package tool receipt")
    receipt_path = Path(receipt_path)
    producer = json.loads(receipt_path.read_text())
    original = (receipt_path.parent / producer["root"]).resolve()
    copied = tmp_path / "meson"
    shutil.copytree(original, copied)
    producer["root"] = str(copied)
    executable = copied / producer["executable"]
    if producer["schema_version"] == 2:
        executable.write_text(executable.read_text().replace(str(original), str(copied)))
        from ports._support.sdk_products import file_hash

        producer["files"][producer["executable"]] = file_hash(executable)
    verify_host_files(receipt_path, producer, "meson", executable)
    source = copied / "mesonbuild/mesonmain.py"
    source.write_bytes(source.read_bytes() + b"\nraise RuntimeError('changed tool code')\n")
    with pytest.raises(ValueError):
        verify_host_files(receipt_path, producer, "meson", executable)


def test_compile_flags_keep_linker_selection_in_link_flags(tmp_path):
    from ports._support.sdk_products import MaterializedSDK, Receipt

    sdk = Receipt(tmp_path / "sdk", tmp_path / "sdk.json", "sdk-digest", {})
    compiler = Receipt(tmp_path / "compiler", tmp_path / "compiler.json", "compiler-digest", {})
    sysroot = Receipt(tmp_path / "platform", tmp_path / "platform.json", "platform-digest", {})
    cohort = MaterializedSDK(
        sdk, compiler, sysroot, None, None, None, {}, {}, "cohort-digest", True, "wasm32-wasip1-threads", "test-abi"
    )
    assert not any(flag.startswith("-fuse-ld=") for flag in cohort.compiler_flags)
    assert "-fuse-ld=" + str(compiler.root / "bin/wasm-ld") in cohort.linker_flags
    assert "-resource-dir=" + str(sdk.root / "lib/clang/23") in cohort.compiler_flags


def test_historical_python_driver_pins_do_not_change_source_admission():
    from ports._support.sdk_products import local_recipe, verify_cpython_recipe

    historical = local_recipe("python/cpython/threaded-recipe.json")
    historical["build_scripts"][0]["sha256"] = "0" * 64
    verify_cpython_recipe(historical)


@pytest.mark.parametrize(
    "source",
    [
        "../../toolchain/wasi_process/process.c",
        "../../toolchain/wasi_sdk/posix.h",
        "../../toolchain/wasi_process/cpython-3.13.7-process.patch",
    ],
)
def test_historical_python_receipt_rejects_changed_compiled_inputs(source):
    from ports._support.sdk_products import local_recipe, verify_cpython_recipe

    historical = local_recipe("python/cpython/threaded-recipe.json")
    entry = next(item for item in historical["build_scripts"] if item["file"] == source)
    entry["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_cpython_recipe(historical)


@pytest.mark.parametrize("replacement", [None, "threaded.py"])
def test_historical_python_receipt_cannot_hide_a_compiled_input(replacement):
    from ports._support.sdk_products import local_recipe, verify_cpython_recipe

    historical = local_recipe("python/cpython/threaded-recipe.json")
    entry = next(
        item for item in historical["build_scripts"] if item["file"] == "../../toolchain/wasi_process/process.c"
    )
    if replacement is None:
        historical["build_scripts"].remove(entry)
    else:
        entry["file"] = replacement
    with pytest.raises(ValueError):
        verify_cpython_recipe(historical)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", {"url": "https://example.invalid/python.tar.xz", "sha256": "0" * 64}),
        ("dynamic_abi", "different-abi"),
        ("version", "3.13.8"),
        ("patches", []),
    ],
)
def test_historical_python_receipt_requires_the_accepted_source_and_abi(field, value):
    from ports._support.sdk_products import local_recipe, verify_cpython_recipe

    historical = local_recipe("python/cpython/threaded-recipe.json")
    historical[field] = value
    with pytest.raises(ValueError, match="source or ABI profile"):
        verify_cpython_recipe(historical)
