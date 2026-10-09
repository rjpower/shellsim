"""Build unmodified CPython 3.13.7 _ctypes against pinned SDK34 libffi."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ports._support.build import check_build_scripts
from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import needed_libraries
from ports.native.dependencies import (
    artifact_input,
    digest,
    file_hash,
    seal_artifact,
    target_environment,
    target_profile,
    toolchain_identity,
    verify_artifact,
)
from ports.toolchain.wasi_sdk.build import dynamic_toolchain

PORT = Path(__file__).resolve().parent
SOURCES = ("_ctypes.c", "callbacks.c", "callproc.c", "stgdict.c", "cfield.c")
DL_REMAP = (
    "-Ddlopen=__wrap_dlopen",
    "-Ddlsym=__wrap_dlsym",
    "-Ddlerror=__wrap_dlerror",
    "-Ddlclose=__wrap_dlclose",
)
FFI_FEATURES = (
    "-DHAVE_FFI_CLOSURE_ALLOC=1",
    "-DHAVE_FFI_PREP_CLOSURE_LOC=1",
    "-DHAVE_FFI_PREP_CIF_VAR=1",
)


def _run(command: list[str], directory: Path, environment: dict[str, str], log: Path) -> None:
    with log.open("w") as output:
        subprocess.run(command, cwd=directory, env=environment, stdout=output, stderr=subprocess.STDOUT, check=True)


def build_ctypes(base: Path, runtime: Path, libffi: Path, work: Path) -> tuple[Path, dict]:
    """Seal a side module bound to the real main bridge, Python and libffi."""
    recipe = json.loads((PORT / "stdlib_ctypes_recipe.json").read_text())
    check_build_scripts(recipe, PORT)
    libffi_manifest = verify_artifact(libffi)
    libffi_recipe = libffi_manifest["inputs"]["recipe"]
    shared_recipe = json.loads((PORT.parents[1] / "native/libffi/shared/recipe.json").read_text())
    if (
        libffi_recipe != shared_recipe
        or libffi_recipe["target"] != recipe["target"]
        or libffi_recipe["target_profile"] != recipe["target_profile"]
        or libffi_manifest["inputs"]["dependency_artifacts"].keys() != {"native/libffi"}
    ):
        raise ValueError("_ctypes requires the pinned libffi provider")
    if needed_libraries(libffi / "lib/libffi.so") != libffi_recipe["needed_libraries"]:
        raise ValueError("libffi provider has undeclared native dependencies")
    base_manifest = json.loads((base / "manifest.json").read_text())
    runtime_manifest = json.loads((runtime / "manifest.json").read_text())
    if (
        base_manifest["recipe"]["version"] != "3.13.7"
        or base_manifest["recipe"]["source"]["sha256"] != recipe["source"]["sha256"]
        or runtime_manifest["dynamic_abi"] != recipe["abi"]
        or runtime_manifest["source_bundle_sha256"] != file_hash(base / "manifest.json")
        or runtime_manifest["runtime_sources"]["dynamic.c"] != recipe["main_bridge_sha256"]
        or not runtime_manifest.get("process_port")
        or file_hash(runtime / "rootfs/usr/bin/python3.wasm") != runtime_manifest["files"]["/usr/bin/python3.wasm"]
    ):
        raise ValueError("_ctypes source bundle and main bridge are incompatible")
    sdk = base / "wasi-sdk-34.0-x86_64-linux"
    toolchain, dynamic_identity, _ = dynamic_toolchain(sdk)
    if (
        toolchain["abi"] != recipe["abi"]
        or toolchain["sdk"] != recipe["sdk"]
        or toolchain["target"] != recipe["target"]
        or libffi_manifest["inputs"]["toolchain"] != toolchain_identity(libffi_recipe, sdk)
        or runtime_manifest["dynamic_toolchain"]["identity"] != dynamic_identity
    ):
        raise ValueError("_ctypes SDK and target ABI differ from the pin")
    python_source = base / "Python-3.13.7"
    headers = sorted((*python_source.glob("Include/**/*.h"), *python_source.glob("Modules/_ctypes/*.h")))
    header_digest = digest({path.relative_to(python_source).as_posix(): file_hash(path) for path in headers})
    if len(headers) != recipe["python_headers"]["count"] or header_digest != recipe["python_headers"]["sha256"]:
        raise ValueError("_ctypes CPython header closure differs from its pin")
    for name, expected in recipe["python_files_sha256"].items():
        path = base / name if name.startswith("wasi-build/") else python_source / name
        if file_hash(path) != expected:
            raise ValueError(f"_ctypes CPython source or generated config differs: {name}")
    inputs = artifact_input(recipe, PORT, dynamic_identity, {"native/libffi/shared": libffi_manifest})
    inputs["base_manifest_sha256"] = file_hash(base / "manifest.json")
    inputs["runtime_manifest_sha256"] = file_hash(runtime / "manifest.json")
    identity = digest(inputs)
    prefix = work / "native-artifacts" / identity
    if prefix.exists():
        return prefix, verify_artifact(prefix, inputs)
    build = work / ("ctypes-build-" + identity[:12])
    build.mkdir(parents=True)
    temporary = prefix.with_name(prefix.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "lib-dynload").mkdir(parents=True)
    (temporary / "licenses").mkdir()
    environment = target_environment(sdk)
    profile = target_profile(toolchain)
    include = [
        "-I" + str(python_source / "Include"),
        "-I" + str(python_source / "Include/internal"),
        "-I" + str(base / "wasi-build"),
        "-I" + str(libffi / "include"),
    ]
    output = temporary / "lib-dynload/_ctypes.so"
    _run(
        [
            str(sdk / "bin/clang"),
            *profile["compiler_flags"],
            *profile["cpp_flags"],
            *toolchain["side_link_flags"],
            *DL_REMAP,
            *FFI_FEATURES,
            *include,
            *(str(python_source / "Modules/_ctypes" / name) for name in SOURCES),
            "-L" + str(libffi / "lib"),
            "-lffi",
            "-o",
            str(output),
        ],
        build,
        environment,
        build / "compile.log",
    )
    mark_abi(output, recipe["abi"].encode())
    if needed_libraries(output) != ["libffi.so"]:
        raise ValueError("_ctypes must declare its independent libffi.so dependency")
    shutil.copyfile(python_source / "LICENSE", temporary / "licenses/cpython.txt")
    shutil.copyfile(libffi / "licenses/libffi.txt", temporary / "licenses/libffi.txt")
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("libffi", type=Path)
    parser.add_argument("work", type=Path)
    arguments = parser.parse_args()
    prefix, manifest = build_ctypes(
        arguments.base.resolve(),
        arguments.runtime.resolve(),
        arguments.libffi.resolve(),
        arguments.work.resolve(),
    )
    print(json.dumps({"prefix": str(prefix), "artifact_sha256": manifest["artifact_sha256"]}))


if __name__ == "__main__":
    main()
