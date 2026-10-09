"""Assemble an unchanged threaded CPython image with its verified FFI closure."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ports._support.build import check_build_scripts
from ports._support.wasm_metadata import needed_libraries
from ports.native.dependencies import file_hash, verify_artifact
from ports.python.cpython.assembly import _copy, _verified_rootfs


def build(runtime, libffi, ctypes, output):
    """Verify both dependency edges before publishing a new rootfs bundle."""
    directory = Path(__file__).resolve().parent
    recipe = json.loads((directory / "threaded_ffi_assembly_recipe.json").read_bytes())
    check_build_scripts(recipe, directory)
    if output.exists() or output.is_symlink():
        raise ValueError("threaded FFI output already exists")
    base = json.loads((runtime / "manifest.json").read_bytes())
    _verified_rootfs(runtime, base)
    provider = verify_artifact(libffi)
    extension = verify_artifact(ctypes)
    provider_recipe = json.loads((directory.parents[1] / "native/libffi/threaded/recipe.json").read_bytes())
    extension_recipe = json.loads((directory / "stdlib_ctypes_threaded_recipe.json").read_bytes())
    if (
        provider["inputs"]["recipe"] != provider_recipe
        or extension["inputs"]["recipe"] != extension_recipe
        or base["dynamic_abi"] != "shellsim-wasi-sdk34-cpython3137-threads-v3"
        or extension["inputs"]["runtime_manifest_sha256"] != file_hash(runtime / "manifest.json")
        or extension["inputs"]["dependency_artifacts"] != {"native/libffi/threaded": provider["artifact_sha256"]}
        or provider["inputs"]["toolchain"] != extension["inputs"]["toolchain"]
        or base["build_profile"]["sysroot"] != provider["inputs"]["toolchain"]["overlay"]
        or base["build_profile"]["compiler"] != provider["inputs"]["toolchain"]["compiler"]
        or needed_libraries(libffi / "lib/libffi.so") != []
        or needed_libraries(ctypes / "lib-dynload/_ctypes.so") != ["libffi.so"]
    ):
        raise ValueError("threaded FFI dependency graph or runtime differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".partial-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        shutil.copytree(runtime / "rootfs", stage / "rootfs")
        _verified_rootfs(stage, base)
        files = dict(base["files"])
        for prefix, manifest, source, destination in (
            (libffi, provider, "lib/libffi.so", "/lib/libffi.so"),
            (ctypes, extension, "lib-dynload/_ctypes.so", "/usr/lib/python3.13/lib-dynload/_ctypes.so"),
            (libffi, provider, "licenses/libffi.txt", "/TOOLCHAIN-LICENSES/libffi.txt"),
            (ctypes, extension, "licenses/cpython.txt", "/TOOLCHAIN-LICENSES/ctypes-cpython.txt"),
        ):
            expected = manifest["files"][source]
            _copy(stage / "rootfs", destination, prefix / source, expected)
            files[destination] = expected
        result = {**base, "files": files}
        result["ffi_runtime"] = {
            "runtime_manifest_sha256": file_hash(runtime / "manifest.json"),
            "libffi_artifact_sha256": provider["artifact_sha256"],
            "ctypes_artifact_sha256": extension["artifact_sha256"],
            "assembly_recipe": recipe,
        }
        _verified_rootfs(stage, result)
        (stage / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        stage.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("runtime", "libffi", "ctypes", "output")
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(build(*(getattr(args, name).resolve() for name in names)))


if __name__ == "__main__":
    main()
