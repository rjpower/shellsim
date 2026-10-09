"""Assemble verified process CPython, stdlib zlib, and shared ctypes artifacts."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ports._support.build import check_build_scripts
from ports._support.wasm_metadata import needed_libraries
from ports.native.dependencies import file_hash, toolchain_identity, verify_artifact

PORT = Path(__file__).resolve().parent
MAX_FILES = 10_000
MAX_BYTES = 128 * 1024 * 1024
ADDITIONS = {
    "zlib.so": "/usr/lib/python3.13/lib-dynload/zlib.so",
    "libz.so": "/lib/libz.so",
    "_ctypes.so": "/usr/lib/python3.13/lib-dynload/_ctypes.so",
    "libffi.so": "/lib/libffi.so",
}


def _verified_rootfs(bundle: Path, manifest: dict) -> None:
    rootfs = bundle / "rootfs"
    if rootfs.is_symlink() or not rootfs.is_dir():
        raise ValueError("input CPython rootfs must be a real directory")
    files = manifest["files"]
    if not isinstance(files, dict) or len(files) > MAX_FILES:
        raise ValueError("input CPython rootfs has too many files")
    actual = {}
    total = 0
    for path in rootfs.rglob("*"):
        if path.is_symlink():
            raise ValueError("input CPython rootfs contains a symbolic link")
        if path.is_file():
            total += path.stat().st_size
            if total > MAX_BYTES or len(actual) >= MAX_FILES:
                raise ValueError("input CPython rootfs exceeds its size limit")
            actual["/" + path.relative_to(rootfs).as_posix()] = file_hash(path)
        elif not path.is_dir():
            raise ValueError("input CPython rootfs contains a special file")
    if actual != files:
        raise ValueError("input CPython rootfs differs from its manifest")


def _copy(rootfs: Path, destination: str, source: Path, expected_hash: str) -> None:
    if file_hash(source) != expected_hash:
        raise ValueError(f"native artifact changed: {source}")
    target = rootfs / destination.lstrip("/")
    if target.exists():
        raise ValueError(f"input CPython already contains {destination}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if file_hash(target) != expected_hash:
        raise ValueError(f"native artifact changed while copying: {source}")


def build_runtime(process: Path, zlib: Path, libffi: Path, ctypes: Path, output: Path) -> None:
    """Stage only the pinned native closure into a fresh, integrity-checked bundle."""
    recipe = json.loads((PORT / "assembly_recipe.json").read_text())
    check_build_scripts(recipe, PORT)
    if (process / "manifest.json").stat().st_size > 1024 * 1024:
        raise ValueError("input CPython manifest exceeds 1 MiB")
    base = json.loads((process / "manifest.json").read_text())
    _verified_rootfs(process, base)
    if (
        base["dynamic_abi"] != recipe["abi"]
        or base["recipe"]["version"] != "3.13.7"
        or base["recipe"]["target"] != recipe["target"]
        or base["recipe"]["prefix"] != "/usr"
        or base["recipe"]["target_profile"] != "wasi-cpython-v2"
        or base["site_packages"] != "/usr/lib/python3.13/site-packages"
        or base["runtime_sources"]["dynamic.c"] != recipe["main_bridge_sha256"]
        or base["process_port"]["recipe"]["patch_sha256"] != recipe["process_patch_sha256"]
    ):
        raise ValueError("CPython process image is outside the pinned ctypes cohort")
    zlib_manifest = json.loads((zlib / "manifest.json").read_text())
    zlib_files = zlib_manifest["files"]
    if (
        zlib_manifest["schema_version"] != 1
        or zlib_manifest["abi"] != recipe["abi"]
        or zlib_manifest["recipe"]["source"]["sha256"] != recipe["cpython_source_sha256"]
        or zlib_manifest["recipe"]["zlib_source"]["sha256"] != recipe["zlib_source_sha256"]
        or {item["path"]: item["native_dependencies"] for item in zlib_manifest["artifacts"]}
        != {"zlib.so": ["libz.so"], "libz.so": []}
        or set(zlib_files) != {"zlib.so", "libz.so", "PYTHON-LICENSE", "ZLIB-LICENSE"}
    ):
        raise ValueError("stdlib zlib artifact differs from its pinned ABI and source")
    for name, expected in zlib_files.items():
        if file_hash(zlib / name) != expected:
            raise ValueError(f"stdlib zlib artifact changed: {name}")
    shared = verify_artifact(libffi)
    extension = verify_artifact(ctypes)
    shared_recipe = json.loads((PORT.parents[1] / "native/libffi/shared/recipe.json").read_text())
    ctypes_recipe = json.loads((PORT / "stdlib_ctypes_recipe.json").read_text())
    sdk = Path(base["source_bundle"]) / "wasi-sdk-34.0-x86_64-linux"
    if (
        shared["inputs"]["recipe"] != shared_recipe
        or shared_recipe["target"] != recipe["target"]
        or shared_recipe["target_profile"] != "wasi-cpython-v2"
        or shared["inputs"]["toolchain"] != toolchain_identity(shared_recipe, sdk)
        or shared["inputs"]["dependency_artifacts"].keys() != {"native/libffi"}
        or extension["inputs"]["recipe"] != ctypes_recipe
        or extension["inputs"]["toolchain"] != base["dynamic_toolchain"]["identity"]
        or extension["inputs"]["dependency_artifacts"] != {"native/libffi/shared": shared["artifact_sha256"]}
        or extension["inputs"]["runtime_manifest_sha256"] != file_hash(process / "manifest.json")
        or zlib_manifest["toolchain"] != base["dynamic_toolchain"]["identity"]
        or needed_libraries(libffi / "lib/libffi.so") != []
        or needed_libraries(ctypes / "lib-dynload/_ctypes.so") != ["libffi.so"]
        or needed_libraries(zlib / "zlib.so") != ["libz.so"]
        or needed_libraries(zlib / "libz.so") != []
    ):
        raise ValueError("ctypes, libffi, zlib or process dependency closure differs from its pin")
    sources = {
        "zlib.so": (zlib / "zlib.so", zlib_files["zlib.so"]),
        "libz.so": (zlib / "libz.so", zlib_files["libz.so"]),
        "_ctypes.so": (ctypes / "lib-dynload/_ctypes.so", extension["files"]["lib-dynload/_ctypes.so"]),
        "libffi.so": (libffi / "lib/libffi.so", shared["files"]["lib/libffi.so"]),
    }
    if output.exists():
        raise ValueError("output CPython bundle already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        rootfs = temporary / "rootfs"
        shutil.copytree(process / "rootfs", rootfs)
        _verified_rootfs(temporary, base)
        expected_files = dict(base["files"])
        for name, destination in ADDITIONS.items():
            source, expected = sources[name]
            _copy(rootfs, destination, source, expected)
            expected_files[destination] = expected
        for source, name, expected in (
            (zlib / "PYTHON-LICENSE", "stdlib-zlib-PYTHON-LICENSE", zlib_files["PYTHON-LICENSE"]),
            (zlib / "ZLIB-LICENSE", "stdlib-zlib-ZLIB-LICENSE", zlib_files["ZLIB-LICENSE"]),
            (libffi / "licenses/libffi.txt", "libffi-LICENSE", shared["files"]["licenses/libffi.txt"]),
        ):
            destination = "/TOOLCHAIN-LICENSES/" + name
            _copy(rootfs, destination, source, expected)
            expected_files[destination] = expected
        result = dict(base)
        result["files"] = expected_files
        _verified_rootfs(temporary, result)
        result["ffi_runtime"] = {
            "recipe": recipe,
            "process_manifest_sha256": file_hash(process / "manifest.json"),
            "zlib_manifest_sha256": file_hash(zlib / "manifest.json"),
            "libffi_artifact_sha256": shared["artifact_sha256"],
            "ctypes_artifact_sha256": extension["artifact_sha256"],
        }
        (temporary / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("process", "zlib", "libffi", "ctypes", "output"):
        parser.add_argument(name, type=Path)
    arguments = parser.parse_args()
    build_runtime(*(getattr(arguments, name).resolve() for name in ("process", "zlib", "libffi", "ctypes", "output")))
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
