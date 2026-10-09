"""Build upstream CPython zlib as a side module against shared WASI zlib.

The interpreter owns Python and libc. This extension imports their canonical
symbols and records libz.so as an independently staged native dependency.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ports.cpython.build import fetch_extract
from ports.dynamic.build import mark_abi
from ports.native.dependencies import file_hash, target_environment, target_profile
from ports.numpy.build import check_build_scripts
from ports.toolchain.wasi_sdk.build import dynamic_toolchain


def build(runtime, work):
    directory = Path(__file__).parent
    recipe = json.loads((directory / "stdlib_zlib_recipe.json").read_text())
    check_build_scripts(recipe, directory)
    base = runtime / "base"
    sdk = base / "wasi-sdk-34.0-x86_64-linux"
    toolchain, identity, _ = dynamic_toolchain(sdk)
    if toolchain["abi"] != recipe["abi"]:
        raise ValueError("zlib requires the reviewed SDK 34 runtime ABI")
    library = runtime / "libz.so"
    previous = json.loads((runtime / "manifest.json").read_text())
    if file_hash(library) != previous["proof_artifacts"]["libz.so"]:
        raise ValueError("shared zlib provider changed")
    work.mkdir(parents=True, exist_ok=True)
    source = fetch_extract(recipe["source"], work)
    zlib_source = fetch_extract(recipe["zlib_source"], work)
    profile = target_profile(toolchain)
    target = work / "zlib.so"
    environment = target_environment(sdk)
    command = [
        str(sdk / "bin/clang"),
        *profile["compiler_flags"],
        *toolchain["side_link_flags"],
        "-fPIC",
        "-DPy_BUILD_CORE_MODULE",
        "-I" + str(base / "Python-3.13.7/Include"),
        "-I" + str(base / "Python-3.13.7/Include/internal"),
        "-I" + str(base / "wasi-build"),
        "-I" + str(zlib_source),
        str(source / "Modules/zlibmodule.c"),
        "-L" + str(runtime),
        "-lz",
        "-o",
        str(target),
    ]
    subprocess.run(command, env=environment, check=True)
    mark_abi(target, recipe["abi"].encode())
    shutil.copyfile(library, work / "libz.so")
    shutil.copyfile(source / "LICENSE", work / "PYTHON-LICENSE")
    shutil.copyfile(zlib_source / "LICENSE", work / "ZLIB-LICENSE")
    manifest = {
        "schema_version": 1,
        "abi": recipe["abi"],
        "name": "cpython-stdlib-zlib",
        "version": "3.13.7",
        "recipe": recipe,
        "toolchain": identity,
        "interpreter_sha256": file_hash(runtime / "rootfs/usr/bin/python3.wasm"),
        "artifacts": [
            {
                "path": "zlib.so",
                "destination": "/usr/lib/python3.13/lib-dynload/zlib.so",
                "sha256": file_hash(target),
                "native_dependencies": ["libz.so"],
            },
            {"path": "libz.so", "destination": "/lib/libz.so", "sha256": file_hash(library), "native_dependencies": []},
        ],
        "files": {name: file_hash(work / name) for name in ("zlib.so", "libz.so", "PYTHON-LICENSE", "ZLIB-LICENSE")},
        "target_headers": {
            str(path.relative_to(base)): file_hash(path)
            for path in (base / "Python-3.13.7/Include/Python.h", base / "wasi-build/pyconfig.h")
        },
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.runtime.resolve(), args.work_dir.resolve()))


if __name__ == "__main__":
    main()
