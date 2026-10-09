"""Describe the pinned SDK 34 main-runtime and independent side-module ABI.

The executable owns libc++, libc++abi and libunwind once. Shared C++ modules
import that runtime and its exception tag; they never link private runtime copies.
"""

import json
from pathlib import Path

from ports.native.dependencies import file_hash, toolchain_identity
from ports.numpy.build import check_build_scripts


def dynamic_toolchain(sdk):
    """Verify the reviewed SDK contract and return its flags and provenance."""
    directory = Path(__file__).parent
    recipe = json.loads((directory / "recipe.json").read_text())
    check_build_scripts(recipe, directory)
    for notice in recipe["notices"]:
        if file_hash(directory / notice["file"]) != notice["sha256"]:
            raise ValueError("A pinned runtime license notice has changed")
    identity = toolchain_identity(recipe, sdk)
    identity["llvm_nm_sha256"] = file_hash(sdk / "bin/llvm-nm")
    runtime = sdk / "share/wasi-sysroot/lib/wasm32-wasip1/eh"
    archives = [runtime / name for name in recipe["runtime_archives"]]
    if not all(path.is_file() for path in archives):
        raise ValueError("The pinned SDK exception runtime archives are missing")
    return recipe, identity, archives
