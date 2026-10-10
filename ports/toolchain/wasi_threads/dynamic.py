"""Build the separately versioned threaded dynamic WASI sysroot.

Both source archives are verified before extraction. The LLVM artifact must
match the production threaded compiler recipe; diagnostic binaries are rejected.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ports._support.build import apply_patch, check_build_scripts
from ports._support.producer_tools import extract, run
from ports.native.dependencies import digest, file_hash, target_environment


def sdk_tooling(sdk):
    """Hash the frontend, archive tools and their resource headers/runtime."""
    names = [
        "bin/" + name
        for name in ("clang", "clang++", "clang.cfg", "clang++.cfg", "llvm-ar", "llvm-ranlib", "llvm-nm", "llvm-strip")
    ]
    names.extend(str(path.relative_to(sdk)) for path in sorted((sdk / "lib/clang/23").rglob("*")) if path.is_file())
    return {name: file_hash(sdk / name) for name in names}


def verify_sdk(sdk, manifest):
    """Admit an external SDK only against the recipe-pinned tooling receipt."""
    receipt = manifest["identity"]["sdk_tooling"]
    if digest(receipt) != manifest["identity"]["recipe"]["sdk_tooling_digest"]:
        raise ValueError("SDK tooling receipt differs from the recipe")
    if sdk_tooling(sdk) != receipt:
        raise ValueError("SDK frontend or resource files differ")


def compiler_identity(prefix, recipe_path=None):
    """Admit the recorded compiler policy and its immutable product inventory."""
    from ports._support.producer_policy import verify_policy
    from ports._support.sdk_products import Receipt, verify_product

    manifest_path = prefix / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verify_policy(manifest["identity"]["recipe"], "toolchain/llvm:host")
    verify_product(Receipt(prefix, manifest_path, file_hash(manifest_path), manifest))
    return manifest


def build(sdk_archive, libc_archive, llvm_prefix, cmake, ninja, work):
    directory = Path(__file__).resolve().parent
    from ports._support.producer_policy import load_policy

    recipe = load_policy("toolchain/wasi_threads")
    check_build_scripts(recipe, directory)
    compiler = compiler_identity(llvm_prefix)
    if work.exists():
        raise ValueError("use a fresh threaded sysroot output directory")
    work.mkdir(parents=True)
    sdk = work / "sdk"
    source = work / "libc-source"
    extract(sdk_archive, sdk, recipe["sdk"]["sha256"], set())
    extract(libc_archive, source, recipe["wasi_libc"]["sha256"], set())
    tooling = sdk_tooling(sdk)
    if digest(tooling) != recipe["sdk_tooling_digest"]:
        raise ValueError("SDK tooling inputs differ")
    for patch in recipe["patches"]:
        for name, expected in patch["inputs"].items():
            if file_hash(source / name) != expected:
                raise ValueError("threaded libc patch source differs: " + name)
        apply_patch(source, directory / patch["file"], patch["sha256"])
    environment = target_environment(sdk)
    output = work / "libc-build"
    commands = [
        [
            str(cmake),
            "-G",
            "Ninja",
            "-S",
            str(source),
            "-B",
            str(output),
            "-DCMAKE_MAKE_PROGRAM=" + str(ninja),
            "-DTARGET_TRIPLE=wasm32-wasip1-threads",
            "-DBUILD_SHARED=OFF",
            "-DBUILD_TESTS=OFF",
            "-DCMAKE_C_COMPILER=" + str(sdk / "bin/clang"),
            "-DCMAKE_ASM_COMPILER=" + str(sdk / "bin/clang"),
            "-DCMAKE_AR=" + str(sdk / "bin/llvm-ar"),
            "-DCMAKE_RANLIB=" + str(sdk / "bin/llvm-ranlib"),
            "-DBUILTINS_LIB=" + str(sdk / "lib/clang/23/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a"),
        ],
        [str(cmake), "--build", str(output), "--parallel", "4"],
    ]
    for index, command in enumerate(commands):
        run(command, work / f"build-{index}.log", environment)
    prefix = work / "prefix"
    # Preserve the SDK's pinned C++ headers and threaded EH archives, then
    # replace its C runtime and C headers with the scheduler-aware build.
    shutil.copytree(sdk / "share/wasi-sysroot", prefix / "sysroot")
    shutil.copytree(output / "sysroot", prefix / "sysroot", dirs_exist_ok=True)
    (prefix / "licenses").mkdir()
    shutil.copyfile(source / "LICENSE", prefix / "licenses/wasi-libc-LICENSE.txt")
    shutil.copyfile(llvm_prefix / "licenses/LLVM-LICENSE.txt", prefix / "licenses/LLVM-LICENSE.txt")
    manifest = {
        "schema_version": 1,
        "identity": {
            "recipe": recipe,
            "compiler": compiler,
            "sdk_tooling": tooling,
            "tools": {str(path): file_hash(path) for path in (cmake, ninja)},
        },
        "commands": commands,
        "artifacts": {
            str(path.relative_to(prefix)): file_hash(path) for path in sorted(prefix.rglob("*")) if path.is_file()
        },
    }
    (prefix / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return prefix
