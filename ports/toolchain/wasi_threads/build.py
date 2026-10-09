"""Build the pinned LLD/libc scheduler thread toolchain outside the repository."""

import argparse
import json
import os
import resource
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ports._support.build import apply_patch, check_build_scripts
from ports.native.dependencies import file_hash


def toolchain_identity(recipe):
    """Bind toolchain caches to every production recipe input and driver hash.

    Fixture builders have a separate recipe. Renaming or changing a production
    driver invalidates this identity even when compiler source inputs are equal.
    """
    return dict(recipe)


def extract(archive, destination, digest, selected):
    if file_hash(archive) != digest:
        raise ValueError("toolchain source archive SHA-256 mismatch")
    if destination.exists():
        raise ValueError("use a fresh toolchain work directory")
    destination.mkdir()
    with tarfile.open(archive) as source:
        members = []
        for member in source:
            parts = Path(member.name).parts
            if len(parts) < 2 or (selected and parts[1] not in selected):
                continue
            member.name = str(Path(*parts[1:]))
            if member.islnk():
                member.linkname = str(Path(*Path(member.linkname).parts[1:]))
            members.append(member)
        source.extractall(destination, members=members, filter="data")


def run(command, log, environment):
    with log.open("w") as output:
        subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=3600)


def build(sdk, llvm_archive, libc_archive, ninja, work):
    directory = Path(__file__).parent
    recipe = json.loads((directory / "recipe.json").read_text())
    check_build_scripts(recipe, directory)
    for name, digest in recipe["sdk_binaries"].items():
        if file_hash(sdk / "bin" / name) != digest:
            raise ValueError("toolchain build requires the pinned SDK 34 binaries")
    for name, digest in recipe["sdk_files"].items():
        if file_hash(sdk / name) != digest:
            raise ValueError("toolchain build requires the pinned SDK 34 runtime inputs")
    work.mkdir(parents=True, exist_ok=True)
    llvm = work / "llvm-source"
    libc = work / "libc-source"
    extract(
        llvm_archive,
        llvm,
        recipe["llvm_source"]["sha256"],
        {"llvm", "lld", "libc", "libunwind", "cmake", "third-party", "LICENSE.TXT"},
    )
    extract(libc_archive, libc, recipe["wasi_libc"]["sha256"], set())
    for item in recipe["build_scripts"]:
        if item["file"].endswith(".patch"):
            apply_patch(llvm if item["file"].startswith("llvm") else libc, directory / item["file"], item["sha256"])
    environment = dict(os.environ)
    environment.pop("CFLAGS", None)
    environment.pop("CXXFLAGS", None)
    environment.pop("LDFLAGS", None)
    llvm_build = work / "lld-build"
    libc_build = work / "libc-build"
    commands = [
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            str(llvm / "llvm"),
            "-B",
            str(llvm_build),
            "-DCMAKE_MAKE_PROGRAM=" + str(ninja),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLVM_ENABLE_PROJECTS=lld",
            "-DLLVM_TARGETS_TO_BUILD=WebAssembly",
            "-DLLVM_INCLUDE_TESTS=OFF",
            "-DLLVM_INCLUDE_EXAMPLES=OFF",
            "-DLLVM_INCLUDE_BENCHMARKS=OFF",
            "-DLLVM_ENABLE_ZLIB=OFF",
            "-DLLVM_ENABLE_ZSTD=OFF",
            "-DLLVM_ENABLE_LIBXML2=OFF",
            "-DLLVM_ENABLE_BINDINGS=OFF",
            "-DLLVM_PARALLEL_LINK_JOBS=1",
        ],
        ["cmake", "--build", str(llvm_build), "--target", "lld", "FileCheck", "--parallel", "8"],
        [
            "cmake",
            "-S",
            str(libc),
            "-B",
            str(libc_build),
            "-DTARGET_TRIPLE=wasm32-wasip1-threads",
            "-DBUILD_SHARED=OFF",
            "-DBUILD_TESTS=OFF",
            "-DCMAKE_C_COMPILER=" + str(sdk / "bin/clang"),
            "-DCMAKE_ASM_COMPILER=" + str(sdk / "bin/clang"),
            "-DCMAKE_AR=" + str(sdk / "bin/llvm-ar"),
            "-DCMAKE_RANLIB=" + str(sdk / "bin/llvm-ranlib"),
            "-DBUILTINS_LIB=" + str(sdk / "lib/clang/23/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a"),
        ],
        ["cmake", "--build", str(libc_build), "--parallel", "8"],
    ]
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))
    for index, command in enumerate(commands):
        run(command, work / f"build-{index}.log", environment)
    tools = {
        name: subprocess.check_output([name, "--version"], text=True).splitlines()[0]
        for name in ("cmake", "gcc", "g++")
    }
    tools["ninja"] = subprocess.check_output([str(ninja), "--version"], text=True).strip()
    manifest = {
        "recipe": recipe,
        "host_tools": tools,
        "commands": commands,
        "build_limits": {
            "compile_jobs": 8,
            "link_jobs": 1,
            "address_space_bytes_per_process": 12 * 1024**3,
            "command_timeout_seconds": 3600,
        },
        "artifacts": {
            str(path.relative_to(work)): file_hash(path)
            for path in (
                llvm_build / "bin/lld",
                llvm_build / "bin/FileCheck",
                libc_build / "sysroot/lib/wasm32-wasip1-threads/libc.a",
            )
        },
    }
    (work / "toolchain-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sdk", "llvm-archive", "libc-archive", "ninja", "work-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(
        args.sdk.resolve(),
        args.llvm_archive.resolve(),
        args.libc_archive.resolve(),
        args.ninja.resolve(),
        args.work_dir.resolve(),
    )


if __name__ == "__main__":
    main()
