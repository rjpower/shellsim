"""Build the pinned WASI shared-library linker with explicit initialization."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ports._support.build import apply_patch, check_build_scripts


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def extract(archive, destination, expected):
    if digest(archive) != expected:
        raise ValueError("LLVM source archive identity differs")
    destination.mkdir()
    with tarfile.open(archive) as source:
        members = []
        for member in source:
            parts = Path(member.name).parts
            if len(parts) < 2 or parts[1] not in {
                "llvm",
                "lld",
                "libc",
                "libunwind",
                "cmake",
                "third-party",
                "LICENSE.TXT",
            }:
                continue
            member.name = str(Path(*parts[1:]))
            if member.islnk():
                member.linkname = str(Path(*Path(member.linkname).parts[1:]))
            members.append(member)
        source.extractall(destination, members=members, filter="data")


def run(command, log, environment):
    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

    with log.open("w") as output:
        subprocess.run(
            command,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=3600,
            preexec_fn=limit,
        )


def build(archive, cc, cxx, cmake, ninja, work):
    directory = Path(__file__).resolve().parent
    recipe = json.loads((directory / "recipe.json").read_text())
    check_build_scripts(recipe, directory)
    tools = {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in {"cc": cc, "cxx": cxx, "cmake": cmake, "ninja": ninja}.items()
    }
    identity = {"recipe": recipe, "tools": tools}
    if work.exists():
        raise ValueError("use a fresh LLVM artifact directory; recipe and tool changes invalidate artifacts")
    work.mkdir(parents=True)
    source = work / "source"
    extract(archive, source, recipe["source"]["sha256"])
    for item in recipe["patches"]:
        for name, expected in item["inputs"].items():
            if digest(source / "lld/wasm" / name) != expected:
                raise ValueError("LLVM patch source identity differs: " + name)
        apply_patch(source, directory / item["file"], item["sha256"])
    environment = dict(os.environ)
    for name in ("CFLAGS", "CXXFLAGS", "LDFLAGS"):
        environment.pop(name, None)
    output = work / "build"
    commands = [
        [
            str(cmake),
            "-G",
            "Ninja",
            "-S",
            str(source / "llvm"),
            "-B",
            str(output),
            "-DCMAKE_MAKE_PROGRAM=" + str(ninja),
            "-DCMAKE_C_COMPILER=" + str(cc),
            "-DCMAKE_CXX_COMPILER=" + str(cxx),
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
        [str(cmake), "--build", str(output), "--target", "lld", "--parallel", "4"],
    ]
    for index, command in enumerate(commands):
        run(command, work / f"build-{index}.log", environment)
    prefix = work / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "licenses").mkdir()
    shutil.copyfile(output / "bin/lld", prefix / "bin/lld")
    (prefix / "bin/lld").chmod(0o755)
    (prefix / "bin/wasm-ld").symlink_to("lld")
    shutil.copyfile(source / "LICENSE.TXT", prefix / "licenses/LLVM-LICENSE.txt")
    manifest = {
        "schema_version": 1,
        "identity": identity,
        "commands": commands,
        "build_limits": recipe["build_limits"],
        "artifacts": {
            "bin/lld": digest(prefix / "bin/lld"),
            "licenses/LLVM-LICENSE.txt": digest(prefix / "licenses/LLVM-LICENSE.txt"),
        },
    }
    (prefix / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "cc", "cxx", "cmake", "ninja", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        build(
            *(getattr(args, name).resolve() for name in ("archive", "cc", "cxx", "cmake", "ninja", "work")),
        )
    )


if __name__ == "__main__":
    main()
