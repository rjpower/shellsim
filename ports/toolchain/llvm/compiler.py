"""Build the pinned Clang toolchain for threaded WASI ports."""

import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ports._support.build import apply_patch, check_build_scripts
from ports.toolchain.llvm.build import digest, run


def build(archive, cc, cxx, cmake, ninja, work):
    directory = Path(__file__).resolve().parent
    recipe = json.loads((directory / "compiler-recipe.json").read_text())
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
    if digest(archive) != recipe["source"]["sha256"]:
        raise ValueError("LLVM source archive identity differs")
    source.mkdir()
    with tarfile.open(archive) as upstream:
        upstream.extractall(source, filter="data")
    roots = list(source.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("LLVM archive must contain one source root")
    source = roots[0]
    for item in recipe["patches"]:
        for name, expected in item["inputs"].items():
            if digest(source / name) != expected:
                raise ValueError("LLVM patch source identity differs: " + name)
        apply_patch(source, directory / item["file"], item["sha256"])
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C", "SOURCE_DATE_EPOCH": "1756857600"}
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
            "-DLLVM_ENABLE_PROJECTS=clang;lld",
            "-DCLANG_INCLUDE_TESTS=OFF",
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
        [
            str(cmake),
            "--build",
            str(output),
            "--target",
            "clang",
            "lld",
            "llc",
            "llvm-ar",
            "llvm-nm",
            "llvm-objcopy",
            "--parallel",
            "4",
        ],
    ]
    for index, command in enumerate(commands):
        run(command, work / f"build-{index}.log", environment)
    prefix = work / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "licenses").mkdir()
    for name in ("clang", "lld", "llc", "llvm-ar", "llvm-nm", "llvm-objcopy"):
        shutil.copyfile(output / "bin" / name, prefix / "bin" / name)
        (prefix / "bin" / name).chmod(0o755)
    for alias, name in {
        "clang++": "clang",
        "wasm-ld": "lld",
        "llvm-ranlib": "llvm-ar",
        "llvm-strip": "llvm-objcopy",
    }.items():
        (prefix / "bin" / alias).symlink_to(name)
    shutil.copytree(output / "lib/clang", prefix / "lib/clang")
    shutil.copyfile(source / "LICENSE.TXT", prefix / "licenses/LLVM-LICENSE.txt")
    manifest = {
        "schema_version": 1,
        "identity": identity,
        "commands": commands,
        "build_limits": recipe["build_limits"],
        "artifacts": {
            str(path.relative_to(prefix)): digest(path)
            for path in sorted(prefix.rglob("*"))
            if path.is_file() and not path.is_symlink()
        },
        "symlinks": {
            str(path.relative_to(prefix)): os.readlink(path) for path in sorted(prefix.rglob("*")) if path.is_symlink()
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
