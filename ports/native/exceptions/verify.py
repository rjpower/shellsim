"""Build real C/C++ exception probes using the versioned static target profile."""

import argparse
import json
import subprocess
from pathlib import Path

from ports.native.dependencies import digest, file_hash, target_environment, toolchain_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).parent
    recipe = json.loads((directory.parents[1] / "cpython/recipe.json").read_text())
    toolchain = toolchain_identity(recipe, args.sdk)
    profile = toolchain["profile"]
    env = target_environment(args.sdk)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for name, library, consumer, cpp in (
        ("setjmp", "jump.c", "setjmp.c", False),
        ("cpp", "throw.cpp", "catch.cpp", True),
    ):
        compiler = str(args.sdk / "bin" / ("clang++" if cpp else "clang"))
        flags = [*profile["compiler_flags"], *(profile["cpp_flags"] if cpp else [])]
        object_path = args.work_dir / (name + ".o")
        archive = args.work_dir / ("lib" + name + ".a")
        subprocess.run([compiler, *flags, "-c", str(directory / library), "-o", str(object_path)], env=env, check=True)
        subprocess.run([str(args.sdk / "bin/llvm-ar"), "rcs", str(archive), str(object_path)], env=env, check=True)
        subprocess.run(
            [
                compiler,
                *flags,
                str(directory / consumer),
                str(archive),
                *profile["link_flags"],
                *(profile["cpp_link_flags"] if cpp else []),
                "-o",
                str(args.work_dir / (name + ".wasm")),
            ],
            env=env,
            check=True,
        )
    inputs = {
        "toolchain": toolchain,
        "sources": {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()},
        "outputs": {p.name: file_hash(p) for p in sorted(args.work_dir.glob("*.wasm"))},
    }
    (args.work_dir / "manifest.json").write_text(
        json.dumps({"inputs": inputs, "sha256": digest(inputs)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
