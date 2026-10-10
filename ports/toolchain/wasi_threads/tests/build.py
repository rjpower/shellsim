"""Compile and audit a pinned SDK pthread fixture without enabling host threads."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ports._support.build import check_build_scripts
from ports.native.dependencies import file_hash, target_environment


def build_probe(sdk, output, toolchain=None, fixture="two_pthreads"):
    if fixture not in {"two_pthreads", "sequential_pthreads"}:
        raise ValueError("unknown pthread fixture")
    directory = Path(__file__).parent
    fixture_recipe = json.loads((directory / "recipe.json").read_text())
    check_build_scripts(fixture_recipe, directory)
    recipe = json.loads((directory / "platform-policy.json").read_text())
    for name, digest in recipe["sdk_binaries"].items():
        if file_hash(sdk / "bin" / name) != digest:
            raise ValueError("pthread probe requires the pinned SDK 34 binaries")
    for name, digest in recipe["sdk_files"].items():
        if file_hash(sdk / name) != digest:
            raise ValueError("pthread probe requires the pinned SDK 34 runtime inputs")
    output.mkdir(parents=True, exist_ok=True)
    target = output / (fixture.replace("_", "-") + (".wasm" if toolchain else "-raw.wasm"))
    provenance = None
    extra_flags = []
    if toolchain is not None:
        provenance = json.loads((toolchain / "toolchain-manifest.json").read_text())
        if {key: provenance["recipe"][key] for key in recipe} != recipe:
            raise ValueError("patched toolchain recipe differs from the fixture recipe")
        for path, digest in provenance["artifacts"].items():
            if file_hash(toolchain / path) != digest:
                raise ValueError("patched toolchain artifact SHA-256 mismatch")
        linker = toolchain / "lld-build/bin/wasm-ld"
        if not linker.exists():
            linker.symlink_to("lld")
        extra_flags = [
            "--sysroot=" + str(toolchain / "libc-build/sysroot"),
            "-fuse-ld=" + str(linker),
            "-Wl,--shared-memory,--serial-memory-init",
        ]
    environment = target_environment(sdk)
    command = [
        str(sdk / "bin/clang"),
        "--target=wasm32-wasip1-threads",
        "-pthread",
        "-O2",
        "-g0",
        *extra_flags,
        str(directory / (fixture + ".c")),
        "-Wl,--import-memory,--export-memory,--export=__stack_pointer,--export=__tls_base",
        "-Wl,--initial-memory=16777216,--max-memory=16777216",
        "-o",
        str(target),
    ]
    subprocess.run(command, env=environment, check=True)
    disassembly = subprocess.check_output(
        [str(sdk / "bin/llvm-objdump"), "-d", str(target)], env=environment, text=True
    )
    (output / "fixture.disasm").write_text(disassembly)
    raw_operations = {
        operation: len(re.findall(r"\bmemory\.atomic\." + operation + r"\b", disassembly))
        for operation in ("wait32", "wait64", "notify")
    }
    if toolchain is not None and any(raw_operations.values()):
        raise ValueError("patched pthread fixture contains raw atomic wait/notify")
    manifest = {
        "schema_version": 1,
        "recipe": fixture_recipe,
        "toolchain_recipe": recipe,
        "fixture_implementation_sha256": file_hash(Path(__file__)),
        "execution_verified": False,
        "artifact": {"path": target.name, "sha256": file_hash(target)},
        "raw_atomic_operations": raw_operations,
        "toolchain_provenance": provenance,
        "compile_command": command,
        "memory_reservation_bytes": 16 * 1024 * 1024,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--fixture", choices=("two_pthreads", "sequential_pthreads"), default="two_pthreads")
    args = parser.parse_args()
    print(
        build_probe(
            args.sdk.resolve(),
            args.output.resolve(),
            args.toolchain.resolve() if args.toolchain else None,
            args.fixture,
        )
    )


if __name__ == "__main__":
    main()
