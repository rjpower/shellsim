"""Compile and run the pinned libffi scalar boundary through a C side provider."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import needed_libraries
from ports.native.dependencies import target_environment, target_profile, toolchain_identity, verify_artifact
from ports.toolchain.wasi_sdk.build import dynamic_toolchain

PORT = Path(__file__).resolve().parent.parent


def build_probe(artifact: Path, sdk: Path, output: Path) -> tuple[Path, Path]:
    """Link the verified archive into a main image with a separate C provider."""
    recipe = json.loads((PORT / "recipe.json").read_text())
    manifest = verify_artifact(artifact)
    if manifest["inputs"]["recipe"] != recipe or manifest["inputs"]["toolchain"] != toolchain_identity(recipe, sdk):
        raise ValueError("libffi fixture requires the pinned archive and SDK")
    runtime, _, _ = dynamic_toolchain(sdk)
    if runtime["abi"] != "shellsim-wasi-sdk34-cpython3137-v2":
        raise ValueError("libffi fixture requires the reviewed dynamic ABI")
    profile = target_profile(runtime)
    environment = target_environment(sdk)
    output.mkdir(parents=True, exist_ok=True)
    clang = str(sdk / "bin/clang")
    flags = [*profile["compiler_flags"], *profile["cpp_flags"]]
    provider = output / "libffi_probe.so"
    subprocess.run(
        [
            clang,
            *flags,
            *runtime["side_link_flags"],
            "-Wl,--export-all",
            str(PORT / "tests/provider.c"),
            "-o",
            str(provider),
        ],
        env=environment,
        check=True,
    )
    mark_abi(provider, runtime["abi"].encode())
    if needed_libraries(provider):
        raise ValueError("libffi C probe provider has undeclared native dependencies")
    bridge = output / "bridge.o"
    subprocess.run(
        [
            clang,
            *flags,
            f'-DSHELLSIM_DYLINK_NAMESPACE="{runtime["loader_namespace"]}"',
            "-c",
            str(PORT.parents[1] / "toolchain/wasi_sdk/dynamic.c"),
            "-o",
            str(bridge),
        ],
        env=environment,
        check=True,
    )
    main = output / "probe.wasm"
    subprocess.run(
        [
            clang,
            *flags,
            "-I" + str(artifact / "include"),
            str(PORT / "tests/probe.c"),
            str(artifact / "lib/libffi.a"),
            str(bridge),
            *profile["link_flags"],
            "-Wl,--export-all,--export-table,--growable-table,--export=__stack_pointer",
            "-Wl,--undefined=malloc",
            "-Wl,--wrap=dlopen,--wrap=dlsym,--wrap=dlerror,--wrap=dlclose",
            "-o",
            str(main),
        ],
        env=environment,
        check=True,
    )
    mark_abi(main, runtime["abi"].encode())
    return main, provider


def run_probe(main: Path, provider: Path):
    """Execute the two compiled images inside shellsim's virtual filesystem."""
    import shellsim

    environment = shellsim.Environment(cpu=500_000_000, memory=256 * 1024**2, disk=32 * 1024**2)
    environment.mkdir("/lib", parents=True)
    environment.write_file("/lib/libffi_probe.so", provider.read_bytes())
    environment.write_file("/probe", main.read_bytes(), mode=0o755)
    return environment.run("/probe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    program, provider = build_probe(arguments.artifact, arguments.sdk, arguments.output)
    result = run_probe(program, provider)
    if result.returncode or result.stderr:
        raise RuntimeError(f"libffi guest probe failed: {result.returncode}: {result.stderr!r}")
    if result.stdout != b"upstream libffi common code and SDK34 scalar backend: ok\n":
        raise RuntimeError("libffi guest probe did not complete")
    print(result.stdout.decode(), end="")


if __name__ == "__main__":
    main()
