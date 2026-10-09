"""Build the SDK 34 scalar FFI proof with a separately linked WASI provider."""

import argparse
import json
from pathlib import Path

from ports.dynamic.build import mark_abi, run
from ports.native.dependencies import file_hash, target_environment, target_profile
from ports.toolchain.wasi_sdk.build import dynamic_toolchain

PORT = Path(__file__).resolve().parent


def build(bundle: Path, output: Path) -> None:
    recipe = json.loads((PORT / "recipe.json").read_text())
    if file_hash(PORT / "build.py") != recipe["build_script_sha256"]:
        raise ValueError("FFI proof builder differs from its pin")
    base = json.loads((bundle / "manifest.json").read_text())
    if base["recipe"]["target_profile"] != "wasi-cpython-v2" or base["native_ports"]:
        raise ValueError("FFI proof requires the bare SDK 34 bundle")
    sdk = bundle / "wasi-sdk-34.0-x86_64-linux"
    toolchain, identity, _ = dynamic_toolchain(sdk)
    if toolchain["abi"] != recipe["abi"] or toolchain["sdk"]["sha256"] != recipe["sdk_sha256"]:
        raise ValueError("FFI proof SDK and target ABI differ from the pin")
    profile = target_profile(toolchain)
    for name, expected in recipe["sources_sha256"].items():
        if file_hash(PORT / name) != expected:
            raise ValueError(f"FFI proof source differs from its pin: {name}")
    bridge = PORT.parent / "dynamic/fixtures/bridge.c"
    if file_hash(bridge) != recipe["bridge_sha256"]:
        raise ValueError("Dynamic bridge source differs from its pin")
    environment = target_environment(sdk)
    output.mkdir(parents=True, exist_ok=True)
    clang = str(sdk / "bin/clang")
    flags = [*profile["compiler_flags"], *profile["cpp_flags"]]
    provider = output / "libffi_proof.so"
    run(
        [
            clang,
            *flags,
            *toolchain["side_link_flags"],
            "-Wl,--export-all",
            str(PORT / "fixtures/provider.c"),
            "-o",
            str(provider),
        ],
        env=environment,
    )
    mark_abi(provider, recipe["abi"].encode())

    bridge_object = output / "bridge.o"
    run(
        [
            clang,
            *flags,
            f'-DSHELLSIM_DYLINK_NAMESPACE="{toolchain["loader_namespace"]}"',
            "-c",
            str(bridge),
            "-o",
            str(bridge_object),
        ],
        env=environment,
    )
    main = output / "ffi_proof.wasm"
    run(
        [
            clang,
            *flags,
            str(PORT / "fixtures/main.c"),
            str(bridge_object),
            *profile["link_flags"],
            "-Wl,--export-all,--export-table,--growable-table,--export=__stack_pointer",
            "-Wl,--undefined=malloc",
            "-Wl,--wrap=dlopen,--wrap=dlsym,--wrap=dlerror,--wrap=dlclose",
            "-o",
            str(main),
        ],
        env=environment,
    )
    mark_abi(main, recipe["abi"].encode())
    manifest = {
        "abi": recipe["abi"],
        "source_bundle": str(bundle.resolve()),
        "source_bundle_sha256": file_hash(bundle / "manifest.json"),
        "toolchain_identity": identity,
        "recipe_sha256": file_hash(PORT / "recipe.json"),
        "build_script_sha256": recipe["build_script_sha256"],
        "sources_sha256": recipe["sources_sha256"],
        "files": {path.name: file_hash(path) for path in (main, provider)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.bundle, args.output)


if __name__ == "__main__":
    main()
