"""Seal the pinned libffi archive as an independent SDK 34 WASI provider."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import needed_libraries
from ports.native.dependencies import (
    artifact_input,
    digest,
    seal_artifact,
    target_environment,
    target_profile,
    toolchain_identity,
    verify_artifact,
)

PORT = Path(__file__).resolve().parent


def build_shared(provider: Path, sdk: Path, work: Path) -> tuple[Path, dict]:
    """Link only the verified PIC archive and record its exact provider identity."""
    recipe = json.loads((PORT / "recipe.json").read_text())
    static_recipe = json.loads((PORT.parent / "recipe.json").read_text())
    native = verify_artifact(provider)
    if native["inputs"]["recipe"] != static_recipe:
        raise ValueError("libffi archive differs from the pinned static provider")
    toolchain = toolchain_identity(recipe, sdk)
    if native["inputs"]["toolchain"] != toolchain:
        raise ValueError("libffi archive and shared provider use different SDKs")
    inputs = artifact_input(recipe, PORT, toolchain, {"native/libffi": native})
    prefix = work / "native-artifacts" / digest(inputs)
    if prefix.exists():
        return prefix, verify_artifact(prefix, inputs)
    temporary = prefix.with_name(prefix.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    for directory in ("include", "lib/pkgconfig", "licenses"):
        (temporary / directory).mkdir(parents=True, exist_ok=True)
    profile = target_profile(recipe)
    library = temporary / "lib/libffi.so"
    subprocess.run(
        [
            str(sdk / "bin/clang"),
            *profile["cpp_flags"],
            *recipe["link_flags"],
            "-Wl,--whole-archive",
            str(provider / "lib/libffi.a"),
            "-Wl,--no-whole-archive",
            "-o",
            str(library),
        ],
        env=target_environment(sdk),
        check=True,
    )
    mark_abi(library, recipe["abi"].encode())
    if needed_libraries(library) != recipe["needed_libraries"]:
        raise ValueError("libffi provider has undeclared native dependencies")
    for name in ("ffi.h", "ffitarget.h"):
        shutil.copyfile(provider / "include" / name, temporary / "include" / name)
    shutil.copyfile(provider / "licenses/libffi.txt", temporary / "licenses/libffi.txt")
    (temporary / "lib/pkgconfig/libffi.pc").write_text(
        "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n"
        "Name: libffi\nDescription: Shellsim WASI scalar libffi provider\nVersion: 3.5.2\n"
        "Libs: -L${libdir} -lffi\nCflags: -I${includedir}\n"
    )
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", type=Path)
    parser.add_argument("sdk", type=Path)
    parser.add_argument("work", type=Path)
    arguments = parser.parse_args()
    prefix, manifest = build_shared(arguments.provider.resolve(), arguments.sdk.resolve(), arguments.work.resolve())
    print(json.dumps({"prefix": str(prefix), "artifact_sha256": manifest["artifact_sha256"]}))


if __name__ == "__main__":
    main()
