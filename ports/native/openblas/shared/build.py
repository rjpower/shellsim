"""Link verified PIC OpenBLAS into an independent SDK 34 shared provider."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ports.dynamic.build import mark_abi
from ports.native.dependencies import (
    artifact_input,
    digest,
    file_hash,
    seal_artifact,
    target_environment,
    target_profile,
    toolchain_identity,
    verify_artifact,
)


def build_shared(provider, sdk, work):
    """Consume an unchanged sealed archive and retain its complete numerical ABI."""
    directory = Path(__file__).parent
    recipe = json.loads((directory / "recipe.json").read_text())
    static_recipe = json.loads((directory.parent / "recipe.json").read_text())
    native = verify_artifact(provider)
    if native["inputs"]["recipe"] != static_recipe:
        raise ValueError("OpenBLAS archive provider differs from the pinned static recipe")
    toolchain = toolchain_identity(recipe, sdk)
    if native["inputs"]["toolchain"] != toolchain:
        raise ValueError("OpenBLAS archive provider has a different SDK identity")
    runtime = sdk / recipe["compiler_support"]["path"]
    inputs = artifact_input(recipe, directory, toolchain, {"native/openblas": native})
    inputs["compiler_support"] = {
        **recipe["compiler_support"],
        "sha256": file_hash(runtime),
    }
    prefix = work / "native-artifacts" / digest(inputs)
    if prefix.exists():
        return prefix, verify_artifact(prefix, inputs)
    temporary = prefix.with_name(prefix.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "lib/pkgconfig").mkdir(parents=True)
    (temporary / "include").mkdir()
    (temporary / "licenses").mkdir()
    profile = target_profile(recipe)
    library = temporary / "lib/libopenblas.so"
    subprocess.run(
        [
            str(sdk / "bin/clang"),
            *profile["cpp_flags"],
            *recipe["link_flags"],
            "-Wl,--whole-archive",
            str(provider / "lib/libopenblas.a"),
            "-Wl,--no-whole-archive",
            str(runtime),
            "-o",
            str(library),
        ],
        env=target_environment(sdk),
        check=True,
    )
    mark_abi(library, recipe["abi"].encode())
    for name in ("cblas.h", "openblas_config.h"):
        shutil.copyfile(provider / "include" / name, temporary / "include" / name)
    shutil.copyfile(provider / "licenses/openblas.txt", temporary / "licenses/openblas.txt")
    shutil.copyfile(
        directory.parents[2] / "toolchain/wasi_sdk/notices/llvm-LICENSE.TXT", temporary / "licenses/compiler-rt.txt"
    )
    (temporary / "lib/pkgconfig/openblas.pc").write_text(
        "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n"
        "Name: OpenBLAS\nDescription: Scalar WASI shared BLAS and translated LAPACK\n"
        "Version: 0.3.31\nLibs: -L${libdir} -lopenblas\nCflags: -I${includedir}\n"
    )
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    prefix, manifest = build_shared(args.provider, args.sdk, args.work_dir)
    print(json.dumps({"prefix": str(prefix), "artifact_sha256": manifest["artifact_sha256"]}))


if __name__ == "__main__":
    main()
