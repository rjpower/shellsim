"""Compile a FreeType guest probe from the sealed library closure in a built image."""

import argparse
import json
import subprocess
from pathlib import Path

from ports.native.dependencies import dependency_prefix, digest, target_environment, target_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    recipe = json.loads(Path(__file__).with_name("recipe.json").read_text())
    providers = {
        name: args.bundle / "native-artifacts" / digest(artifact["inputs"])
        for name, artifact in manifest["native_libraries"].items()
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.work_dir / "dependencies"
    _, links = dependency_prefix(
        [{"port": "native/freetype", "version": recipe["version"]}],
        providers,
        prefix,
        recipe["target_profile"],
    )
    sdk = args.bundle / ("wasi-sdk-" + recipe["sdk"]["version"] + "-x86_64-linux")
    profile = target_profile(recipe)
    subprocess.run(
        [
            str(sdk / "bin/clang"),
            *profile["compiler_flags"],
            "-I" + str(prefix / "include/freetype2"),
            str(Path(__file__).with_name("probe.c")),
            *links,
            *profile["link_flags"],
            "-o",
            str(args.work_dir / "probe.wasm"),
        ],
        env=target_environment(sdk),
        check=True,
    )


if __name__ == "__main__":
    main()
