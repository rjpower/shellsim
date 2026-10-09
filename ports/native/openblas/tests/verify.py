"""Compile the numerical ABI probe from a verified OpenBLAS artifact closure."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from ports.native.dependencies import dependency_prefix, target_environment, target_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads((Path(__file__).parent.parent / "recipe.json").read_text())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.work_dir / "dependencies"
    _, links = dependency_prefix(
        [{"port": "native/openblas", "version": recipe["version"]}],
        {"native/openblas": args.artifact},
        prefix,
        recipe["target_profile"],
    )
    profile = target_profile(recipe)
    for name, link_inputs in (
        ("probe", links),
        (
            "whole-archive",
            [
                "-Wl,--whole-archive",
                str(prefix / "lib/libopenblas.a"),
                "-Wl,--no-whole-archive,--no-gc-sections,--fatal-warnings",
                *links,
            ],
        ),
    ):
        subprocess.run(
            [
                str(args.sdk / "bin/clang"),
                *profile["compiler_flags"],
                "-Wl,--initial-memory=20971520,--max-memory=67108864",
                "-I" + str(prefix / "include"),
                str(Path(__file__).with_name("probe.c")),
                *link_inputs,
                *profile["link_flags"],
                "-o",
                str(args.work_dir / (name + ".wasm")),
            ],
            env=target_environment(args.sdk),
            check=True,
        )


if __name__ == "__main__":
    main()
