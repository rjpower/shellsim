"""Build and probe using explicit local source and SDK paths; never download."""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

from ports.native.dependencies import file_hash, target_environment, toolchain_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).parent
    recipe = json.loads((directory / "recipe.json").read_text())
    if file_hash(args.source_archive) != recipe["source"]["sha256"]:
        raise ValueError("Source archive hash mismatch")
    cpython = json.loads((directory.parents[1] / "cpython/recipe.json").read_text())
    spec = importlib.util.spec_from_file_location("libjpeg_build", directory / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run(command, cwd, env, log):
        with log.open("w") as output:
            subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    prefix, artifact = module.build_libjpeg_turbo(
        recipe,
        args.source.resolve(),
        args.sdk.resolve(),
        args.work_dir.resolve(),
        toolchain_identity(cpython, args.sdk),
        run,
    )
    run(
        [
            str(args.sdk / "bin/clang"),
            "-O2",
            "-I" + str(prefix / "include"),
            str(directory / "probe.c"),
            str(prefix / "lib/libjpeg.a"),
            "-o",
            str(args.work_dir / "probe.wasm"),
        ],
        Path.cwd(),
        target_environment(args.sdk),
        args.work_dir / "probe.log",
    )
    print(json.dumps({"prefix": str(prefix), "artifact_sha256": artifact["artifact_sha256"]}))


if __name__ == "__main__":
    main()
