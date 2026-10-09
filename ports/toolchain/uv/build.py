"""Build the pinned uv resolver with shellsim's WASI target patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECIPE = json.loads((HERE / "recipe.json").read_text())


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination for the built uv executable")
    parser.add_argument("--source", help="local Git source repository instead of the upstream URL")
    parser.add_argument("--offline", action="store_true", help="use only cached Cargo dependencies")
    args = parser.parse_args()

    patch = HERE / RECIPE["patch"]["file"]
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    if digest != RECIPE["patch"]["sha256"]:
        raise ValueError(f"uv patch hash mismatch: {digest}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_dir = output.parent / f"uv-{RECIPE['version']}-source"
    if source_dir.exists():
        raise FileExistsError(source_dir)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            RECIPE["source"]["tag"],
            args.source or RECIPE["source"]["url"],
            str(source_dir),
        ],
        check=True,
    )
    commit = run("git", "rev-parse", "HEAD", cwd=source_dir)
    if commit != RECIPE["source"]["commit"]:
        raise ValueError(f"uv source commit mismatch: {commit}")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=source_dir, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=source_dir, check=True)
    env = os.environ.copy()
    target_dir = Path(env.get("CARGO_TARGET_DIR", output.parent / "cargo-target")).resolve()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    command = ["cargo", "build", "-p", "uv", "--bin", "uv", "--locked"]
    if args.offline:
        command.append("--offline")
    subprocess.run(command, cwd=source_dir, env=env, check=True)
    shutil.copy2(target_dir / "debug" / "uv", output)


if __name__ == "__main__":
    main()
