"""Reproduce the pinned Wasmtime fuel-resume source overlay."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(archive: Path, destination: Path) -> dict:
    directory = Path(__file__).parent
    recipe = json.loads((directory / "recipe.json").read_text())
    if sha256(Path(__file__)) != recipe["build_driver_sha256"]:
        raise ValueError("Wasmtime overlay builder identity differs")
    patch = directory / recipe["patch"]["file"]
    if sha256(archive) != recipe["source"]["sha256"] or sha256(patch) != recipe["patch"]["sha256"]:
        raise ValueError("Wasmtime overlay input identity differs")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    crate = destination / "wasmtime-49.0.0"
    # Cargo generates this reserved filename itself when packaging local sources.
    (crate / "Cargo.toml.orig").unlink()
    for name, digest in recipe["patch"]["inputs"].items():
        if sha256(crate / name) != digest:
            raise ValueError("Wasmtime patch source identity differs")
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch.resolve())], cwd=crate, check=True)
    manifest = {
        "recipe": recipe,
        "files": {str(path.relative_to(crate)): sha256(path) for path in sorted(crate.rglob("*")) if path.is_file()},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.archive.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
