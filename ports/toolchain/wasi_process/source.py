"""Prepare the pinned CPython process facade sources for the runtime producer."""

import shutil
import subprocess
from pathlib import Path

from ports.native.dependencies import file_hash

PORT = Path(__file__).resolve().parent


def _patched_source(source: Path, output: Path, recipe: dict[str, object]) -> Path:
    staging = output / "patched-source"
    for name in ("Modules/posixmodule.c", "Modules/signalmodule.c", "Modules/faulthandler.c", "Lib/subprocess.py"):
        original = source / name
        if file_hash(original) != recipe["cpython_source_sha256"][name]:
            raise ValueError(f"Pinned CPython source differs: {name}")
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    patch = PORT / "cpython-3.13.7-process.patch"
    if file_hash(patch) != recipe["patch_sha256"]:
        raise ValueError("Pinned CPython process patch differs")
    subprocess.run(["git", "apply", "--check", "--whitespace=error", str(patch)], cwd=staging, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=staging, check=True)
    return staging
