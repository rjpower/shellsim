"""Admit the final threaded compiler/sysroot before a provider uses its files."""

import json
import os
from pathlib import Path, PurePosixPath

from ports.native.dependencies import file_hash
from ports.toolchain.wasi_threads.dynamic import compiler_identity, verify_sdk

MAX_MANIFEST = 8 * 1024**2
MAX_FILES = 32768
MAX_FILE = 512 * 1024**2


def read_manifest(path):
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST:
        raise ValueError("threaded toolchain manifest is missing or oversized")
    return json.loads(path.read_bytes())


def validate_files(prefix, manifest):
    """Bound every admitted path before hashing compiler or sysroot bytes."""
    artifacts = manifest["artifacts"]
    if not 1 <= len(artifacts) <= MAX_FILES:
        raise ValueError("threaded toolchain file count is outside the bounded profile")
    root = prefix.resolve()
    for name, expected in artifacts.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise ValueError("unsafe threaded toolchain artifact path")
        path = prefix / name
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size > MAX_FILE:
            raise ValueError("threaded toolchain artifact is missing, outside its prefix, or oversized")
        if file_hash(path) != expected:
            raise ValueError("threaded toolchain artifact differs: " + name)


def validate_tree(prefix, directory, artifacts):
    """Reject extra include/library inputs and bound empty-directory traversal."""
    pending = [prefix / directory]
    visited = 0
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                visited += 1
                if visited > MAX_FILES * 2:
                    raise ValueError("threaded sysroot traversal exceeds the bounded profile")
                path = Path(entry.path)
                if entry.is_symlink():
                    raise ValueError("threaded sysroot contains an undeclared symlink")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif not entry.is_file(follow_symlinks=False) or path.relative_to(prefix).as_posix() not in artifacts:
                    raise ValueError("threaded sysroot contains an undeclared input")


def admit(sdk: Path, compiler: Path, overlay: Path, *, compiler_recipe: Path, overlay_recipe: Path):
    """Return exact immutable receipts; diagnostic or tampered products fail."""
    expected_overlay = json.loads(overlay_recipe.read_bytes())
    manifest = read_manifest(overlay / "manifest.json")
    if manifest["schema_version"] != 1 or manifest["identity"]["recipe"] != expected_overlay:
        raise ValueError("threaded sysroot recipe differs from the admitted production recipe")
    if file_hash(compiler_recipe) != expected_overlay["compiler_recipe_sha256"]:
        raise ValueError("threaded compiler source recipe differs")
    validate_files(compiler, read_manifest(compiler / "manifest.json"))
    compiled = compiler_identity(compiler, compiler_recipe)
    if manifest["identity"]["compiler"] != compiled:
        raise ValueError("threaded compiler/sysroot cohort differs")
    if expected_overlay["dynamic_abi"] != "shellsim-wasi-sdk34-cpython3137-threads-v3":
        raise ValueError("threaded provider requires the exact v3 ABI")
    validate_files(overlay, manifest)
    validate_tree(overlay, "sysroot", manifest["artifacts"])
    verify_sdk(sdk, manifest)
    return {"compiler": compiled, "overlay": manifest}
