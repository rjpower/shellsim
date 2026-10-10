"""Verify trusted build inputs and apply exact source patches for ports."""

import hashlib
import subprocess


def apply_patch(source, patch, expected_hash):
    """Apply a verified patch once, refusing mismatched cached build inputs."""
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"Patch hash mismatch: {patch}")
    marker = source / f".shellsim-{patch.name}.sha256"
    if marker.exists():
        if marker.read_text().strip() != digest:
            raise ValueError(f"Patch changed; use a clean build directory: {source}")
        return
    command = ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)]
    probe = subprocess.run([*command, "--dry-run"], cwd=source, text=True, capture_output=True, check=True)
    if "offset" in probe.stdout or "offset" in probe.stderr:
        raise ValueError(f"Patch does not match the pinned source exactly: {patch}")
    subprocess.run(command, cwd=source, check=True)
    marker.write_text(digest + "\n")


def check_build_scripts(recipe, directory):
    """Bind the cached profile to its reviewed build tooling as well as sources."""
    for item in recipe.get("build_scripts", []):
        path = directory / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Build script hash mismatch: {path}")

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name, expected in recipe.get("inputs", {}).items():
        path = root / name
        if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Compiled port input differs: {name}")
