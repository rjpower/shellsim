"""Project canonical SDK metadata into the immutable producer contract.

Legacy receipts remain immutable. Their source, protocol and compiled auxiliary
inputs must match the current policy; importing a prior implementation also
requires the explicit frozen migration registry, never ordinary cache fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from ports._support.graph import _document, _reference, select_variant
from ports._support.store import file_hash

_FIELDS = {
    "llvm-host": {"name", "version", "source", "protocol", "patches", "build_limits", "main_tls_protocol"},
    "wasi-sysroot": {
        "name",
        "version",
        "dynamic_abi",
        "scheduler_namespace",
        "sdk",
        "wasi_libc",
        "patches",
        "build_limits",
        "sdk_tooling_digest",
    },
    "sdk-tooling": {
        "name",
        "version",
        "dynamic_abi",
        "scheduler_namespace",
        "sdk",
        "wasi_libc",
        "patches",
        "build_limits",
        "sdk_tooling_digest",
    },
    "cpython-threaded": {
        "name",
        "version",
        "target",
        "dynamic_abi",
        "prefix",
        "source",
        "patches",
        "target_dependencies",
        "build_limits",
        "maximum_memory_bytes",
    },
    "uv-host": {
        "name",
        "version",
        "host_target",
        "rust_toolchain",
        "guest_python",
        "guest_target",
        "wheel_platform_tag",
        "source",
        "patch",
    },
}
_CROSS_POLICY_PINS = {"compiler_recipe_sha256", "sysroot_recipe_sha256"}


def metadata(reference: str, ports: Path | None = None) -> tuple[Path, dict]:
    """Read one canonical producer selection without importing its builder."""
    root = ports or Path(__file__).resolve().parents[1]
    reference = _reference(reference)
    path, _, document = _document(root, reference)
    return path.parent, select_variant(document, reference.partition(":")[2])


def policy(recipe: Mapping) -> dict:
    """Select all build-affecting producer fields, excluding graph annotations."""
    system = recipe["build_system"]
    source = dict(recipe)
    if "producer_identity" in recipe:
        source.update(recipe["producer_identity"])
    if "sdk_archive" in recipe:
        source["sdk"] = recipe["sdk_archive"]
    result = {key: source[key] for key in _FIELDS[system]}
    if system == "cpython-threaded" and "subdirectory" in result["source"]:
        # The retained runtime producer selects this directory from its version.
        # Other extraction roots are not equivalent to that original behavior.
        if result["source"]["subdirectory"] != "Python-" + result["version"]:
            raise ValueError("CPython source directory differs from its producer contract")
        result["source"] = {key: value for key, value in result["source"].items() if key != "subdirectory"}
    result["inputs"] = dict(recipe.get("inputs", {}))
    return result


def historical_policy(recipe: dict, directory: Path, system: str) -> dict:
    """Normalize only the reviewed legacy receipt schema for explicit migration.

    Cross-recipe pins are replaced by verified dependency receipt edges during
    admission. Unrecognized policy fields fail closed. Driver code compatibility
    is checked separately against the frozen migration implementation identity.
    """
    allowed = _FIELDS[system] | _CROSS_POLICY_PINS | {"build_scripts", "inputs"}
    if set(recipe) - allowed:
        raise ValueError("unknown historical producer policy fields")
    result = {key: recipe[key] for key in _FIELDS[system]}
    root = Path(__file__).resolve().parents[1]
    inputs = dict(recipe.get("inputs", {}))
    patches = {item["file"] for item in recipe.get("patches", [])}
    for item in recipe.get("build_scripts", []):
        name = item["file"]
        if Path(name).suffix in {".py", ".json"} or name in patches:
            continue
        path = (directory / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("producer auxiliary input escapes ports")
        inputs[path.relative_to(root).as_posix()] = item["sha256"]
    result["inputs"] = inputs
    return result


def verify_policy(recorded: dict, reference: str, ports: Path | None = None) -> None:
    """Check exact source/runtime policy while retaining the recorded identity."""
    directory, recipe = metadata(reference, ports)
    if historical_policy(recorded, directory, recipe["build_system"]) != policy(recipe):
        raise ValueError("producer source, auxiliary inputs or runtime policy differs")


def load_policy(reference: str, ports: Path | None = None) -> dict:
    """Supply existing producer internals with automatically hashed code inputs."""
    from ports._support.graph import Port
    from ports.api import implementation

    root = ports or Path(__file__).resolve().parents[1]
    directory, recipe = metadata(reference, root)
    port = Port(reference, directory, recipe["name"], recipe["version"], "", (), recipe)
    result = policy(recipe)
    files = result.pop("inputs")
    for name, expected in files.items():
        if file_hash(root / name) != expected:
            raise ValueError("producer compiled auxiliary input differs")
    code = implementation(port)
    for name, digest in code.items():
        if name == "builder":
            relative = (directory.resolve() / "build.py").relative_to(root.resolve()).as_posix()
        else:
            relative = name.removeprefix("local:")
        files[relative] = digest
    result["build_scripts"] = [
        {"file": os.path.relpath(root / name, directory), "sha256": digest} for name, digest in sorted(files.items())
    ]
    return result
