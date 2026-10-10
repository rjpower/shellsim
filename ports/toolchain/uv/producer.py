"""Build and seal a verified host uv resolver for the fixed CPython/WASI guest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ports._support.producer_policy import load_policy

HERE = Path(__file__).resolve().parent

RECIPE = load_policy("toolchain/uv")


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _tool(name: str, version_flag: str = "--version") -> dict[str, str]:
    path = shutil.which(name)
    if path is None:
        raise ValueError(f"required host build tool is missing: {name}")
    return {
        "path": str(Path(path).absolute()),
        "sha256": file_hash(Path(path)),
        "version": subprocess.check_output([path, version_flag], text=True).strip(),
    }


def _pinned_rust_tool(name: str, rustup: str) -> dict[str, str]:
    path = Path(
        subprocess.check_output([rustup, "which", "--toolchain", RECIPE["rust_toolchain"], name], text=True).strip()
    )
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"pinned Rust tool is missing: {name}")
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "version": subprocess.check_output([str(path), "--version"], text=True).strip(),
    }


def _inputs() -> dict[str, str]:
    files = {}
    for item in RECIPE["build_scripts"]:
        path = HERE / item["file"]
        actual = file_hash(path)
        if actual != item["sha256"]:
            raise ValueError(f"uv port build input changed: {item['file']}")
        files[item["file"]] = actual
    return files


def _host_elf(path: Path, *, objdump: str, readelf: str) -> tuple[str, list[str]]:
    with path.open("rb") as source:
        header = source.read(20)
    if len(header) != 20 or header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 62:
        raise ValueError("uv release is not an x86-64 Linux ELF executable")
    symbols = subprocess.check_output([objdump, "-T", str(path)], text=True)
    versions = {tuple(map(int, item.split("."))) for item in re.findall(r"GLIBC_([0-9]+\.[0-9]+)", symbols)}
    if not versions:
        raise ValueError("uv release has no measured glibc requirement")
    dynamic = subprocess.check_output([readelf, "-d", str(path)], text=True)
    needed = sorted(set(re.findall(r"Shared library: \[([^]]+)\]", dynamic)))
    return ".".join(map(str, max(versions))), needed


def build(output: Path, *, source: str | None = None, offline: bool = False, jobs: int = 2) -> Path:
    """Publish executable and provenance together after a real WASI target probe."""
    if jobs < 1 or jobs > 32:
        raise ValueError("Cargo jobs must be between 1 and 32")
    if sys.version_info < (3, 11):
        raise ValueError("the pinned uv target verifier requires host Python 3.11 or newer")
    inputs = _inputs()
    patch = HERE / RECIPE["patch"]["file"]
    if file_hash(patch) != RECIPE["patch"]["sha256"]:
        raise ValueError("uv source patch differs from the pinned recipe")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tools = {name: _tool(name) for name in ("git", "rustup", "strip", "objdump", "readelf")}
    for name in ("cargo", "rustc"):
        tools[name] = _pinned_rust_tool(name, tools["rustup"]["path"])
    host = subprocess.check_output([tools["rustc"]["path"], "-vV"], text=True)
    if (
        f"host: {RECIPE['host_target']}" not in host.splitlines()
        or tools["rustc"]["version"].split()[1] != RECIPE["rust_toolchain"]
    ):
        raise ValueError("host Rust target or toolchain differs from the uv port recipe")
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", output.parent / "cargo-target")).resolve()
    environment = os.environ.copy()
    for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"):
        environment.pop(key, None)
    for key in tuple(environment):
        if key.startswith("CARGO_PROFILE_RELEASE_"):
            environment.pop(key)
    environment["CARGO_TARGET_DIR"] = str(target_dir)
    environment["CARGO_INCREMENTAL"] = "0"
    environment["RUSTC"] = tools["rustc"]["path"]
    with tempfile.TemporaryDirectory(prefix=".shellsim-uv-release-", dir=output.parent) as temporary:
        work = Path(temporary)
        source_dir = work / "source"
        subprocess.run(
            [
                tools["git"]["path"],
                "clone",
                "--depth",
                "1",
                "--branch",
                RECIPE["source"]["tag"],
                source or RECIPE["source"]["url"],
                str(source_dir),
            ],
            check=True,
        )
        commit = subprocess.check_output([tools["git"]["path"], "rev-parse", "HEAD"], cwd=source_dir, text=True).strip()
        if commit != RECIPE["source"]["commit"]:
            raise ValueError("uv source commit differs from the pinned recipe")
        subprocess.run([tools["git"]["path"], "apply", "--check", str(patch)], cwd=source_dir, check=True)
        subprocess.run([tools["git"]["path"], "apply", str(patch)], cwd=source_dir, check=True)
        lock_hash = file_hash(source_dir / "Cargo.lock")
        command = [tools["cargo"]["path"], "build", "-p", "uv", "--bin", "uv", "--release", "--locked", "-j", str(jobs)]
        if offline:
            command.append("--offline")
        subprocess.run(command, cwd=source_dir, env=environment, check=True)
        stage = work / "release"
        stage.mkdir()
        binary = stage / "uv"
        shutil.copy2(target_dir / "release/uv", binary)
        subprocess.run([tools["strip"]["path"], "--strip-debug", str(binary)], check=True)
        floor, needed = _host_elf(binary, objdump=tools["objdump"]["path"], readelf=tools["readelf"]["path"])
        verifier = HERE / "tests/verify_target.py"
        probe = subprocess.run(
            [sys.executable, str(verifier), str(binary)],
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        manifest = {
            "schema_version": 1,
            "recipe": RECIPE,
            "source_commit": commit,
            "cargo_lock_sha256": lock_hash,
            "build": {
                "profile": "release",
                "locked": True,
                "incremental": False,
                "jobs": jobs,
                "tools": tools,
                "inputs": inputs,
            },
            "executable": {
                "file": "uv",
                "sha256": file_hash(binary),
                "size": binary.stat().st_size,
                "host_target": RECIPE["host_target"],
                "min_glibc": floor,
                "needed_libraries": needed,
            },
            "verification": {"script_sha256": inputs["tests/verify_target.py"], "stdout": probe.stdout.strip()},
        }
        (stage / "artifact.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        stage.replace(output)
    return output / "uv"
