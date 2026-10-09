"""Check pinned thread probe inputs; an SDK build is explicitly compile only."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from ports._support.build import check_build_scripts
from ports.toolchain.wasi_threads.build import toolchain_identity
from ports.toolchain.wasi_threads.tests.build import build_probe


def test_toolchain_identity_preserves_every_production_input():
    recipe = {"llvm_commit": "pinned", "build_scripts": [{"file": "build.py", "sha256": "production-driver"}]}
    assert toolchain_identity(recipe) == recipe
    changed = dict(recipe, build_scripts=[{"file": "build.py", "sha256": "changed-driver"}])
    assert toolchain_identity(changed) != toolchain_identity(recipe)
    assert toolchain_identity(dict(recipe, llvm_commit="different")) != toolchain_identity(recipe)


def test_thread_recipe_pins_production_and_separate_fixture_inputs():
    directory = Path(__file__).resolve().parent.parent
    recipe = json.loads((directory / "recipe.json").read_text())
    check_build_scripts(recipe, directory)
    assert recipe["name"] == "wasi-pthreads"
    assert recipe["sdk"]["version"] == "34.0"
    assert recipe["scheduler_namespace"] == "shellsim_threads_v1"
    assert "status" not in recipe
    assert not any(
        item["file"].startswith("runner/") or item["file"].endswith("_pthreads.c") for item in recipe["build_scripts"]
    )
    fixtures = directory / "tests"
    check_build_scripts(json.loads((fixtures / "recipe.json").read_text()), fixtures)


def test_stock_sdk_pthread_fixture_records_unsafe_waits(tmp_path):
    sdk = os.environ.get("SHELLSIM_WASI_SDK34")
    if sdk is None:
        pytest.skip("set SHELLSIM_WASI_SDK34 to audit the real SDK pthread output")
    target = build_probe(Path(sdk), tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert target.read_bytes().startswith(b"\0asm\x01\0\0\0")
    assert manifest["execution_verified"] is False
    assert manifest["raw_atomic_operations"] == {"wait32": 2, "wait64": 0, "notify": 3}
    assert manifest["memory_reservation_bytes"] == 16 * 1024 * 1024


def test_patched_pthread_fixture_executes_with_virtual_timeout(tmp_path):
    sdk = os.environ.get("SHELLSIM_WASI_SDK34")
    toolchain = os.environ.get("SHELLSIM_PTHREADS_TOOLCHAIN")
    runner = os.environ.get("SHELLSIM_PTHREADS_RUNNER")
    if sdk is None or toolchain is None or runner is None:
        pytest.skip("set SDK34, PTHREADS_TOOLCHAIN and PTHREADS_RUNNER for the standalone proof")
    target = build_probe(Path(sdk), tmp_path, Path(toolchain))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["raw_atomic_operations"] == {"wait32": 0, "wait64": 0, "notify": 0}
    proof = subprocess.run([runner, str(target)], capture_output=True, text=True, check=True, timeout=120)
    assert "two pthreads: join, mutex, condition, TLS, virtual timeout passed" in proof.stdout
    assert "virtual_ns=5000000 spawned=2" in proof.stdout
    stock = build_probe(Path(sdk), tmp_path / "stock")
    rejected = subprocess.run([runner, str(stock)], capture_output=True, text=True, timeout=120)
    assert rejected.returncode != 0
    assert "raw atomic wait/notify is outside the scheduler ABI" in rejected.stderr
