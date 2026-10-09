"""Resolve and run magiccube through a static NumPy recipe.

The native index contains resolution metadata only. Its wheel is never installed:
NumPy code comes exclusively from the verified source-built CPython bundle.
"""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

import tomllib

MARKERS = {
    "implementation_name": "cpython",
    "implementation_version": "3.13.7",
    "os_name": "posix",
    "platform_machine": "wasm32",
    "platform_python_implementation": "CPython",
    "platform_release": "0.0.0",
    "platform_system": "wasi",
    "platform_version": "0.0.0",
    "python_full_version": "3.13.7",
    "python_version": "3.13",
    "sys_platform": "wasi",
}
PURE_RECIPE = json.loads(Path(__file__).with_name("magiccube").joinpath("recipe.json").read_text())
WHEEL = PURE_RECIPE["source"]
NUMPY_VERSION = "2.3.5"
VERIFY = """
import json, magiccube, numpy as np
import os, platform, sys
markers = dict(implementation_name=sys.implementation.name,
               implementation_version='.'.join(str(value) for value in sys.implementation.version[:3]),
               os_name=os.name, platform_machine=platform.machine(),
               platform_python_implementation=platform.python_implementation(),
               platform_release=platform.release(), platform_system=platform.system(),
               platform_version=platform.version(), python_full_version=platform.python_version(),
               python_version='.'.join(platform.python_version_tuple()[:2]), sys_platform=sys.platform)
import numpy._core._multiarray_umath as core
assert np.__version__ == '2.3.5'
assert core.__spec__.origin == 'built-in'
a = np.arange(12, dtype=np.int64).reshape(3, 4)
assert np.array_equal(a.sum(axis=0), [12, 15, 18, 21])
cube = magiccube.Cube(3)
assert cube.is_done()
cube.rotate('R U F')
assert not cube.is_done()
cube.rotate("F' U' R'")
assert cube.is_done()
assert cube.cube.shape == (3, 3, 3)
assert cube.cube.dtype == np.dtype(object)
print(json.dumps({'guest_markers': markers, 'numpy': np.__version__, 'native_origin': core.__spec__.origin,
                  'sum': a.sum(axis=0).tolist(), 'cube_shape': list(cube.cube.shape),
                  'cube_restored': cube.is_done()}, sort_keys=True))
"""


def fetch_wheel(work: Path) -> Path:
    """Download the pinned pure wheel and verify its complete digest."""
    path = work / "downloads" / WHEEL["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(WHEEL["url"]) as response:
            path.write_bytes(response.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != WHEEL["sha256"]:
        raise ValueError("magiccube wheel SHA256 mismatch")
    return path


def index_wheel(directory: Path, wheel: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / wheel.name
    if destination != wheel:
        shutil.copyfile(wheel, destination)
    (directory / "index.html").write_text(f'<a href="{wheel.name}">{wheel.name}</a>\n')


def numpy_metadata(recipe: dict, bundle: Path) -> bytes:
    """Read actual metadata from the digest-verified NumPy source archive."""
    source = recipe["source"]
    archive_path = bundle / "downloads" / source["url"].rsplit("/", 1)[-1]
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        with urllib.request.urlopen(source["url"]) as response:
            archive_path.write_bytes(response.read())
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("NumPy source SHA256 mismatch")
    with tarfile.open(archive_path) as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
        ]
        if len(members) != 1 or members[0].size > 1024 * 1024:
            raise ValueError("NumPy archive has no unique bounded package metadata")
        with archive.extractfile(members[0]) as metadata:
            content = metadata.read()
    headers = email.message_from_bytes(content)
    if (
        headers["Name"] != "numpy"
        or headers["Version"] != NUMPY_VERSION
        or headers.get_all("Requires-Dist", []) != recipe["requires_dist"]
    ):
        raise ValueError("NumPy source metadata differs from its native recipe")
    return content.split(b"\n\n", 1)[0] + b"\n\n"


def resolve(work: Path, requirement: str, numpy_requirement: str, recipe: dict, bundle: Path) -> tuple[dict, Path]:
    """Lock real magiccube metadata to one curated native provider without public fallback."""
    wheel = fetch_wheel(work)
    with zipfile.ZipFile(wheel) as archive:
        headers = email.message_from_bytes(archive.read("magiccube-0.3.0.dist-info/METADATA"))
    dependencies = headers.get_all("Requires-Dist", [])
    if headers["Name"] != "magiccube" or headers["Version"] != "0.3.0" or dependencies != PURE_RECIPE["requires_dist"]:
        raise ValueError("magiccube release metadata differs from the measured graph")
    public = work / "pure-index"
    native = work / "native-index"
    index_wheel(public / "magiccube", wheel)
    provider = work / "numpy-2.3.5-py3-none-any.whl"
    info = "numpy-2.3.5.dist-info"
    with zipfile.ZipFile(provider, "w") as archive:
        archive.writestr(info + "/METADATA", numpy_metadata(recipe, bundle))
        archive.writestr(info + "/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(info + "/RECORD", "")
    index_wheel(native / "numpy", provider)
    marker = " and ".join(f"{name} == '{value}'" for name, value in sorted(MARKERS.items()))
    project = work / "resolution"
    project.mkdir(exist_ok=True)
    dependencies = [requirement]
    if numpy_requirement:
        dependencies.append(numpy_requirement)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "numpy-workflow-spike"\nversion = "0.1.0"\nrequires-python = "==3.13.*"\n'
        + "dependencies = "
        + json.dumps(dependencies)
        + "\n[tool.uv]\nenvironments = "
        + json.dumps([marker])
        + '\n[[tool.uv.index]]\nname = "native-recipes"\nurl = '
        + json.dumps(native.as_uri())
        + '\n[[tool.uv.index]]\nname = "pure-wheels"\ndefault = true\nurl = '
        + json.dumps(public.as_uri())
        + "\n"
    )
    (project / "uv.lock").unlink(missing_ok=True)
    command = ["uv", "lock", "--offline", "--directory", str(project), "--python", sys.executable]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (work / "resolution.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError("graph cannot resolve to approved NumPy 2.3.5; see " + str(work / "resolution.log"))
    lock = tomllib.loads((project / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    if set(packages) != {"magiccube", "numpy", "numpy-workflow-spike"}:
        raise ValueError("resolved graph contains an unapproved package")
    if packages["numpy"]["version"] != NUMPY_VERSION or packages["numpy"]["source"] != {"registry": str(native)}:
        raise ValueError("NumPy did not resolve to the curated recipe metadata index")
    if packages["magiccube"]["version"] != "0.3.0" or packages["magiccube"]["dependencies"] != [{"name": "numpy"}]:
        raise ValueError("resolved magiccube graph differs from release metadata")
    plan = {
        "phase": "metadata-only",
        "root": requirement,
        "guest_markers": MARKERS,
        "pure_wheel": WHEEL,
        "native_provider": {"name": "numpy", "version": NUMPY_VERSION, "index": str(native)},
        "native_recipe_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
        "native_source_sha256": recipe["source"]["sha256"],
        "dependencies": {"magiccube==0.3.0": ["numpy==2.3.5"], "numpy==2.3.5": []},
    }
    (work / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan, wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirement", nargs="?", default="magiccube==0.3.0")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/shellsim-numpy-workflow"))
    parser.add_argument("--bundle", type=Path, default=Path("/tmp/shellsim-numpy"))
    parser.add_argument("--numpy-requirement", default="", help="exercise an additional native-version constraint")
    parser.add_argument(
        "--resolve-only", action="store_true", help="write metadata evidence without claiming execution"
    )
    args = parser.parse_args()
    if args.requirement != "magiccube==0.3.0":
        parser.error("this measured spike supports magiccube==0.3.0 only")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    for name in ("plan.json", "result.json", "guest.stdout", "guest.stderr"):
        (work / name).unlink(missing_ok=True)
    recipe = json.loads(Path(__file__).with_name("numpy").joinpath("recipe.json").read_text())
    bundle = args.bundle.resolve()
    plan, wheel = resolve(work, args.requirement, args.numpy_requirement, recipe, bundle)
    cpython_recipe = json.loads(Path(__file__).with_name("cpython").joinpath("recipe.json").read_text())
    plan["cpython_recipe"] = cpython_recipe
    plan["cpython_recipe_sha256"] = hashlib.sha256(json.dumps(cpython_recipe, sort_keys=True).encode()).hexdigest()
    (work / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(
        "Resolved real magiccube metadata to curated NumPy 2.3.5; native metadata wheel is not executable.", flush=True
    )
    if args.resolve_only:
        return
    if not (bundle / "manifest.json").exists():
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "cpython/build.py"),
                "--with-numpy",
                "--work-dir",
                str(bundle),
            ],
            check=True,
        )
    from shellsim import CPythonRuntime, Environment

    runtime = CPythonRuntime(bundle)
    if runtime.manifest["recipe"] != cpython_recipe:
        raise ValueError("cached bundle CPython recipe or SDK differs from the resolved native profile")
    ports = [port for port in runtime.manifest["native_ports"] if port["name"] == "numpy"]
    if len(ports) != 1 or ports[0]["version"] != NUMPY_VERSION or not ports[0]["builtin_modules"]:
        raise ValueError("bundle does not provide the resolved NumPy native recipe")
    if any(ports[0].get(key) != value for key, value in recipe.items()):
        raise ValueError("cached NumPy bundle recipe differs from the resolved recipe; rebuild this profile")
    if not set(ports[0]["builtin_modules"]).issubset(runtime.builtin_modules):
        raise ValueError("NumPy native recipe modules are missing from the guest image")
    environment = Environment(cpu=3_000_000_000, memory=256 * 1024 * 1024, disk=128 * 1024 * 1024)
    runtime.mount(environment)
    runtime.install_wheel(environment, wheel)
    result = runtime.run(environment, ["-c", VERIFY])
    (work / "guest.stdout").write_bytes(result.stdout)
    (work / "guest.stderr").write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f"real guest verification failed: {result.stderr_text}")
    guest = json.loads(result.stdout)
    if guest["guest_markers"] != MARKERS:
        raise ValueError("actual guest marker profile differs from the resolved graph")
    plan.update(phase="executed", native_provider=ports[0], guest=guest, usage=asdict(result.usage))
    (work / "result.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(result.stdout_text, end="")


if __name__ == "__main__":
    main()
