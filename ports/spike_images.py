"""Resolve and execute the pinned imageio -> NumPy/Pillow -> zlib graph.

Native metadata wheels are used only by uv's resolver. Execution uses verified
source-built native providers and installs only the actual imageio pure wheel.
"""

from __future__ import annotations

import argparse
import base64
import csv
import email
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ports.spike_numpy import MARKERS, index_wheel

DIRECTORY = Path(__file__).with_name("imaging_graph")
RECIPE = json.loads((DIRECTORY / "recipe.json").read_text())
VERIFY = """
import imageio, imageio.v3 as iio, numpy as np, PIL, json
import os, platform, sys
markers = dict(implementation_name=sys.implementation.name,
               implementation_version='.'.join(str(v) for v in sys.implementation.version[:3]),
               os_name=os.name, platform_machine=platform.machine(),
               platform_python_implementation=platform.python_implementation(),
               platform_release=platform.release(), platform_system=platform.system(),
               platform_version=platform.version(), python_full_version=platform.python_version(),
               python_version='.'.join(platform.python_version_tuple()[:2]), sys_platform=sys.platform)
import numpy._core._multiarray_umath as core
import PIL._imaging as imaging
assert core.__spec__.origin == 'built-in'
assert imaging.__spec__.origin == 'built-in'
assert imageio.__version__ == '2.37.0'
assert np.__version__ == '2.3.5' and PIL.__version__ == '12.3.0'
pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
iio.imwrite('/tmp/graph.png', pixels, plugin='pillow')
restored = iio.imread('/tmp/graph.png', plugin='pillow')
assert restored.shape == (3, 4, 3) and restored.dtype == np.uint8
assert np.array_equal(restored, pixels)
from imageio.core.findlib import load_lib
try:
    load_lib(['forbidden'], ['forbidden'])
except NotImplementedError:
    pass
else:
    raise AssertionError('dynamic library loading accepted')
try:
    iio.imread(b'not an image', plugin='pillow')
except OSError:
    pass
else:
    raise AssertionError('invalid image accepted')
print(json.dumps(dict(imageio=imageio.__version__, numpy=np.__version__, pillow=PIL.__version__,
                     shape=list(restored.shape), dtype=str(restored.dtype), pixels=restored.tolist(),
                     invalid_rejected=True, guest_markers=markers), sort_keys=True))
"""


def verified_metadata(recipe: dict, bundle: Path) -> bytes:
    """Read the native release's real metadata from its verified source archive."""
    source = recipe["source"]
    path = bundle / "downloads" / source["url"].rsplit("/", 1)[-1]
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("native source digest mismatch")
    with tarfile.open(path) as archive:
        members = [m for m in archive.getmembers() if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")]
        if len(members) != 1 or members[0].size > 1024 * 1024:
            raise ValueError("native source lacks unique bounded metadata")
        with archive.extractfile(members[0]) as stream:
            data = stream.read()
    headers = email.message_from_bytes(data)
    if headers["Name"].lower() != recipe["name"] or headers["Version"] != recipe["version"]:
        raise ValueError("native release metadata differs from recipe")
    if headers.get_all("Requires-Dist", []) != recipe["requires_dist"]:
        raise ValueError("native release dependencies differ from recipe")
    return data.split(b"\n\n", 1)[0] + b"\n\n"


def resolve(work: Path, bundle: Path) -> tuple[dict, Path]:
    """Lock real release metadata to the two explicitly selected native providers."""
    work.mkdir(parents=True, exist_ok=True)
    source = RECIPE["source"]
    wheel = work / "downloads" / source["filename"]
    wheel.parent.mkdir(exist_ok=True)
    if not wheel.exists():
        with urllib.request.urlopen(source["url"]) as response:
            wheel.write_bytes(response.read())
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("imageio wheel digest mismatch")
    public, native = work / "pure-index", work / "native-index"
    index_wheel(public / "imageio", wheel)
    native_recipes = {}
    for name, version in RECIPE["providers"].items():
        recipe = json.loads(Path(__file__).with_name(name).joinpath("recipe.json").read_text())
        if recipe["version"] != version:
            raise ValueError("native version differs from graph")
        native_recipes[name] = recipe
        metadata = verified_metadata(recipe, bundle)
        provider = work / f"{name}-{version}-py3-none-any.whl"
        info = f"{name}-{version}.dist-info"
        with zipfile.ZipFile(provider, "w") as archive:
            archive.writestr(info + "/METADATA", metadata)
            archive.writestr(info + "/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            archive.writestr(info + "/RECORD", "")
        index_wheel(native / name, provider)
    project = work / "resolution"
    project.mkdir(exist_ok=True)
    marker = " and ".join(f"{name} == '{value}'" for name, value in sorted(MARKERS.items()))
    (project / "pyproject.toml").write_text(
        '[project]\nname="imaging-graph-spike"\nversion="0.1.0"\nrequires-python="==3.13.*"\n'
        'dependencies=["imageio==2.37.0"]\n[tool.uv]\nenvironments='
        + json.dumps([marker])
        + '\n[[tool.uv.index]]\nname="native-recipes"\nurl='
        + json.dumps(native.as_uri())
        + '\n[[tool.uv.index]]\nname="pure-wheels"\ndefault=true\nurl='
        + json.dumps(public.as_uri())
        + "\n"
    )
    (project / "uv.lock").unlink(missing_ok=True)
    result = subprocess.run(
        ["uv", "lock", "--offline", "--directory", str(project), "--python", sys.executable],
        capture_output=True,
        text=True,
        check=False,
    )
    (work / "resolution.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError("graph resolution failed; see resolution.log")
    lock = tomllib.loads((project / "uv.lock").read_text())
    packages = {p["name"]: p for p in lock["package"]}
    if set(packages) != {"imaging-graph-spike", "imageio", "numpy", "pillow"}:
        raise ValueError("graph contains an unapproved package")
    for name, version in RECIPE["providers"].items():
        if packages[name]["version"] != version or packages[name]["source"] != {"registry": str(native)}:
            raise ValueError("graph selected an unapproved native provider")
    plan = {
        "phase": "resolved",
        "graph_recipe": RECIPE,
        "guest_markers": MARKERS,
        "native_recipes": native_recipes,
        "lock_sha256": hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest(),
    }
    (work / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan, wheel


def adapted_wheel(wheel: Path, work: Path) -> tuple[Path, dict]:
    """Disable unsupported dynamic loaders while preserving the pure PNG provider."""
    member = "imageio/core/findlib.py"
    patch = DIRECTORY / "imageio-wasi.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != RECIPE["adaptation"]["sha256"]:
        raise ValueError("imageio adaptation patch digest mismatch")
    destination = work / "adapted" / wheel.name
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(wheel) as original:
        files = {entry.filename: original.read(entry) for entry in original.infolist()}
    source = files[member].decode()
    if source.count("import ctypes\n") != 1 or source.count("    # Checks\n") != 1:
        raise ValueError("imageio dynamic loader differs from reviewed patch")
    source = source.replace("import ctypes\n", "").replace(
        "    # Checks\n",
        '    if sys.platform == "wasi":\n'
        '        raise NotImplementedError("imageio dynamic libraries are unsupported on WASI")\n\n'
        "    import ctypes\n\n    # Checks\n",
    )
    files[member] = source.encode()
    plugin = "imageio/plugins/pillow.py"
    source = files[plugin].decode()
    eager = "from PIL import ExifTags, GifImagePlugin, Image, ImageSequence, UnidentifiedImageError"
    branch = '        if self._image.format == "GIF":\n            # Converting GIF'
    if source.count(eager) != 1 or source.count(branch) != 1:
        raise ValueError("imageio Pillow plugin differs from reviewed patch")
    source = source.replace(eager, eager.replace("GifImagePlugin, ", "")).replace(
        branch,
        '        if self._image.format == "GIF":\n            from PIL import GifImagePlugin\n\n            # Converting GIF',
    )
    files[plugin] = source.encode()
    record = "imageio-2.37.0.dist-info/RECORD"
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    for name, data in sorted(files.items()):
        if name != record:
            sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow([name, "sha256=" + sha, len(data)])
    writer.writerow([record, "", ""])
    files[record] = output.getvalue().encode()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as derived:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2025, 1, 20, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            derived.writestr(entry, data)
    return destination, {
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "wheel_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "disabled": ["dynamic-library-loading"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/shellsim-imaging-graph"))
    parser.add_argument("--bundle", type=Path, default=Path("/tmp/shellsim-imaging-graph"))
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args()
    work, bundle = args.work_dir.resolve(), args.bundle.resolve()
    for name in ("result.json", "guest.stdout", "guest.stderr"):
        (work / name).unlink(missing_ok=True)
    plan, wheel = resolve(work, bundle)
    if args.resolve_only:
        return
    from shellsim import CPythonRuntime, Environment

    runtime = CPythonRuntime(bundle)
    providers = {p["name"]: p for p in runtime.manifest["native_ports"]}
    for name, recipe in plan["native_recipes"].items():
        if name not in providers or any(providers[name].get(k) != v for k, v in recipe.items()):
            raise ValueError("bundle differs from selected native recipe")
    libraries = runtime.manifest["native_libraries"]
    if libraries["native/zlib"]["inputs"]["recipe"]["version"] != "1.3.1":
        raise ValueError("bundle does not provide pinned zlib")
    environment = Environment(cpu=3_000_000_000, memory=256 * 1024 * 1024, disk=128 * 1024 * 1024)
    runtime.mount(environment)
    installed_wheel, adaptation = adapted_wheel(wheel, work)
    runtime.install_wheel(environment, installed_wheel)
    result = runtime.run(environment, ["-c", VERIFY])
    (work / "guest.stdout").write_bytes(result.stdout)
    (work / "guest.stderr").write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f"guest verification failed: {result.stderr_text}")
    if not environment.read_file("/tmp/graph.png").startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("guest did not create PNG in VFS")
    guest = json.loads(result.stdout)
    if guest["guest_markers"] != MARKERS:
        raise ValueError("executed guest markers differ from resolved graph")
    plan.update(
        phase="executed",
        adaptation=adaptation,
        guest=guest,
        usage=asdict(result.usage),
        runtime_manifest_sha256=hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
        native_libraries=libraries,
        link_consumers=runtime.manifest["link_consumers"],
    )
    (work / "result.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(result.stdout_text, end="")


if __name__ == "__main__":
    main()
