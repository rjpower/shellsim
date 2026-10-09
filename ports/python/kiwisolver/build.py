"""Build upstream Kiwi as an independent SDK 34 CPython C++ extension.

The fixed interpreter owns libc and the C++ exception runtime. The source patch
corrects CPython callback signatures; it does not alter solver behavior.
"""

import argparse
import base64
import csv
import email
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ports._support.build import apply_patch, check_build_scripts
from ports._support.wasm import mark_abi
from ports.native.dependencies import file_hash, target_environment, target_profile
from ports.python.cellpylib.build import _relative, _unpack_tool
from ports.toolchain.wasi_sdk.build import dynamic_toolchain

PORT = Path(__file__).resolve().parent


def write_wheel(stage, destination):
    """Write deterministic entries with a complete PEP 427 RECORD."""
    record = "kiwisolver-1.5.1.dist-info/RECORD"
    rows = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            rows.append((path.relative_to(stage).as_posix(), "sha256=" + digest, len(data)))
    rows.append((record, "", ""))
    text = io.StringIO(newline="")
    csv.writer(text, lineterminator="\n").writerows(rows)
    (stage / record).write_text(text.getvalue())
    with zipfile.ZipFile(destination, "w") as wheel:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                entry = zipfile.ZipInfo(path.relative_to(stage).as_posix(), (1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o644 << 16
                wheel.writestr(entry, path.read_bytes())


def unpack_source(archive_path, destination, recipe):
    """Extract only bounded regular files from the pinned upstream source."""
    if file_hash(archive_path) != recipe["source"]["sha256"]:
        raise ValueError("Kiwi source archive differs from its pin")
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if len(members) > 2048 or sum(member.size for member in members) > 64 * 1024**2:
            raise ValueError("Kiwi source archive exceeds its bound")
        seen = set()
        for member in members:
            relative = _relative(member.name)
            if (
                relative.parts[0] != "kiwisolver-" + recipe["version"]
                or relative in seen
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Kiwi source contains unsupported archive members")
            seen.add(relative)
        archive.extractall(destination, filter="data")
    return destination / ("kiwisolver-" + recipe["version"])


def version_header(source, recipe):
    """Use upstream's SCM template with the verified sdist release identity."""
    metadata = email.message_from_bytes((source / "PKG-INFO").read_bytes())
    if metadata["Name"] != recipe["name"] or metadata["Version"] != recipe["version"]:
        raise ValueError("Kiwi sdist metadata differs from the recipe")
    if metadata.get_all("Requires-Dist", []) or metadata["Requires-Python"] != ">=3.10":
        raise ValueError("Kiwi requirements differ from the reviewed release")
    config = tomllib.loads((source / "pyproject.toml").read_text())
    scm = config["tool"]["setuptools_scm"]
    if scm["write_to"] != "py/src/version.h":
        raise ValueError("Kiwi version-header output changed")
    header = scm["write_to_template"].format(version=metadata["Version"])
    (source / scm["write_to"]).write_text(header)
    return hashlib.sha256(header.encode()).hexdigest()


def build(archive, cppy_wheel, cpython_source, cpython_build, sdk, runtime, output):
    """Verify build inputs and emit a wheel without changing the interpreter."""
    recipe = json.loads((PORT / "recipe.json").read_text())
    check_build_scripts(recipe, PORT)
    tool = recipe["cppy"]
    if cppy_wheel.name != tool["filename"] or file_hash(cppy_wheel) != tool["sha256"]:
        raise ValueError("CPPy headers differ from their pin")
    if file_hash(cpython_build / "pyconfig.h") != recipe["cpython"]["pyconfig_sha256"]:
        raise ValueError("Target CPython configuration differs from the reviewed cohort")
    if (cpython_source / "Include/patchlevel.h").read_text().find('"3.13.7"') < 0:
        raise ValueError("Kiwi requires target CPython 3.13.7 headers")
    inputs = {
        path.relative_to(cpython_source).as_posix(): file_hash(path)
        for path in sorted((cpython_source / "Include").rglob("*.h"))
    }
    headers_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if headers_hash != recipe["cpython"]["headers_sha256"]:
        raise ValueError("Target CPython source headers differ from their pin")
    runtime_manifest = json.loads((runtime / "manifest.json").read_text())
    if runtime_manifest.get("dynamic_abi") != recipe["abi"]:
        raise ValueError("Kiwi requires the SDK 34 dynamic ABI v2 runtime")
    toolchain, identity, _ = dynamic_toolchain(sdk)
    profile = target_profile(toolchain)
    output.mkdir(parents=True)
    source = unpack_source(archive, output / "source", recipe)
    _unpack_tool(cppy_wheel, output / "cppy")
    generated_hash = version_header(source, recipe)
    patch = recipe["patch"]
    apply_patch(source, PORT / patch["file"], patch["sha256"])
    stage = output / "wheel-root"
    shutil.copytree(source / "py/kiwisolver", stage / "kiwisolver")
    environment = target_environment(sdk)
    objects = []
    with (output / "build.log").open("w") as log:
        for name in recipe["sources"]:
            obj = output / (Path(name).stem + ".o")
            command = [
                str(sdk / "bin/clang++"),
                *profile["compiler_flags"],
                *profile["cpp_flags"],
                "-fPIC",
                "-I" + str(cpython_source / "Include"),
                "-I" + str(cpython_build),
                "-I" + str(output / "cppy/cppy/include"),
                "-I" + str(source),
                "-c",
                str(source / name),
                "-o",
                str(obj),
            ]
            subprocess.run(command, check=True, env=environment, stdout=log, stderr=log)
            objects.append(obj)
        library = stage / "kiwisolver/_cext.so"
        subprocess.run(
            [
                str(sdk / "bin/clang++"),
                *profile["cpp_flags"],
                *toolchain["side_link_flags"],
                "-Wl,--export=PyInit__cext,--export-all,--fatal-warnings",
                *(str(obj) for obj in objects),
                "-o",
                str(library),
            ],
            check=True,
            env=environment,
            stdout=log,
            stderr=log,
        )
    mark_abi(library, recipe["abi"].encode())
    dist = stage / "kiwisolver-1.5.1.dist-info"
    dist.mkdir()
    shutil.copyfile(source / "PKG-INFO", dist / "METADATA")
    licenses = dist / "licenses"
    licenses.mkdir()
    shutil.copyfile(source / "LICENSE", licenses / "LICENSE")
    shutil.copyfile(output / "cppy/cppy-1.3.1.dist-info/LICENSE", licenses / "CPPy-LICENSE")
    manifest = {
        "schema_version": 1,
        "name": recipe["name"],
        "version": recipe["version"],
        "abi": recipe["abi"],
        "recipe": recipe,
        "toolchain": identity,
        "cpython_headers": inputs,
        "generated_version_header_sha256": generated_hash,
        "runtime_manifest_sha256": file_hash(runtime / "manifest.json"),
        "runtime_interpreter_sha256": file_hash(runtime / "rootfs/usr/bin/python3.wasm"),
        "build_python": {"version": sys.version, "sha256": file_hash(Path(sys.executable))},
        "artifacts": [{"path": "kiwisolver/_cext.so", "sha256": file_hash(library), "native_dependencies": []}],
    }
    (dist / "shellsim-native.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dist / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: shellsim-kiwisolver\nRoot-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1\n"
    )
    wheel = output / "kiwisolver-1.5.1-cp313-cp313-wasm32_wasip1.whl"
    write_wheel(stage, wheel)
    (output / "manifest.json").write_text(json.dumps({**manifest, "wheel_sha256": file_hash(wheel)}, indent=2) + "\n")
    return wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "cppy-wheel", "cpython-source", "cpython-build", "sdk", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        build(
            *(
                getattr(args, name).resolve()
                for name in ("archive", "cppy_wheel", "cpython_source", "cpython_build", "sdk", "runtime", "output")
            )
        )
    )


if __name__ == "__main__":
    main()
