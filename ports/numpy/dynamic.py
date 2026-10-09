"""Package PIC NumPy archives as independent SDK 34 CPython extensions.

The reviewed archive provider supplies target objects and generated Python files.
This builder never modifies or relinks the fixed CPython runtime bundle.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ports.cpython.build import install_numpy_notices
from ports.dynamic.build import mark_abi
from ports.native.dependencies import file_hash, target_environment, target_profile
from ports.numpy.build import check_build_scripts, install_numpy
from ports.toolchain.wasi_sdk.build import dynamic_toolchain


def provider_inputs(provider, recipe):
    """Refuse unknown cached objects and map their original qualified modules."""
    manifest = json.loads((provider / "manifest.json").read_text())
    native = next(port for port in manifest["native_ports"] if port["name"] == "numpy")
    identity = hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest()
    if identity != recipe["archive_provider_sha256"]:
        raise ValueError("NumPy archive provider differs from the pinned cohort")
    if (provider / "numpy-profile.sha256").read_text().strip() != identity:
        raise ValueError("NumPy archive cache has a different recipe identity")
    archive = provider / "downloads" / recipe["source"]["url"].rsplit("/", 1)[1]
    if file_hash(archive) != recipe["source"]["sha256"]:
        raise ValueError("NumPy upstream archive hash mismatch")
    build = provider / "numpy-build"
    targets = json.loads((build / "meson-info/intro-targets.json").read_text())
    modules = {}
    support = {}
    for target in targets:
        if target["type"] != "static library":
            continue
        path = Path(target["filename"][0])
        if target["name"] in ("npymath", "npyrandom"):
            support[target["name"]] = path
            continue
        destinations = target.get("install_filename")
        if not destinations or ".cpython-" not in target["name"]:
            continue
        name = target["name"].split(".cpython-", 1)[0]
        parent = Path(destinations[0]).parent.as_posix().split("/site-packages/", 1)[1]
        qualified = parent.replace("/", ".") + "." + name
        if qualified in recipe["modules"]:
            modules[qualified] = path
    if set(modules) != set(recipe["modules"]) or set(support) != {"npymath", "npyrandom"}:
        raise ValueError("NumPy target archives do not match the dynamic module closure")
    return native, modules, support


def write_wheel(stage, destination):
    """Write deterministic wheel entries and a complete PEP 427 RECORD."""
    records = []
    record = "numpy-2.3.5.dist-info/RECORD"
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.relative_to(stage).as_posix() != record:
            content = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            records.append((path.relative_to(stage).as_posix(), "sha256=" + digest, len(content)))
    records.append((record, "", ""))
    text = io.StringIO(newline="")
    csv.writer(text, lineterminator="\n").writerows(records)
    (stage / record).write_text(text.getvalue())
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                entry = zipfile.ZipInfo(path.relative_to(stage).as_posix(), (1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o644 << 16
                wheel.writestr(entry, path.read_bytes())


def build_dynamic(provider, runtime, output):
    directory = Path(__file__).parent
    recipe = json.loads((directory / "dynamic-recipe.json").read_text())
    check_build_scripts(recipe, directory)
    runtime_manifest = json.loads((runtime / "manifest.json").read_text())
    if runtime_manifest.get("dynamic_abi") != recipe["abi"] or runtime_manifest["native_ports"]:
        raise ValueError("NumPy requires the fixed bare dynamic ABI v2 runtime")
    native, modules, support = provider_inputs(provider, recipe)
    sdk = provider / "wasi-sdk-34.0-x86_64-linux"
    toolchain, identity, _ = dynamic_toolchain(sdk)
    profile = target_profile(toolchain)
    compiler_runtime = Path(
        subprocess.check_output(
            [str(sdk / "bin/clang"), "--print-libgcc-file-name"],
            text=True,
            env=target_environment(sdk),
        ).strip()
    )
    stage = output / "wheel-root"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    install_numpy(provider / "numpy-build", stage)
    artifacts = []
    archive_hashes = {}
    for qualified, archive in modules.items():
        name = qualified.rsplit(".", 1)[1]
        target = stage / (qualified.replace(".", "/") + ".so")
        target.parent.mkdir(parents=True, exist_ok=True)
        libraries = [support[name] for name in recipe["modules"][qualified]]
        command = [
            str(sdk / "bin/clang++"),
            *profile["compiler_flags"],
            *profile["cpp_flags"],
            *toolchain["side_link_flags"],
            f"-Wl,--export=PyInit_{name}",
            "-Wl,--whole-archive",
            str(archive),
            "-Wl,--no-whole-archive",
            *(str(path) for path in libraries),
            str(compiler_runtime),
            "-o",
            str(target),
        ]
        subprocess.run(command, env=target_environment(sdk), check=True)
        mark_abi(target, recipe["abi"].encode())
        artifacts.append(
            {
                "path": target.relative_to(stage).as_posix(),
                "sha256": file_hash(target),
                "native_dependencies": [],
            }
        )
        for path in (archive, *libraries):
            archive_hashes[str(path.relative_to(provider))] = file_hash(path)
    source = provider / "numpy-2.3.5"
    dist = stage / "numpy-2.3.5.dist-info"
    dist.mkdir()
    shutil.copy2(source / "PKG-INFO", dist / "METADATA")
    licenses = dist / "licenses"
    licenses.mkdir()
    shutil.copy2(source / "LICENSE.txt", licenses / "LICENSE.txt")
    shutil.copy2(source / "LICENSES_bundled.txt", licenses / "LICENSES_bundled.txt")
    install_numpy_notices(source, licenses)
    shutil.copy2(directory.parent / "toolchain/wasi_sdk/notices/llvm-LICENSE.TXT", licenses / "LLVM-LICENSE.TXT")
    manifest = {
        "schema_version": 1,
        "name": recipe["name"],
        "version": recipe["version"],
        "abi": recipe["abi"],
        "recipe": recipe,
        "artifacts": artifacts,
        "archive_provider": native,
        "archive_inputs": archive_hashes,
        "toolchain": identity,
        "compiler_support": {
            "file": "libclang_rt.builtins.a",
            "sha256": file_hash(compiler_runtime),
            "source": toolchain["runtime_sources"]["llvm"],
        },
        "runtime_manifest_sha256": file_hash(runtime / "manifest.json"),
        "runtime_interpreter_sha256": file_hash(runtime / "rootfs/usr/bin/python3.wasm"),
    }
    (dist / "shellsim-native.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (dist / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: shellsim-numpy-dynamic\nRoot-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1\n"
    )
    wheel = output / "numpy-2.3.5-cp313-cp313-wasm32_wasip1.whl"
    write_wheel(stage, wheel)
    (output / "manifest.json").write_text(json.dumps({**manifest, "wheel_sha256": file_hash(wheel)}, indent=2) + "\n")
    return wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_dynamic(args.provider.resolve(), args.runtime.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
