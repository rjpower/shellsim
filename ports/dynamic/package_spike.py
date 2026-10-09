"""Build a local WASI package universe from the independently built v2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wheel(path: Path, name: str, version: str, tag: str, files: dict[str, bytes], requirements: list[str]) -> None:
    stem = name.replace("-", "_")
    info = f"{stem}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\nRequires-Python: >=3.13\n"
    for requirement in requirements:
        metadata += f"Requires-Dist: {requirement}\n"
    with ZipFile(path, "w") as archive:
        for filename, content in sorted(files.items()):
            archive.writestr(filename, content)
        archive.writestr(f"{info}/METADATA", metadata)
        archive.writestr(
            f"{info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: shellsim-v2-package-spike\n"
            f"Root-Is-Purelib: {str(tag == 'py3-none-any').lower()}\nTag: {tag}\n",
        )
        archive.writestr(f"{info}/RECORD", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numpy-wheel", type=Path)
    parser.add_argument("--magiccube-wheel", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    abi = manifest.get("dynamic_abi")
    if not isinstance(abi, str) or manifest["recipe"]["version"] != "3.13.7":
        raise ValueError("the package spike requires an SDK 34 dynamic CPython 3.13.7 bundle")
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("package universe destination must be empty")
    output.mkdir(parents=True, exist_ok=True)
    wheels = output / "wheels"
    wheels.mkdir()
    providers = output / "providers"
    providers.mkdir()
    native = args.bundle / "zlib_consumer.so"
    library = args.bundle / "libz.so"
    for name, path in (("zlib_consumer.so", native), ("libz.so", library)):
        expected = manifest["proof_artifacts"][name]
        if sha256(path) != expected:
            raise ValueError(f"dynamic bundle proof artifact differs from manifest: {name}")
    native_wheel = wheels / "zlib_consumer-0.1-cp313-cp313-wasm32_wasip1.whl"
    native_manifest = {
        "schema_version": 1,
        "name": "zlib-consumer",
        "version": "0.1",
        "abi": abi,
        "recipe": {
            "source": "ports/dynamic/fixtures/zlib_extension.c",
            "compiler": manifest["dynamic_toolchain"]["identity"],
        },
        "artifacts": [{"path": "zlib_consumer.so", "sha256": sha256(native), "native_dependencies": ["libz.so"]}],
    }
    wheel(
        native_wheel,
        "zlib-consumer",
        "0.1",
        "cp313-cp313-wasm32_wasip1",
        {
            "zlib_consumer.so": native.read_bytes(),
            "zlib_consumer-0.1.dist-info/shellsim-native.json": json.dumps(native_manifest, sort_keys=True).encode(),
        },
        [],
    )
    shutil.copy2(library, providers / "libz.so")
    pure_wheels = output / "pure-wheels"
    pure_wheels.mkdir()
    wrapper = pure_wheels / "wasm_zlib_wrapper-0.1-py3-none-any.whl"
    wheel(
        wrapper,
        "wasm-zlib-wrapper",
        "0.1",
        "py3-none-any",
        {"wasm_zlib_wrapper/__init__.py": b"from zlib_consumer import roundtrip\n"},
        ["zlib-consumer==0.1"],
    )
    simple = output / "pure-simple/wasm-zlib-wrapper"
    simple.mkdir(parents=True)
    (simple / "index.html").write_text(f'<a href="{wrapper.as_uri()}#sha256={sha256(wrapper)}">{wrapper.name}</a>\n')
    if args.magiccube_wheel is not None:
        recipe = json.loads((Path(__file__).parents[1] / "magiccube/recipe.json").read_text())
        source = recipe["source"]
        if args.magiccube_wheel.name != source["filename"] or sha256(args.magiccube_wheel) != source["sha256"]:
            raise ValueError("magiccube wheel differs from its pinned upstream release")
        magiccube = pure_wheels / source["filename"]
        shutil.copy2(args.magiccube_wheel, magiccube)
        magiccube_index = output / "pure-simple/magiccube"
        magiccube_index.mkdir(parents=True)
        (magiccube_index / "index.html").write_text(
            f'<a href="{magiccube.as_uri()}#sha256={source["sha256"]}">{magiccube.name}</a>\n'
        )
    packages = [
        {
            "name": "zlib-consumer",
            "version": "0.1",
            "wheel": "wheels/" + native_wheel.name,
            "sha256": sha256(native_wheel),
        }
    ]
    if args.numpy_wheel is not None:
        numpy = wheels / args.numpy_wheel.name
        shutil.copy2(args.numpy_wheel, numpy)
        packages.append({"name": "numpy", "version": "2.3.5", "wheel": "wheels/" + numpy.name, "sha256": sha256(numpy)})
    catalog = {
        "schema_version": 1,
        "abi": abi,
        "target": "wasm32-wasip1",
        "python_version": "3.13.7",
        "pure_index": "pure-simple",
        "packages": packages,
        "native_providers": [
            {
                "name": "libz.so",
                "path": "providers/libz.so",
                "destination": "/lib/libz.so",
                "sha256": sha256(providers / "libz.so"),
                "native_dependencies": [],
            }
        ],
    }
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
