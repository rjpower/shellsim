"""Seal complete image-native packages in the retained bootstrap host seed.

Run inside the published image with the retained prepared root mounted at its
original absolute path. Only task-owned files are created; existing caller
utilities are verified in place. This never materializes any SDK product.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from ports._support import host_tools, sdk
from ports._support.sdk_products import file_hash

PACKAGES = {
    "cc": ("gcc", "bin/gcc"),
    "cmake": ("cmake", "bin/cmake"),
    "ninja": ("ninja", "bin/ninja"),
    "pkg-config": ("pkgconf", "bin/pkg-config"),
}


def write_new(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.write("\n")
    path.chmod(0o444)
    return path


def stage(image: str) -> Path:
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("published image must be pinned by digest")
    inputs = json.loads(Path("/opt/buildomatic/inputs.json").read_text())
    root = Path(inputs["prepared_root"])
    if root.resolve() != root or (root / "launch.json").exists():
        raise ValueError("native staging requires the original unlaunched task root")
    prepared = root / "host-seed.prepared.json"
    if file_hash(prepared) != inputs["prepared_seed_sha256"]:
        raise ValueError("prepared seed differs")
    sdk.load_seed(prepared)
    seed = json.loads(prepared.read_text())
    for name, binding in inputs["system_bindings"].items():
        path = Path(binding["path"])
        alias = os.readlink(path) if path.is_symlink() else None
        if file_hash(path) != binding["sha256"] or alias != binding["symlink"]:
            raise ValueError("system utility differs: " + name)
        seed["tools"][name] = {"path": str(path), "sha256": binding["sha256"], "receipt": None}
    destination = root / "host/native"
    if destination.exists() or (root / "host-seed.ready.json").exists():
        raise FileExistsError(destination)
    destination.mkdir()
    for name, (package, executable) in PACKAGES.items():
        original = Path("/opt/buildomatic/native") / package
        copied = destination / package
        aliases = {}
        files = host_tools.inventory(original, symlinks=aliases)
        shutil.copytree(original, copied, symlinks=True)
        copied_aliases = {}
        if host_tools.inventory(copied, symlinks=copied_aliases) != files or copied_aliases != aliases:
            raise ValueError("native package changed during staging")
        host_tools.make_read_only(copied)
        receipt = write_new(
            root / "host/native-receipts" / (name + ".json"),
            {
                "schema_version": 2,
                "kind": "host-tool-files",
                "name": name,
                "version": "published-image",
                "root": str(copied),
                "executable": executable,
                "source": {"image": image, "origin": "complete-image-native-package"},
                "files": files,
                "symlinks": aliases,
            },
        )
        seed["tools"][name] = {
            "path": str(copied / executable),
            "sha256": file_hash(copied / executable),
            "receipt": {"path": str(receipt), "sha256": file_hash(receipt)},
        }
    helper = destination / "python-helper"
    original_helper = Path("/opt/buildomatic/python-helper")
    aliases = {}
    expected = host_tools.inventory(original_helper, symlinks=aliases)
    shutil.copytree(original_helper, helper, symlinks=True)
    copied_aliases = {}
    if host_tools.inventory(helper, symlinks=copied_aliases) != expected or copied_aliases != aliases:
        raise ValueError("native helper changed during staging")
    host_tools.make_read_only(destination)
    paths = {
        "cc": destination / "gcc/bin/gcc",
        "cxx": destination / "gcc/bin/g++",
        "cmake": destination / "cmake/bin/cmake",
        "ninja": destination / "ninja/bin/ninja",
    }
    seed["compiler_tools"] = {name: {"path": str(path), "sha256": file_hash(path)} for name, path in paths.items()}
    seed["python_helper"] = {"path": str(helper / "bin/python3.13"), "sha256": file_hash(helper / "bin/python3.13")}
    ready = write_new(root / "host-seed.ready.json", seed)
    sdk.load_seed(ready)
    write_new(
        root / "native-staged.json",
        {"schema_version": 1, "image": image, "host_seed_sha256": file_hash(ready), "compile_started": False},
    )
    return ready


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(stage(args.image))
