"""Compile a requested native guest probe beside its remote product artifacts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from ports._support.wasm import mark_abi
from ports.buildomatic.artifacts import install
from ports.buildomatic.remote_store import RemoteStore
from ports.buildomatic.sdk_artifacts import load_sdk


def main() -> None:
    specification = json.loads((Path(os.environ["BUILD_INPUT_recipe"]) / "node.json").read_text())
    store = RemoteStore(specification["prefix"])
    cache = Path("/app/.buildomatic/artifacts")
    sdk = load_sdk(store, specification["sdk"], cache)
    product = Path("probe-sysroot")
    product.mkdir()
    for reference in specification["products"].values():
        native = install(store, reference["artifact"], cache) / "native"
        if native.is_dir():
            shutil.copytree(native, product, dirs_exist_ok=True, symlinks=True)
    ports = Path(__file__).resolve().parents[1]
    directory = ports / Path(specification["reference"].partition(":")[0]).parent
    test = specification["test"]
    output = Path(os.environ["BUILD_OUTPUT_DIR"]) / "acceptance.wasm"
    command = [
        str(sdk.compiler()),
        *sdk.compiler_flags,
        *sdk.linker_flags,
        *sdk.executable_flags,
        "-I" + str(product / "include"),
        *("-I" + str(product / name) for name in test.get("include_directories", [])),
        "-L" + str(product / "lib"),
        str(directory / test["source"]),
        *(("-Wl,-Bdynamic",) if any(name.endswith(".so") for name in test["link_inputs"]) else ()),
        *(str(product / name) for name in test["link_inputs"]),
        *(str(sdk.sysroot.root / "sysroot" / name) for name in test.get("sdk_link_inputs", [])),
        *test.get("link_flags", []),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    mark_abi(output, sdk.dynamic_abi.encode())
    libraries = output.parent / "lib"
    libraries.mkdir()
    for library in (product / "lib").glob("*.so"):
        shutil.copyfile(library, libraries / library.name)


if __name__ == "__main__":
    main()
