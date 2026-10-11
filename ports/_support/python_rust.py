"""Build locked upstream PyO3 extensions against cached PIC Rust libraries."""

import email
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping

from packaging.utils import canonicalize_name

from ports._support.build_context import BuildContext
from ports._support.native_adapters import NativeBuildCommand
from ports._support.python_adapters import PythonBuildOutput, _source_file, _wheel
from ports._support.rust_tools import run_cargo, rust_environment, rust_tools
from ports._support.store import file_hash, identity
from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import validate_provider_signatures


def linker_script(linker: Path, inputs: tuple[Path, ...]) -> str:
    """Adapt Rust's executable-oriented WASI link to the shared SDK protocol."""
    options = [
        "--shared",
        "--shared-memory",
        "--import-memory",
        "--import-table",
        "--serial-memory-init",
        "--defer-shared-init",
        "--no-export-dynamic",
        "--unresolved-symbols=import-dynamic",
        *(str(path) for path in inputs),
    ]
    return (
        "#!" + sys.executable + "\nimport os, sys\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['-flavor', 'wasm']: args = args[2:]\n"
        "for index in range(len(args) - 2, -1, -1):\n"
        "    if args[index:index + 2] == ['-l', 'c']: del args[index:index + 2]\n"
        "args = [arg for arg in args if arg not in ('--export-memory', '--stack-first', '-lc')"
        " and not arg.startswith('--max-memory=')]\n"
        + "os.execv("
        + repr(str(linker))
        + ", ["
        + repr(str(linker))
        + "] + "
        + repr(options)
        + " + args)\n"
    )


def generated_file(path: Path, contents: str, *, executable: bool = False) -> None:
    """Publish immutable generated text atomically for concurrent Cargo users.

    The caller keys the path by its contents. Existing complete entries need
    no reads or rewrites; the executable mode is set before publication.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".generated-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(contents)
        staged.chmod(0o755 if executable else 0o644)
        os.replace(staged, path)


def build_rust_extension(
    ctx: BuildContext,
    *,
    module: str,
    licenses: tuple[str, ...],
    stubs: tuple[str, ...] = (),
    python_sources: tuple[tuple[str, str], ...] = (),
    manifest_path: str = "Cargo.toml",
    configure_environment: Mapping[str, str] | None = None,
) -> PythonBuildOutput:
    """Preserve source metadata and stage an ABI-marked native extension wheel."""
    native, python = ctx.require_native(), ctx.require_cpython()
    if (
        python.version.split(".")[:2] != ["3", "13"]
        or python.wheel_platform != "wasm32_wasip1"
        or python.runtime_target != native.target
    ):
        raise ValueError("Rust extension needs the admitted CPython 3.13 WASI target")
    environment = rust_environment(ctx)
    rustc, cargo, _, _ = rust_tools("1.97.1", environment)
    native.build.mkdir(parents=True, exist_ok=True)
    configuration_text = (
        "implementation=CPython\nversion="
        + ".".join(python.version.split(".")[:2])
        + "\nshared=true\nabi3=false\npointer_width=32\nsuppress_build_script_link_lines=true\n"
    )
    linker_text = linker_script(native.linker, native.shared_library_inputs)
    # Stable generated paths keep Cargo dependency fingerprints warm between
    # private source workspaces. These keys cover text, not host executable bytes.
    tools = Path(environment["CARGO_TARGET_DIR"]) / "shellsim-tools"
    configuration = tools / identity({"pyo3": configuration_text}) / "pyo3-config.txt"
    generated_file(configuration, configuration_text)
    linker = tools / identity({"linker": linker_text}) / "rust-shared-linker"
    generated_file(linker, linker_text, executable=True)
    stdlib = ctx.build_artifacts["toolchain/rust-stdlib"]
    scheduler = ctx.build_artifacts["toolchain/wasm-scheduler"] / "bin/shellsim-wasm-scheduler"
    environment.update(
        {
            "RUSTC": str(rustc),
            "PYO3_CONFIG_FILE": str(configuration),
            "RUSTFLAGS": shlex.join(
                [
                    "-C",
                    "relocation-model=pic",
                    "-C",
                    "link-self-contained=no",
                    "--sysroot",
                    str(stdlib),
                    "-C",
                    "linker=" + str(linker),
                ]
            ),
        }
    )
    environment.update(configure_environment or {})
    command = [
        str(cargo),
        "build",
        "--manifest-path",
        str(native.source / manifest_path),
        "--release",
        "--locked",
        "--target",
        native.target,
        "--message-format=json",
    ]
    if ctx.offline:
        command.append("--offline")
    result = run_cargo(command, native.source, environment, native.build / "cargo.log")
    libraries = [
        Path(filename)
        for line in result.stdout.splitlines()
        if (artifact := json.loads(line)).get("reason") == "compiler-artifact"
        and "cdylib" in artifact["target"]["crate_types"]
        for filename in artifact["filenames"]
        if filename.endswith(".wasm")
    ]
    if len(libraries) != 1:
        raise ValueError("Rust project must produce one target extension")
    stage = native.build / "wheel"
    stage.mkdir()
    for source, destination in python_sources:
        shutil.copytree(native.source / source, stage / destination)
    extension = stage / (module.replace(".", "/") + ".so")
    extension.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(scheduler), str(libraries[0]), str(extension), "shellsim_threads_v2"], check=True)
    mark_abi(extension, native.abi.encode())
    needed = validate_provider_signatures(extension, {})
    metadata = _source_file(native.source, "PKG-INFO")
    message = email.message_from_bytes(metadata.read_bytes())
    recipe = ctx.metadata
    if (
        canonicalize_name(message["Name"]) != canonicalize_name(recipe["name"])
        or message["Version"] != recipe["version"]
        or sorted(message.get_all("Requires-Dist", [])) != sorted(recipe.get("requires_dist", []))
    ):
        raise ValueError("Rust source metadata differs from recipe")
    name = recipe["name"].replace("-", "_")
    dist = name + "-" + recipe["version"] + ".dist-info"
    info = stage / dist
    info.mkdir()
    shutil.copyfile(metadata, info / "METADATA")
    (info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: shellsim-python-rust\nRoot-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1\n"
    )
    (info / "licenses").mkdir()
    for name in licenses:
        shutil.copyfile(_source_file(native.source, name), info / "licenses" / Path(name).name)
    for name in stubs:
        shutil.copyfile(_source_file(native.source, name), stage / Path(name).name)
    for path in (stdlib / "licenses").iterdir():
        shutil.copyfile(path, info / "licenses" / ("Rust-" + path.name))
    manifest = {
        "schema_version": 1,
        "name": recipe["name"],
        "version": recipe["version"],
        "abi": native.abi,
        "recipe": dict(recipe),
        "cpython_receipt": dict(python.receipt),
        "artifacts": [
            {
                "path": extension.relative_to(stage).as_posix(),
                "sha256": file_hash(extension),
                "native_dependencies": needed,
            }
        ],
    }
    (info / "shellsim-native.json").write_text(json.dumps(manifest) + "\n")
    wheel = (
        native.staging_prefix
        / "wheels"
        / f"{recipe['name'].replace('-', '_')}-{recipe['version']}-cp313-cp313-wasm32_wasip1.whl"
    )
    wheel.parent.mkdir(parents=True)
    _wheel(stage, wheel, dist + "/RECORD")
    return PythonBuildOutput(native.staging_prefix, (NativeBuildCommand(tuple(command), native.source),))
