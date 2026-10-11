"""Package upstream RDKit chemistry modules and their real C++ wrappers."""

import hashlib
import json
import shutil

from ports._support.python_adapters import PythonBuildOutput, _wheel
from ports._support.wasm import mark_abi
from ports._support.wasm_metadata import needed_libraries
from ports.api import BuildContext
from ports.native.rdkit_bindings import build_support


def build(ctx: BuildContext) -> PythonBuildOutput:
    native = ctx.require_native()
    python = ctx.require_cpython()
    output = build_support(ctx, bindings=True)
    payload = native.staging_prefix / "bindings/rdkit"
    stage = native.build / "wheel"
    package = stage / "rdkit"
    package.mkdir(parents=True)
    for source in sorted((native.source / "rdkit").rglob("*.py")):
        relative = source.relative_to(native.source / "rdkit")
        if any(part in {"test_data", "testData", "__pycache__"} for part in relative.parts) or source.name.startswith(
            "UnitTest"
        ):
            continue
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    # Upstream selects this module when its optional InChI component is disabled.
    shutil.copyfile(native.source / "External/INCHI-API/python/noinchi.py", package / "Chem/inchi.py")
    shutil.copyfile(native.build / "cmake-build/rdkit/RDPaths.py", package / "RDPaths.py")
    artifacts = []
    for source in sorted(payload.rglob("*.so")):
        relative = source.relative_to(payload)
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        mark_abi(destination, native.abi.encode())
        artifacts.append(
            {
                "path": destination.relative_to(stage).as_posix(),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "native_dependencies": needed_libraries(destination),
            }
        )
    dist = f"rdkit-{ctx.port.version}.dist-info"
    info = stage / dist
    info.mkdir()
    metadata = ["Metadata-Version: 2.1", "Name: rdkit", f"Version: {ctx.port.version}"]
    metadata.extend("Requires-Dist: " + value for value in ctx.metadata["requires_dist"])
    (info / "METADATA").write_text("\n".join(metadata) + "\n\n")
    (info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: shellsim-rdkit\nRoot-Is-Purelib: false\nTag: cp313-cp313-wasm32_wasip1\n"
    )
    manifest = {
        "schema_version": 1,
        "name": "rdkit",
        "version": ctx.port.version,
        "abi": native.abi,
        "recipe": dict(ctx.metadata),
        "cpython_receipt": dict(python.receipt),
        "artifacts": artifacts,
    }
    (info / "shellsim-native.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    licenses = info / "licenses"
    licenses.mkdir()
    shutil.copyfile(native.source / "license.txt", licenses / "rdkit.txt")
    wheels = native.staging_prefix / "wheels"
    wheels.mkdir()
    _wheel(stage, wheels / f"rdkit-{ctx.port.version}-cp313-cp313-wasm32_wasip1.whl", dist + "/RECORD")
    return PythonBuildOutput(native.staging_prefix, output.commands)
