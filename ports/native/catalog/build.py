"""Assemble verified guest tools and the measured C/zlib development graph.

Wrapping existing artifacts preserves their original provenance and byte hashes;
this command neither compiles them nor executes any target program on the host.
"""

import argparse
import json
import shutil
from pathlib import Path

import tomllib
from shellsim_c_toolchain import _verified_assets

from ports.native.dependencies import digest, file_hash, seal_artifact, verify_artifact

ROOT = Path(__file__).resolve().parents[3]


def _write(prefix, name, payload):
    path = prefix / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _record(root, prefix, manifest, name, kind, destinations):
    recipe = manifest["inputs"]["recipe"]
    return {
        "name": name,
        "version": recipe["version"],
        "kind": kind,
        "artifact": prefix.relative_to(root).as_posix(),
        "artifact_sha256": manifest["artifact_sha256"],
        "destinations": destinations,
        "dependencies": [],
        "recipe_name": recipe["name"],
        "target_profile": recipe["target_profile"],
        "toolchain_sha256": digest(manifest["inputs"]["toolchain"]),
        "abi": recipe.get("abi"),
    }


def build_catalog(make: Path, zlib: Path, proof: Path, destination: Path):
    """Seal the already measured TinyCC/SDK34 C graph and pinned make release."""
    distribution = tomllib.loads((ROOT / "toolchain/pyproject.toml").read_text())["project"]
    if distribution["name"] != "shellsim-c-toolchain" or distribution["version"] != "0.1.30":
        raise ValueError("C toolchain distribution differs from its catalog pin")
    destination.mkdir(parents=True)
    artifacts = destination / "artifacts"
    artifacts.mkdir()
    records = []
    manifest = verify_artifact(make)
    prefix = artifacts / manifest["artifact_sha256"]
    shutil.copytree(make, prefix)
    mapping = {
        name: "/usr/bin/make" if name == "bin/make" else "/usr/share/shellsim/native/make/" + name
        for name in manifest["files"]
    }
    records.append(_record(destination, prefix, manifest, "make", "build-tool", mapping))

    prefix = artifacts / "c-toolchain"
    prefix.mkdir()
    exports = []
    for tree, files in _verified_assets():
        for name, payload in files:
            path = tree + "/" + name
            _write(prefix, path, payload)
            exports.append(path)
    _write(prefix, "bin/cc", (prefix / "tcc/tcc-shellsim.wasm").read_bytes())
    assets = ROOT / "toolchain/src/shellsim_c_toolchain/_assets"
    archive = assets / "tinycc-22a2e10-source.tar.gz"
    if file_hash(archive) != "0be27686ffa17cbac5c95827941bd1715bbc88348f5bde381a8cb596ce7b427b":
        raise ValueError("TinyCC corresponding source archive differs from its pin")
    notices = ["source/" + archive.name]
    _write(prefix, notices[0], archive.read_bytes())
    for source in sorted((ROOT / "toolchain/LICENSES").iterdir()):
        path = "licenses/" + source.name
        _write(prefix, path, source.read_bytes())
        notices.append(path)
    recipe = {
        "name": "shellsim-c-toolchain",
        "version": "0.1.30",
        "target": "wasm32-wasip1",
        "target_profile": "tinycc-c-wasi-v1",
        "target_dependencies": [],
        "source": {"commit": "22a2e10d6fb5be75af2863a1b9cc07b9260fe99e", "sha256": file_hash(archive)},
        "exports": {
            "tools": ["bin/cc", "tcc/tcc-shellsim.wasm"],
            "development": [name for name in exports if name != "tcc/tcc-shellsim.wasm"],
            "notices": notices,
        },
    }
    inputs = {
        "recipe": recipe,
        "recipe_sha256": digest(recipe),
        "source_sha256": file_hash(archive),
        "toolchain": {
            "profile": {"name": "tinycc-c-wasi-v1", "target": "wasm32-wasip1"},
            "compiler_sha256": file_hash(prefix / "bin/cc"),
            "sysroot_tree_sha256": "c8db6acd30553b74e149cf9bf23387107e7587761cc930933026000af3a03e63",
        },
        "dependency_artifacts": {},
        "catalog_builder_sha256": file_hash(Path(__file__)),
    }
    manifest = seal_artifact(prefix, inputs)
    mapping = {
        name: "/usr/bin/cc"
        if name == "bin/cc"
        else "/" + name
        if name.startswith(("tcc/", "wasi-sysroot/"))
        else "/usr/share/shellsim/native/shellsim-c-toolchain/" + name
        for name in manifest["files"]
    }
    records.append(_record(destination, prefix, manifest, "shellsim-c-toolchain", "build-tool", mapping))

    original = verify_artifact(zlib)
    evidence = json.loads(proof.read_text())
    if evidence["zlib_artifact"] != original["artifact_sha256"] or any(run["status"] for run in evidence["runs"]):
        raise ValueError("C/zlib compatibility proof does not match the source artifact")
    prefix = artifacts / "zlib-devel"
    shutil.copytree(zlib, prefix)
    (prefix / "artifact.json").unlink()
    _write(prefix, "provenance/original-artifact.json", (zlib / "artifact.json").read_bytes())
    _write(prefix, "provenance/c-zlib-proof.json", proof.read_bytes())
    inputs = json.loads(json.dumps(original["inputs"]))
    inputs["recipe"]["exports"]["provenance"] = ["provenance/original-artifact.json", "provenance/c-zlib-proof.json"]
    inputs["recipe_sha256"] = digest(inputs["recipe"])
    inputs["original_artifact_sha256"] = original["artifact_sha256"]
    inputs["catalog_builder_sha256"] = file_hash(Path(__file__))
    inputs["compatibility"] = {
        "scope": "tinycc-sdk34-c-zlib-v1",
        "proof_sha256": file_hash(proof),
        "compiler_sha256": "b81a97bb6630cce02d9bebcb465983f38b9cc5895e0af57d09612221494808f9",
        "sysroot_tree_sha256": "c8db6acd30553b74e149cf9bf23387107e7587761cc930933026000af3a03e63",
    }
    manifest = seal_artifact(prefix, inputs)
    mapping = {
        name: "/opt/zlib/" + name
        if name.startswith(("include/", "lib/"))
        else "/usr/share/shellsim/native/zlib-devel/" + name
        for name in manifest["files"]
    }
    records.append(_record(destination, prefix, manifest, "zlib-devel", "devel", mapping))
    catalog = {"format": 1, "target": "wasm32-wasip1", "packages": records}
    (destination / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    return destination / "catalog.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("make", "zlib", "proof", "destination"):
        parser.add_argument(name, type=Path)
    arguments = parser.parse_args()
    print(build_catalog(arguments.make, arguments.zlib, arguments.proof, arguments.destination))


if __name__ == "__main__":
    main()
