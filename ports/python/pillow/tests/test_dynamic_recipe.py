"""Reject mismatched shared providers and unsafe source inputs before compiling."""

import hashlib
import io
import json
import tarfile

import pytest

from ports._support.graph import plan
from ports._support.runner import _admit_recipe
from ports._support.sdk_products import Receipt, file_hash, verify_product
from ports._support.store import extract


def analyze(roots):
    from pathlib import Path

    return plan(Path(__file__).resolve().parents[4] / "ports", roots)


def admit_linker(recipe, directory, prefix):
    path = prefix / "manifest.json"
    verify_product(Receipt(prefix, path, file_hash(path), json.loads(path.read_text())))
    return prefix / "bin/wasm-ld", {}


def test_dynamic_recipe_pins_builders_and_provider_build_scripts():
    for port in analyze(["python/pillow"]).ports:
        _admit_recipe(port)


@pytest.mark.parametrize("member", ["/absolute", "source/../../escape"])
def test_source_archive_rejects_unsafe_paths_before_extraction(tmp_path, member):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as output:
        info = tarfile.TarInfo(member)
        info.size = 1
        output.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        extract(archive, tmp_path / "source", subdirectory="source")
    assert not (tmp_path / "escape").exists()


@pytest.fixture
def dependency_graph(tmp_path):
    # Use the production artifact fixture so these checks exercise sealed bytes.
    from ports.native.tests.test_shared_dependencies import graph

    add, consumer, providers, manifests = graph.__wrapped__(tmp_path)
    add("zlib", soname="libz.so")
    return consumer("zlib"), providers, manifests["native/zlib"]


@pytest.mark.parametrize(
    "field,value",
    [("abi", "wrong"), ("name", "wrong"), ("version", "1.0-other"), ("target", "wasm64"), ("target_profile", "wrong")],
)
def test_shared_dependency_rejects_wrong_identity_before_copy(tmp_path, dependency_graph, field, value):
    from ports.native.dependencies import seal_artifact
    from ports.native.tests.test_shared_dependencies import TOOLCHAIN, stage_dependencies

    consumer, providers, manifest = dependency_graph
    manifest["inputs"]["recipe"][field] = value
    seal_artifact(providers["native/zlib"], manifest["inputs"])
    with pytest.raises(ValueError):
        stage_dependencies(consumer, providers, tmp_path / "selected", TOOLCHAIN)
    assert not (tmp_path / "selected").exists()


def test_shared_dependency_rejects_missing_declared_child(tmp_path, dependency_graph):
    from ports.native.dependencies import seal_artifact
    from ports.native.tests.test_shared_dependencies import TOOLCHAIN, stage_dependencies

    consumer, providers, manifest = dependency_graph
    manifest["inputs"]["recipe"]["target_dependencies"] = [{"port": "native/missing", "version": "1.0"}]
    manifest["inputs"]["dependency_artifacts"] = {"native/missing": "0" * 64}
    seal_artifact(providers["native/zlib"], manifest["inputs"])
    with pytest.raises(ValueError):
        stage_dependencies(consumer, providers, tmp_path / "selected", TOOLCHAIN)


def test_shared_dependency_rejects_conflicting_header(tmp_path, dependency_graph):
    from ports.native.tests.test_shared_dependencies import TOOLCHAIN, stage_dependencies

    consumer, providers, manifest = dependency_graph
    destination = tmp_path / "selected"
    (destination / "usr/local/include").mkdir(parents=True)
    (destination / "usr/local/include/zlib.h").write_bytes(b"conflicting header")
    with pytest.raises(ValueError):
        stage_dependencies(consumer, providers, destination, TOOLCHAIN)
    assert (destination / "usr/local/include/zlib.h").read_bytes() == b"conflicting header"


@pytest.fixture
def verified_linker(tmp_path):
    upstream = {
        "name": "llvm-wasi-threaded",
        "protocol": {
            "dylink_subsection_type": 128,
            "type_encoding": "uint8",
            "vendor": "shellsim.deferred-init",
            "version": 1,
            "reject_module_start": True,
        },
    }
    recipe_path = tmp_path / "llvm-recipe.json"
    recipe_path.write_text(json.dumps(upstream))
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "licenses").mkdir()
    (prefix / "bin/lld").write_bytes(b"verified test linker")
    (prefix / "licenses/LLVM-LICENSE.txt").write_bytes(b"test license")
    (prefix / "bin/wasm-ld").symlink_to("lld")
    manifest = {
        "schema_version": 1,
        "identity": {"recipe": upstream},
        "artifacts": {
            relative: hashlib.sha256((prefix / relative).read_bytes()).hexdigest()
            for relative in ("bin/lld", "licenses/LLVM-LICENSE.txt")
        },
    }
    (prefix / "manifest.json").write_text(json.dumps(manifest))
    recipe = {
        "linker": {
            "recipe": recipe_path.name,
            "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        }
    }
    return recipe, tmp_path, prefix


def test_linker_admission_rejects_changed_binary_before_invocation(verified_linker):
    recipe, directory, prefix = verified_linker
    command, _ = admit_linker(recipe, directory, prefix)
    assert command == prefix / "bin/wasm-ld"
    (prefix / "bin/lld").write_bytes(b"changed linker")
    with pytest.raises(ValueError):
        admit_linker(recipe, directory, prefix)


def test_linker_admission_rejects_redirected_command(verified_linker):
    recipe, directory, prefix = verified_linker
    command = prefix / "bin/wasm-ld"
    command.unlink()
    command.symlink_to("../../outside")
    with pytest.raises(ValueError):
        admit_linker(recipe, directory, prefix)
