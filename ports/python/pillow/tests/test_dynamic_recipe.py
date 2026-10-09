"""Reject mismatched shared providers and unsafe source inputs before compiling."""

import hashlib
import io
import json
import tarfile

import pytest

from ports._support.build import check_build_scripts
from ports.native import imaging_shared
from ports.python.pillow.dynamic import PORT, unpack


def test_dynamic_recipe_pins_builders_and_provider_build_scripts():
    recipe = json.loads((PORT / "dynamic-recipe.json").read_text())
    check_build_scripts(recipe, PORT)
    for name in ("zlib", "libjpeg-turbo", "freetype"):
        directory = PORT.parents[1] / "native" / name
        provider = json.loads((directory / "shared-recipe.json").read_text())
        check_build_scripts(provider, directory)


@pytest.mark.parametrize("member", ["/absolute", "source/../../escape"])
def test_source_archive_rejects_unsafe_paths_before_extraction(tmp_path, member):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as output:
        info = tarfile.TarInfo(member)
        info.size = 1
        output.addfile(info, io.BytesIO(b"x"))
    spec = {"url": "https://example.invalid/source.tar", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    with pytest.raises(ValueError):
        unpack(spec, tmp_path, tmp_path / "source")
    assert not (tmp_path / "source").exists()


@pytest.fixture
def dependency_graph(tmp_path, monkeypatch):
    provider = {
        "name": "zlib",
        "version": "1.3.1",
        "target_profile": "wasi-cpython-v2",
        "target": "wasm32-wasip1",
        "linkage": "shared",
        "abi": imaging_shared.ABI,
        "soname": "libz.so",
        "target_dependencies": [],
        "exports": {"headers": ["include/zlib.h"]},
    }
    manifest = {
        "inputs": {"recipe": provider, "toolchain": {"sdk": "pinned"}, "dependency_artifacts": {}},
        "artifact_sha256": "identity",
    }
    prefix = tmp_path / "zlib"
    (prefix / "include").mkdir(parents=True)
    (prefix / "include/zlib.h").write_bytes(b"real header fixture")
    monkeypatch.setattr(imaging_shared, "verify_artifact", lambda path: manifest)
    monkeypatch.setattr(imaging_shared, "needed_libraries", lambda path: [])
    consumer = {
        "target_profile": provider["target_profile"],
        "target": provider["target"],
        "abi": provider["abi"],
        "target_dependencies": [{"port": "native/zlib", "version": "1.3.1"}],
    }
    return consumer, {"native/zlib": prefix}, manifest


@pytest.mark.parametrize(
    "field,value",
    [
        ("linkage", "static"),
        ("abi", "wrong"),
        ("name", "wrong"),
        ("version", "1.0"),
        ("target", "wasm64"),
        ("target_profile", "wrong"),
    ],
)
def test_shared_dependency_rejects_wrong_identity_before_copy(tmp_path, dependency_graph, field, value):
    consumer, providers, manifest = dependency_graph
    manifest["inputs"]["recipe"][field] = value
    with pytest.raises(ValueError):
        imaging_shared.shared_dependencies(consumer, providers, tmp_path / "selected")
    assert not (tmp_path / "selected").exists()


def test_shared_dependency_rejects_missing_declared_child(tmp_path, dependency_graph):
    consumer, providers, manifest = dependency_graph
    manifest["inputs"]["recipe"]["target_dependencies"] = [{"port": "native/missing", "version": "1.0"}]
    with pytest.raises(ValueError):
        imaging_shared.shared_dependencies(consumer, providers, tmp_path / "selected")


def test_shared_dependency_rejects_conflicting_header(tmp_path, dependency_graph):
    consumer, providers, _ = dependency_graph
    destination = tmp_path / "selected"
    (destination / "include").mkdir(parents=True)
    (destination / "include/zlib.h").write_bytes(b"conflicting header")
    with pytest.raises(ValueError):
        imaging_shared.shared_dependencies(consumer, providers, destination)
    assert (destination / "include/zlib.h").read_bytes() == b"conflicting header"


@pytest.fixture
def verified_linker(tmp_path):
    upstream = {
        "protocol": {
            "dylink_subsection_type": 128,
            "type_encoding": "uint8",
            "vendor": "shellsim.deferred-init",
            "version": 1,
            "reject_module_start": True,
        }
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
    command, _ = imaging_shared.admit_linker(recipe, directory, prefix)
    assert command == prefix / "bin/wasm-ld"
    (prefix / "bin/lld").write_bytes(b"changed linker")
    with pytest.raises(ValueError):
        imaging_shared.admit_linker(recipe, directory, prefix)


def test_linker_admission_rejects_redirected_command(verified_linker):
    recipe, directory, prefix = verified_linker
    command = prefix / "bin/wasm-ld"
    command.unlink()
    command.symlink_to("../../outside")
    with pytest.raises(ValueError):
        imaging_shared.admit_linker(recipe, directory, prefix)
