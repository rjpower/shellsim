"""Check the native-library boundary without network access or compiler work."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[2] / "ports/native/dependencies.py"
_SPEC = importlib.util.spec_from_file_location("native_dependencies", _PATH)
DEPENDENCIES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(DEPENDENCIES)


@pytest.fixture
def provider(tmp_path):
    prefix = tmp_path / "zlib"
    (prefix / "include").mkdir(parents=True)
    (prefix / "lib").mkdir()
    (prefix / "include/zlib.h").write_text("target header")
    (prefix / "lib/libz.a").write_bytes(b"target archive")
    recipe = {
        "name": "zlib",
        "version": "1.3.1",
        "target_profile": "wasi-cpython-v1",
        "exports": {"headers": ["include/zlib.h"], "archives": ["lib/libz.a"]},
        "target_dependencies": [],
        "transitive_link_flags": [],
    }
    DEPENDENCIES.seal_artifact(prefix, {"recipe": recipe, "dependency_artifacts": {}})
    return prefix


def test_two_consumers_share_the_same_artifact(provider, tmp_path):
    requirement = [{"port": "native/zlib", "version": "1.3.1"}]
    first, links = DEPENDENCIES.dependency_prefix(
        requirement, {"native/zlib": provider}, tmp_path / "first", "wasi-cpython-v1"
    )
    second, _ = DEPENDENCIES.dependency_prefix(
        requirement, {"native/zlib": provider}, tmp_path / "second", "wasi-cpython-v1"
    )
    assert first == second
    assert links == [str(tmp_path / "first/lib/libz.a")]
    assert (tmp_path / "second/include/zlib.h").read_text() == "target header"
    assert not (tmp_path / "second/artifact.json").exists()


@pytest.mark.parametrize(
    "version,profile,providers",
    [
        ("1.3.1", "wasi-cpython-v1", False),
        ("1.2.0", "wasi-cpython-v1", True),
        ("1.3.1", "other-target", True),
    ],
)
def test_missing_and_conflicting_target_dependencies_fail_before_staging(
    provider, tmp_path, version, profile, providers
):
    destination = tmp_path / "consumer"
    with pytest.raises(ValueError):
        DEPENDENCIES.dependency_prefix(
            [{"port": "native/zlib", "version": version}],
            {"native/zlib": provider} if providers else {},
            destination,
            profile,
        )
    assert not destination.exists()


@pytest.mark.parametrize("mutation", ["bytes", "missing", "extra", "nested-manifest", "special", "symlink", "inputs"])
def test_artifact_tampering_and_changed_inputs_are_rejected(provider, mutation):
    expected = DEPENDENCIES.verify_artifact(provider)["inputs"]
    if mutation == "bytes":
        (provider / "lib/libz.a").write_bytes(b"host archive")
    elif mutation == "missing":
        (provider / "include/zlib.h").unlink()
    elif mutation == "extra":
        (provider / "include/host.h").touch()
    elif mutation == "nested-manifest":
        (provider / "include/artifact.json").write_text("undeclared")
    elif mutation == "special":
        import os

        os.mkfifo(provider / "include/fifo")
    elif mutation == "symlink":
        (provider / "include/host.h").symlink_to("zlib.h")
    else:
        expected = {**expected, "changed": True}
    with pytest.raises(ValueError):
        DEPENDENCIES.verify_artifact(provider, expected)


def test_host_include_and_library_overrides_are_not_inherited(monkeypatch):
    for name in (
        "CPATH",
        "C_INCLUDE_PATH",
        "LIBRARY_PATH",
        "LDFLAGS",
        "CFLAGS",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
    ):
        monkeypatch.setenv(name, "/host/undeclared")
    environment = DEPENDENCIES.target_environment(Path("/sdk"))
    assert "/host/undeclared" not in environment.values()
    assert environment["PKG_CONFIG_PATH"] == environment["PKG_CONFIG_LIBDIR"] == ""


def test_changed_dependency_identity_invalidates_consumer_inputs(provider, tmp_path):
    recipe = {"source": {"sha256": "source"}, "build_scripts": []}
    first = DEPENDENCIES.verify_artifact(provider)
    before = DEPENDENCIES.artifact_input(recipe, tmp_path, {"sdk": "24"}, {"native/zlib": first})
    (provider / "lib/libz.a").write_bytes(b"new target archive")
    second = DEPENDENCIES.seal_artifact(provider, first["inputs"])
    after = DEPENDENCIES.artifact_input(recipe, tmp_path, {"sdk": "24"}, {"native/zlib": second})
    assert DEPENDENCIES.digest(before) != DEPENDENCIES.digest(after)


def test_prefix_rejects_duplicate_export_owners(provider, tmp_path):
    with pytest.raises(ValueError):
        DEPENDENCIES.dependency_prefix(
            [{"port": "one", "version": "1.3.1"}, {"port": "two", "version": "1.3.1"}],
            {"one": provider, "two": provider},
            tmp_path / "consumer",
            "wasi-cpython-v1",
        )
    assert not (tmp_path / "consumer").exists()


@pytest.fixture
def transitive_provider(provider, tmp_path):
    prefix = tmp_path / "codec"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "lib/libcodec.a").write_bytes(b"codec archive")
    recipe = {
        "name": "codec",
        "version": "1",
        "target_profile": "wasi-cpython-v1",
        "exports": {"archives": ["lib/libcodec.a"]},
        "target_dependencies": [{"port": "native/zlib", "version": "1.3.1"}],
        "transitive_link_flags": ["-lm"],
    }
    dependency = DEPENDENCIES.verify_artifact(provider)["artifact_sha256"]
    DEPENDENCIES.seal_artifact(prefix, {"recipe": recipe, "dependency_artifacts": {"native/zlib": dependency}})
    return prefix


@pytest.mark.parametrize("direct_shared_dependency", [False, True])
def test_transitive_archives_follow_their_consumer(provider, transitive_provider, tmp_path, direct_shared_dependency):
    destination = tmp_path / "consumer"
    requirements = [{"port": "native/codec", "version": "1"}]
    if direct_shared_dependency:
        requirements.insert(0, {"port": "native/zlib", "version": "1.3.1"})
    selected, links = DEPENDENCIES.dependency_prefix(
        requirements,
        {"native/zlib": provider, "native/codec": transitive_provider},
        destination,
        "wasi-cpython-v1",
    )
    assert set(selected) == {"native/zlib", "native/codec"}
    assert links == [str(destination / "lib/libcodec.a"), str(destination / "lib/libz.a"), "-lm"]


def test_changed_transitive_artifact_is_rejected(provider, transitive_provider, tmp_path):
    inputs = DEPENDENCIES.verify_artifact(provider)["inputs"]
    (provider / "lib/libz.a").write_bytes(b"replacement")
    DEPENDENCIES.seal_artifact(provider, inputs)
    with pytest.raises(ValueError):
        DEPENDENCIES.dependency_prefix(
            [{"port": "native/codec", "version": "1"}],
            {"native/zlib": provider, "native/codec": transitive_provider},
            tmp_path / "consumer",
            "wasi-cpython-v1",
        )
    assert not (tmp_path / "consumer").exists()


def test_recipe_build_script_changes_are_rejected(tmp_path):
    script = tmp_path / "build.py"
    script.write_text("original")
    recipe = {"build_scripts": [{"file": "build.py", "sha256": DEPENDENCIES.file_hash(script)}]}
    DEPENDENCIES.recipe_identity(recipe, tmp_path)
    script.write_text("changed")
    with pytest.raises(ValueError):
        DEPENDENCIES.recipe_identity(recipe, tmp_path)


def test_pillow_source_lists_include_upstream_internal_library(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    (tmp_path / "setup.py").write_text(
        "raise AssertionError('setup.py must not execute')\n"
        "_IMAGING = ('encode',)\n_LIB_IMAGING = ('ZipEncode',)\n"
        "libraries: list = [('pil_imaging_mode', {'sources': ['src/libImaging/Mode.c']})]\n"
    )
    path = Path(__file__).parents[2] / "ports/pillow/build.py"
    spec = importlib.util.spec_from_file_location("pillow_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert [p.relative_to(tmp_path).as_posix() for p in module.pillow_sources(tmp_path)] == [
        "src/_imaging.c",
        "src/encode.c",
        "src/libImaging/ZipEncode.c",
        "src/libImaging/Mode.c",
    ]
