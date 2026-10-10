"""Check upstream install-plan paths independently of package build layouts."""

import pytest

from ports._support.python_meson import (
    extension_destination,
    shared_providers,
    stage_development_exports,
    stage_install_plan,
    wheel_destination,
)


@pytest.mark.parametrize("destination", ["{py_platlib}/../escape", "{py_purelib}//absolute", "/host/site-packages/x"])
def test_reject_install_destination_escape(destination):
    with pytest.raises(ValueError):
        wheel_destination(destination)


def test_qualified_extensions_and_upstream_exclusions(tmp_path):
    source = tmp_path / "source"
    build = tmp_path / "build"
    wheel = tmp_path / "wheel"
    package = source / "package"
    (package / "tests").mkdir(parents=True)
    (package / "tests/uninstalled.py").write_text("excluded")
    (package / "__init__.py").write_text("upstream package")
    build.mkdir()
    extension = build / "_example.cpython-313-x86_64-linux-gnu.so"
    extension.write_bytes(b"compiled extension")
    skipped = build / "test_extension.so"
    skipped.write_bytes(b"test-only")
    plan = {
        "targets": {
            str(extension): {
                "destination": "{py_platlib}/package/private/_example.cpython-313-x86_64-linux-gnu.so",
                "tag": "python-runtime",
            },
            str(skipped): {"destination": "{py_platlib}/package/test_extension.so", "tag": "tests"},
        },
        "install_subdirs": {
            str(package): {
                "destination": "{py_purelib}/package",
                "tag": "python-runtime",
                "exclude_dirs": ["tests"],
                "exclude_files": [],
            },
        },
    }
    extensions = stage_install_plan(plan, source, build, wheel, ("python-runtime",))
    assert extensions == [wheel / "package/private/_example.so"]
    assert (wheel / "package/private/_example.so").read_bytes() == b"compiled extension"
    assert (wheel / "package/__init__.py").read_text() == "upstream package"
    assert not (wheel / "package/tests").exists()
    assert not (wheel / "package/test_extension.so").exists()


def test_reject_install_source_outside_project(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("host source")
    plan = {"data": {str(outside): {"destination": "{py_purelib}/package.py", "tag": "runtime"}}}
    with pytest.raises(ValueError):
        stage_install_plan(plan, tmp_path / "source", tmp_path / "build", tmp_path / "wheel", ("runtime",))


def test_extension_destination_preserves_qualified_name():
    assert extension_destination("scipy/special/_ufuncs.cpython-313-x86_64-linux-gnu.so") == "scipy/special/_ufuncs.so"


def test_development_archives_use_native_payload(tmp_path):
    wheel = tmp_path / "wheel"
    library = wheel / "numpy/_core/lib/libnpymath.a"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"target archive")
    staging = tmp_path / "stage"
    stage_development_exports(wheel, staging, [{"source": "numpy/_core/lib", "destination": "lib"}])
    assert (staging / "usr/local/lib/libnpymath.a").read_bytes() == b"target archive"
    assert not library.exists()


def test_undeclared_archive_cannot_enter_curated_wheel(tmp_path):
    wheel = tmp_path / "wheel"
    wheel.mkdir()
    (wheel / "hidden.a").write_bytes(b"archive")
    with pytest.raises(ValueError):
        stage_development_exports(wheel, tmp_path / "stage", [])


def test_provider_diamond_uses_same_admitted_prefix_once(tmp_path):
    prefix = tmp_path / "merged"
    prefix.mkdir()
    provider = prefix / "libprovider.so"
    provider.write_bytes(b"one admitted provider")
    assert shared_providers({"first": prefix, "second": prefix / "."}) == {"libprovider.so": provider}


@pytest.mark.parametrize("contents", [b"first provider", b"different provider"])
def test_distinct_provider_identities_with_same_name_are_rejected(tmp_path, contents):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "libprovider.so").write_bytes(b"first provider")
    (second / "libprovider.so").write_bytes(contents)
    with pytest.raises(ValueError):
        shared_providers({"first": first, "second": second})
