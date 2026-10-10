"""Reject invalid build requests before creating an unpublished staging tree."""

import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from ports._support.native_adapters import NativeAdapter, NativeBuildContext, NativeBuildRequest, build_native


@pytest.fixture
def build_request(tmp_path):
    context = NativeBuildContext(
        source=tmp_path / "source",
        build=tmp_path / "build",
        staging_prefix=tmp_path / "stage",
        sdk=tmp_path / "sdk",
        compiler_prefix=tmp_path / "compiler",
        sysroot=tmp_path / "sysroot",
        target="wasm32-wasip1",
        compiler_flags=(),
        linker_flags=(),
        dependencies={},
        host_tools={},
        target_tools={},
        dependency_sysroot=tmp_path / "dependencies",
        abi="test-abi",
        compiler_resource_directory=tmp_path / "sdk/lib/clang/23",
        linker=tmp_path / "compiler/bin/wasm-ld",
    )
    return NativeBuildRequest(NativeAdapter.CMAKE, context)


@pytest.mark.parametrize("jobs", [0, 17])
def test_invalid_parallelism_leaves_staging_absent(build_request, jobs):
    with pytest.raises(ValueError):
        build_native(replace(build_request, jobs=jobs))
    assert not build_request.context.build.exists()
    assert not build_request.context.staging_prefix.exists()


def test_unknown_target_leaves_staging_absent(build_request):
    context = replace(build_request.context, target="wasm32-wasip1-unadmitted")
    with pytest.raises(ValueError):
        build_native(replace(build_request, context=context))
    assert not context.build.exists()


def test_missing_tool_leaves_staging_absent(build_request):
    with pytest.raises(ValueError):
        build_native(build_request)
    assert not build_request.context.staging_prefix.exists()


def test_unsupported_meson_install_target_fails_before_build(build_request):
    with pytest.raises(ValueError):
        build_native(replace(build_request, adapter=NativeAdapter.MESON, install_targets=("custom",)))
    assert not build_request.context.build.exists()


def test_unapproved_install_prefix_fails_before_build(build_request):
    with pytest.raises(ValueError):
        build_native(replace(build_request, install_prefix=PurePosixPath("/usr")))
    assert not build_request.context.staging_prefix.exists()


@pytest.mark.parametrize("bindings", [{"CC": "/ambient/compiler"}, {"PATH": "/ambient/bin"}, {"LIBS": 12}, []])
def test_configure_environment_cannot_replace_admitted_tools(build_request, bindings):
    request = replace(build_request, adapter=NativeAdapter.CONFIGURE_MAKE, configure_environment=bindings)
    with pytest.raises(ValueError, match="configure environment"):
        build_native(request)
    assert not request.context.build.exists()


@pytest.mark.parametrize(
    "argument", ["CC=/ambient/compiler", "CXX=/ambient/compiler", "PATH=/ambient/bin", "SHELL=/ambient/sh"]
)
def test_make_arguments_cannot_replace_admitted_tools(build_request, argument):
    request = replace(build_request, adapter=NativeAdapter.CONFIGURE_MAKE, build_args=(argument,))
    with pytest.raises(ValueError, match="make build argument"):
        build_native(request)
    assert not request.context.build.exists()


def test_plain_make_runs_ordered_targets_with_separate_host_compiler(build_request):
    context = build_request.context
    context.source.mkdir()
    (context.source / "host.c").write_text("int main(void) { return 0; }\n")
    (context.source / "Makefile").write_text(
        "first:\n\t$(HOSTCC) host.c -o host-generator\n\t./host-generator\n\tprintf first > order\n"
        "second:\n\ttest -f order\n\tprintf second >> order\n"
    )
    host_tools = {name: Path("/usr/bin/" + name) for name in ("make", "cc", "sh", "rm")}
    host_tools.update({"python": Path(sys.executable), "pkg-config": Path("/usr/bin/true")})
    # This Makefile only needs its native generator. Target entrypoints must
    # remain supplied and must never be substituted for HOSTCC.
    context = replace(
        context,
        host_tools=host_tools,
        target_tools={name: Path("/usr/bin/false") for name in ("cc", "cxx", "ar", "ranlib")},
    )
    result = build_native(
        replace(
            build_request,
            adapter=NativeAdapter.PLAIN_MAKE,
            context=context,
            build_targets=("first", "second"),
            install_targets=(),
        )
    )
    assert (context.source / "order").read_text() == "firstsecond"
    assert (context.source / "host-generator").is_file()
    assert len(result.commands) == 2


def test_plain_make_rejects_configure_phase_before_staging(build_request):
    with pytest.raises(ValueError, match="no configure phase"):
        build_native(replace(build_request, adapter=NativeAdapter.PLAIN_MAKE, configure_args=("--static",)))
    assert not build_request.context.staging_prefix.exists()


def _commit_fixture(git, directory):
    subprocess.run(
        [
            git,
            "-C",
            str(directory),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "--no-gpg-sign",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )


def test_extracted_source_vcs_fallback_ignores_parent_commits(build_request, monkeypatch):
    git = shutil.which("git")
    if git is None:
        pytest.skip("real Git is required for source discovery regression")
    parent = build_request.context.source.parent / "parent-repository"
    parent.mkdir()
    source, build = parent / "extracted/source", parent / "generated/build"
    source.mkdir(parents=True)
    build.mkdir(parents=True)

    subprocess.run([git, "init", str(parent)], check=True, capture_output=True)
    _commit_fixture(git, parent)
    # Establish the actual leak before applying the adapter boundary.
    assert subprocess.run([git, "describe", "--always"], cwd=source, capture_output=True).returncode == 0
    (source / "Makefile").write_text(
        "all:\n\t"
        + shlex.quote(git)
        + " describe --always > source-version 2>/dev/null || printf unknown > source-version\n"
        + "\tcd "
        + shlex.quote(str(build))
        + "; "
        + shlex.quote(git)
        + " describe --always > build-version 2>/dev/null || printf unknown > build-version\n"
    )
    host = {name: Path(shutil.which(name)) for name in ("make", "cc", "sh", "rm")}
    host.update({"python": Path(sys.executable), "pkg-config": Path(shutil.which("true"))})
    context = replace(
        build_request.context,
        source=source,
        build=build,
        host_tools=host,
        target_tools={name: Path(shutil.which("false")) for name in ("cc", "cxx", "ar", "ranlib")},
    )
    request = replace(build_request, adapter=NativeAdapter.PLAIN_MAKE, context=context, install_targets=())
    monkeypatch.setenv("GIT_DIR", str(parent / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(parent))
    build_native(request)
    assert (source / "source-version").read_text() == "unknown"
    assert (build / "build-version").read_text() == "unknown"
    _commit_fixture(git, parent)
    request = replace(request, context=replace(context, retained_workspace=True))
    build_native(request)
    assert (source / "source-version").read_text() == "unknown"
    assert (build / "build-version").read_text() == "unknown"
    # A checkout genuinely inside the admitted source boundary remains visible.
    subprocess.run([git, "init", str(source)], check=True, capture_output=True, env={"PATH": str(Path(git).parent)})
    # Explicit -C is insufficient with inherited overrides; remove the injected
    # overrides for fixture creation while the adapter continues to exclude them.
    monkeypatch.delenv("GIT_DIR")
    monkeypatch.delenv("GIT_WORK_TREE")
    _commit_fixture(git, source)
    expected = subprocess.check_output([git, "describe", "--always"], cwd=source).decode()
    build_native(request)
    assert (source / "source-version").read_text() == expected


def test_executable_archive_tail_resolves_real_symbol_only_for_links(build_request):
    from ports._support.native_adapters import compiler_wrapper_text

    context = build_request.context
    context.source.mkdir()
    provider = context.source / "provider.c"
    provider.write_text("int supplied(void) { return 37; }\n")
    consumer = context.source / "consumer.c"
    consumer.write_text("extern int supplied(void); int main(void) { return supplied() != 37; }\n")
    compiler, archiver = shutil.which("cc"), shutil.which("ar")
    assert compiler is not None and archiver is not None
    object_file, archive = context.source / "provider.o", context.source / "provider.a"
    subprocess.run([compiler, "-c", str(provider), "-o", str(object_file)], check=True)
    subprocess.run([archiver, "rcs", str(archive), str(object_file)], check=True)
    context = replace(
        context,
        host_tools={"python": Path(sys.executable)},
        target_tools={"cc": Path(compiler)},
        executable_link_inputs=(archive,),
    )
    response = Path(__file__).parents[1] / "compiler_response.py"
    wrapper = context.source / "cc"
    wrapper.write_text(compiler_wrapper_text(context, "cc", response.read_text()))
    wrapper.chmod(0o755)
    output = context.source / "consumer.o"
    subprocess.run([str(wrapper), "-Werror", "-c", str(consumer), "-o", str(output)], check=True)
    executable = context.source / "consumer"
    subprocess.run([str(wrapper), str(output), "-o", str(executable)], check=True)
    assert subprocess.run([str(executable)], check=False).returncode == 0
    # A shared link receives only its shared inputs, and must not silently borrow
    # an archive declared for executable ownership.
    result = subprocess.run(
        [str(wrapper), "-shared", "-Wl,--no-undefined", str(output), "-o", str(context.source / "consumer.so")],
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
