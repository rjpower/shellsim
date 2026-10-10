"""Reject invalid build requests before creating an unpublished staging tree."""

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import uuid
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


def test_admitted_cache_launcher_handles_only_object_compilation(build_request, monkeypatch):
    from ports._support.native_adapters import CompilerCacheLauncher, build_environment, compiler_wrapper_text
    from ports._support.store import file_hash

    context = build_request.context
    context.source.mkdir()
    compiler = shutil.which("cc")
    assert compiler is not None
    log = context.source / "cache-calls.jsonl"
    launcher = context.source / "sccache"
    launcher.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        f"with open({str(log)!r}, 'a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "os.execv(sys.argv[1], sys.argv[1:])\n"
    )
    launcher.chmod(0o755)
    context = replace(
        context,
        host_tools={"python": Path(sys.executable), "pkg-config": Path("/admitted/pkg-config"), "cc": Path(compiler)},
        target_tools={
            "cc": Path(compiler),
            "cxx": Path(compiler),
            "ar": Path("/admitted/ar"),
            "ranlib": Path("/admitted/ranlib"),
        },
        compiler_cache=CompilerCacheLauncher(
            launcher, file_hash(launcher), (("SCCACHE_DIR", str(context.source / "cache")),)
        ),
    )
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret")
    environment = build_environment(context, {})
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert environment["SCCACHE_CLIENT_SIDE"] == "1"
    assert environment["SCCACHE_BASEDIRS"] == ":".join(
        map(str, (context.source, context.build, context.dependency_sysroot))
    )
    wrapper = context.source / "cc"
    wrapper.write_text(
        compiler_wrapper_text(context, "cc", (Path(__file__).parents[1] / "compiler_response.py").read_text())
    )
    wrapper.chmod(0o755)
    source = context.source / "main.c"
    source.write_text("int main(void) { return 0; }\n")
    response = context.source / "compile.rsp"
    response.write_text("-c " + str(source) + " -o " + str(context.source / "main.o"))
    subprocess.run([str(wrapper), "@" + str(response)], env=environment, check=True)
    subprocess.run(
        [str(wrapper), str(context.source / "main.o"), "-o", str(context.source / "main")], env=environment, check=True
    )
    subprocess.run([str(wrapper), "-E", str(source)], env=environment, stdout=subprocess.DEVNULL, check=True)
    import json

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0][0] == compiler
    assert calls[0][1:] == [
        "-c",
        os.path.relpath(source),
        "-o",
        os.path.relpath(context.source / "main.o"),
    ]


@pytest.mark.parametrize("debug_flags,stable_debug", [([], False), (["-g"], True), (["-g", "-g0"], False)])
def test_cache_wrapper_normalizes_owned_paths_and_preserves_link_responses(build_request, debug_flags, stable_debug):
    from ports._support.native_adapters import CompilerCacheLauncher, compiler_wrapper_text
    from ports._support.store import file_hash

    context = build_request.context
    context.source.mkdir()
    context.build.mkdir()
    context.compiler_prefix.mkdir()
    log = context.build / "arguments.json"
    compiler = context.compiler_prefix / "clang-driver"
    compiler.write_text(f"#!{sys.executable}\nimport json, sys\nopen({str(log)!r}, 'w').write(json.dumps(sys.argv))\n")
    compiler.chmod(0o755)
    driver = context.compiler_prefix / "clang"
    driver.symlink_to(compiler.name)
    launcher = context.compiler_prefix / "sccache"
    launcher.write_text(f"#!{sys.executable}\nimport os, sys\nos.execv(sys.argv[1], sys.argv[1:])\n")
    launcher.chmod(0o755)
    external = context.source.parent / "source-other/header.h"
    context = replace(
        context,
        host_tools={"python": Path(sys.executable)},
        target_tools={"cc": driver},
        compiler_flags=("--sysroot=" + str(context.sysroot), "-I" + str(context.dependency_sysroot / "include")),
        compiler_cache=CompilerCacheLauncher(launcher, file_hash(launcher)),
    )
    wrapper = context.build / "cc"
    wrapper.write_text(
        compiler_wrapper_text(context, "cc", (Path(__file__).parents[1] / "compiler_response.py").read_text())
    )
    wrapper.chmod(0o755)
    source = context.source / "space name.c"
    nested = context.build / "nested.rsp"
    nested.write_text(shlex.join(["-c", str(source), "-I" + str(context.source / "include"), *debug_flags]))
    arguments = [
        "@nested.rsp",
        "-isystem",
        str(context.dependency_sysroot / "include"),
        "-iquote" + str(context.source / "quotes"),
        "-include",
        str(external),
        "-MF" + str(context.build / "dep.d"),
        "-o",
        str(context.build / "obj.o"),
        "-DORIGINAL=" + str(context.source),
    ]
    outer = context.build / "outer.rsp"
    outer.write_text(shlex.join(arguments))
    subprocess.run([str(wrapper), "@outer.rsp"], cwd=context.build, check=True)
    observed = json.loads(log.read_text())
    assert observed == [
        str(driver),
        "--sysroot=" + str(context.sysroot),
        "-I../dependencies/include",
        "-c",
        "../source/space name.c",
        "-I../source/include",
        *debug_flags,
        "-isystem",
        "../dependencies/include",
        "-iquote../source/quotes",
        "-include",
        str(external),
        "-MFdep.d",
        "-o",
        "obj.o",
        "-DORIGINAL=" + str(context.source),
        *(["-fdebug-compilation-dir=."] if stable_debug else []),
    ]
    link = context.build / "link.rsp"
    link.write_text(shlex.join([str(context.build / "obj.o"), "-o", str(context.build / "program")]))
    subprocess.run([str(wrapper), "@link.rsp"], cwd=context.build, check=True)
    assert json.loads(log.read_text()) == [str(driver), *context.compiler_flags, "@link.rsp"]


@pytest.mark.parametrize("target,debug", [("host", False), ("wasm", False), ("wasm", True)])
def test_real_cache_wrapper_reuses_objects_across_private_roots(build_request, tmp_path, target, debug):
    """Opt-in pinned-binary acceptance exercises generated wrappers, not a cache model."""
    from ports._support.native_adapters import CompilerCacheLauncher, build_environment, compiler_wrapper_text
    from ports._support.store import file_hash

    binary = os.environ.get("SHELLSIM_TEST_SCCACHE")
    compiler = os.environ.get("SHELLSIM_TEST_WASM_CLANG") if target == "wasm" else shutil.which("cc")
    if not binary or not compiler:
        pytest.skip("provide pinned sccache and an existing Wasm Clang for real cache acceptance")
    sccache = Path(binary).absolute()
    compiler = Path(compiler).absolute()
    pinned = "973cb15f6a986d84ca334bbed3bbe2eb8f1ee8fd81bf9e115b8539a293bf8d59"
    assert file_hash(sccache) == pinned
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = str(reservation.getsockname()[1])
    client = {"PATH": os.defpath, "HOME": str(tmp_path), "LC_ALL": "C", "SCCACHE_SERVER_PORT": port}
    daemon = {**client, "SCCACHE_DIR": str(tmp_path / "cache")}

    def run(argv, directory, environment=client):
        return subprocess.run(argv, cwd=directory, env=environment, capture_output=True, timeout=30, check=True)

    def stats():
        report = json.loads(run([str(sccache), "--show-stats", "--stats-format=json"], tmp_path).stdout)
        assert report["basedirs"] == []
        values = report["stats"]
        return [sum(values[name]["counts"].values()) for name in ("cache_hits", "cache_misses")] + [
            values["cache_writes"]
        ]

    run([str(sccache), "--start-server"], tmp_path, daemon)
    try:
        nonce = uuid.uuid4().hex
        source_text = (
            '#include "value.h"\nconst char probe_file[] = __FILE__;\n'
            f'const char nonce[] = "{nonce}";\nint probe(void) {{ return VALUE; }}\n'
            + ("int main(void) { return probe() != 42; }\n" if target == "host" else "")
        )
        before = stats()
        objects = []
        for index in range(2):
            workspace = tmp_path / (f"private-{index}-" + nonce)
            source, build, dependencies = (workspace / name for name in ("source", "build", "dependencies"))
            source.mkdir(parents=True)
            build.mkdir()
            (dependencies / "include").mkdir(parents=True)
            (source / "probe.c").write_text(source_text)
            (dependencies / "include/value.h").write_text("#define VALUE 42\n")
            flags = ("-O2", *(("--target=wasm32-wasip1",) if target == "wasm" else ()), *(("-g",) if debug else ()))
            context = replace(
                build_request.context,
                source=source,
                build=build,
                dependency_sysroot=dependencies,
                host_tools={"python": Path(sys.executable), "pkg-config": Path("/admitted/pkg-config"), "cc": compiler},
                target_tools={"cc": compiler, "ar": Path("/admitted/ar"), "ranlib": Path("/admitted/ranlib")},
                compiler_flags=flags,
                compiler_cache=CompilerCacheLauncher(sccache, pinned, (("SCCACHE_SERVER_PORT", port),)),
            )
            wrapper = build / "cc"
            wrapper.write_text(
                compiler_wrapper_text(context, "cc", (Path(__file__).parents[1] / "compiler_response.py").read_text())
            )
            wrapper.chmod(0o755)
            response = build / "compile.rsp"
            response.write_text(
                shlex.join(
                    ["-I" + str(dependencies / "include"), "-c", str(source / "probe.c"), "-o", str(build / "probe.o")]
                )
            )
            environment = build_environment(context, {})
            assert environment["SCCACHE_CLIENT_SIDE"] == "1"
            assert "SCCACHE_ERROR_LOG" not in environment and "SCCACHE_DIST_SCHEDULER_URL" not in environment
            run([str(wrapper), "@compile.rsp"], build, environment)
            after = stats()
            assert [value - previous for value, previous in zip(after, before)] == (
                [0, 1, 1] if index == 0 else [1, 0, 0]
            )
            objects.append((build / "probe.o").read_bytes())
            assert str(workspace).encode() not in objects[-1]
            assert b"../source/probe.c" in objects[-1]
            direct = [
                str(compiler),
                *flags,
                *(("-fdebug-compilation-dir=.",) if debug else ()),
                "-I../dependencies/include",
                "-c",
                "../source/probe.c",
                "-o",
                "direct.o",
            ]
            run(direct, build)
            assert (build / "direct.o").read_bytes() == objects[-1]
            link = [
                str(wrapper),
                *(("-nostdlib", "-Wl,--no-entry,--export=probe") if target == "wasm" else ()),
                str(build / "probe.o"),
                "-o",
                str(build / "probe"),
            ]
            run(link, build, environment)
            if target == "host":
                run([str(build / "probe")], build)
            assert stats() == after
            before = after
        assert objects[0] == objects[1]
    finally:
        run([str(sccache), "--stop-server"], tmp_path)


@pytest.mark.parametrize(
    "binding",
    [
        ("AWS_SECRET_ACCESS_KEY", "secret"),
        ("CC", "ambient"),
        ("SCCACHE_DIR", "bad\0path"),
        ("SCCACHE_CLIENT_SIDE", "0"),
        ("SCCACHE_ERROR_LOG", "/tmp/log"),
        ("SCCACHE_DIST_SCHEDULER_URL", "https://scheduler.invalid"),
    ],
)
def test_cache_launcher_rejects_unadmitted_environment(tmp_path, binding):
    from ports._support.native_adapters import CompilerCacheLauncher, compiler_cache_identity
    from ports._support.store import file_hash

    launcher = tmp_path / "sccache"
    launcher.write_text("executable fixture")
    launcher.chmod(0o755)
    with pytest.raises(ValueError):
        compiler_cache_identity(CompilerCacheLauncher(launcher, file_hash(launcher), (binding,)))


def test_worker_image_cache_identity_does_not_require_caller_executable(tmp_path):
    from ports._support.native_adapters import CompilerCacheLauncher, compiler_cache_identity

    launcher = CompilerCacheLauncher(tmp_path / "worker-image-only-sccache", "a" * 64)
    assert compiler_cache_identity(launcher, verify_executable=False)["sha256"] == "a" * 64
    with pytest.raises(FileNotFoundError):
        compiler_cache_identity(launcher)


@pytest.mark.parametrize(
    "environment",
    [
        (("SCCACHE_SERVER_UDS", "/app/.buildomatic/sccache.sock"),),
        (("SCCACHE_SERVER_PORT", "4226"),),
        (("SCCACHE_DIR", "/local/cache"),),
    ],
)
def test_cache_launcher_accepts_one_endpoint_or_default_local_cache(tmp_path, environment):
    from ports._support.native_adapters import CompilerCacheLauncher, compiler_cache_identity

    launcher = CompilerCacheLauncher(tmp_path / "worker-sccache", "a" * 64, environment)
    identity = compiler_cache_identity(launcher, verify_executable=False)
    assert identity["environment"] == {**dict(environment), "SCCACHE_CLIENT_SIDE": "1"}


@pytest.mark.parametrize(
    "environment",
    [
        (("SCCACHE_SERVER_PORT", "4226"), ("SCCACHE_SERVER_UDS", "/app/.buildomatic/sccache.sock")),
        (("SCCACHE_SERVER_UDS", "relative.sock"),),
        (("SCCACHE_SERVER_UDS", ""),),
    ],
)
def test_cache_launcher_rejects_conflicting_or_relative_socket_endpoints(tmp_path, environment):
    from ports._support.native_adapters import CompilerCacheLauncher, compiler_cache_identity

    launcher = CompilerCacheLauncher(tmp_path / "worker-sccache", "a" * 64, environment)
    with pytest.raises(ValueError):
        compiler_cache_identity(launcher, verify_executable=False)


def test_cache_launcher_rejects_changed_binary_and_endpoint_credentials(tmp_path):
    from ports._support.native_adapters import CompilerCacheLauncher, compiler_cache_identity
    from ports._support.store import file_hash

    path = tmp_path / "sccache"
    path.write_bytes(b"admitted")
    path.chmod(0o755)
    launcher = CompilerCacheLauncher(path, file_hash(path))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        compiler_cache_identity(launcher)
    credential = replace(launcher, environment=(("SCCACHE_ENDPOINT", "https://user:secret@cache.invalid"),))
    with pytest.raises(ValueError):
        compiler_cache_identity(credential, verify_executable=False)
