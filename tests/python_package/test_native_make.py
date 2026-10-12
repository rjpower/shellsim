"""Opt-in public Clang and make regressions exercise graph and virtual libc behavior."""

import os
from pathlib import Path

import pytest
import shellsim


def test_catalog_make_builds_two_c_units_and_reloads_generated_rules():
    catalog = os.environ.get("SHELLSIM_GUEST_CATALOG")
    if not catalog:
        pytest.skip("set SHELLSIM_GUEST_CATALOG to an exported Clang and make catalog")
    env = shellsim.Context(
        pypi_catalog=catalog,
        trusted_precompiled=os.environ.get("SHELLSIM_GUEST_TRUSTED_PRECOMPILED") == "1",
    ).Environment(tools=["clang", "make"], cpu=500_000_000_000, memory=8 * 1024**3, disk=768 * 1024**2)
    env.write_file("/work/main.c", '#include <stdio.h>\nint answer(void);\nint main(void) { printf("guest C %d\\n", answer()); return 0; }\n')
    env.write_file("/work/answer.c", "int answer(void) { return 42; }\n")
    env.write_file("/work/Makefile", "all: answer.wasm\nanswer.wasm: main.o answer.o\n\tcc $^ -o $@\n\tchmod +x $@\n%.o: %.c\n\tcc -c $< -o $@\n")
    build = env.run("cd /work && make -j2 && make -q && ./answer.wasm")
    assert build.returncode == 0, build.stderr
    assert build.stdout.endswith(b"guest C 42\n")
    env.write_file(
        "/work/reload.mk",
        b'include generated.mk\nall:\n\tprintf "reload-$(RELOADED)-$(MAKE_RESTARTS)\\n"\ngenerated.mk:\n\tprintf "RELOADED = yes\\n" > generated.mk\n',
    )
    reload = env.run("cd /work && make -f reload.mk")
    assert reload.returncode == 0, reload.stderr
    assert reload.stdout.endswith(b"reload-yes-1\n")
    env.write_file("/work/parallel.mk", b'all: first second\nfirst second:\n\tsleep 1; printf "$@\\n"\n')
    parallel = env.run("cd /work; date +%s; make -j2 -f parallel.mk; date +%s")
    assert parallel.returncode == 0, parallel.stderr
    lines = parallel.stdout.splitlines()
    assert int(lines[-1]) - int(lines[0]) == 1
    assert b"first" in lines and b"second" in lines


def test_virtual_libc_accounts_and_temporary_files(tmp_path):
    import subprocess

    sdk = os.environ.get("SHELLSIM_NATIVE_SDK")
    build = os.environ.get("SHELLSIM_MAKE_BUILD")
    if not sdk or not build:
        pytest.skip("set SDK and sealed make build paths for target libc acceptance")
    root = Path(__file__).resolve().parents[2]
    fixture = root / "tests/fixtures/native_make/libc-proof.c"
    binary = tmp_path / "libc-proof.wasm"
    # Host compiler is a build tool only; this target fixture runs solely in VFS.
    subprocess.run(
        [
            str(Path(sdk) / "bin/clang"),
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-deprecated-declarations",
            "-I" + str(root / "ports/toolchain/wasi_accounts"),
            "-I" + str(root / "ports/toolchain/wasi_process"),
            str(fixture),
            *[str(Path(build) / (name + ".o")) for name in ("accounts", "tempfile", "process", "posix")],
            "-Wl,--wrap=signal,--wrap=open,--wrap=openat",
            "-lwasi-emulated-signal",
            "-o",
            str(binary),
        ],
        check=True,
    )

    def guest(passwd):
        env = shellsim.Context().Environment()
        env.run("mkdir -p /etc /tmp /home/alice").check_returncode()
        env.write_file("/etc/passwd", passwd)
        env.write_file("/proof", binary.read_bytes(), mode=0o755)
        return env

    valid = b"alice:x:123:456:Alice:/home/alice:/bin/sh\n"
    env = guest(valid)
    result = env.run("/proof")
    assert result.returncode == 0, result.stderr
    candidate = result.stdout.splitlines()[0].decode()
    assert env.run("stat -c %a " + candidate).stdout == b"600\n"
    # The same seeded guest's first random suffix collides with existing data.
    # Exclusive creation must retry without overwriting that file.
    collision = guest(valid)
    collision.write_file(candidate, b"preserved")
    retried = collision.run("/proof")
    assert retried.returncode == 0, retried.stderr
    assert retried.stdout.splitlines()[0].decode() != candidate
    assert collision.read_file(candidate) == b"preserved"
    for passwd, mode in [
        (b"bob:x:1:1:B:/home/bob:/bin/sh\n", "absent"),
        (b"alice:x:bad:1:A:/home/a:/bin/sh\n", "bad"),
        (b"alice\x00:x:1:1:A:/home/a:/bin/sh\n", "bad"),
        (b"x" * 4096, "long"),
    ]:
        rejected = guest(passwd).run("/proof " + mode)
        assert rejected.returncode == 0, rejected.stderr
