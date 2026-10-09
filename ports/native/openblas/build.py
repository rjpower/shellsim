"""Build scalar OpenBLAS and its translated LAPACK for the WASI static ABI."""

import re
import shutil
import subprocess
from pathlib import Path

from ports.native.dependencies import (
    artifact_input,
    digest,
    file_hash,
    seal_artifact,
    target_environment,
    target_profile,
    verify_artifact,
)


def convert_interface_returns(text):
    """Give translated subroutines a defined zero return without changing complex ABI."""
    pattern = re.compile(
        r"\bvoid\s+(?:C?NAME|(?:LAED4|LACPY|LASET)|(?!(?:[cz](?:dotc|dotu|ladiv))_)[a-z0-9_]+_)\s*\([^;{]*\)\s*\{"
    )
    spans = []
    for match in pattern.finditer(text):
        if spans and match.start() < spans[-1][1]:
            continue
        start = match.end()
        # Upstream interfaces share bodies across #if CBLAS signatures. Their
        # function closing braces start in column zero; nested blocks do not.
        closing = re.search(r"^}", text[start:], re.M)
        if closing is not None:
            spans.append((start, start + closing.start()))
            continue
        depth = 1
        token = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
        for brace in token.finditer(text, start):
            if brace.group() == "{":
                depth += 1
            elif brace.group() == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, brace.start()))
                    break
    for start, end in reversed(spans):
        body = re.sub(r"\breturn\s*;", "return 0;", text[start:end])
        text = text[:start] + body + "\nreturn 0;\n" + text[end:]
    return text


def normalize_xerbla(text):
    """Supply Fortran character lengths omitted by newer C-translated LAPACK files."""
    text = re.sub(r"xerbla_\(\s*char\s*\*,\s*integer\s*\*\s*\)", "xerbla_(char *, integer *, ftnlen)", text)
    return re.sub(
        r'xerbla_\(\s*"([^"\n]+)",\s*(&[a-z0-9_]+)\)',
        lambda match: f'xerbla_("{match[1]}", {match[2]}, (ftnlen){len(match[1])})',
        text,
    )


def prepare_source(source, build):
    """Select generic C kernels while retaining wasm32 sizes and a coherent f2c ABI."""
    shutil.copytree(source, build)
    check = build / "ctest.c"
    check.write_text(check.read_text() + "\n#if defined(__wasi__)\nOS_EMBEDDED\nARCH_WASM32\n#endif\n")
    check = build / "c_check"
    text = check.read_text().replace(
        'case "$data" in *OS_LINUX*)', 'case "$data" in *OS_EMBEDDED*) os=EMBEDDED ;; esac\ncase "$data" in *OS_LINUX*)'
    )
    text = text.replace("    *ARCH_X86_64*)", "    *ARCH_WASM32*) architecture=wasm32 ;;\n    *ARCH_X86_64*)")
    text = text.replace(
        "    arm|arm64) defined=1 ;;", "    wasm32) defined=1; BINARY=32 ;;\n    arm|arm64) defined=1 ;;"
    )
    check.write_text(text)
    (build / "Makefile.wasm32").write_text("# Scalar generic kernels need no architecture flags.\n")
    (build / "kernel/wasm32").mkdir()
    for name in ("KERNEL", "KERNEL.RISCV64_GENERIC"):
        shutil.copyfile(build / "kernel/riscv64" / name, build / "kernel/wasm32" / name)
    shutil.copytree(build / "lapack/laswp/riscv64", build / "lapack/laswp/wasm32")
    prebuild = build / "Makefile.prebuild"
    prebuild.write_text(
        prebuild.read_text().replace("ifeq ($(TARGET), RISCV64_GENERIC)", "ifeq ($(TARGET), DISABLED_RISCV64_GENERIC)")
    )
    common = build / "common.h"
    common.write_text(
        common.read_text().replace(
            '#ifdef ARCH_RISCV64\n#include "common_riscv64.h"',
            '#ifdef ARCH_WASM32\n#include "common_wasm32.h"\n#endif\n#ifdef ARCH_RISCV64\n#include "common_riscv64.h"',
        )
    )
    (build / "common_wasm32.h").write_text(
        "#define MB\n#define WMB\n#define RMB\n#define YIELDING\n"
        "#define RETURN_BY_STACK\n#define BUFFER_SIZE (32 << 20)\n#define SEEK_ADDRESS\n"
        "static inline int blas_quickdivide(blasint x, blasint y) { return x / y; }\n"
    )
    memory = build / "driver/others/memory.c"
    memory.write_text(
        re.sub(
            r"^inline (?:int|char \*)(?:\s*)?(?:puts|printf|getenv|atoi)\([^\n]*\) \{[^\n]*\}\n",
            "",
            memory.read_text(),
            flags=re.M,
        )
    )
    # f2c subroutines return int. Wasm requires declarations and definitions to agree.
    paths = [build / "common_interface.h", build / "cblas.h", *sorted((build / "interface").glob("*.c"))]
    for path in paths:
        text = path.read_text()
        if path.name != "zdot.c":
            text = convert_interface_returns(text)
        text = re.sub(r"void(\s+)BLASFUNC", r"int\1BLASFUNC", text)
        text = re.sub(r"void(\s+)cblas_", r"int\1cblas_", text)
        if path.name != "zdot.c":
            text = re.sub(r"void(\s+)(C?NAME)\b", r"int\1\2", text)
        text = re.sub(r"int(\s+)BLASFUNC\(([czx]dot[cu])\)", r"void\1BLASFUNC(\2)", text)
        text = re.sub(r"int(\s+)(cblas_[cz]dot[cu]_sub)", r"void\1\2", text)
        path.write_text(text)
    for path in sorted((build / "lapack-netlib").rglob("*.c")):
        text = re.sub(r"\bvoid ([a-z0-9_]+_)", r"int \1", convert_interface_returns(normalize_xerbla(path.read_text())))
        text = re.sub(r"\bint ([cz](?:dotc|dotu|ladiv))", r"void \1", text)
        path.write_text(text)
    for path in sorted((build / "lapack/laed3").glob("*.c")):
        path.write_text(
            re.sub(r"^void (LAED4|LACPY|LASET)\(", r"int \1(", convert_interface_returns(path.read_text()), flags=re.M)
        )


def build_openblas(recipe, source, sdk, work, toolchain, run):
    """Seal the full scalar LP64 archive without host library discovery."""
    env = target_environment(sdk)
    env["PATH"] = "/usr/bin:/bin"
    host_tools = {}
    for name in ("cc", "make"):
        executable = Path(shutil.which(name, path=env["PATH"])).resolve()
        host_tools[name] = {
            "path": str(executable),
            "sha256": file_hash(executable),
            "version": subprocess.check_output([str(executable), "--version"], env=env, text=True).splitlines()[0],
        }
    inputs = artifact_input(recipe, Path(__file__).parent, toolchain, {})
    inputs["host_tools"] = host_tools
    inputs["source_tree_sha256"] = digest(
        {p.relative_to(source).as_posix(): file_hash(p) for p in sorted(source.rglob("*")) if p.is_file()}
    )
    prefix = work / "native-artifacts" / digest(inputs)
    if prefix.exists():
        return prefix, verify_artifact(prefix, inputs)
    build = work / ("openblas-build-" + digest(inputs)[:12])
    if build.exists():
        shutil.rmtree(build)
    prepare_source(source, build)
    flags = " ".join(target_profile(recipe)["compiler_flags"])
    arguments = [
        host_tools["make"]["path"],
        "-j4",
        "NOFORTRAN=1",
        "NO_LAPACKE=1",
        "USE_THREAD=0",
        "NUM_THREADS=1",
        "TARGET=RISCV64_GENERIC",
        "BINARY=32",
        "ARCH=wasm32",
        "OSNAME=EMBEDDED",
        "CC=" + str(sdk / "bin/clang"),
        "HOSTCC=" + host_tools["cc"]["path"],
        "AR=" + str(sdk / "bin/llvm-ar"),
        "RANLIB=" + str(sdk / "bin/llvm-ranlib"),
        "CFLAGS=" + flags,
    ]
    # Both targets append to the same archive; complete BLAS before netlib.
    for target in ("libs", "netlib"):
        run([*arguments, target], build, env, work / ("openblas-" + target + ".log"))
    temporary = prefix.with_name(prefix.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "lib/pkgconfig").mkdir(parents=True)
    (temporary / "include").mkdir()
    (temporary / "licenses").mkdir()
    shutil.copyfile(build / "libopenblas.a", temporary / "lib/libopenblas.a")
    (temporary / "include/cblas.h").write_text(
        (build / "cblas.h").read_text().replace('"common.h"', '"openblas_config.h"')
    )
    # The public header includes config.h under its installed name.
    config = "\n".join(
        " ".join([parts[0], "OPENBLAS_" + parts[1], *parts[2:]])
        for line in (build / "config.h").read_text().splitlines()
        if (parts := line.split())
    )
    (temporary / "include/openblas_config.h").write_text(
        "#ifndef OPENBLAS_CONFIG_H\n#define OPENBLAS_CONFIG_H\n"
        + config
        + "\n"
        + (build / "openblas_config_template.h").read_text()
        + "\n#endif\n"
    )
    shutil.copyfile(build / "LICENSE", temporary / "licenses/openblas.txt")
    (temporary / "lib/pkgconfig/openblas.pc").write_text(
        "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\nName: OpenBLAS\nDescription: Scalar BLAS and translated LAPACK\nVersion: 0.3.31\nLibs: -L${libdir} -lopenblas\nLibs.private: -lm\nCflags: -I${includedir}\n"
    )
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)
