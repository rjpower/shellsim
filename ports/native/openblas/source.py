"""Normalize scalar OpenBLAS and translated LAPACK interfaces for wasm32."""

import re
import shutil


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
