"""Select one process-owned runtime for native mains and installed guest Clang.

The GNU archive index supplies the public libc roots from the admitted archive.
Retaining libc by roots lets explicit providers and the long-double implementation
take precedence before the driver extracts ordinary libc members at its default
late archive position.
"""

import struct
from pathlib import Path

PROFILE_NAME = "shellsim-executable-runtime-v1.json"
SYSROOT_TOKEN = "@SYSROOT@"
ARCHIVES = (
    "eh/libc++.a",
    "eh/libc++abi.a",
    "eh/libunwind.a",
    "libsetjmp.a",
    "libc-printscan-long-double.a",
)


def public_archive_symbols(archive: Path) -> tuple[str, ...]:
    """Read a bounded LLVM/GNU archive index without invoking host utilities."""
    with archive.open("rb") as stream:
        header = stream.read(68)
        if header[:8] != b"!<arch>\n" or header[8:24].strip() != b"/":
            raise ValueError("runtime libc requires a GNU archive symbol index")
        if header[66:68] != b"`\n":
            raise ValueError("invalid GNU archive symbol index header")
        size = int(header[56:66].strip())
        if not 4 <= size <= 1024 * 1024:
            raise ValueError("runtime libc archive index exceeds its bound")
        index = stream.read(size)
    if len(index) != size:
        raise ValueError("truncated runtime libc archive index")
    count = struct.unpack_from(">I", index)[0]
    if count > 16384 or 4 + count * 4 > size:
        raise ValueError("invalid runtime libc archive symbol count")
    names = index[4 + count * 4 :].split(b"\0", count)
    if len(names) != count + 1 or names[-1] not in (b"", b"\0"):
        raise ValueError("invalid runtime libc archive symbol names")
    symbols = set()
    for name in names[:-1]:
        if not name or len(name) > 256:
            raise ValueError("invalid runtime libc archive symbol")
        text = name.decode("ascii")
        if not all(c.isalnum() or c in "_.$" for c in text):
            raise ValueError("unsupported runtime libc archive symbol")
        symbols.add(text)
    return tuple(sorted(symbols))


def executable_runtime_arguments(sysroot: Path, target: str) -> tuple[str, ...]:
    """Return linker arguments before user inputs, with exact admitted archives."""
    library = sysroot / "lib" / target
    archives = [library / name for name in ARCHIVES]
    if any(not archive.is_file() for archive in archives):
        raise ValueError("canonical executable runtime archive is missing")
    libc = library / "libc.a"
    return (
        "--whole-archive",
        *(str(archive) for archive in archives),
        "--no-whole-archive",
        *("--undefined=" + symbol for symbol in public_archive_symbols(libc)),
    )


COMMON_COMPILER_FLAGS = (
    "-fPIC",
    "-fwasm-exceptions",
    "-mllvm",
    "-wasm-enable-wasi-dynamic-tls",
    "-mllvm",
    "-wasm-enable-sjlj",
    "-mllvm",
    "-wasm-use-legacy-eh=false",
)
MAIN_LINKER_FLAGS = (
    "--shared-memory",
    "--serial-memory-init",
    "--import-memory",
    "--export-memory",
    "--initial-memory=16777216",
    "--max-memory=67108864",
    "--export-all",
    "--export-table",
    "--growable-table",
    "--export=__stack_pointer",
    "--export=__tls_base",
    "--emit-main-tls-info",
    "--undefined=pthread_create",
    "--split-runtime-ctors",
    "-Bdynamic",
)
SIDE_LINKER_FLAGS = (
    "--shared-memory",
    "--serial-memory-init",
    "--defer-shared-init",
    "--fatal-warnings",
    "--import-memory",
    "--import-table",
    "--export-all",
    "--no-entry",
    "--unresolved-symbols=import-dynamic",
)


def guest_runtime_profile(sysroot: Path, target: str) -> dict:
    """Generate SDK opt-in data without embedding host build paths."""
    arguments = executable_runtime_arguments(sysroot, target)
    relative = []
    for argument in arguments:
        if argument.startswith("-"):
            relative.append(argument)
            continue
        archive = Path(argument).relative_to(sysroot)
        relative.append(SYSROOT_TOKEN + "/" + archive.as_posix())
    return {
        "version": 1,
        "main": [
            *relative,
            *MAIN_LINKER_FLAGS,
            SYSROOT_TOKEN + "/lib/" + target + "/shellsim-abi.o",
        ],
        "shared": [*SIDE_LINKER_FLAGS, SYSROOT_TOKEN + "/lib/" + target + "/shellsim-abi.o"],
    }


def host_executable_flags(sysroot: Path, target: str) -> tuple[str, ...]:
    """Apply the same ordered main policy through a host compiler wrapper."""
    return tuple(
        "-Wl," + argument if argument.startswith("-") else argument
        for argument in (*executable_runtime_arguments(sysroot, target), *MAIN_LINKER_FLAGS)
    )
