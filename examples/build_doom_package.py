"""Package external Doomgeneric inputs for compilation inside one shellsim action."""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

import shellsim

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLCHAIN = REPO_ROOT / "tests/fixtures/tinycc/tcc-shellsim-package.tar.gz"
SYSROOT = REPO_ROOT / "tests/fixtures/wasi-libc/sysroot-34.tar.gz"
PLATFORM = REPO_ROOT / "tests/fixtures/doom_probe/platform.c"
SOURCE_NAME = re.compile(r"[A-Za-z0-9_-]+\.o\Z")
LIMITS = shellsim.Limits(
    cpu=400_000_000_000,
    memory=512 * 1024 * 1024,
    disk=512 * 1024 * 1024,
    output=32 * 1024 * 1024,
)


def doom_sources(source: Path) -> tuple[str, ...]:
    """Return Makefile-listed C files, rejecting paths outside the Doom source directory."""

    makefile = (source / "Makefile.soso").read_text()
    objects = next(
        (line.removeprefix("SRC_DOOM = ") for line in makefile.splitlines() if line.startswith("SRC_DOOM = ")),
        None,
    )
    if objects is None:
        raise ValueError("Doomgeneric Makefile.soso has no SRC_DOOM list")
    names = [name for name in objects.split() if name != "doomgeneric_soso.o"]
    if not names or any(SOURCE_NAME.fullmatch(name) is None for name in names):
        raise ValueError("Doomgeneric source list contains an invalid object name")
    sources = tuple(name[:-2] + ".c" for name in names)
    if any(not (source / name).is_file() for name in sources):
        raise ValueError("Doomgeneric source list references a missing C file")
    return sources


def guest_script(sources: tuple[str, ...]) -> str:
    """Build and launch Doom in the same guest shell action."""

    commands = [
        "set -e",
        "mkdir -p /tcc /wasi-sysroot",
        "tar -xzf /work/tcc.tar.gz -C /tcc",
        "tar -xzf /work/sysroot.tar.gz -C /wasi-sysroot",
        "chmod +x /tcc/tcc-shellsim.wasm",
        "rm /usr/bin/cc",
        "ln -s /tcc/tcc-shellsim.wasm /usr/bin/cc",
    ]
    for index, name in enumerate(sources):
        commands.append(f"cc -DSHELLSIM_WASI -c /work/doom/{name} -o /work/doom/{index}.o")
    objects = " ".join(f"/work/doom/{index}.o" for index in range(len(sources)))
    commands += [
        f"cc {objects} /work/doom/doomgeneric_shellsim.c -o /work/doom/doom.wasm",
        "chmod +x /work/doom/doom.wasm",
        "/work/doom/doom.wasm -iwad /work/doom/freedoom1.wad",
    ]
    return "\n".join(commands) + "\n"


def build_package(source: Path, wad: Path) -> shellsim.Package:
    """Make an immutable blob; no compiler or Doom code runs on the host."""

    sources = doom_sources(source)
    with wad.open("rb") as file:
        if file.read(4) != b"IWAD":
            raise ValueError("expected an IWAD file")
    with tempfile.TemporaryDirectory(prefix="shellsim-doom-package-") as directory:
        root = Path(directory)
        shutil.copytree(source, root / "doom", symlinks=True)
        system_path = root / "doom/i_system.c"
        system = system_path.read_text()
        if "!defined(__DJGPP__)" not in system or "#elif defined(__DJGPP__)" not in system:
            raise ValueError("Doomgeneric i_system.c does not match the supported source revision")
        system_path.write_text(
            system.replace("!defined(__DJGPP__)", "!defined(__DJGPP__) && !defined(SHELLSIM_WASI)").replace(
                "#elif defined(__DJGPP__)", "#elif defined(__DJGPP__) || defined(SHELLSIM_WASI)"
            )
        )
        shutil.copyfile(wad, root / "doom/freedoom1.wad")
        shutil.copyfile(PLATFORM, root / "doom/doomgeneric_shellsim.c")
        shutil.copyfile(TOOLCHAIN, root / "tcc.tar.gz")
        shutil.copyfile(SYSROOT, root / "sysroot.tar.gz")
        (root / "run.sh").write_text(guest_script(sources))
        spec = shellsim.PackageSpec(
            name="doomgeneric-freedoom",
            version="0.13.0",
            entrypoint=("sh", "/work/run.sh"),
            limits=LIMITS,
            requested_clock="real_time",
        )
        return shellsim.Package.build_from_directory(root, spec=spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Doomgeneric doomgeneric/ source directory")
    parser.add_argument("wad", type=Path, help="Freedoom IWAD")
    parser.add_argument("output", type=Path, help="output .shl file")
    args = parser.parse_args()
    package = build_package(args.source, args.wad)
    args.output.write_bytes(package.to_bytes())
    print(f"{package.sha256}  {args.output}")


if __name__ == "__main__":
    main()
