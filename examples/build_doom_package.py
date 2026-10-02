"""Package external Doomgeneric inputs for compilation inside one shellsim action."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import shellsim

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATFORM = REPO_ROOT / "tests/fixtures/doom_probe/platform.c"
SHELLSIM_LICENSE = REPO_ROOT / "LICENSE"
FREEDOOM_RELEASE_URL = "https://github.com/freedoom/freedoom/releases/download/v0.13.0/freedoom-0.13.0.zip"
FREEDOOM_WAD_SHA256 = "7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d"
SOURCE_NAME = re.compile(r"[A-Za-z0-9_-]+\.o\Z")
DOOM_REVISION = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
FREEDOOM_FILES = ("freedoom1.wad", "COPYING.txt", "CREDITS.txt", "CREDITS-MUSIC.txt")
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

    commands = ["set -e"]
    for index, name in enumerate(sources):
        commands.append(f"cc -DSHELLSIM_WASI -c /work/doom/{name} -o /work/doom/{index}.o")
    objects = " ".join(f"/work/doom/{index}.o" for index in range(len(sources)))
    commands += [
        f"cc {objects} /work/doom/doomgeneric_shellsim.c -o /work/doom/doom.wasm",
        "chmod +x /work/doom/doom.wasm",
        "/work/doom/doom.wasm -iwad /work/doom/freedoom1.wad",
    ]
    return "\n".join(commands) + "\n"


def _release_files(archive_path: Path) -> dict[str, bytes]:
    """Read the IWAD and its notices from the same bounded release archive."""

    if archive_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("Freedoom release ZIP exceeds 128 MiB")
    with zipfile.ZipFile(archive_path) as archive:
        names = [member.filename for member in archive.infolist()]
        if len(names) != len(set(names)):
            raise ValueError("duplicate Freedoom release ZIP member")
        files = {}
        for name in FREEDOOM_FILES:
            member = archive.getinfo(f"freedoom-0.13.0/{name}")
            limit = 64 * 1024 * 1024 if name == "freedoom1.wad" else 1024 * 1024
            if member.file_size > limit:
                raise ValueError(f"oversized Freedoom release member: {name}")
            with archive.open(member) as source:
                data = source.read(limit + 1)
            if len(data) != member.file_size or len(data) > limit:
                raise ValueError(f"incorrect Freedoom release member size: {name}")
            files[name] = data
    if not files["freedoom1.wad"].startswith(b"IWAD"):
        raise ValueError("expected an IWAD file")
    return files


def build_package(
    source: Path,
    release: Path,
    *,
    doom_revision: Optional[str] = None,
    expected_wad_sha256: Optional[str] = None,
) -> shellsim.Package:
    """Make a source-bearing blob with license notices; no Doom code runs on the host."""

    sources = doom_sources(source)
    if doom_revision is not None and DOOM_REVISION.fullmatch(doom_revision) is None:
        raise ValueError("doom_revision must be a 40-character lowercase commit hash")
    if expected_wad_sha256 is not None and DIGEST.fullmatch(expected_wad_sha256) is None:
        raise ValueError("expected_wad_sha256 must be a lowercase SHA-256 digest")
    release_files = _release_files(release)
    wad_sha256 = hashlib.sha256(release_files["freedoom1.wad"]).hexdigest()
    if expected_wad_sha256 is not None and wad_sha256 != expected_wad_sha256:
        raise ValueError("Freedoom IWAD SHA-256 mismatch")
    doom_license = source.parent / "LICENSE"
    if doom_license.is_symlink() or not doom_license.is_file():
        raise ValueError("Doomgeneric source tree is missing its parent LICENSE")
    with tempfile.TemporaryDirectory(prefix="shellsim-doom-package-") as directory:
        root = Path(directory)
        shutil.copytree(source, root / "doom", symlinks=True)
        licenses = root / "LICENSES"
        licenses.mkdir()
        system_path = root / "doom/i_system.c"
        system = system_path.read_text()
        if "!defined(__DJGPP__)" not in system or "#elif defined(__DJGPP__)" not in system:
            raise ValueError("Doomgeneric i_system.c does not match the supported source revision")
        system_path.write_text(
            "/* Modified by shellsim on 2026-09-24 to use Doomgeneric's console error path under WASI. */\n"
            + system.replace("!defined(__DJGPP__)", "!defined(__DJGPP__) && !defined(SHELLSIM_WASI)").replace(
                "#elif defined(__DJGPP__)", "#elif defined(__DJGPP__) || defined(SHELLSIM_WASI)"
            )
        )
        (root / "doom/freedoom1.wad").write_bytes(release_files["freedoom1.wad"])
        shutil.copyfile(PLATFORM, root / "doom/doomgeneric_shellsim.c")
        shutil.copyfile(doom_license, licenses / "doomgeneric-GPL-2.0.txt")
        shutil.copyfile(SHELLSIM_LICENSE, licenses / "shellsim-Apache-2.0.txt")
        for name in FREEDOOM_FILES[1:]:
            (licenses / f"freedoom-{name}").write_bytes(release_files[name])
        provenance = {
            "doomgeneric": {
                "source": "doom/",
                "upstream": "https://github.com/ozkl/doomgeneric",
                "revision": doom_revision,
                "license": "LICENSES/doomgeneric-GPL-2.0.txt",
                "modified_files": ["doom/i_system.c"],
                "added_files": ["doom/doomgeneric_shellsim.c"],
            },
            "freedoom": {
                "release": FREEDOOM_RELEASE_URL if wad_sha256 == FREEDOOM_WAD_SHA256 else None,
                "wad_sha256": wad_sha256,
                "license": "LICENSES/freedoom-COPYING.txt",
            },
            "runtime_toolchain": {"provider": "shellsim Python distribution", "required": True},
            "shellsim_adapter": {
                "source": "doom/doomgeneric_shellsim.c",
                "license": "Apache-2.0 OR GPL-2.0-or-later",
                "license_texts": ["LICENSES/shellsim-Apache-2.0.txt", "LICENSES/doomgeneric-GPL-2.0.txt"],
            },
        }
        (root / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
        (root / "run.sh").write_text(guest_script(sources))
        spec = shellsim.PackageSpec(
            name="doomgeneric-freedoom",
            version="0.13.0",
            entrypoint=("sh", "/work/run.sh"),
            limits=LIMITS,
            requested_clock="real_time",
            requires_c_toolchain=True,
        )
        return shellsim.Package.build_from_directory(root, spec=spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Doomgeneric doomgeneric/ source directory")
    parser.add_argument("release", type=Path, help="Freedoom 0.13.0 release ZIP")
    parser.add_argument("output", type=Path, help="output .shl file")
    parser.add_argument("--doom-revision", help="pinned Doomgeneric source commit")
    parser.add_argument("--wad-sha256", help="expected SHA-256 of the release IWAD")
    args = parser.parse_args()
    package = build_package(
        args.source,
        args.release,
        doom_revision=args.doom_revision,
        expected_wad_sha256=args.wad_sha256,
    )
    args.output.write_bytes(package.to_bytes())
    print(f"{package.sha256}  {args.output}")


if __name__ == "__main__":
    main()
