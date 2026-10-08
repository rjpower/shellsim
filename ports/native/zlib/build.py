"""Compile the pinned zlib source set without host discovery or build downloads."""

import shutil
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

SOURCES = (
    "adler32",
    "compress",
    "crc32",
    "deflate",
    "gzclose",
    "gzlib",
    "gzread",
    "gzwrite",
    "inflate",
    "infback",
    "inftrees",
    "inffast",
    "trees",
    "uncompr",
    "zutil",
)


def build_zlib(recipe, source, sdk, work, toolchain, run):
    """Build once per complete input identity, then verify the shared artifact."""
    inputs = artifact_input(recipe, Path(__file__).parent, toolchain, {})
    inputs["source_tree_sha256"] = digest(
        {
            path.name: file_hash(path)
            for path in sorted(source.iterdir())
            if path.suffix in (".c", ".h") or path.name == "LICENSE"
        }
    )
    prefix = work / "native-artifacts" / digest(inputs)
    if prefix.exists():
        return prefix, verify_artifact(prefix, inputs)
    build = work / "zlib-build"
    build.mkdir(exist_ok=True)
    env = target_environment(sdk)
    objects = []
    for name in SOURCES:
        output = build / (name + ".o")
        run(
            [
                str(sdk / "bin/clang"),
                *target_profile(recipe)["compiler_flags"],
                "-DZ_HAVE_UNISTD_H",
                "-c",
                str(source / (name + ".c")),
                "-o",
                str(output),
            ],
            build,
            env,
            build / (name + ".log"),
        )
        objects.append(output)
    temporary = prefix.with_name(prefix.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "lib/pkgconfig").mkdir(parents=True)
    (temporary / "include").mkdir()
    (temporary / "licenses").mkdir()
    run(
        [str(sdk / "bin/llvm-ar"), "rcs", str(temporary / "lib/libz.a"), *map(str, objects)],
        build,
        env,
        build / "archive.log",
    )
    for name in ("zlib.h", "zconf.h"):
        shutil.copyfile(source / name, temporary / "include" / name)
    shutil.copyfile(source / "LICENSE", temporary / "licenses/zlib.txt")
    (temporary / "lib/pkgconfig/zlib.pc").write_text(
        "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n"
        "Name: zlib\nDescription: zlib compression library\nVersion: 1.3.1\n"
        "Libs: -L${libdir} -lz\nCflags: -I${includedir}\n"
    )
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)
