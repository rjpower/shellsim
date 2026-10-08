"""Build scalar 8-bit libjpeg with the pinned WASI toolchain."""

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
    "jcapimin",
    "jcapistd",
    "jccoefct",
    "jccolor",
    "jcdctmgr",
    "jchuff",
    "jcicc",
    "jcinit",
    "jcmainct",
    "jcmarker",
    "jcmaster",
    "jcomapi",
    "jcparam",
    "jcphuff",
    "jcprepct",
    "jcsample",
    "jctrans",
    "jdapimin",
    "jdapistd",
    "jdatadst",
    "jdatasrc",
    "jdcoefct",
    "jdcolor",
    "jddctmgr",
    "jdhuff",
    "jdicc",
    "jdinput",
    "jdmainct",
    "jdmarker",
    "jdmaster",
    "jdmerge",
    "jdphuff",
    "jdpostct",
    "jdsample",
    "jdtrans",
    "jerror",
    "jfdctflt",
    "jfdctfst",
    "jfdctint",
    "jidctflt",
    "jidctfst",
    "jidctint",
    "jidctred",
    "jquant1",
    "jquant2",
    "jutils",
    "jmemmgr",
    "jmemnobs",
    "jsimd_none",
)
CONFIG = """#define JPEG_LIB_VERSION 62
#define LIBJPEG_TURBO_VERSION 2.1.5.1
#define LIBJPEG_TURBO_VERSION_NUMBER 2001005
#define MEM_SRCDST_SUPPORTED 1
#define BITS_IN_JSAMPLE 8
"""
INTERNAL_CONFIG = """#define BUILD "20230208"
#undef inline
#define INLINE inline __attribute__((always_inline))
#define THREAD_LOCAL
#define PACKAGE_NAME "libjpeg-turbo"
#define VERSION "2.1.5.1"
#define SIZEOF_SIZE_T 4
#define HAVE_BUILTIN_CTZL
#define FALLTHROUGH __attribute__((fallthrough));
"""


def build_libjpeg_turbo(recipe, source, sdk, work, toolchain, run):
    """Publish an immutable archive; error recovery requires a consumer setjmp ABI."""
    inputs = artifact_input(recipe, Path(__file__).parent, toolchain, {})
    inputs["source_tree_sha256"] = digest(
        {p.relative_to(source).as_posix(): file_hash(p) for p in sorted(source.rglob("*")) if p.is_file()}
    )
    prefix = work / "native-artifacts" / digest(inputs)
    if prefix.exists():
        return prefix, verify_artifact(prefix, inputs)
    build = work / "libjpeg-turbo-build"
    build.mkdir(exist_ok=True)
    (build / "jconfig.h").write_text(CONFIG)
    (build / "jconfigint.h").write_text(INTERNAL_CONFIG)
    (build / "jversion.h").write_text((source / "jversion.h.in").read_text().replace("@COPYRIGHT_YEAR@", "1991-2023"))
    env = target_environment(sdk)
    objects = []
    for name in SOURCES:
        output = build / (name + ".o")
        run(
            [
                str(sdk / "bin/clang"),
                *target_profile(recipe)["compiler_flags"],
                "-I" + str(build),
                "-I" + str(source),
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
        [str(sdk / "bin/llvm-ar"), "rcs", str(temporary / "lib/libjpeg.a"), *map(str, objects)],
        build,
        env,
        build / "archive.log",
    )
    for name in ("jpeglib.h", "jmorecfg.h", "jerror.h"):
        shutil.copyfile(source / name, temporary / "include" / name)
    shutil.copyfile(build / "jconfig.h", temporary / "include/jconfig.h")
    shutil.copyfile(source / "LICENSE.md", temporary / "licenses/libjpeg-turbo.txt")
    shutil.copyfile(source / "README.ijg", temporary / "licenses/README.ijg")
    (temporary / "lib/pkgconfig/libjpeg.pc").write_text(
        "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n"
        "Name: libjpeg\nDescription: Scalar 8-bit JPEG codec\nVersion: 2.1.5.1\n"
        "Libs: -L${libdir} -ljpeg\nCflags: -I${includedir}\n"
    )
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)
