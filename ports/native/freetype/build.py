"""Build explicit TrueType/CFF FreeType modules against declared WASI zlib."""

import shutil
from pathlib import Path

from ports.native.dependencies import (
    artifact_input,
    dependency_prefix,
    digest,
    file_hash,
    seal_artifact,
    target_environment,
    target_profile,
    verify_artifact,
)

SOURCES = (
    "base/ftbase.c",
    "base/ftinit.c",
    "base/ftsystem.c",
    "base/ftdebug.c",
    "base/ftbbox.c",
    "base/ftbitmap.c",
    "base/ftglyph.c",
    "base/ftstroke.c",
    "base/ftmm.c",
    "base/ftfstype.c",
    "base/ftgasp.c",
    "base/ftotval.c",
    "base/ftpatent.c",
    "base/ftpfr.c",
    "base/fttype1.c",
    "base/ftwinfnt.c",
    "truetype/truetype.c",
    "cff/cff.c",
    "sfnt/sfnt.c",
    "autofit/autofit.c",
    "psaux/psaux.c",
    "psnames/psnames.c",
    "pshinter/pshinter.c",
    "smooth/smooth.c",
    "raster/raster.c",
    "gzip/ftgzip.c",
)
MODULES = """FT_USE_MODULE( FT_Module_Class, autofit_module_class )
FT_USE_MODULE( FT_Driver_ClassRec, tt_driver_class )
FT_USE_MODULE( FT_Driver_ClassRec, cff_driver_class )
FT_USE_MODULE( FT_Module_Class, psaux_module_class )
FT_USE_MODULE( FT_Module_Class, psnames_module_class )
FT_USE_MODULE( FT_Module_Class, pshinter_module_class )
FT_USE_MODULE( FT_Module_Class, sfnt_module_class )
FT_USE_MODULE( FT_Renderer_Class, ft_smooth_renderer_class )
FT_USE_MODULE( FT_Renderer_Class, ft_raster1_renderer_class )
"""


def build_freetype(recipe, source, sdk, work, toolchain, providers, run):
    """Seal headers and static archive with their exact zlib dependency identity."""
    build = work / "freetype-build"
    build.mkdir(parents=True, exist_ok=True)
    dependencies, _ = dependency_prefix(
        recipe["target_dependencies"], providers, build / "dependencies", recipe["target_profile"]
    )
    inputs = artifact_input(recipe, Path(__file__).parent, toolchain, dependencies)
    inputs["source_tree_sha256"] = digest(
        {
            p.relative_to(source).as_posix(): file_hash(p)
            for root in ("include", "src", "docs")
            for p in sorted((source / root).rglob("*"))
            if p.is_file()
        }
    )
    prefix = work / "native-artifacts" / digest(inputs)
    if prefix.exists():
        return prefix, verify_artifact(prefix, inputs)
    include = build / "include"
    if include.exists():
        shutil.rmtree(include)
    shutil.copytree(source / "include", include)
    config = include / "freetype/config"
    (config / "ftmodule.h").write_text(MODULES)
    options = (config / "ftoption.h").read_text()
    options = options.replace("/* #define FT_CONFIG_OPTION_SYSTEM_ZLIB */", "#define FT_CONFIG_OPTION_SYSTEM_ZLIB")
    options = options.replace("#define FT_CONFIG_OPTION_USE_LZW", "/* LZW font compression is outside this profile. */")
    (config / "ftoption.h").write_text(options)
    env = target_environment(sdk)
    objects = []
    for name in SOURCES:
        output = build / (name.replace("/", "_") + ".o")
        run(
            [
                str(sdk / "bin/clang"),
                *target_profile(recipe)["compiler_flags"],
                "-DFT2_BUILD_LIBRARY",
                "-I" + str(include),
                "-I" + str(build / "dependencies/include"),
                "-c",
                str(source / "src" / name),
                "-o",
                str(output),
            ],
            build,
            env,
            output.with_suffix(".log"),
        )
        objects.append(output)
    temporary = prefix.with_name(prefix.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "lib/pkgconfig").mkdir(parents=True)
    (temporary / "licenses").mkdir()
    shutil.copytree(include, temporary / "include/freetype2")
    run(
        [str(sdk / "bin/llvm-ar"), "rcs", str(temporary / "lib/libfreetype.a"), *map(str, objects)],
        build,
        env,
        build / "archive.log",
    )
    for name in ("FTL.TXT", "GPLv2.TXT", "LICENSE.TXT"):
        shutil.copyfile(
            source / name if name == "LICENSE.TXT" else source / "docs" / name, temporary / "licenses" / name
        )
    (temporary / "lib/pkgconfig/freetype2.pc").write_text(
        "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include/freetype2\n"
        "Name: FreeType 2\nDescription: Minimal TrueType/CFF font rasterizer\nVersion: 2.13.3\n"
        "Requires.private: zlib\nLibs: -L${libdir} -lfreetype\nLibs.private: -lm\nCflags: -I${includedir}\n"
    )
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)
