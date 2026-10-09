"""Build explicit TrueType/CFF FreeType modules against declared WASI zlib."""

import re
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


def pkg_config(source, recipe):
    """Render upstream's template with its libtool version and declared links.

    FreeType's pkg-config Version is its libtool version, not the semantic
    release version used by FT_Library_Version and our source recipe.
    """
    header = (source / "include/freetype/freetype.h").read_text()
    release = [
        re.search(r"^#define FREETYPE_" + part + r"\s+(\d+)$", header, re.M) for part in ("MAJOR", "MINOR", "PATCH")
    ]
    if any(match is None for match in release) or ".".join(match[1] for match in release) != recipe["version"]:
        raise ValueError("FreeType source release differs from the recipe")
    configure = (source / "builds/unix/configure.raw").read_text()
    versions = re.findall(r"^version_info='(\d+):(\d+):(\d+)'$", configure, re.M)
    if len(versions) != 1 or "ft_version=`echo $version_info | tr : .`" not in configure:
        raise ValueError("FreeType pkg-config version derivation changed")
    values = {
        "prefix": "${pcfiledir}/../..",
        "exec_prefix": "${prefix}",
        "libdir": "${exec_prefix}/lib",
        "includedir": "${prefix}/include",
        "ft_version": ".".join(versions[0]),
        "PKGCONFIG_REQUIRES": "",
        "PKGCONFIG_REQUIRES_PRIVATE": "zlib",
        "PKGCONFIG_LIBS": "-L${libdir} -lfreetype",
        "PKGCONFIG_LIBS_PRIVATE": " ".join(recipe["transitive_link_flags"]),
    }
    template = (source / "builds/unix/freetype2.in").read_text()
    if set(re.findall(r"%([A-Za-z_]+)%", template)) != set(values):
        raise ValueError("FreeType pkg-config template fields changed")
    return re.sub(r"%([A-Za-z_]+)%", lambda match: values[match[1]], template)


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
    inputs["pkg_config_sources"] = {
        name: file_hash(source / name) for name in ("builds/unix/configure.raw", "builds/unix/freetype2.in")
    }
    package_metadata = pkg_config(source, recipe)
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
    (temporary / "lib/pkgconfig/freetype2.pc").write_text(package_metadata)
    seal_artifact(temporary, inputs)
    temporary.rename(prefix)
    return prefix, verify_artifact(prefix, inputs)
