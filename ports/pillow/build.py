"""Compile Pillow's upstream core and FreeType bindings for a static WASI profile.

The pinned source lists are read as literals, without executing setup.py and its
ambient host-library discovery. No optional external codecs or host headers enter
target objects. CPython headers describe the target development configuration.
"""

import ast
import shutil
from pathlib import Path

from ports.native.dependencies import artifact_input, digest, file_hash, target_environment, target_profile


def pillow_sources(source):
    """Read the core and internal-library source lists from pinned upstream setup.py."""
    tree = ast.parse((source / "setup.py").read_text())
    lists = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in ("_IMAGING", "_LIB_IMAGING")
    }
    libraries = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "libraries"
    )
    if [name for name, _ in libraries] != ["pil_imaging_mode"]:
        raise ValueError("Pillow internal libraries differ from the reviewed source profile")
    sources = [source / "src/_imaging.c"]
    sources += [source / "src" / (name + ".c") for name in lists["_IMAGING"]]
    sources += [source / "src/libImaging" / (name + ".c") for name in lists["_LIB_IMAGING"]]
    sources += [source / name for _, library in libraries for name in library["sources"]]
    return sources


def build_pillow(recipe, source, cpython_source, cpython_build, sdk, work, dependencies, prefix, run):
    """Return one builtin archive, invalidating it on any declared input change."""
    inputs = artifact_input(
        recipe, Path(__file__).parent, dependencies["native/zlib"]["inputs"]["toolchain"], dependencies
    )
    inputs["cpython_development"] = {
        "pyconfig_sha256": file_hash(cpython_build / "pyconfig.h"),
        "headers_sha256": digest(
            {
                str(p.relative_to(cpython_source)): file_hash(p)
                for p in sorted((cpython_source / "Include").rglob("*.h"))
            }
        ),
    }
    sources = pillow_sources(source)
    inputs["upstream_source_list"] = [str(path.relative_to(source)) for path in sources]
    inputs["source_tree_sha256"] = digest(
        {
            str(path.relative_to(source)): file_hash(path)
            for path in sorted((source / "src").rglob("*"))
            if path.suffix in (".c", ".h")
        }
    )
    build = work / "pillow-build"
    build.mkdir(exist_ok=True)
    marker = build / "inputs.json"
    identity = digest(inputs)
    archives = {"PIL._imaging": build / "libpillow.a", "PIL._imagingft": build / "libpillow-freetype.a"}
    if marker.exists():
        recorded = marker.read_text().splitlines()
        if recorded[0] != identity:
            raise ValueError("Pillow dependency inputs changed; use a clean build directory")
        if len(recorded) == 2 and recorded[1] == digest({name: file_hash(path) for name, path in archives.items()}):
            return archives, inputs
        raise ValueError("Pillow archive integrity failure")
    env = target_environment(sdk)
    objects = []
    for index, item in enumerate(sources):
        output = build / f"{index}.o"
        run(
            [
                str(sdk / "bin/clang"),
                *target_profile(recipe)["compiler_flags"],
                "-DHAVE_LIBZ",
                "-DHAVE_LIBJPEG",
                f'-DPILLOW_VERSION="{recipe["version"]}"',
                f"-I{prefix / 'include'}",
                f"-I{cpython_source / 'Include'}",
                f"-I{cpython_build}",
                f"-I{source / 'src/libImaging'}",
                "-c",
                str(item),
                "-o",
                str(output),
            ],
            build,
            env,
            build / f"{index}.log",
        )
        objects.append(output)
    run(
        [str(sdk / "bin/llvm-ar"), "rcs", str(archives["PIL._imaging"]), *map(str, objects)],
        build,
        env,
        build / "archive.log",
    )
    output = build / "imagingft.o"
    run(
        [
            str(sdk / "bin/clang"),
            *target_profile(recipe)["compiler_flags"],
            f'-DPILLOW_VERSION="{recipe["version"]}"',
            f"-I{prefix / 'include/freetype2'}",
            f"-I{cpython_source / 'Include'}",
            f"-I{cpython_build}",
            "-c",
            str(source / "src/_imagingft.c"),
            "-o",
            str(output),
        ],
        build,
        env,
        build / "imagingft.log",
    )
    run(
        [str(sdk / "bin/llvm-ar"), "rcs", str(archives["PIL._imagingft"]), str(output)],
        build,
        env,
        build / "freetype-archive.log",
    )
    marker.write_text(identity + "\n" + digest({name: file_hash(path) for name, path in archives.items()}) + "\n")
    return archives, inputs


def install_pillow(source, site_packages):
    """Stage upstream Python modules without build helpers or test payloads."""
    shutil.copytree(source / "src/PIL", site_packages / "PIL", ignore=shutil.ignore_patterns("__pycache__"))
