"""Build the pinned NumPy static extensions with target CPython headers.

The host Python runs Meson and Cython only. Every compiled object uses the WASI
SDK, and native extension archives become qualified CPython builtin modules.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from ports._support.build import apply_patch, check_build_scripts
from ports.native.dependencies import target_environment, target_profile


def check_profile(recipe, work):
    """Refuse cached Meson options or objects from a different native recipe."""
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    marker = work / "numpy-profile.sha256"
    if not marker.exists() and (work / "numpy-build").exists():
        raise ValueError(f"Unidentified cached NumPy build; use a clean build directory: {work}")
    if marker.exists() and marker.read_text().strip() != digest:
        raise ValueError(f"NumPy profile changed; use a clean build directory: {work}")
    marker.write_text(digest + "\n")


def build_numpy(recipe, source, cpython_source, cpython_build, sdk, work, jobs, run):
    """Return target extension archives after compiling a closed native module set."""
    directory = Path(__file__).parent
    check_build_scripts(recipe, directory)
    check_profile(recipe, work)
    for patch in recipe["patches"]:
        if patch["file"] == "numpy-wasi.patch":
            apply_patch(source, directory / patch["file"], patch["sha256"])
    tools = work / "numpy-build-tools"
    env = target_environment(sdk)
    if not (tools / "bin/python").exists():
        subprocess.run(["uv", "venv", "--python", "3.13", str(tools)], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(tools / "bin/python"),
            f"ninja=={recipe['build_tools']['ninja']}",
            f"Cython=={recipe['build_tools']['cython']}",
        ],
        check=True,
    )
    env["PATH"] = str(tools / "bin") + os.pathsep + env["PATH"]
    sysroot = sdk / "share/wasi-sysroot"
    # NumPy's meson.find_installation() uses host Python for generators. A target
    # pkg-config provider is necessary to keep its headers out of target objects.
    pkg_config = work / "numpy-python-pkg-config"
    target_headers = f"-I{cpython_source / 'Include'} -I{cpython_build}"
    pkg_config.write_text(
        f"#!{tools / 'bin/python'}\n"
        "import sys\nargs = sys.argv[1:]\n"
        "if '--version' in args:\n print('1.9.5'); raise SystemExit(0)\n"
        "if not any(arg.startswith('python-') for arg in args):\n raise SystemExit(1)\n"
        "if '--modversion' in args:\n print('3.13')\n"
        f"elif '--cflags' in args:\n print({target_headers!r})\n"
        "elif '--libs' in args:\n print('')\n"
        f"elif any(arg.startswith('--variable=') for arg in args):\n print({str(cpython_build)!r})\n"
    )
    pkg_config.chmod(0o755)
    cross = work / "numpy-wasi.cross"
    profile = target_profile(recipe)
    c_args = [f"--sysroot={sysroot}", *profile["compiler_flags"]]
    cross.write_text(
        "[binaries]\n"
        + "".join(
            f"{key} = {str(path)!r}\n"
            for key, path in {
                "c": sdk / "bin/clang",
                "cpp": sdk / "bin/clang++",
                "ar": sdk / "bin/llvm-ar",
                "strip": sdk / "bin/llvm-strip",
                "cython": tools / "bin/cython",
                "pkg-config": pkg_config,
            }.items()
        )
        + "[host_machine]\nsystem = 'wasi'\ncpu_family = 'wasm32'\ncpu = 'wasm32'\nendian = 'little'\n"
        + "[properties]\nneeds_exe_wrapper = true\nlongdouble_format = 'IEEE_QUAD_LE'\n"
        + f"[built-in options]\nc_args = {c_args!r}\ncpp_args = {[*c_args, *profile['cpp_flags']]!r}\n"
    )
    longdouble = work / "numpy-longdouble.c"
    longdouble.write_text(
        '_Static_assert(sizeof(long double) == 16, "WASI long double must be quad");\n'
        '_Static_assert(__LDBL_MANT_DIG__ == 113, "WASI long double must have 113 mantissa bits");\n'
    )
    subprocess.run(
        [str(sdk / "bin/clang"), *c_args, "-c", str(longdouble), "-o", str(work / "numpy-longdouble.o")], check=True
    )
    build = work / "numpy-build"
    meson = [str(tools / "bin/python"), str(source / "vendored-meson/meson/meson.py")]
    if not (build / "build.ninja").exists():
        run(
            [
                *meson,
                "setup",
                str(build),
                str(source),
                "--cross-file",
                str(cross),
                "-Dblas=none",
                "-Dlapack=none",
                "-Ddisable-threading=true",
                "-Ddisable-optimization=true",
                "-Ddisable-highway=true",
                "-Ddisable-intel-sort=true",
                "-Dcpu-baseline=none",
                "-Dcpu-dispatch=none",
                "--buildtype",
                "release",
                "--prefix",
                "/usr",
            ],
            work,
            env,
            work / "numpy-configure.log",
        )
    targets = json.loads((build / "meson-info/intro-targets.json").read_text())
    archives = {}
    support = []
    for target in targets:
        if target["type"] != "static library":
            continue
        install = target.get("install_filename")
        if install and ".cpython-" in target["name"]:
            destination = Path(install[0])
            name = target["name"].split(".cpython-", 1)[0]
            qualified = destination.parent.as_posix().split("/site-packages/", 1)[1].replace("/", ".") + "." + name
            if qualified in recipe["builtin_modules"]:
                archives[qualified] = Path(target["filename"][0])
        elif target["name"] in ("npymath", "npyrandom"):
            support.append(Path(target["filename"][0]))
    if set(archives) != set(recipe["builtin_modules"]) or len(support) != 2:
        raise ValueError("NumPy Meson targets do not match the pinned native module recipe")
    run(
        [
            str(tools / "bin/ninja"),
            "-C",
            str(build),
            f"-j{jobs}",
            *(str(path.relative_to(build)) for path in [*archives.values(), *support]),
        ],
        work,
        env,
        work / "numpy-make.log",
    )
    return archives, support


def install_numpy(build, site_packages):
    """Install Meson's Python runtime payload, omitting native archives and tests."""
    plan = json.loads((build / "meson-info/intro-install_plan.json").read_text())
    for section, entries in plan.items():
        for source, spec in entries.items():
            if spec["tag"] != "python-runtime":
                continue
            relative = spec["destination"].removeprefix("{py_platlib}/")
            if relative == spec["destination"] or not relative.startswith("numpy/"):
                raise ValueError(f"Unexpected NumPy install destination: {spec['destination']}")
            destination = site_packages / relative
            if section == "install_subdirs":
                shutil.copytree(
                    source,
                    destination,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", *spec["exclude_dirs"], *spec["exclude_files"]),
                )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
