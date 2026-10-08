"""Build a pinned, static CPython WASI command and its virtual filesystem image.

This is trusted host build tooling. The resulting command receives capabilities
only through shellsim's WASI adapter; it cannot invoke the host helper interpreter.
"""

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ports.native.dependencies import dependency_prefix, digest, file_hash, target_environment, toolchain_identity
from ports.native.freetype.build import build_freetype
from ports.native.zlib.build import build_zlib
from ports.numpy.build import apply_patch, build_numpy, check_build_scripts, install_numpy
from ports.pillow.build import build_pillow, install_pillow


def build_jpeg(recipe, source, sdk, work, toolchain, run):
    """Load the hyphenated catalog entry without adding a second public name."""
    import importlib.util

    path = Path(__file__).parent.parent / "native/libjpeg-turbo/build.py"
    spec = importlib.util.spec_from_file_location("libjpeg_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_libjpeg_turbo(recipe, source, sdk, work, toolchain, run)


def fetch_extract(spec, work):
    """Verify the complete upstream archive before extracting build inputs."""
    archive = work / "downloads" / spec["url"].rsplit("/", 1)[-1]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen(spec["url"]) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != spec["sha256"]:
        raise ValueError(f"SHA256 mismatch for {archive}: {actual}")
    with tarfile.open(archive) as source:
        root = work / source.getnames()[0].split("/", 1)[0]
        if not root.exists():
            source.extractall(work, filter="data")
    return root


def run(command, cwd, env, log):
    """Retain compiler diagnostics without printing thousands of build lines."""
    print(f"Building in {cwd}; log: {log}", flush=True)
    with log.open("w") as output:
        subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)


def install_numpy_notices(source, root):
    """Preserve upstream and bundled third-party license notices in the image."""
    destination = root / "NUMPY-LICENSES"
    destination.mkdir()
    shutil.copyfile(source / "LICENSES_bundled.txt", destination / "LICENSES_bundled.txt")
    notices = []
    for path in sorted((source / "numpy").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.name.upper().startswith(("LICENSE", "COPYING")):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        elif path.suffix in (".c", ".cpp", ".h"):
            text = path.read_text(errors="replace")
            header = re.match(r"\s*/\*(.*?)\*/", text, re.DOTALL)
            if header is not None and re.search("copyright|license", header[1], re.IGNORECASE):
                notices.append(str(relative) + "\n" + header[0] + "\n")
    (destination / "SOURCE-NOTICES.txt").write_text("\n".join(notices))


def install_metadata(port, source, root):
    """Describe a static provider to the resolver without staging a native wheel."""
    destination = root / port["dist_info"].lstrip("/")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "PKG-INFO", destination / "METADATA")
    (destination / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: shellsim-static-port\nRoot-Is-Purelib: true\nTag: py313-none-any\n"
    )
    # Native code lives in python.wasm. These files describe its distribution;
    # they are never a standalone substitute for that verified interpreter.
    with (destination / "RECORD").open("w", newline="") as output:
        writer = csv.writer(output)
        for path in sorted(destination.iterdir()):
            if path.name == "RECORD":
                continue
            contents = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode()
            writer.writerow([destination.name + "/" + path.name, "sha256=" + digest, len(contents)])
        writer.writerow([destination.name + "/RECORD", "", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/shellsim-cpython"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--target-profile", choices=("wasi-cpython-v1", "wasi-cpython-v2"), default="wasi-cpython-v2")
    parser.add_argument("--with-pycosat", action="store_true", help="statically link the pinned native pycosat port")
    parser.add_argument("--with-numpy", action="store_true", help="statically link the pinned native NumPy port")
    parser.add_argument("--with-zlib", action="store_true", help="link the pinned shared target zlib artifact")
    parser.add_argument(
        "--with-pillow", action="store_true", help="link Pillow PNG/JPEG/FreeType and its declared target libraries"
    )
    parser.add_argument("--build-python", type=Path, help="reuse a trusted native CPython 3.13 helper")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.target_profile == "wasi-cpython-v1" and (args.with_numpy or args.with_pillow or args.with_zlib):
        raise ValueError("The current native library recipes require wasi-cpython-v2")
    recipe_file = "recipe-v1.json" if args.target_profile == "wasi-cpython-v1" else "recipe.json"
    recipe = json.loads(Path(__file__).with_name(recipe_file).read_text())
    source = fetch_extract(recipe["source"], work)
    sdk = fetch_extract(recipe["sdk"], work)
    host = work / "host-build"
    guest = work / "wasi-build"
    host.mkdir(exist_ok=True)
    guest.mkdir(exist_ok=True)
    env = dict(os.environ, SOURCE_DATE_EPOCH="1756857600")
    if args.build_python is None:
        if not (host / "Makefile").exists():
            run([str(source / "configure"), "--without-ensurepip"], host, env, work / "host-configure.log")
        run(["make", f"-j{args.jobs}"], host, env, work / "host-make.log")
    helper = args.build_python.resolve() if args.build_python is not None else host / "python"
    if (
        subprocess.check_output([str(helper), "-c", "import sys; print(sys.version.split()[0])"], text=True).strip()
        != recipe["version"]
    ):
        raise ValueError("The native build helper must match the pinned CPython version")
    sysroot = sdk / "share" / "wasi-sysroot"
    toolchain = toolchain_identity(recipe, sdk)
    target_artifacts = {}
    target_links = []
    providers = {}
    prefix = work / "dependency-prefix"
    if args.with_zlib or args.with_pillow:
        directory = Path(__file__).parent.parent / "native/zlib"
        zlib_recipe = json.loads((directory / "recipe.json").read_text())
        zlib_source = fetch_extract(zlib_recipe["source"], work)
        zlib_prefix, _ = build_zlib(zlib_recipe, zlib_source, sdk, work, toolchain, run)
        providers["native/zlib"] = zlib_prefix
        if args.with_pillow:
            for name, builder in (("libjpeg-turbo", build_jpeg), ("freetype", build_freetype)):
                library_recipe = json.loads(
                    (Path(__file__).parent.parent / "native" / name / "recipe.json").read_text()
                )
                library_source = fetch_extract(library_recipe["source"], work)
                arguments = [library_recipe, library_source, sdk, work, toolchain]
                if name == "freetype":
                    arguments.append(providers)
                providers["native/" + name], _ = builder(*arguments, run)
        target_artifacts, target_links = dependency_prefix(
            recipe["optional_target_dependencies"]["zlib"],
            {"native/zlib": zlib_prefix},
            prefix,
            recipe["target_profile"],
        )
        if args.with_pillow:
            pillow_recipe = json.loads(Path(__file__).parent.parent.joinpath("pillow/recipe.json").read_text())
            target_artifacts, _ = dependency_prefix(
                pillow_recipe["target_dependencies"], providers, prefix, recipe["target_profile"]
            )
    profile = {
        "recipe": recipe,
        "builder_sha256": file_hash(Path(__file__)),
        "toolchain": toolchain,
        "host_helper_sha256": file_hash(helper),
        "host_tools": {
            "driver": {"version": sys.version.split()[0], "sha256": file_hash(Path(sys.executable))},
            "make": {"sha256": file_hash(Path(shutil.which("make")))},
            "python-build-helper": {"version": recipe["version"], "sha256": file_hash(helper)},
        },
        "ports": {
            name: file_hash(Path(__file__).parent.parent / name / "recipe.json")
            for name, enabled in (
                ("numpy", args.with_numpy),
                ("pycosat", args.with_pycosat),
                ("pillow", args.with_pillow),
            )
            if enabled
        },
        "dependencies": {name: item["artifact_sha256"] for name, item in target_artifacts.items()},
    }
    marker = work / "cpython-profile.sha256"
    profile_hash = digest(profile)
    if marker.exists() and marker.read_text().strip() != profile_hash:
        raise ValueError("CPython build inputs changed; use a clean build directory")
    if not marker.exists() and (guest / "Makefile").exists():
        raise ValueError("Unidentified cached CPython build; use a clean build directory")
    marker.write_text(profile_hash + "\n")
    guest_env = dict(
        target_environment(sdk),
        CC=f"{sdk / 'bin/clang'} --sysroot={sysroot}",
        AR=str(sdk / "bin/llvm-ar"),
        RANLIB=str(sdk / "bin/llvm-ranlib"),
        CONFIG_SITE=str(source / "Tools/wasm/config.site-wasm32-wasi"),
        CFLAGS=" ".join(toolchain["profile"]["compiler_flags"]),
        LDFLAGS=" ".join(
            [
                *toolchain["profile"]["link_flags"],
                *(toolchain["profile"]["cpp_flags"] if args.with_numpy else []),
            ]
        ),
        PKG_CONFIG_PATH="",
        PKG_CONFIG_LIBDIR=str(prefix / "lib/pkgconfig") if target_artifacts else "",
        PKG_CONFIG="/bin/false",
    )
    if target_artifacts:
        guest_env["ZLIB_CFLAGS"] = f"-I{prefix / 'include'}"
        guest_env["ZLIB_LIBS"] = " ".join(target_links)
    if not (guest / "Makefile").exists():
        build = subprocess.check_output([str(source / "config.guess")], text=True).strip()
        run(
            [
                str(source / "configure"),
                "--host=wasm32-wasip1",
                f"--build={build}",
                f"--with-build-python={helper}",
                "--prefix=/usr",
                "--without-ensurepip",
                "--disable-test-modules",
            ],
            guest,
            guest_env,
            work / "wasi-configure.log",
        )
    native_ports = []
    port_sources = {}
    setup = guest / "Modules/Setup.local"
    setup_text = "# shellsim static native ports\n"
    link_consumers = {}
    if target_artifacts:
        setup_text += f"*static*\nzlib zlibmodule.c -I{prefix / 'include'} {' '.join(target_links)}\n"
        link_consumers["cpython.zlib"] = {
            "dependency_artifacts": {"native/zlib": profile["dependencies"]["native/zlib"]},
            "link_inputs": target_links,
        }
    if args.with_pycosat:
        port = json.loads(Path(__file__).parent.parent.joinpath("pycosat/recipe.json").read_text())
        port_source = fetch_extract(port["source"], work)
        for name in ("pycosat.c", "picosat.c", "picosat.h"):
            shutil.copyfile(port_source / name, source / "Modules" / name)
        setup_text += "*static*\n" + port["setup"] + "\n"
        native_ports.append(port)
        port_sources[port["name"]] = port_source
    if args.with_pillow:
        directory = Path(__file__).parent.parent / "pillow"
        port = json.loads((directory / "recipe.json").read_text())
        if port["target_development"] != {"cpython": recipe["version"]}:
            raise ValueError("Pillow's pinned CPython development configuration does not match")
        for patch in port["patches"]:
            apply_patch(source, directory / patch["file"], patch["sha256"])
        pillow_source = fetch_extract(port["source"], work)
        dependencies, links = dependency_prefix(
            port["target_dependencies"], providers, prefix, recipe["target_profile"]
        )
        target_artifacts.update(dependencies)
        archives, inputs = build_pillow(port, pillow_source, source, guest, sdk, work, dependencies, prefix, run)
        for name, archive in archives.items():
            setup_text += f"*static*\n{name} {archive} {' '.join(links)} -lm\n"
            link_consumers[name] = {
                "inputs": inputs,
                "archive_sha256": file_hash(archive),
                "dependency_artifacts": {key: item["artifact_sha256"] for key, item in dependencies.items()},
                "link_inputs": [str(archive), *links, "-lm"],
            }
        native_ports.append(port)
        port_sources[port["name"]] = pillow_source
    if args.with_numpy:
        directory = Path(__file__).parent.parent / "numpy"
        port = json.loads((directory / "recipe.json").read_text())
        check_build_scripts(port, directory)
        if port["dependencies"] != {"cpython": recipe["version"], "wasi_sdk": recipe["sdk"]["version"]}:
            raise ValueError("NumPy's pinned toolchain does not match the CPython recipe")
        for patch in port["patches"]:
            if patch["file"] == "cpython-qualified-builtins.patch":
                apply_patch(source, directory / patch["file"], patch["sha256"])
        port_source = fetch_extract(port["source"], work)
        archives, support = build_numpy(port, port_source, source, guest, sdk, work, args.jobs, run)
        libraries = (
            " ".join(str(path) for path in [*archives.values(), *support])
            + " "
            + " ".join(toolchain["profile"]["cpp_link_flags"])
            + " -lc-printscan-long-double"
        )
        setup_text += (
            "*static*\n"
            + "\n".join(
                name + " " + (libraries if index == 0 else str(archives[name]))
                for index, name in enumerate(port["builtin_modules"])
            )
            + "\n"
        )
        native_ports.append(port)
        port_sources[port["name"]] = port_source
    if not setup.exists() or setup.read_text() != setup_text:
        setup.write_text(setup_text)
    run(["make", f"-j{args.jobs}"], guest, guest_env, work / "wasi-make.log")
    root = work / "rootfs"
    if root.exists():
        shutil.rmtree(root)
    binary = root / "usr/bin/python3.wasm"
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(guest / "python.wasm", binary)
    subprocess.run([str(sdk / "bin/llvm-strip"), str(binary)], check=True)
    binary.chmod(0o755)
    stdlib = root / "usr/lib/python3.13"
    shutil.copytree(
        source / "Lib",
        stdlib,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "test", "tests", "idlelib", "tkinter", "ensurepip"),
    )
    (stdlib / "site-packages").mkdir(exist_ok=True)
    (stdlib / "lib-dynload").mkdir(exist_ok=True)
    for config in (guest / "build").glob("lib.*/_sysconfigdata_*.py"):
        shutil.copyfile(config, stdlib / config.name)
    shutil.copyfile(source / "LICENSE", root / "CPYTHON-LICENSE")
    if target_artifacts:
        shutil.copyfile(prefix / "licenses/zlib.txt", root / "ZLIB-LICENSE")
    if args.with_pillow:
        for name in ("libjpeg-turbo.txt", "README.ijg", "FTL.TXT", "GPLv2.TXT", "LICENSE.TXT"):
            shutil.copyfile(prefix / "licenses" / name, root / name)
    for port in native_ports:
        port_source = port_sources[port["name"]]
        license_file = "LICENSE.txt" if port["name"] == "numpy" else "LICENSE"
        shutil.copyfile(port_source / license_file, root / f"{port['name'].upper()}-LICENSE")
        if port["name"] == "numpy":
            install_numpy(work / "numpy-build", stdlib / "site-packages")
            install_numpy_notices(port_source, root)
        elif port["name"] == "pillow":
            install_pillow(port_source, stdlib / "site-packages")
        install_metadata(port, port_source, root)
        if port["name"] == "pycosat":
            notice = (port_source / "picosat.c").read_text().split("*/", 1)[0]
            (root / "PICOSAT-LICENSE").write_text(notice.strip("/*\n") + "\n")
    builtin_modules = sorted(set(re.findall(r'\{"([^" ]+)",', (guest / "Modules/config.c").read_text())))
    files = {
        "/" + path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "recipe": recipe,
        "site_packages": "/usr/lib/python3.13/site-packages",
        "builtin_modules": builtin_modules,
        "native_ports": native_ports,
        "native_libraries": target_artifacts,
        "link_consumers": link_consumers,
        "build_profile": profile,
        "build_profile_sha256": profile_hash,
        "files": files,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Guest image: {root}\nManifest: {work / 'manifest.json'}")


if __name__ == "__main__":
    main()
