"""Build a pinned, static CPython WASI command and its virtual filesystem image.

This is trusted host build tooling. The resulting command receives capabilities
only through shellsim's WASI adapter; it cannot invoke the host helper interpreter.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/shellsim-cpython"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--with-pycosat", action="store_true", help="statically link the pinned native pycosat port")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    recipe = json.loads(Path(__file__).with_name("recipe.json").read_text())
    source = fetch_extract(recipe["source"], work)
    sdk = fetch_extract(recipe["sdk"], work)
    host = work / "host-build"
    guest = work / "wasi-build"
    host.mkdir(exist_ok=True)
    guest.mkdir(exist_ok=True)
    env = dict(os.environ, SOURCE_DATE_EPOCH="1756857600")
    if not (host / "Makefile").exists():
        run([str(source / "configure"), "--without-ensurepip"], host, env, work / "host-configure.log")
    run(["make", f"-j{args.jobs}"], host, env, work / "host-make.log")
    sysroot = sdk / "share" / "wasi-sysroot"
    guest_env = dict(
        env,
        CC=f"{sdk / 'bin/clang'} --sysroot={sysroot}",
        AR=str(sdk / "bin/llvm-ar"),
        RANLIB=str(sdk / "bin/llvm-ranlib"),
        CONFIG_SITE=str(source / "Tools/wasm/config.site-wasm32-wasi"),
        CFLAGS="-O2 -g0",
        PKG_CONFIG_PATH="",
        PKG_CONFIG_LIBDIR=str(sysroot / "lib/pkgconfig"),
    )
    if not (guest / "Makefile").exists():
        build = subprocess.check_output([str(source / "config.guess")], text=True).strip()
        run(
            [
                str(source / "configure"),
                "--host=wasm32-wasip1",
                f"--build={build}",
                f"--with-build-python={host / 'python'}",
                "--prefix=/usr",
                "--without-ensurepip",
                "--disable-test-modules",
            ],
            guest,
            guest_env,
            work / "wasi-configure.log",
        )
    native_ports = []
    setup = guest / "Modules/Setup.local"
    setup_text = "# shellsim static native ports\n"
    if args.with_pycosat:
        port = json.loads(Path(__file__).parent.parent.joinpath("pycosat/recipe.json").read_text())
        port_source = fetch_extract(port["source"], work)
        for name in ("pycosat.c", "picosat.c", "picosat.h"):
            shutil.copyfile(port_source / name, source / "Modules" / name)
        setup_text += "*static*\n" + port["setup"] + "\n"
        native_ports.append(port)
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
    for port in native_ports:
        port_source = work / f"{port['name']}-{port['version']}"
        shutil.copyfile(port_source / "LICENSE", root / f"{port['name'].upper()}-LICENSE")
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
        "files": files,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Guest image: {root}\nManifest: {work / 'manifest.json'}")


if __name__ == "__main__":
    main()
