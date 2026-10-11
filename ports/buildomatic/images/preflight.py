"""Verify image inputs and native runtime paths without compiling source.

The prepared host seed remains partial. Final image provenance belongs to fresh
native receipts staged after publication at the retained task root.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/opt/buildomatic")

from ports._support import sdk
from ports._support.host_tools import inventory, verify_read_only
from ports._support.sdk_products import file_hash


def query(*argv: str) -> str:
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def verify() -> dict:
    inputs = json.loads(Path("/opt/buildomatic/inputs.json").read_text())
    root = Path(inputs["prepared_root"])
    seed = root / "host-seed.prepared.json"
    if file_hash(seed) != inputs["prepared_seed_sha256"]:
        raise ValueError("prepared seed differs")
    admitted = sdk.load_seed(seed)
    if admitted.compiler_tools or admitted.python_helper is not None:
        raise ValueError("image must retain the unlaunched prepared seed")
    for name, binding in inputs["system_bindings"].items():
        path = Path(binding["path"])
        alias = os.readlink(path) if path.is_symlink() else None
        if file_hash(path) != binding["sha256"] or alias != binding["symlink"]:
            raise ValueError("caller utility differs: " + name)
    for archive in json.loads((root / "prepared.json").read_text())["archives"].values():
        if file_hash(Path(archive["path"])) != archive["sha256"]:
            raise ValueError("source archive differs")
    helper = Path("/opt/buildomatic/python-helper")
    verify_read_only(helper)
    helper_aliases = {}
    helper_files = inventory(helper, symlinks=helper_aliases)
    helper_digest = hashlib.sha256(
        json.dumps({"files": helper_files, "symlinks": helper_aliases}, sort_keys=True).encode()
    ).hexdigest()
    if helper_digest != inputs["python_helper_inventory_sha256"]:
        raise ValueError("native helper inventory differs")
    helper_version = query(str(helper / "bin/python3.13"), "-I", "-B", "--version")
    if helper_version != "Python " + inputs["python_helper_version"]:
        raise ValueError("native build helper version differs")
    query(str(helper / "bin/python3.13"), "-I", "-B", "-c", "import _ssl,_ctypes,zlib,_bz2,_lzma")
    sccache = Path("/usr/local/bin/sccache")
    if file_hash(sccache) != inputs["sccache"]["sha256"]:
        raise ValueError("sccache differs")
    sccache_version = query(str(sccache), "--version")
    if sccache_version != "sccache " + inputs["sccache"]["version"]:
        raise ValueError("sccache version differs")
    native = Path("/opt/buildomatic/native")
    cc, cxx = native / "gcc/bin/gcc", native / "gcc/bin/g++"
    cc1, cc1plus = (
        Path(query(str(cc), "-print-prog-name=cc1")).resolve(),
        Path(query(str(cxx), "-print-prog-name=cc1plus")).resolve(),
    )
    if any(not path.is_file() or not path.is_relative_to(native / "gcc") for path in (cc1, cc1plus)):
        raise ValueError("GCC does not use its private support tree")
    cmake = native / "cmake/bin/cmake"
    cmake_version = query(str(cmake), "--version").splitlines()[0]
    if cmake_version != "cmake version " + inputs["cmake"]["version"]:
        raise ValueError("CMake version differs")
    subprocess.run([str(cmake), "--help-module", "FindThreads"], stdout=subprocess.DEVNULL, check=True, timeout=30)
    generator_versions = query(
        str(root / "host/python/environment-0/bin/python"),
        "-I",
        "-B",
        "-c",
        "import sys,numpy,Cython,pybind11; print(sys.version.split()[0],numpy.__version__,Cython.__version__,pybind11.__version__)",
    )
    meson_version = query(str(root / "host/meson/bin/meson"), "--version")
    # These checks load dynamic libraries but never launch a compiler invocation.
    runtime = {}
    for executable in (
        cc1.parent / "gcc",
        cc1.parent / "g++",
        cc1,
        cc1plus,
        cmake,
        native / "ninja/bin/ninja",
    ):
        linked = query("ldd", str(executable))
        if "not found" in linked:
            raise ValueError("native runtime dependency missing: " + str(executable))
        runtime[str(executable)] = linked
    pkgconf = native / "pkgconf"
    linked = query("env", "LD_LIBRARY_PATH=" + str(pkgconf / "lib"), "ldd", str(pkgconf / "lib/pkgconf"))
    if "not found" in linked or str(pkgconf / "lib/libpkgconf.so.3") not in linked:
        raise ValueError("pkgconf does not use its packaged library")
    runtime[str(pkgconf / "lib/pkgconf")] = linked
    sdk.load_seed(seed)
    if (root / "store/sdk-work").exists() or (root / "launch.json").exists():
        raise ValueError("image unexpectedly contains a bootstrap attempt")
    gcc_lib = cc1.parent
    return {
        "state": "native_preflight_passed",
        "compile_started": False,
        "helper_version": helper_version,
        "sccache_version": sccache_version,
        "cmake_version": cmake_version,
        "gcc_version": query(str(cc), "--version").splitlines()[0],
        "ninja_version": query(str(native / "ninja/bin/ninja"), "--version"),
        "pkgconf_version": query(str(native / "pkgconf/bin/pkg-config"), "--version"),
        "generator_versions": generator_versions,
        "meson_version": meson_version,
        "helper_inventory_sha256": helper_digest,
        "installed_packages": query("dpkg-query", "-W", "-f=${binary:Package}=${Version}\n").splitlines(),
        "prepared_seed_sha256": file_hash(seed),
        "runtime": runtime,
        "native_config": {
            "tools": {
                "cc": str(cc),
                "cxx": str(cxx),
                "cmake": str(cmake),
                "ninja": str(native / "ninja/bin/ninja"),
                "pkg-config": str(native / "pkgconf/bin/pkg-config"),
            },
            "system_bindings": inputs["system_bindings"],
            "gcc_lib": str(gcc_lib),
            "cmake_share": str(native / "cmake/share/cmake-3.31"),
            "python_helper_base": str(helper),
        },
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True, indent=2))
