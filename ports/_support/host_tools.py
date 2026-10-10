"""Produce explicit Meson and Python generator receipts without changing tools.

Pinned upstream archives are checked against the installed code. Receipts also
bind the Python environment and base interpreter tree used by each entrypoint.
Production writes fresh descriptors; existing receipts and build trees stay intact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import tempfile
import zipfile
from pathlib import Path

from ports._support.build import apply_patch
from ports._support.store import extract, fetch, file_hash, relative_path

_DEFINITION = Path(__file__).with_name("host-tools.json")
_PYTHON_TOOLS = {
    "python": "bin/python",
    "cython": "bin/cython",
    "f2py": "bin/f2py",
    "pybind11-config": "bin/pybind11-config",
}


def definition() -> dict:
    """Read the in-tree source pins required by receipt admission."""
    return json.loads(_DEFINITION.read_text())


def inventory(root: Path, *, symlinks: dict[str, str] | None = None) -> dict[str, str]:
    """Bound regular files in an explicitly supplied host code tree."""
    if not root.is_dir():
        raise ValueError("host code root is not a directory")
    files = {}
    size = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            if symlinks is None or not path.resolve().is_relative_to(root.resolve()) or not path.resolve().exists():
                raise ValueError("host code tree contains an unsupported symlink")
            symlinks[path.relative_to(root).as_posix()] = os.readlink(path)
            continue
        if path.is_dir():
            continue
        size += path.stat().st_size
        if size > 2 * 1024**3 or len(files) >= 100_000:
            raise ValueError("host code tree exceeds inventory bounds")
        files[path.relative_to(root).as_posix()] = file_hash(path)
    return files


def verify_read_only(root: Path) -> None:
    """Reject host code trees that ordinary interpreter use could mutate."""
    for path in (root, *root.rglob("*")):
        if not path.is_symlink() and path.stat().st_mode & 0o222:
            raise ValueError("base Python code tree is writable")


def make_read_only(root: Path) -> None:
    """Preserve execute bits while removing write permissions from a snapshot."""
    for path in (*root.rglob("*"), root):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)


def snapshot_python(base: Path, environments: tuple[Path, ...], destination: Path) -> tuple[Path, tuple[Path, ...]]:
    """Copy explicit Python inputs coherently, rebinding venv launch paths.

    These private copies keep shared uv interpreter use from invalidating host
    receipts. Existing trees are preserved; no interpreter is executed here.
    """
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    base = base.resolve()
    private_base = destination.resolve() / "base"
    aliases = {}
    expected = inventory(base, symlinks=aliases)
    shutil.copytree(base, private_base, symlinks=True)
    copied_aliases = {}
    if inventory(private_base, symlinks=copied_aliases) != expected or aliases != copied_aliases:
        raise ValueError("base Python changed while copying")
    make_read_only(private_base)
    copied = []
    for index, original in enumerate(environments):
        original = original.resolve()
        environment = destination.resolve() / ("environment-" + str(index))
        aliases = {}
        expected = inventory(original, symlinks=aliases)
        shutil.copytree(original, environment, symlinks=True)
        copied_aliases = {}
        if inventory(environment, symlinks=copied_aliases) != expected or aliases != copied_aliases:
            raise ValueError("Python environment changed while copying")
        config_path = environment / "pyvenv.cfg"
        config = dict(line.split(" = ", 1) for line in config_path.read_text().splitlines())
        if Path(config["home"]).resolve() != base / "bin":
            raise ValueError("Python environment has a different base")
        config["home"] = str(private_base / "bin")
        config_path.write_text("".join(name + " = " + value + "\n" for name, value in config.items()))
        for script in (environment / "bin").iterdir():
            if script.is_file() and not script.is_symlink() and not script.name.startswith("python"):
                data = script.read_bytes()
                rewritten = data.replace(str(original).encode(), str(environment).encode())
                if rewritten != data:
                    script.write_bytes(rewritten)
        make_read_only(environment)
        copied.append(environment)
    return private_base, tuple(copied)


def python_closure(environment: Path, base: Path) -> dict:
    """Bind a venv's explicit base interpreter, stdlib and extension modules."""
    environment, base = environment.resolve(), base.resolve()
    config = dict(line.split(" = ", 1) for line in (environment / "pyvenv.cfg").read_text().splitlines())
    if Path(config["home"]).resolve() != base / "bin" or config["include-system-site-packages"] != "false":
        raise ValueError("Python environment uses a different base or system packages")
    if file_hash(environment / "bin/python") != file_hash(base / "bin/python3.13"):
        raise ValueError("Python environment interpreter differs from its base")
    version = re.search(
        r'^#define\s+PY_VERSION\s+"([^"\n]+)"',
        (base / "include/python3.13/patchlevel.h").read_text(),
        re.MULTILINE,
    )
    if version is None:
        raise ValueError("base Python version header is missing")
    version = version.group(1)
    if version != definition()["python"]["version"]:
        raise ValueError("base Python version differs")
    verify_read_only(base)
    aliases = {}
    files = inventory(base, symlinks=aliases)
    return {"root": str(base), "files": files, "symlinks": aliases}


def verify_python_closure(environment: Path, closure: dict) -> None:
    """Recheck every base Python file before admitting a package tool."""
    if set(closure) != {"root", "files", "symlinks"}:
        raise ValueError("base Python receipt fields differ")
    if python_closure(environment, Path(closure["root"])) != closure:
        raise ValueError("base Python code differs from its receipt")


def _receipt(name: str, version: str, root: Path, executable: str, source: dict) -> dict:
    aliases = {}
    files = inventory(root, symlinks=aliases)
    return {
        "schema_version": 2,
        "kind": "host-tool-files",
        "name": name,
        "version": version,
        "root": str(root.resolve()),
        "executable": executable,
        "source": source,
        "files": files,
        "symlinks": aliases,
    }


def _verify_wheel(archive: Path, environment: Path) -> None:
    site = environment / "lib/python3.13/site-packages"
    with zipfile.ZipFile(archive) as wheel:
        for member in wheel.infolist():
            if member.is_dir() or member.filename.endswith(".dist-info/RECORD"):
                continue
            name = relative_path(member.filename)
            if ".data/" in name:
                raise ValueError("host wheel data relocation is unsupported")
            path = site / name
            if not path.is_file() or path.read_bytes() != wheel.read(member):
                raise ValueError("installed host wheel differs: " + name)


def meson_receipt(
    root: Path, environment: Path, base: Path, cache: Path, *, offline: bool = False, materialize: bool = False
) -> dict:
    """Reproduce pinned patched Meson and verify the existing consumed tree."""
    pinned = definition()["meson"]
    archive = fetch(pinned["source"], cache, offline=offline)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".meson-proof-", dir=cache) as temporary:
        source = extract(archive, Path(temporary) / "source", subdirectory=pinned["source"]["subdirectory"])
        if ("version = '" + pinned["version"] + "'") not in (source / "mesonbuild/coredata.py").read_text():
            raise ValueError("vendored Meson version differs")
        patch = pinned["patch"]
        for name, digest in patch["inputs"].items():
            if file_hash(source / name) != digest:
                raise ValueError("Meson patch input differs")
        apply_patch(source, _DEFINITION.parent / patch["file"], patch["sha256"])
        expected = {name: file_hash(source / name) for name in ("COPYING", "meson.py")}
        expected.update({"mesonbuild/" + name: digest for name, digest in inventory(source / "mesonbuild").items()})
        launcher = (
            "#!/usr/bin/sh\nexec "
            + shlex.quote(str(environment.resolve() / "bin/python"))
            + " -I -B "
            + shlex.quote(str(root.resolve() / "meson.py"))
            + ' "$@"\n'
        )
        if materialize:
            if root.exists():
                raise FileExistsError(root)
            root.mkdir(parents=True)
            for name in ("COPYING", "meson.py"):
                shutil.copyfile(source / name, root / name)
            shutil.copytree(source / "mesonbuild", root / "mesonbuild")
            (root / "bin").mkdir()
            (root / "bin/meson").write_text(launcher)
            (root / "bin/meson").chmod(0o755)
        if (root / "bin/meson").read_text() != launcher:
            raise ValueError("Meson launcher differs from the explicit interpreter")
        expected["bin/meson"] = file_hash(root / "bin/meson")
        if inventory(root) != expected:
            raise ValueError("installed Meson differs from pinned patched source")
    aliases = {}
    interpreter = {
        "root": str(environment.resolve()),
        "files": inventory(environment, symlinks=aliases),
        "symlinks": aliases,
        "base_python": python_closure(environment, base),
    }
    return _receipt("meson", pinned["version"], root, "bin/meson", {"definition": pinned, "interpreter": interpreter})


def generator_receipts(environment: Path, base: Path, cache: Path, *, offline: bool = False) -> dict[str, dict]:
    """Verify pinned installed wheels and record all generator code and Python."""
    pinned = definition()["python"]
    for package in pinned["packages"].values():
        _verify_wheel(fetch(package, cache, offline=offline), environment)
    source = {"definition": pinned, "base_python": python_closure(environment, base)}
    versions = {
        "python": pinned["version"],
        "cython": pinned["packages"]["Cython"]["version"],
        "f2py": pinned["packages"]["numpy"]["version"],
        "pybind11-config": pinned["packages"]["pybind11"]["version"],
    }
    return {
        name: _receipt(name, versions[name], environment, executable, source)
        for name, executable in _PYTHON_TOOLS.items()
    }


def verify_source(root: Path, producer: dict, name: str) -> None:
    """Require current source pins and the complete interpreter closure."""
    source = producer["source"]
    pinned = definition()
    if name == "meson":
        if producer["version"] != pinned["meson"]["version"] or producer["executable"] != "bin/meson":
            raise ValueError("Meson entrypoint profile differs")
        if set(source) != {"definition", "interpreter"} or source["definition"] != pinned["meson"]:
            raise ValueError("Meson source definition differs")
        patch = pinned["meson"]["patch"]
        if file_hash(_DEFINITION.parent / patch["file"]) != patch["sha256"]:
            raise ValueError("Meson producer patch differs")
        for path, digest in patch["outputs"].items():
            if file_hash(root / path) != digest:
                raise ValueError("Meson patched code differs")
        interpreter = source["interpreter"]
        if set(interpreter) != {"root", "files", "symlinks", "base_python"}:
            raise ValueError("Meson interpreter fields differ")
        environment = Path(interpreter["root"])
        aliases = {}
        files = inventory(environment, symlinks=aliases)
        if files != interpreter["files"] or aliases != interpreter["symlinks"]:
            raise ValueError("Meson interpreter environment differs")
        launcher = (
            "#!/usr/bin/sh\nexec "
            + shlex.quote(str(environment.resolve() / "bin/python"))
            + " -I -B "
            + shlex.quote(str(root.resolve() / "meson.py"))
            + ' "$@"\n'
        )
        if (root / "bin/meson").read_text() != launcher:
            raise ValueError("Meson launcher interpreter differs")
        verify_python_closure(environment, interpreter["base_python"])
        return
    if (
        name not in _PYTHON_TOOLS
        or set(source) != {"definition", "base_python"}
        or source["definition"] != pinned["python"]
    ):
        raise ValueError("Python generator source definition differs")
    versions = {
        "python": pinned["python"]["version"],
        "cython": pinned["python"]["packages"]["Cython"]["version"],
        "f2py": pinned["python"]["packages"]["numpy"]["version"],
        "pybind11-config": pinned["python"]["packages"]["pybind11"]["version"],
    }
    if producer["version"] != versions[name] or producer["executable"] != _PYTHON_TOOLS[name]:
        raise ValueError("Python generator entrypoint profile differs")
    verify_python_closure(root, source["base_python"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--python-environment", type=Path, required=True)
    parser.add_argument("--meson-root", type=Path, required=True)
    parser.add_argument("--meson-python-environment", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--private-python", type=Path, help="Create private read-only base and environment copies at a new directory"
    )
    parser.add_argument(
        "--materialize-meson",
        action="store_true",
        help="Create the pinned Meson tree at a previously absent --meson-root",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.private_python is not None:
        args.base_python, environments = snapshot_python(
            args.base_python, (args.python_environment, args.meson_python_environment), args.private_python
        )
        args.python_environment, args.meson_python_environment = environments
    receipts = generator_receipts(args.python_environment, args.base_python, args.cache, offline=args.offline)
    receipts["meson"] = meson_receipt(
        args.meson_root,
        args.meson_python_environment,
        args.base_python,
        args.cache,
        offline=args.offline,
        materialize=args.materialize_meson,
    )
    value = json.loads(args.cohort.read_text())
    # Resolve every old binding before moving the descriptor to its new directory.
    for name in ("sdk", "llvm", "sysroot", "cpython", "runtime"):
        if value[name] is not None:
            for field in ("root", "manifest"):
                value[name][field] = str((args.cohort.parent / value[name][field]).resolve())
    for item in value["host_tools"].values():
        item["path"] = str((args.cohort.parent / item["path"]).absolute())
        if item["receipt"] is not None:
            item["receipt"]["path"] = str((args.cohort.parent / item["receipt"]["path"]).resolve())
    args.output.mkdir(parents=True)
    for name, receipt in receipts.items():
        path = (args.output / (name + "-receipt.json")).resolve()
        path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        item = value["host_tools"][name]
        executable = Path(receipt["root"]) / receipt["executable"]
        item["path"] = str(executable)
        item["sha256"] = file_hash(executable)
        item["receipt"] = {"path": str(path), "sha256": file_hash(path)}
    descriptor = args.output / "cohort.json"
    descriptor.write_text(json.dumps(value, sort_keys=True) + "\n")
    from ports._support.cohort import load_cohort

    cohort = load_cohort(descriptor)
    print(str(descriptor.resolve()), cohort.identity)


if __name__ == "__main__":
    main()
