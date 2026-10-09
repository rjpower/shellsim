"""Build the pinned Clang toolchain for threaded WASI ports."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ports._support.build import apply_patch, check_build_scripts
from ports.toolchain.llvm.build import digest, run

TARGETS = ("clang", "lld", "llc", "llvm-ar", "llvm-nm", "llvm-objcopy")


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_workspace(path, value):
    """Publish recoverable workspace state without truncating its last receipt."""
    temporary = path.with_name(path.name + ".preparing")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def configure_and_build(commands, output, workspace, receipt, attempts, environment):
    """Recover configuration or Ninja interruption under the workspace lock."""
    cache = output / "CMakeCache.txt"
    phase = workspace.get("phase", "ready")
    if phase not in {"ready", "configuring", "building"}:
        raise ValueError("unsupported LLVM workspace phase")
    if phase == "ready" and "configuration_sha256" in workspace:
        if digest(cache) != workspace["configuration_sha256"]:
            raise ValueError("LLVM workspace configuration changed outside its producer")
    attempt = len(list(attempts.glob("configure-*.log")))
    workspace["phase"] = "configuring"
    write_workspace(receipt, workspace)
    run(commands[0], attempts / f"configure-{attempt}.log", environment)
    workspace["configuration_sha256"] = digest(cache)
    workspace["phase"] = "building"
    write_workspace(receipt, workspace)
    run(commands[1], attempts / f"build-{attempt}.log", environment)
    workspace["configuration_sha256"] = digest(cache)
    workspace["phase"] = "ready"
    write_workspace(receipt, workspace)


def verify_source(archive, source, recipe, directory):
    """Verify retained sources against the pinned archive and applied patches.

    Only files changed by patches need a temporary reference tree. No generated
    Ninja graph or historical build inventory becomes a source of authority.
    """
    changed = {name for patch in recipe["patches"] for name in patch["inputs"]}
    expected = {}
    with tempfile.TemporaryDirectory(dir=source.parent) as temporary:
        reference = Path(temporary)
        with tarfile.open(archive) as upstream:
            for member in upstream:
                pieces = Path(member.name).parts
                if len(pieces) < 2 or member.isdir():
                    continue
                name = str(Path(*pieces[1:]))
                if member.issym():
                    path = source / name
                    if not path.is_symlink() or os.readlink(path) != member.linkname:
                        raise ValueError("retained source symlink differs: " + name)
                    expected[name] = None
                    continue
                if not member.isfile():
                    raise ValueError("unsupported LLVM source archive entry")
                stream = upstream.extractfile(member)
                if name in changed:
                    path = reference / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("wb") as output:
                        shutil.copyfileobj(stream, output)
                else:
                    value = hashlib.sha256()
                    for chunk in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
                        value.update(chunk)
                    expected[name] = value.hexdigest()
        for patch in recipe["patches"]:
            for name, expected_input in patch["inputs"].items():
                if digest(reference / name) != expected_input:
                    raise ValueError("LLVM patch source identity differs: " + name)
            apply_patch(reference, directory / patch["file"], patch["sha256"])
        for name in changed:
            expected.pop(name, None)
        expected.update(
            {str(path.relative_to(reference)): digest(path) for path in reference.rglob("*") if path.is_file()}
        )
    actual = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file() or path.is_symlink()}
    if actual != set(expected):
        raise ValueError("retained LLVM source contains unrecorded or missing files")
    for name, expected_hash in expected.items():
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("retained source escapes its root")
        if expected_hash is not None and digest(path) != expected_hash:
            raise ValueError("retained LLVM source differs: " + name)
    return identity_hash(expected)


def verify_product(prefix, identity):
    """Return an exact sealed product, without executing a compiler."""
    manifest = json.loads((prefix / "manifest.json").read_text())
    if manifest["identity"] != identity:
        raise ValueError("sealed compiler identity differs")
    for name, expected in manifest["artifacts"].items():
        path = prefix / name
        if not path.resolve().is_relative_to(prefix.resolve()) or digest(path) != expected:
            raise ValueError("sealed compiler bytes differ: " + name)
    for name, target in manifest["symlinks"].items():
        path = prefix / name
        if not path.is_symlink() or os.readlink(path) != target or not path.resolve().is_relative_to(prefix.resolve()):
            raise ValueError("sealed compiler alias differs: " + name)
    for name in TARGETS:
        if (prefix / "bin" / name).stat().st_mode & 0o777 != 0o755:
            raise ValueError("sealed compiler executable mode differs: " + name)
    actual = {str(p.relative_to(prefix)) for p in prefix.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(manifest["artifacts"]) | set(manifest["symlinks"]) | {"manifest.json"}:
        raise ValueError("sealed compiler contains unrecorded or missing files")
    return prefix


def build(archive, cc, cxx, cmake, ninja, work):
    """Serialize workspace mutation while keeping sealed products immutable."""
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".compiler.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return build_locked(archive, cc, cxx, cmake, ninja, work)


def build_locked(archive, cc, cxx, cmake, ninja, work):
    directory = Path(__file__).resolve().parent
    recipe = json.loads((directory / "compiler-recipe.json").read_text())
    check_build_scripts(recipe, directory)
    tools = {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in {"cc": cc, "cxx": cxx, "cmake": cmake, "ninja": ninja}.items()
    }
    compatibility = {"source": recipe["source"], "patches": recipe["patches"], "tools": tools}
    work.mkdir(parents=True, exist_ok=True)
    receipt = work / "workspace.json"
    workspace = {"schema_version": 1, "compatibility": compatibility}
    if receipt.exists():
        workspace = json.loads(receipt.read_text())
        if workspace["compatibility"] != compatibility:
            raise ValueError("LLVM workspace source, patches or host tools differ; select another workspace")
    elif (work / "prefix/manifest.json").exists():
        old = json.loads((work / "prefix/manifest.json").read_text())["identity"]
        previous = {"source": old["recipe"]["source"], "patches": old["recipe"]["patches"], "tools": old["tools"]}
        if previous != compatibility:
            raise ValueError("existing compiler workspace inputs differ")
    elif any(path.name not in {".compiler.lock", "workspace.json.preparing"} for path in work.iterdir()):
        raise ValueError("existing LLVM workspace has no input receipt")
    write_workspace(receipt, workspace)
    if digest(archive) != recipe["source"]["sha256"]:
        raise ValueError("LLVM source archive identity differs")
    with tarfile.open(archive) as upstream:
        root_name = Path(next(iter(upstream)).name).parts[0]
    source = work / "source" / root_name
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C", "SOURCE_DATE_EPOCH": "1756857600"}
    output = work / "build"
    commands = [
        [
            str(cmake),
            "-G",
            "Ninja",
            "-S",
            str(source / "llvm"),
            "-B",
            str(output),
            "-DCMAKE_MAKE_PROGRAM=" + str(ninja),
            "-DCMAKE_C_COMPILER=" + str(cc),
            "-DCMAKE_CXX_COMPILER=" + str(cxx),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLVM_ENABLE_PROJECTS=clang;lld",
            "-DLLVM_APPEND_VC_REV=OFF",
            "-DCLANG_INCLUDE_TESTS=OFF",
            "-DLLVM_TARGETS_TO_BUILD=WebAssembly",
            "-DLLVM_INCLUDE_TESTS=OFF",
            "-DLLVM_INCLUDE_EXAMPLES=OFF",
            "-DLLVM_INCLUDE_BENCHMARKS=OFF",
            "-DLLVM_ENABLE_ZLIB=OFF",
            "-DLLVM_ENABLE_ZSTD=OFF",
            "-DLLVM_ENABLE_LIBXML2=OFF",
            "-DLLVM_ENABLE_BINDINGS=OFF",
            "-DLLVM_PARALLEL_LINK_JOBS=1",
        ],
        [
            str(cmake),
            "--build",
            str(output),
            "--target",
            *TARGETS,
            "--parallel",
            "4",
        ],
    ]
    identity = {"recipe": recipe, "tools": tools, "commands": commands}
    key = identity_hash(identity)
    prefix = work / "products" / key
    if prefix.exists():
        return verify_product(prefix, identity)
    if not (work / "source").exists():
        preparing = work / "source-preparing"
        if preparing.exists():
            shutil.rmtree(preparing)
        preparing.mkdir()
        with tarfile.open(archive) as upstream:
            upstream.extractall(preparing, filter="data")
        roots = list(preparing.iterdir())
        if len(roots) != 1 or roots[0].name != root_name or not roots[0].is_dir():
            raise ValueError("LLVM archive must contain one source root")
        for item in recipe["patches"]:
            for name, expected in item["inputs"].items():
                if digest(roots[0] / name) != expected:
                    raise ValueError("LLVM patch source identity differs: " + name)
            apply_patch(roots[0], directory / item["file"], item["sha256"])
        preparing.rename(work / "source")
    source_hash = verify_source(archive, source, recipe, directory)
    attempts = work / "attempts" / key
    attempts.mkdir(parents=True, exist_ok=True)
    configure_and_build(commands, output, workspace, receipt, attempts, environment)
    staging = work / "products" / (key + ".preparing")
    if staging.exists():
        shutil.rmtree(staging)
    prefix = staging
    (prefix / "bin").mkdir(parents=True)
    (prefix / "licenses").mkdir()
    for name in ("clang", "lld", "llc", "llvm-ar", "llvm-nm", "llvm-objcopy"):
        shutil.copyfile(output / "bin" / name, prefix / "bin" / name)
        (prefix / "bin" / name).chmod(0o755)
    for alias, name in {
        "clang++": "clang",
        "wasm-ld": "lld",
        "llvm-ranlib": "llvm-ar",
        "llvm-strip": "llvm-objcopy",
    }.items():
        (prefix / "bin" / alias).symlink_to(name)
    shutil.copytree(output / "lib/clang", prefix / "lib/clang")
    shutil.copyfile(source / "LICENSE.TXT", prefix / "licenses/LLVM-LICENSE.txt")
    manifest = {
        "schema_version": 1,
        "identity": identity,
        "source_tree_sha256": source_hash,
        "configuration_sha256": workspace["configuration_sha256"],
        "commands": commands,
        "build_limits": recipe["build_limits"],
        "artifacts": {
            str(path.relative_to(prefix)): digest(path)
            for path in sorted(prefix.rglob("*"))
            if path.is_file() and not path.is_symlink()
        },
        "symlinks": {
            str(path.relative_to(prefix)): os.readlink(path) for path in sorted(prefix.rglob("*")) if path.is_symlink()
        },
    }
    (prefix / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    destination = work / "products" / key
    prefix.rename(destination)
    return verify_product(destination, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "cc", "cxx", "cmake", "ninja", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        build(
            *(getattr(args, name).resolve() for name in ("archive", "cc", "cxx", "cmake", "ninja", "work")),
        )
    )


if __name__ == "__main__":
    main()
