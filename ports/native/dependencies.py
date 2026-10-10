"""Select and verify a small, explicit closure of target library artifacts.

This trusted build utility never searches host libraries. Each consumer receives
only exported files from its declared target dependencies; toolchain libraries
come from the pinned WASI sysroot. Artifacts are content-addressed and immutable.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath


def digest(value):
    """Hash canonical JSON so paths outside an artifact do not affect its identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_environment(sdk):
    """Exclude ambient compiler/search overrides from the target build environment."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "SOURCE_DATE_EPOCH": "1756857600",
        "LC_ALL": "C",
        "CC": str(sdk / "bin/clang"),
        "AR": str(sdk / "bin/llvm-ar"),
        "RANLIB": str(sdk / "bin/llvm-ranlib"),
        "PKG_CONFIG_PATH": "",
        "PKG_CONFIG_LIBDIR": "",
    }


def target_profile(recipe):
    """Read a versioned profile; callers use its compiler and final link flags."""
    return json.loads(
        (
            Path(__file__).resolve().parents[2] / "tests/fixtures/wasm/dynamic" / (recipe["target_profile"] + ".json")
        ).read_text()
    )


def toolchain_identity(recipe, sdk):
    """Bind the declared SDK archive to the tools and sysroot actually consumed."""
    profile = target_profile(recipe)
    if (
        profile["sdk_sha256"] != recipe["sdk"]["sha256"]
        or profile["target"] != recipe["target"]
        or profile["sdk_version"] != recipe["sdk"]["version"]
    ):
        raise ValueError("Target profile conflicts with the selected toolchain")
    compiler_version = subprocess.check_output([str(sdk / "bin/clang"), "--version"], text=True)
    # SDK 24 predates VERSION; its compiler identity is still recorded and hashed.
    version_file = sdk / "VERSION"
    if version_file.exists() and version_file.read_text().splitlines()[0] != profile["sdk_version"]:
        raise ValueError("Installed SDK version conflicts with the target profile")
    files = [
        sdk / "bin/clang",
        sdk / "bin/llvm-ar",
        sdk / "bin/llvm-ranlib",
        sdk / "bin/llvm-strip",
        sdk / "bin/wasm-ld",
    ]
    if version_file.exists():
        files.append(version_file)
    files += sorted(path for path in (sdk / "bin").glob("*.cfg") if path.is_file())
    files += sorted(path for path in (sdk / "lib/clang").rglob("*") if path.is_file())
    files += sorted(path for path in (sdk / "share/wasi-sysroot").rglob("*") if path.is_file())
    return {
        "sdk": recipe["sdk"],
        "profile": profile,
        "compiler_version": compiler_version,
        "files_sha256": digest({str(p.relative_to(sdk)): file_hash(p) for p in files}),
    }


def recipe_identity(recipe, directory):
    """Hash static metadata, checking any original receipt-owned code pins.

    Canonical builders have a separately recorded automatic implementation
    closure. Historical artifact recipes retain their original explicit pins.
    """
    for item in recipe.get("build_scripts", []):
        if file_hash(directory / item["file"]) != item["sha256"]:
            raise ValueError(f"Build script hash mismatch: {directory / item['file']}")
    return digest(recipe)


def artifact_input(recipe, directory, toolchain, dependencies):
    """Record source, features, tooling and the exact dependency artifact closure."""
    return {
        "recipe": recipe,
        "recipe_sha256": recipe_identity(recipe, directory),
        "source_sha256": recipe["source"]["sha256"],
        "toolchain": toolchain,
        "dependency_artifacts": {name: item["artifact_sha256"] for name, item in sorted(dependencies.items())},
    }


def exported_paths(recipe):
    paths = [path for group in recipe["exports"].values() for path in group]
    for path in paths:
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != path:
            raise ValueError(f"Invalid artifact export: {path}")
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate artifact export")
    return paths


def seal_artifact(prefix, inputs):
    """Write the artifact identity after every declared export has been produced."""
    files = {name: file_hash(prefix / name) for name in exported_paths(inputs["recipe"])}
    manifest = {"inputs": inputs, "files": files}
    directories = inputs["recipe"].get("empty_directories", {})
    if directories:
        manifest["directories"] = directories
    manifest["artifact_sha256"] = digest(manifest)
    (prefix / "artifact.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_artifact(prefix, expected_inputs=None):
    """Fail closed on changed inputs, missing exports, extra files or corrupt bytes."""
    manifest = json.loads((prefix / "artifact.json").read_text())
    unsigned = {key: value for key, value in manifest.items() if key != "artifact_sha256"}
    if digest(unsigned) != manifest["artifact_sha256"]:
        raise ValueError(f"Native artifact identity mismatch: {prefix}")
    if expected_inputs is not None and manifest["inputs"] != expected_inputs:
        raise ValueError(f"Native artifact inputs changed: {prefix}")
    files = manifest["files"]
    directories = manifest.get("directories", {})
    if directories != manifest["inputs"]["recipe"].get("empty_directories", {}):
        raise ValueError("Native artifact directories differ from recipe")
    if not isinstance(directories, dict) or len(directories) > 10_000:
        raise ValueError("Invalid native directory exports")
    for name, mode in directories.items():
        if not isinstance(name, str):
            raise ValueError("Native directory path must be a string")
        parsed = PurePosixPath(name)
        path = prefix / name
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or str(parsed) != name
            or name == "."
            or mode != 0o755
            or path.is_symlink()
            or not path.is_dir()
            or path.stat().st_mode & 0o777 != mode
            or any(path.iterdir())
        ):
            raise ValueError("Native empty-directory export is invalid")
    if set(files) != set(exported_paths(manifest["inputs"]["recipe"])):
        raise ValueError(f"Native artifact exports differ from recipe: {prefix}")
    actual = set()
    for path in prefix.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Native artifact contains a symlink: {path}")
        if path.is_dir() or path == prefix / "artifact.json":
            continue
        if not path.is_file():
            raise ValueError(f"Native artifact contains a special file: {path}")
        name = path.relative_to(prefix).as_posix()
        if name not in files or file_hash(path) != files[name]:
            raise ValueError(f"Native artifact integrity failure: {path}")
        actual.add(name)
    if actual != set(files):
        raise ValueError(f"Native artifact has missing files: {prefix}")
    return manifest


def dependency_prefix(requirements, providers, destination, target_profile):
    """Materialize exactly the declared closure and its ordered transitive links.

    Providers map catalog names to artifact directories. Dependencies use exact
    versions and profiles for this foundation; no version solver is implied.
    """
    selected = {}
    links = []
    visiting = set()
    dependency_order = []

    def select(requirement):
        name = requirement["port"]
        if name in visiting:
            raise ValueError(f"Cyclic target dependency: {name}")
        if name not in providers:
            raise ValueError(f"Missing target dependency: {name}")
        prefix = providers[name]
        manifest = verify_artifact(prefix)
        recipe = manifest["inputs"]["recipe"]
        if recipe["version"] != requirement["version"] or recipe["target_profile"] != target_profile:
            raise ValueError(f"Conflicting target dependency: {name}")
        if name in selected:
            if selected[name][1]["artifact_sha256"] != manifest["artifact_sha256"]:
                raise ValueError(f"Conflicting target artifact: {name}")
            return
        selected[name] = (prefix, manifest)
        visiting.add(name)
        for dependency in recipe["target_dependencies"]:
            select(dependency)
            expected = manifest["inputs"]["dependency_artifacts"].get(dependency["port"])
            if expected != selected[dependency["port"]][1]["artifact_sha256"]:
                raise ValueError(f"Conflicting transitive dependency artifact: {dependency['port']}")
        visiting.remove(name)
        dependency_order.append(name)
        for library in recipe["transitive_link_flags"]:
            if library not in ("-lm",):
                raise ValueError(f"Undeclared toolchain link flag: {library}")

    for requirement in requirements:
        select(requirement)
    for name in reversed(dependency_order):
        recipe = selected[name][1]["inputs"]["recipe"]
        links.extend(str(destination / archive) for archive in recipe["exports"].get("archives", []))
    for name in reversed(dependency_order):
        for library in selected[name][1]["inputs"]["recipe"]["transitive_link_flags"]:
            if library not in links:
                links.append(library)
    # Validate the whole closure before creating any consumer files.
    files = {}
    for prefix, manifest in selected.values():
        for relative in manifest["files"]:
            if relative in files:
                raise ValueError(f"Conflicting dependency export: {relative}")
            files[relative] = prefix / relative
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for relative, source in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return {name: manifest for name, (_, manifest) in selected.items()}, links
