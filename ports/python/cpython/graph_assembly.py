"""Overlay verified graph stdlib modules without changing the base interpreter.

The build cohort remains immutable. Publication records a distinct runtime
manifest after validating the complete native closure and all rootfs bytes.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from ports._support.native_artifacts import NativeArtifact, NativeTarget, merge_dependency_sysroot
from ports._support.runtime_files import _copy, _verified_rootfs
from ports.native.dependencies import file_hash


def assemble_stdlib(
    runtime: Path,
    modules: Mapping[str, NativeArtifact],
    closure: Mapping[str, NativeArtifact],
    target: NativeTarget,
    output: Path,
    *,
    runtime_manifest_sha256: str,
) -> tuple[Path, frozenset[str]]:
    """Validate and atomically publish a non-replacing stdlib/runtime overlay.

    ``runtime_manifest_sha256`` comes from the already admitted build cohort.
    Module/provider envelopes must retain the exact linked target identity.
    """
    if output.exists() or output.is_symlink():
        raise ValueError("stdlib assembly output already exists")
    if file_hash(runtime / "manifest.json") != runtime_manifest_sha256:
        raise ValueError("stdlib assembly base runtime differs from admitted receipt")
    base = json.loads((runtime / "manifest.json").read_bytes())
    _verified_rootfs(runtime, base)
    if base["dynamic_abi"] != target.abi or base["recipe"]["target"] != target.target:
        raise ValueError("stdlib assembly target differs from base runtime")
    if not modules:
        raise ValueError("stdlib assembly requires at least one module")
    selected = dict(modules)
    pending = list(modules)
    while pending:
        if len(selected) > 256:
            raise ValueError("stdlib assembly closure exceeds its bound")
        name = pending.pop()
        recipe = selected[name].manifest["inputs"]["recipe"]
        for dependency in recipe.get("target_dependencies", []):
            name = dependency["port"]
            if name not in selected:
                selected[name] = closure[name]
                pending.append(name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".partial-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        merged = merge_dependency_sysroot(modules, selected, temporary / "dependencies", target)
        stage = temporary / "runtime"
        shutil.copytree(runtime / "rootfs", stage / "rootfs")
        files = dict(base["files"])
        records = {}
        providers = set()
        for name, artifact in sorted(selected.items()):
            recipe = artifact.manifest["inputs"]["recipe"]
            stdlib = recipe.get("output") == "stdlib"
            if stdlib and (
                recipe["source"]["sha256"] != base["recipe"]["source"]["sha256"]
                or recipe["version"] != base["recipe"]["version"]
            ):
                raise ValueError("stdlib module source differs from admitted CPython")
            for group in ("shared_libraries", "licenses"):
                for source in recipe.get("exports", {}).get(group, []):
                    filename = Path(source).name
                    if group == "licenses":
                        destination = f"/TOOLCHAIN-LICENSES/graph/{name}/{filename}"
                    elif stdlib:
                        if source != f"lib-dynload/{recipe['module']}.so":
                            raise ValueError("stdlib export differs from declared module")
                        destination = f"/usr/lib/python3.13/lib-dynload/{filename}"
                    else:
                        if filename != recipe.get("soname"):
                            raise ValueError("runtime provider export differs from declared SONAME")
                        destination = f"/lib/{filename}"
                        providers.add(filename)
                    expected = artifact.manifest["files"][source]
                    if destination in files:
                        if files[destination] != expected:
                            raise ValueError("stdlib assembly cannot replace an existing runtime file")
                        continue
                    _copy(stage / "rootfs", destination, merged / "usr/local" / source, expected)
                    files[destination] = expected
            records[name] = artifact.manifest["artifact_sha256"]
        result = {**base, "files": files}
        result["stdlib_graph"] = {
            "base_manifest_sha256": runtime_manifest_sha256,
            "module_artifacts": {name: records[name] for name in modules},
            "dependency_artifacts": records,
        }
        _verified_rootfs(stage, result)
        (stage / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        stage.rename(output)
    return output, frozenset(providers)
