"""Compile a real two-object project across verified appended patch updates."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from ports.toolchain.llvm import compiler


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def project(tmp_path):
    source = tmp_path / "source/upstream"
    source.mkdir(parents=True)
    (source / "value.cpp").write_text("int value() { return 1; }\n")
    (source / "main.cpp").write_text("int value(); int main() { return value(); }\n")
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as output:
        output.add(source, arcname="upstream")
    work = tmp_path / "work"
    build = work / "build"
    build.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text("pinned configuration\n")
    ninja = Path("/tmp/shellsim-scipy-build/tools/bin/ninja")
    if not ninja.is_file():
        pytest.skip("requires admitted Ninja for the compiled producer regression")
    (build / "build.ninja").write_text(
        "rule compile\n  command = /usr/bin/c++ -c $in -o $out\n"
        "rule link\n  command = /usr/bin/c++ $in -o $out\n"
        f"build value.o: compile {source}/value.cpp\n"
        f"build main.o: compile {source}/main.cpp\n"
        "build program: link value.o main.o\n"
    )
    subprocess.run([str(ninja), "-C", str(build)], check=True, capture_output=True)
    old = {"source": {"sha256": hash_file(archive)}, "patches": [], "tools": {"ninja": hash_file(ninja)}}
    workspace = {"compatibility": old, "phase": "ready", "configuration_sha256": hash_file(build / "CMakeCache.txt")}
    compiler.write_workspace(work / "workspace.json", workspace)
    patch = tmp_path / "change.patch"
    patch.write_text(
        "--- a/value.cpp\n+++ b/value.cpp\n@@ -1 +1 @@\n-int value() { return 1; }\n+int value() { return 2; }\n"
    )
    new = {
        **old,
        "patches": [
            {"file": patch.name, "sha256": hash_file(patch), "inputs": {"value.cpp": hash_file(source / "value.cpp")}}
        ],
    }
    return archive, source, new, tmp_path, work, workspace, new, ninja


def test_append_patch_rebuilds_changed_behavior_and_retains_unrelated_object(project):
    archive, source, recipe, directory, work, workspace, compatibility, ninja = project
    unchanged = work / "build/main.o"
    before = unchanged.read_bytes(), unchanged.stat().st_mtime_ns
    assert subprocess.run([str(work / "build/program")]).returncode == 1
    compiler.update_workspace_patches(archive, source, recipe, directory, work, workspace, compatibility)
    subprocess.run([str(ninja), "-C", str(work / "build")], check=True, capture_output=True)
    assert subprocess.run([str(work / "build/program")]).returncode == 2
    assert (unchanged.read_bytes(), unchanged.stat().st_mtime_ns) == before
    assert workspace["patch_updates"][0]["old_source_sha256"] != workspace["patch_updates"][0]["new_source_sha256"]


def test_interrupted_patch_update_recovers_verified_old_and_new_bytes(project, monkeypatch):
    archive, source, recipe, directory, work, workspace, compatibility, _ = project
    replace = compiler.os.replace

    def interrupt_after_write(src, dst):
        result = replace(src, dst)
        if Path(dst) == source / "value.cpp":
            raise OSError("interrupted source update")
        return result

    with monkeypatch.context() as scope:
        scope.setattr(compiler.os, "replace", interrupt_after_write)
        with pytest.raises(OSError):
            compiler.update_workspace_patches(archive, source, recipe, directory, work, workspace, compatibility)
    assert (work / "patch-update.json").is_file()
    compiler.update_workspace_patches(archive, source, recipe, directory, work, workspace, compatibility)
    assert (source / "value.cpp").read_text() == "int value() { return 2; }\n"
    assert not (work / "patch-update.json").exists()
    assert json.loads((work / "workspace.json").read_text())["compatibility"] == compatibility


def test_patch_update_rejects_unpinned_source_edits(project):
    archive, source, recipe, directory, work, workspace, compatibility, _ = project
    (source / "main.cpp").write_text("int main() { return 3; }\n")
    with pytest.raises(ValueError, match="retained LLVM source differs"):
        compiler.update_workspace_patches(archive, source, recipe, directory, work, workspace, compatibility)
    assert not (work / "patch-update.json").exists()


def test_partial_staging_write_does_not_change_retained_source(project, monkeypatch):
    archive, source, recipe, directory, work, workspace, compatibility, _ = project
    original = (source / "value.cpp").read_bytes()
    copy = compiler.shutil.copyfileobj

    def interrupt_staging(stream, output, *args):
        if ".patch-input-" in str(output.name):
            output.write(stream.read(3))
            raise OSError("interrupted staged write")
        return copy(stream, output, *args)

    with monkeypatch.context() as scope:
        scope.setattr(compiler.shutil, "copyfileobj", interrupt_staging)
        with pytest.raises(OSError):
            compiler.update_workspace_patches(archive, source, recipe, directory, work, workspace, compatibility)
    assert (source / "value.cpp").read_bytes() == original
    compiler.update_workspace_patches(archive, source, recipe, directory, work, workspace, compatibility)
    assert (source / "value.cpp").read_text() == "int value() { return 2; }\n"
