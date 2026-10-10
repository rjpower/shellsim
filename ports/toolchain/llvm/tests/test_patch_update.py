"""Compile a real two-object project across verified appended patch updates."""

import hashlib
import json
import shlex
import shutil
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
    ninja_path, cxx = shutil.which("ninja"), shutil.which("c++")
    if ninja_path is None or cxx is None:
        pytest.skip("requires Ninja and a native C++ compiler")
    ninja = Path(ninja_path)
    cxx_command = shlex.quote(cxx).replace("$", "$$")
    (build / "build.ninja").write_text(
        f"rule compile\n  command = {cxx_command} -c $in -o $out\n"
        f"rule link\n  command = {cxx_command} $in -o $out\n"
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


def test_guest_build_directory_preserves_unrelated_object_on_appended_patch(project):
    archive, source, recipe, directory, work, workspace, compatibility, ninja = project
    build = work / "threaded"
    (work / "build").rename(build)
    object_file = build / "main.o"
    before = object_file.read_bytes(), object_file.stat().st_mtime_ns
    compiler.update_workspace_patches(
        archive, source, recipe, directory, work, workspace, compatibility, build_directory=build
    )
    subprocess.run([str(ninja), "-C", str(build)], check=True, capture_output=True)
    assert subprocess.run([str(build / "program")], check=False).returncode == 2
    assert (object_file.read_bytes(), object_file.stat().st_mtime_ns) == before


def test_appended_patch_rejects_changed_nonpatch_compatibility(project):
    archive, source, recipe, directory, work, workspace, compatibility, _ = project
    workspace["compatibility"]["target"] = "admitted target"
    incompatible = {**compatibility, "target": "other target"}
    with pytest.raises(ValueError):
        compiler.update_workspace_patches(archive, source, recipe, directory, work, workspace, incompatible)
    assert subprocess.run([str(work / "build/program")], check=False).returncode == 1


def test_preparation_dry_run_never_builds_or_marks_ready(tmp_path):
    source, build = tmp_path / "source", tmp_path / "threaded"
    source.mkdir()
    build.mkdir()
    (source / "value.cpp").write_text("int main(void) { return 0; }\n")
    ninja, cxx = shutil.which("ninja"), shutil.which("c++")
    if ninja is None or cxx is None:
        pytest.skip("requires Ninja and a native C++ compiler")
    (build / "CMakeCache.txt").write_text("pinned configuration\n")
    (build / "build.ninja").write_text(
        f"rule compile\n  command = {shlex.quote(cxx)} $in -o $out\nbuild program: compile {source}/value.cpp\n"
    )
    state = {"phase": "ready", "configuration_sha256": hash_file(build / "CMakeCache.txt")}
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    receipt = tmp_path / "workspace.json"
    commands = [[shutil.which("true")], [ninja, "-C", str(build), "program"]]
    compiler.configure_and_build(commands, build, state, receipt, attempts, None, prepare_only=True)
    assert not (build / "program").exists()
    assert json.loads(receipt.read_text())["phase"] == "building"
    compiler.configure_and_build(commands, build, state, receipt, attempts, None)
    assert subprocess.run([str(build / "program")], check=False).returncode == 0
    assert json.loads(receipt.read_text())["phase"] == "ready"


def test_interrupted_guest_prepare_relinks_versioned_binary(tmp_path):
    from ports.toolchain.llvm.guest import _invalidate_final_outputs

    cxx, ninja, ar = (shutil.which(name) for name in ("c++", "ninja", "ar"))
    if not all((cxx, ninja, ar)):
        pytest.skip("native C++ compiler, Ninja and ar are required")
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    main = build / "main.cpp"
    main.write_text('#include <cstdio>\nextern int value();\nint main() { std::printf("%d", value()); }\n')
    subprocess.run([cxx, "-c", str(main), "-o", str(build / "main.o")], check=True)
    object_bytes = (build / "main.o").read_bytes()
    object_mtime = (build / "main.o").stat().st_mtime_ns
    provider = build / "provider.cpp"

    def archive(value):
        provider.write_text(f"int value() {{ return {value}; }}\n")
        subprocess.run([cxx, "-c", str(provider), "-o", str(build / "provider.o")], check=True)
        subprocess.run([ar, "rcs", str(build / "provider.a"), str(build / "provider.o")], check=True)

    archive(1)
    # Driver-selected archives are not Ninja dependencies, as with the real SDK.
    (build / "build.ninja").write_text(
        "rule link\n  command = " + shlex.quote(cxx) + " $in provider.a -o $out\nbuild bin/clang-23: link main.o\n"
    )
    subprocess.run([ninja, "-C", str(build), "bin/clang-23"], check=True)
    binary = build / "bin/clang-23"
    previous_bytes = binary.read_bytes()
    assert subprocess.check_output([binary]) == b"1"
    (build / "bin/clang").symlink_to("clang-23")
    receipt = tmp_path / "workspace.json"
    state = {"phase": "ready"}
    recipe = {"version": "23.1.0rc3"}
    _invalidate_final_outputs(build, recipe, state, receipt, True)
    assert not binary.exists()
    assert not (build / "bin/clang").is_symlink()
    # Resume from the durable preparation receipt after archive inputs advance.
    archive(2)
    resumed = json.loads(receipt.read_text())
    resumed["phase"] = "building"
    _invalidate_final_outputs(build, recipe, resumed, receipt, False)
    subprocess.run([ninja, "-C", str(build), "bin/clang-23"], check=True)
    assert subprocess.check_output([binary]) == b"2"
    assert binary.read_bytes() != previous_bytes
    assert (build / "main.o").read_bytes() == object_bytes
    assert (build / "main.o").stat().st_mtime_ns == object_mtime
