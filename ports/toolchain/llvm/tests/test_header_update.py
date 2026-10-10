"""Use a real Ninja dependency graph across verified native-header replacement."""

import copy
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ports._support.producer_tools import digest
from ports.toolchain.llvm.compiler import write_workspace
from ports.toolchain.llvm.guest import _inventory, _update_dependency_headers


@pytest.fixture
def project(tmp_path):
    ninja, cmake, cxx = (shutil.which(name) for name in ("ninja", "cmake", "c++"))
    if not all((ninja, cmake, cxx)):
        pytest.skip("requires CMake, Ninja and a native C++ compiler")
    work = tmp_path / "work"
    source = work / "source"
    source.mkdir(parents=True)
    (source / "value.cpp").write_text('#include "limit.h"\nint value() { return LIMIT; }\n')
    (source / "main.cpp").write_text("int value(); int main() { return value(); }\n")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(header_update CXX)\n"
        "add_executable(program main.cpp value.cpp)\n"
        "target_include_directories(program PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../inputs/posix/include)\n"
    )
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as output:
        output.add(source, arcname="upstream")
    for name in ("posix", "sysroot", "resources"):
        (work / "inputs" / name / "include").mkdir(parents=True)
    (work / "inputs/posix/include/limit.h").write_text("#define LIMIT 1\n")
    build = work / "build"
    subprocess.run(
        [
            cmake,
            "-G",
            "Ninja",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_MAKE_PROGRAM=" + ninja,
            "-DCMAKE_CXX_COMPILER=" + cxx,
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([ninja, "-C", str(build)], check=True, capture_output=True)
    old = {
        "source": {"sha256": digest(archive)},
        "patches": [],
        "target": "fixture",
        "compiler_tools": {},
        "host_tools": {"ninja": digest(Path(ninja))},
        "headers": {
            name: _inventory(work / "inputs" / name, ("include",)) for name in ("posix", "sysroot", "resources")
        },
    }
    state = {
        "compatibility": old,
        "phase": "ready",
        "configuration_sha256": digest(build / "CMakeCache.txt"),
        "inputs": {
            "compiler_tools": {},
            "snapshots": {name: _inventory(work / "inputs" / name) for name in ("posix", "sysroot", "resources")},
        },
    }
    write_workspace(work / "workspace.json", state)
    admitted = tmp_path / "admitted"
    shutil.copytree(work / "inputs/posix", admitted)
    (admitted / "include/limit.h").write_text("#define LIMIT 2\n")
    new = copy.deepcopy(old)
    new["headers"]["posix"] = _inventory(admitted, ("include",))
    snapshots = {"posix": (admitted, _inventory(admitted))}
    return SimpleNamespace(source=archive, build=build), work, state, new, snapshots, ninja


def test_header_update_rebuilds_dependency_and_preserves_unrelated_object(project):
    context, work, state, new, snapshots, ninja = project
    unrelated = context.build / "CMakeFiles/program.dir/main.cpp.o"
    before = unrelated.read_bytes(), unrelated.stat().st_mtime_ns
    assert subprocess.run([str(context.build / "program")]).returncode == 1
    _update_dependency_headers(context, new, work, state, new, snapshots)
    assert state["phase"] == "building"
    subprocess.run([ninja, "-C", str(context.build)], check=True, capture_output=True)
    assert subprocess.run([str(context.build / "program")]).returncode == 2
    assert (unrelated.read_bytes(), unrelated.stat().st_mtime_ns) == before


def test_interrupted_header_update_resumes_only_exact_old_or_new_bytes(project, monkeypatch):
    context, work, state, new, snapshots, ninja = project
    replace = Path.replace

    def interrupt(path, destination):
        result = replace(path, destination)
        if path.name == "header-update-file.preparing":
            raise OSError("interrupted header transition")
        return result

    with monkeypatch.context() as scope:
        scope.setattr(Path, "replace", interrupt)
        with pytest.raises(OSError):
            _update_dependency_headers(context, new, work, state, new, snapshots)
    _update_dependency_headers(context, new, work, state, new, snapshots)
    subprocess.run([ninja, "-C", str(context.build)], check=True, capture_output=True)
    assert subprocess.run([str(context.build / "program")]).returncode == 2


@pytest.mark.parametrize("corrupt", ["source", "configuration", "snapshot", "compiler"])
def test_header_transition_rejects_unadmitted_inputs(project, corrupt):
    context, work, state, new, snapshots, _ = project
    if corrupt == "source":
        (work / "source/main.cpp").write_text("unrecorded source")
    elif corrupt == "configuration":
        (context.build / "CMakeCache.txt").write_text("unrecorded configuration")
    elif corrupt == "snapshot":
        (work / "inputs/posix/include/limit.h").write_text("unrecorded header")
    else:
        new["compiler_tools"] = {"clang": "different"}
    with pytest.raises(ValueError):
        _update_dependency_headers(context, new, work, state, new, snapshots)
    assert not (work / "header-update.json").exists()
