"""Exercise the bundled C toolchain through the public host and package APIs."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from importlib.resources import files
from pathlib import Path

import pytest
import shellsim

LIMITS = shellsim.Limits(cpu=50_000_000_000, memory=256 * 1024 * 1024, disk=128 * 1024 * 1024)
SOURCE = '#include <stdio.h>\nint main(void) { puts("guest C"); return 0; }\n'


def test_installed_c_toolchain_compiles_inside_guest() -> None:
    guest = shellsim.Environment(limits=LIMITS)
    shellsim.install_c_toolchain(guest)
    guest.write_file("/work/hello.c", SOURCE)

    result = guest.run("cd /work && cc -o hello.wasm hello.c && chmod +x hello.wasm && ./hello.wasm")

    assert result.returncode == 0, result.stderr_text
    assert result.stdout == b"guest C\n"
    assert guest.read_file("/tcc/tcc-shellsim.wasm").startswith(b"\0asm")


def test_package_requirement_installs_toolchain_without_embedding_it(tmp_path: Path) -> None:
    (tmp_path / "hello.c").write_text(SOURCE)
    (tmp_path / "run.sh").write_text("cc -o /work/hello.wasm /work/hello.c && chmod +x /work/hello.wasm && /work/hello.wasm\n")
    package = shellsim.Package.build_from_directory(
        tmp_path,
        spec=shellsim.PackageSpec(
            name="c-guest", version="1", entrypoint=("sh", "/work/run.sh"), limits=LIMITS,
            requires_c_toolchain=True,
        ),
    )
    loaded = shellsim.Package.from_bytes(package.to_bytes())

    assert loaded.spec.requires_c_toolchain is True
    assert b"tcc-shellsim-package.tar.gz" not in loaded.to_bytes()
    guest = loaded.instantiate(tools={}, limits=LIMITS)
    assert guest.run_entrypoint().stdout == b"guest C\n"


def test_toolchain_assets_and_corresponding_source_are_in_distribution() -> None:
    assets = files("shellsim").joinpath("_assets")
    expected = {
        "tcc-shellsim-package.tar.gz": "3bd110fbdaa22682fefb89e93bee4b8941a324aa98bb4b3ae554fc50a2fb47f2",
        "sysroot-34.tar.gz": "3d637426ef54d66dfb7a03276ecbf16f925b481145573a127d244093978b65be",
        "tinycc-35c49ec-source.tar.gz": "2f0fb73158f7d64303885bc44a292621fdcd6a088828c8d3a0be98216af0004a",
    }
    for name, digest in expected.items():
        assert hashlib.sha256(assets.joinpath(name).read_bytes()).hexdigest() == digest
    with tarfile.open(fileobj=io.BytesIO(assets.joinpath("tinycc-35c49ec-source.tar.gz").read_bytes()), mode="r:gz") as source:
        assert {"COPYING", "Makefile", ".github/workflows/wasm.yml"} <= set(source.getnames())


def test_package_manifest_rejects_non_boolean_toolchain_requirement(tmp_path: Path) -> None:
    (tmp_path / "run.sh").write_text("true\n")
    package = shellsim.Package.build_from_directory(
        tmp_path,
        spec=shellsim.PackageSpec(name="invalid", version="1", entrypoint=("sh", "/work/run.sh")),
    )
    with zipfile.ZipFile(io.BytesIO(package.to_bytes())) as original:
        members = {member.filename: original.read(member) for member in original.infolist()}
    manifest = json.loads(members["shl.json"])
    manifest.pop("requires_c_toolchain")
    members["shl.json"] = json.dumps(manifest).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as rewritten:
        for name, data in members.items():
            rewritten.writestr(name, data)
    assert shellsim.Package.from_bytes(output.getvalue()).spec.requires_c_toolchain is False

    manifest["requires_c_toolchain"] = "yes"
    members["shl.json"] = json.dumps(manifest).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as rewritten:
        for name, data in members.items():
            rewritten.writestr(name, data)
    with pytest.raises(TypeError, match="requires_c_toolchain"):
        shellsim.Package.from_bytes(output.getvalue())


def test_installation_respects_guest_disk_limit() -> None:
    guest = shellsim.Environment(limits=shellsim.Limits(cpu=50_000_000_000, disk=1024 * 1024))
    with pytest.raises(shellsim.SimulationError):
        shellsim.install_c_toolchain(guest)
