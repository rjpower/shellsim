"""Publish PIC target Rust libraries without changing the installed toolchain."""

import json
import shlex
import shutil
import sys
from pathlib import Path

from ports._support.rust_tools import run_cargo, rust_environment, rust_tools
from ports.api import BuildContext


def build(ctx: BuildContext) -> None:
    environment = rust_environment(ctx)
    rustc, cargo, base, host = rust_tools(ctx.metadata["version"], environment)
    target = ctx.metadata["target"]
    shadow = ctx.source / "rust-src"
    # Build scripts and procedural macros use the ordinary host libraries.
    # Only freshly compiled target libraries enter the resulting product.
    (shadow / "lib/rustlib" / host).symlink_to(base / "lib/rustlib" / host, target_is_directory=True)
    project = ctx.result.parent / "stdlib-probe"
    project.mkdir()
    (project / "src").mkdir()
    (project / "Cargo.toml").write_text(
        '[package]\nname = "shellsim-stdlib-probe"\nversion = "0.0.0"\nedition = "2024"\n'
        '[workspace]\n[profile.release]\npanic = "abort"\n'
    )
    (project / "src/lib.rs").write_text("pub fn probe() -> Vec<u8> { Vec::new() }\n")
    (project / "Cargo.lock").write_text(
        'version = 4\n\n[[package]]\nname = "shellsim-stdlib-probe"\nversion = "0.0.0"\n'
    )
    wrapper = project / "rustc"
    wrapper.write_text(
        "#!"
        + sys.executable
        + "\nimport os, sys\n"
        + "os.execv("
        + repr(str(rustc))
        + ", ["
        + repr(str(rustc))
        + ", '--sysroot', "
        + repr(str(shadow))
        + "] + sys.argv[1:])\n"
    )
    wrapper.chmod(0o755)
    environment.update(
        {
            "RUSTC": str(wrapper),
            "RUSTC_BOOTSTRAP": "1",
            "RUSTFLAGS": shlex.join(["-C", "relocation-model=pic", "-C", "link-self-contained=no"]),
        }
    )
    command = [
        str(cargo),
        "-Z",
        "build-std=std,panic_abort",
        "build",
        "--release",
        "--locked",
        "--lib",
        "--target",
        target,
        "--message-format=json",
    ]
    if ctx.offline:
        command.append("--offline")
    result = run_cargo(command, project, environment, project / "cargo.log")
    destination = ctx.result / "lib/rustlib" / target / "lib"
    destination.mkdir(parents=True)
    for line in result.stdout.splitlines():
        artifact = json.loads(line)
        if artifact.get("reason") != "compiler-artifact" or artifact["target"]["name"] == "shellsim-stdlib-probe":
            continue
        for filename in artifact["filenames"]:
            source = Path(filename)
            if source.suffix == ".rlib" and target in source.parts:
                shutil.copyfile(source, destination / source.name)
    if not any(destination.glob("libstd-*.rlib")):
        raise ValueError("Cargo produced no target standard library")
    licenses = ctx.result / "licenses"
    licenses.mkdir()
    for name in ("LICENSE-MIT", "LICENSE-APACHE"):
        shutil.copyfile(ctx.source / name, licenses / name)
