"""Cross-build upstream Cryptography with static target OpenSSL archives."""

import os
import shlex
import subprocess
from pathlib import Path

from ports._support.native_adapters import build_environment, compiler_wrapper_text
from ports._support.python_rust import build_rust_extension
from ports.api import BuildContext


def build(ctx: BuildContext):
    native, python = ctx.require_native(), ctx.require_cpython()
    native.build.mkdir(parents=True)
    tools = native.build / "adapter-tools"
    tools.mkdir()
    response = Path(__file__).parents[2] / "_support/compiler_response.py"
    for role in ("cc", "cxx"):
        path = tools / role
        path.write_text(compiler_wrapper_text(native, role, response.read_text()))
        path.chmod(0o755)
    generator = native.build / "host-generator"
    command = [
        str(native.host_tools["uv"]),
        "pip",
        "install",
        "--python",
        str(native.host_tools["python"]),
        "--target",
        str(generator),
        "--no-deps",
        "cffi==2.1.1",
        "pycparser==3.11",
    ]
    if ctx.offline:
        command.append("--offline")
    subprocess.run(
        command,
        check=True,
    )
    # Upstream derives target Python headers from this conventional prefix.
    prefix = native.build / "target-python"
    (prefix / "include").mkdir(parents=True)
    (prefix / "include/python3.13").symlink_to(python.include_dir, target_is_directory=True)
    (prefix / "lib/python3.13").mkdir(parents=True)
    openssl = native.dependencies["native/openssl"]
    environment = build_environment(native, {})
    environment.update(
        PYO3_PYTHON=str(native.host_tools["python"]),
        PYO3_CROSS_LIB_DIR=str(prefix / "lib/python3.13"),
        PYTHONPATH=os.pathsep.join([str(generator), *os.environ.get("PYTHONPATH", "").split(os.pathsep)]),
        OPENSSL_STATIC="1",
        OPENSSL_NO_VENDOR="1",
        OPENSSL_LIB_DIR=str(openssl / "lib"),
        OPENSSL_INCLUDE_DIR=str(openssl / "include"),
        CFLAGS=shlex.join(["-I" + str(python.generated_config_dir), "-I" + str(python.include_dir)]),
    )
    return build_rust_extension(
        ctx,
        module="cryptography.hazmat.bindings._rust",
        manifest_path="src/rust/Cargo.toml",
        configure_environment=environment,
        python_sources=(("src/cryptography", "cryptography"),),
        licenses=("LICENSE", "LICENSE.APACHE", "LICENSE.BSD"),
    )
