"""Exercise the standard patched Clang driver and its WASI TLS backend."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def compiler():
    prefix = os.environ.get("SHELLSIM_THREADED_CLANG")
    if prefix is None:
        pytest.skip("requires the normal full threaded compiler artifact")
    return Path(prefix) / "bin/clang"


def test_preprocessor_and_resource_headers_support_wasm32_compilation(compiler, tmp_path):
    source = tmp_path / "value.c"
    source.write_text(
        "#include <stddef.h>\n"
        "#if PORT_VALUE != 37\n#error incorrect preprocessing\n#endif\n"
        '_Static_assert(sizeof(size_t) == 4, "target pointer width");\n'
        "int value(void) { return PORT_VALUE; }\n"
    )
    common = [str(compiler), "--target=wasm32-wasip1-threads", "-DPORT_VALUE=37"]
    preprocessed = tmp_path / "value.i"
    subprocess.run([*common, "-E", str(source), "-o", str(preprocessed)], check=True)
    obj = tmp_path / "value.o"
    subprocess.run([*common, "-c", str(preprocessed), "-o", str(obj)], check=True)
    assert obj.read_bytes().startswith(b"\0asm\x01\0\0\0")


@pytest.mark.parametrize("enabled,relocation", [(False, "external_tls@TLSREL"), (True, "external_tls@GOT@TLS")])
def test_clang_passes_the_opt_in_tls_policy_to_its_backend(compiler, tmp_path, enabled, relocation):
    source = tmp_path / "storage.c"
    source.write_text(
        "extern _Thread_local int external_tls;\n"
        "static _Thread_local int local_tls = 3;\n"
        "int *external_address(void) { return &external_tls; }\n"
        "int *local_address(void) { return &local_tls; }\n"
    )
    command = [str(compiler), "--target=wasm32-wasip1-threads", "-pthread", "-fPIC", "-S", str(source), "-o", "-"]
    if enabled:
        command.extend(["-mllvm", "-wasm-enable-wasi-dynamic-tls"])
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert relocation in result.stdout
    assert "local_tls@TLSREL" in result.stdout
