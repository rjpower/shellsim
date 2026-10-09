"""Exercise TLS lowering through the independently built threaded compiler."""

import os
import subprocess
from pathlib import Path

import pytest

IR = """
@external_tls = external thread_local global i32
@hidden_tls = hidden thread_local global i32 0
define ptr @external_address() {
  %p = call ptr @llvm.threadlocal.address.p0(ptr @external_tls)
  ret ptr %p
}
define ptr @hidden_address() {
  %p = call ptr @llvm.threadlocal.address.p0(ptr @hidden_tls)
  ret ptr %p
}
declare ptr @llvm.threadlocal.address.p0(ptr)
"""


@pytest.fixture
def llc():
    prefix = os.environ.get("SHELLSIM_THREADED_LLVM")
    if prefix is None:
        pytest.skip("requires the normal threaded LLVM artifact")
    return Path(prefix) / "bin/llc"


def compile_ir(llc, triple, enabled, ir=IR):
    command = [str(llc), "-mtriple=" + triple, "-relocation-model=pic", "-mattr=+bulk-memory,+atomics", "-o", "-"]
    if enabled:
        command.append("-wasm-enable-wasi-dynamic-tls")
    return subprocess.run(command, input=ir, text=True, capture_output=True, check=False)


@pytest.mark.parametrize(
    "triple,enabled,relocation",
    [
        ("wasm32-wasip1", False, "external_tls@TLSREL"),
        ("wasm32-wasip1", True, "external_tls@GOT@TLS"),
        ("wasm32-unknown-unknown", True, "external_tls@TLSREL"),
    ],
)
def test_external_tls_lowering_is_opt_in_and_scoped_to_wasi(llc, triple, enabled, relocation):
    result = compile_ir(llc, triple, enabled)
    assert result.returncode == 0, result.stderr
    assert relocation in result.stdout
    assert "hidden_tls@TLSREL" in result.stdout


def test_emscripten_lowering_is_unchanged(llc):
    default = compile_ir(llc, "wasm32-unknown-emscripten", False)
    enabled = compile_ir(llc, "wasm32-unknown-emscripten", True)
    assert default.returncode == enabled.returncode == 0
    assert default.stdout == enabled.stdout


def test_unsupported_initial_exec_reports_a_diagnostic(llc):
    ir = IR.replace("external thread_local global", "external thread_local(initialexec) global")
    result = compile_ir(llc, "wasm32-wasip1", True, ir)
    assert result.returncode != 0
    assert "WASI dynamic TLS does not support initial-exec TLS" in result.stderr
