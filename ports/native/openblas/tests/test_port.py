"""Check the WASI numerical provider's pinned recipe and explicit architecture patch."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def test_openblas_has_no_host_or_fortran_target_dependencies(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from ports.native.dependencies import recipe_identity

    recipe = json.loads((ROOT / "ports/native/openblas/recipe.json").read_text())
    recipe_identity(recipe, ROOT / "ports/native/openblas")
    assert recipe["version"] == "0.3.31"
    assert recipe["target_profile"] == "wasi-cpython-v2"
    assert recipe["target_dependencies"] == []
    assert recipe["features"]["threads"] is False
    assert recipe["features"]["integer_bits"] == 32
    assert recipe["features"]["fortran_compiler"] is False
    assert recipe["features"]["complex_return"] == "hidden-pointer"
    assert recipe["exports"]["archives"] == ["lib/libopenblas.a"]


@pytest.mark.parametrize("name", ["NAME", "slarfb_gett_", "sgetrf2_", "dgeqp3rk_"])
def test_converted_subroutine_returns_zero_at_early_exit_and_end(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("openblas_build", ROOT / "ports/native/openblas/build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = f'void {name}(int n) {{ if (n < 0) {{ return; }} const char *s = "}}"; /* }} */ }}'
    result = builder.convert_interface_returns(source)
    assert "if (n < 0) { return 0; }" in result
    assert result.endswith("return 0;\n}")


def test_return_conversion_preserves_shared_cblas_body(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("openblas_build", ROOT / "ports/native/openblas/build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = "#ifndef CBLAS\nvoid NAME(int *n) {\n  if (!*n) return;\n#else\nvoid CNAME(int n) {\n  if (!n) return;\n#endif\n  run_kernel();\n}\n"
    result = builder.convert_interface_returns(source)
    assert result.count("return 0;") == 3
    assert result.endswith("return 0;\n}\n")


def test_numerical_and_fortran_abi_in_guest():
    import os

    probe = os.environ.get("SHELLSIM_SCIPY_NATIVE_PROBE")
    if probe is None:
        pytest.skip("set SHELLSIM_SCIPY_NATIVE_PROBE to the verified OpenBLAS probe.wasm")
    import shellsim

    environment = shellsim.Environment(cpu=100_000_000, memory=128 * 1024 * 1024, disk=8 * 1024 * 1024)
    environment.write_file("/probe", Path(probe).read_bytes(), mode=0o755)
    result = environment.run("SHELLSIM_OPENBLAS_ABI=123 /probe")
    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith(b"OpenBLAS: dgemm, dgesv, invalid input, complex and REAL ABI passed\n")
    assert result.stderr == b""
    assert result.usage.cpu_used > 0


@pytest.mark.parametrize("name", ["cdotu_", "zdotc_", "cladiv_", "zladiv_"])
def test_complex_hidden_result_functions_keep_void_returns(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("openblas_build", ROOT / "ports/native/openblas/build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = f"static inline void {name}(complex *result, int *n) {{ if (!*n) return; }}"
    assert builder.convert_interface_returns(source) == source


def test_new_lapack_xerbla_calls_receive_character_length(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("openblas_build", ROOT / "ports/native/openblas/build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = 'extern int xerbla_(\n char *, integer \n *);\nxerbla_("SGEQP3RK", &i__1);'
    result = builder.normalize_xerbla(source)
    assert "xerbla_(char *, integer *, ftnlen)" in result
    assert 'xerbla_("SGEQP3RK", &i__1, (ftnlen)8)' in result
    existing = 'xerbla_("DGESV ", &i__1, (ftnlen)6);'
    assert builder.normalize_xerbla(existing) == existing


def test_shared_openblas_recipe_binds_the_archive_and_imports_the_main_runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from ports.native.dependencies import recipe_identity

    directory = ROOT / "ports/native/openblas/shared"
    recipe = json.loads((directory / "recipe.json").read_text())
    recipe_identity(recipe, directory)
    assert recipe["static_provider"] == {"port": "native/openblas", "version": "0.3.31"}
    assert recipe["abi"] == "shellsim-wasi-sdk34-cpython3137-v2"
    assert recipe["soname"] == "libopenblas.so"
    assert recipe["needed_libraries"] == []
    assert "-nostdlib" in recipe["link_flags"]
    assert "-shared" in recipe["link_flags"]
    assert recipe["features"]["runtime_owner"] == "main-executable"
    assert recipe["compiler_support"]["linkage"] == "referenced-members-only"
    assert recipe["exports"]["libraries"] == ["lib/libopenblas.so"]


def test_shared_consumer_dependency_decoder_checks_real_metadata_lengths(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT))
    from ports._support.wasm import leb
    from ports._support.wasm_metadata import needed_libraries

    name = b"dylink.0"
    dependency = b"libopenblas.so"
    needed = leb(1) + leb(len(dependency)) + dependency
    payload = leb(len(name)) + name + b"\x02" + leb(len(needed)) + needed
    path = tmp_path / "consumer.so"
    path.write_bytes(b"\0asm\x01\0\0\0\0" + leb(len(payload)) + payload)
    assert needed_libraries(path) == ["libopenblas.so"]
    path.write_bytes(b"\0asm\x01\0\0\0\0" + leb(len(payload) + 1) + payload)
    with pytest.raises(ValueError):
        needed_libraries(path)


@pytest.mark.parametrize("memory_mib,status", [(512, 0), (256, 137)])
def test_shared_openblas_through_an_independent_consumer_in_guest(monkeypatch, memory_mib, status):
    import os

    provider = os.environ.get("SHELLSIM_SHARED_OPENBLAS_ARTIFACT")
    consumer = os.environ.get("SHELLSIM_SHARED_OPENBLAS_CONSUMER")
    runtime = os.environ.get("SHELLSIM_DYNAMIC_V2_ARTIFACTS")
    if provider is None or consumer is None or runtime is None:
        pytest.skip("set the shared OpenBLAS artifact, consumer and SDK34 runtime paths")
    monkeypatch.syspath_prepend(str(ROOT))
    from ports._support.wasm_metadata import needed_libraries
    from ports.native.openblas.shared.tests.verify import run_probe

    assert needed_libraries(Path(consumer)) == ["libopenblas.so"]
    result = run_probe(Path(runtime), Path(provider), Path(consumer), memory_mib * 1024 * 1024)
    assert result.returncode == status, result.stderr
    assert result.usage.memory_current == 0
    if status == 0:
        assert result.stderr == b""
        assert result.stdout.endswith(b"OpenBLAS: dgemm, dgesv, invalid input, complex and REAL ABI passed\n")
