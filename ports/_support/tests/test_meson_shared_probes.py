"""Exercise real WASI shared-provider link checks without executing probes."""

import os
from pathlib import Path

import pytest

from ports._support.cohort import load_cohort
from ports._support.native_adapters import NativeAdapter, NativeBuildContext, NativeBuildRequest, build_native
from ports.native.dependencies import verify_artifact


def test_meson_shared_provider_symbol_and_call_checks(tmp_path):
    descriptor = os.environ.get("SHELLSIM_BUILD_COHORT")
    prefix = os.environ.get("SHELLSIM_OPENBLAS_PREFIX")
    if descriptor is None or prefix is None:
        pytest.skip("requires an admitted cohort and sealed OpenBLAS provider")
    cohort = load_cohort(Path(descriptor))
    prefix = Path(prefix).resolve()
    artifact = verify_artifact(prefix)
    assert artifact["inputs"]["recipe"]["name"] == "openblas"
    provider = prefix / "lib/libopenblas.so"
    source = tmp_path / "source"
    source.mkdir()
    args = "args: [" + repr(str(provider)) + "]"
    (source / "meson.build").write_text(
        "project('shared-probe', 'c')\n"
        "cc = meson.get_compiler('c')\n"
        "assert(cc.links('extern char *openblas_get_config(void); int main(void) { return openblas_get_config()==0; }', "
        + args
        + "), 'typed provider call must link')\n"
        "assert(cc.links('extern void dgemm_(void); void (*volatile fn)(void)=dgemm_; int main(void) { return fn==0; }', "
        + args
        + "), 'actual symbol address must link')\n"
        "assert(not cc.links('extern void absent_provider_symbol(void); void (*volatile fn)(void)=absent_provider_symbol; int main(void) { return fn==0; }', "
        + args
        + "), 'missing provider symbol must fail')\n"
        "assert(not cc.links('extern void dgemm_(void); int main(void) { dgemm_(); return 0; }', "
        + args
        + "), 'incompatible direct call must fail')\n"
    )
    context = NativeBuildContext(
        source=source,
        build=tmp_path / "build",
        staging_prefix=tmp_path / "staging",
        sdk=cohort.sdk.root,
        compiler_prefix=cohort.llvm.root,
        sysroot=cohort.sysroot.root / "sysroot",
        target=cohort.target,
        compiler_flags=cohort.compiler_flags,
        linker_flags=cohort.linker_flags,
        dependencies={},
        host_tools={name: tool.path for name, tool in cohort.host_tools.items()},
        target_tools={name: tool.path for name, tool in cohort.target_tools.items()},
        dependency_sysroot=tmp_path / "dependencies",
        shared_library_flags=cohort.shared_library_flags,
        executable_flags=cohort.executable_flags,
        shared_library_inputs=(cohort.compiler_runtime_archive,),
    )
    build_native(NativeBuildRequest(NativeAdapter.MESON, context))
