"""Reject changes to a compiler configuration in an otherwise intact SDK."""

import os
from pathlib import Path

import pytest

from ports.native.dependencies import digest
from ports.toolchain.wasi_threads.dynamic import sdk_tooling, verify_sdk


@pytest.mark.parametrize("config", ["clang.cfg", "clang++.cfg"])
def test_external_sdk_configuration_tampering_is_rejected(tmp_path, config):
    source = os.environ.get("SHELLSIM_WASI_SDK34")
    if source is None:
        pytest.skip("requires the pinned SDK for a real tooling receipt")
    source = Path(source)
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    for path in (source / "bin").iterdir():
        (sdk / "bin" / path.name).symlink_to(path)
    (sdk / "lib").symlink_to(source / "lib", target_is_directory=True)
    receipt = sdk_tooling(sdk)
    manifest = {"identity": {"recipe": {"sdk_tooling_digest": digest(receipt)}, "sdk_tooling": receipt}}
    verify_sdk(sdk, manifest)
    changed = sdk / "bin" / config
    changed.unlink()
    changed.write_bytes((source / "bin" / config).read_bytes() + b"\n-DALTERED_SDK_CONFIG=1\n")
    with pytest.raises(ValueError):
        verify_sdk(sdk, manifest)
