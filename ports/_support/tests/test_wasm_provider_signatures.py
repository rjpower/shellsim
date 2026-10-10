"""Check provider calls using core Wasm type/import/export sections."""

import os
import subprocess
from pathlib import Path

import pytest

from ports._support.import_sdk import load_legacy_cohort
from ports._support.wasm_metadata import function_signatures, validate_provider_signatures


def section(kind, data):
    assert len(data) < 128
    return bytes([kind, len(data)]) + data


def string(value):
    encoded = value.encode()
    return bytes([len(encoded)]) + encoded


def module(path, argument, *, provider=False, needed="libprovider.so"):
    data = b"\0asm\x01\0\0\0" + section(1, bytes([1, 0x60, 1, argument, 1, 0x7F]))
    if provider:
        data += section(3, b"\x01\x00")
        data += section(7, b"\x01" + string("calculate") + b"\x00\x00")
        data += section(10, b"\x01\x04\x00\x41\x00\x0b")
    else:
        dependencies = b"\x01" + string(needed)
        data += section(0, string("dylink.0") + bytes([2, len(dependencies)]) + dependencies)
        data += section(2, b"\x01" + string("env") + string("calculate") + b"\x00\x00")
    path.write_bytes(data)
    return path


def test_declared_provider_function_types_match(tmp_path):
    provider = module(tmp_path / "provider.so", 0x7F, provider=True)
    consumer = module(tmp_path / "consumer.so", 0x7F)
    assert validate_provider_signatures(consumer, {"libprovider.so": provider}) == ["libprovider.so"]
    assert function_signatures(provider)[1]["calculate"] == ((0x7F,), (0x7F,))


def test_declared_provider_function_type_mismatch_is_rejected(tmp_path):
    provider = module(tmp_path / "provider.so", 0x7E, provider=True)
    consumer = module(tmp_path / "consumer.so", 0x7F)
    with pytest.raises(ValueError):
        validate_provider_signatures(consumer, {"libprovider.so": provider})


def test_undeclared_provider_is_rejected(tmp_path):
    consumer = module(tmp_path / "consumer.so", 0x7F)
    with pytest.raises(ValueError):
        validate_provider_signatures(consumer, {})


def test_shared_link_rejects_incompatible_object_function_types(tmp_path, monkeypatch):
    descriptor = os.environ.get("SHELLSIM_BUILD_COHORT")
    if descriptor is None:
        pytest.skip("requires an explicitly admitted build cohort")
    cohort = load_legacy_cohort(Path(descriptor))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    provider = tmp_path / "provider.c"
    caller = tmp_path / "caller.c"
    provider.write_text("long long calculate(int x) { return x; }\n")
    caller.write_text("extern int calculate(int); int invoke(void) { return calculate(3); }\n")
    objects = []
    for source in (provider, caller):
        output = source.with_suffix(".o")
        subprocess.run(
            [
                str(cohort.target_tools["cc"].path),
                *cohort.compiler_flags,
                "-fPIC",
                "-c",
                str(source),
                "-o",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        objects.append(str(output))
    flags = cohort.shared_library_flags
    assert any("--fatal-warnings" in flag for flag in flags)
    command = [
        str(cohort.target_tools["cc"].path),
        *cohort.linker_flags,
        *flags,
        *objects,
        "-o",
        str(tmp_path / "mismatch.so"),
    ]
    failed = subprocess.run(command, capture_output=True)
    assert failed.returncode != 0
    permissive = [argument.replace(",--fatal-warnings", "") for argument in command]
    allowed = subprocess.run(permissive, capture_output=True)
    assert allowed.returncode == 0
