"""Compare runtime roots with a real native archive's public definitions."""

import shutil
import subprocess
from pathlib import Path

import pytest

from ports.toolchain.runtime_profile import public_archive_symbols


def test_runtime_roots_include_public_definitions_and_exclude_local_and_undefined(tmp_path: Path):
    cc, ar = shutil.which("cc"), shutil.which("ar")
    if cc is None or ar is None:
        pytest.skip("native C compiler and archiver are unavailable")
    source = tmp_path / "runtime.c"
    source.write_text(
        "extern int dependency(void);\n"
        "static int private_value = 3;\n"
        "int public_value = 7;\n"
        "__attribute__((weak)) int optional(void) { return 9; }\n"
        "int public_call(void) { return private_value + dependency(); }\n"
    )
    obj, archive = tmp_path / "runtime.o", tmp_path / "runtime.a"
    subprocess.run([cc, "-c", str(source), "-o", str(obj)], check=True)
    subprocess.run([ar, "rcs", str(archive), str(obj)], check=True)
    assert public_archive_symbols(archive) == ("optional", "public_call", "public_value")


def test_runtime_archive_truncation_is_rejected(tmp_path: Path):
    archive = tmp_path / "runtime.a"
    archive.write_bytes(
        b"!<arch>\n"
        + b"/".ljust(16)
        + b"0".ljust(12)
        + b"0".ljust(6)
        + b"0".ljust(6)
        + b"0".ljust(8)
        + b"100".ljust(10)
        + b"`\n"
        + b"\x00\x00\x00\x01"
    )
    with pytest.raises(ValueError, match="truncated"):
        public_archive_symbols(archive)
