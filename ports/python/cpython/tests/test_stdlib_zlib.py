"""Exercise the upstream zlib module and independent shared provider in guest."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import shellsim


def test_independent_stdlib_zlib_roundtrip_and_errors():
    bundle = os.environ.get("SHELLSIM_DYNAMIC_V2_ARTIFACTS")
    extension = os.environ.get("SHELLSIM_STDLIB_ZLIB")
    if bundle is None or extension is None:
        pytest.skip("set fixed SDK 34 runtime and SHELLSIM_STDLIB_ZLIB")
    runtime = shellsim.CPythonRuntime(bundle)
    environment = shellsim.Environment(cpu=4_000_000_000, memory=512 * 1024 * 1024)
    runtime.mount(environment)
    root = Path(extension)
    manifest = json.loads((root / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        data = (root / artifact["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
        environment.mkdir(str(Path(artifact["destination"]).parent), parents=True)
        environment.write_file(artifact["destination"], data)
    before = environment.read_file("/usr/bin/python3.wasm")
    result = runtime.run(
        environment,
        [
            "-c",
            """
import zlib
assert zlib.__spec__.origin.endswith('/zlib.so')
payload = b'upstream independent zlib' * 100
assert zlib.decompress(zlib.compress(payload)) == payload
assert zlib.crc32(b'123456789') == 0xcbf43926
compressor = zlib.compressobj(wbits=31)
encoded = compressor.compress(payload[:100]) + compressor.compress(payload[100:]) + compressor.flush()
assert zlib.decompress(encoded, wbits=31) == payload
try:
    zlib.decompress(b'invalid stream')
except zlib.error:
    pass
else:
    raise AssertionError('invalid stream accepted')
print('independent stdlib zlib passed')
""",
        ],
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == b"independent stdlib zlib passed\n"
    assert environment.read_file("/usr/bin/python3.wasm") == before
