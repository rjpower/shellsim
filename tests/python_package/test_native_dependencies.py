"""Exercise real zlib and Pillow through a source-built WASI image and the VFS."""

import os

import pytest
import shellsim


@pytest.fixture
def native_runtime():
    bundle = os.environ.get("SHELLSIM_NATIVE_BUNDLE")
    if bundle is None:
        pytest.skip("set SHELLSIM_NATIVE_BUNDLE to the shared-zlib Pillow WASI image")
    runtime = shellsim.CPythonRuntime(bundle)
    environment = shellsim.Environment(cpu=2_000_000_000, memory=256 * 1024 * 1024, disk=64 * 1024 * 1024)
    runtime.mount(environment)
    return runtime, environment


def test_manifest_has_one_native_zlib_provider_for_both_consumers(native_runtime):
    runtime, _ = native_runtime
    libraries = runtime.manifest["native_libraries"]
    assert list(libraries) == ["native/zlib"]
    identity = libraries["native/zlib"]["artifact_sha256"]
    for consumer in ("cpython.zlib", "PIL._imaging"):
        assert runtime.manifest["link_consumers"][consumer]["dependency_artifacts"] == {"native/zlib": identity}
    assert "zlib" in runtime.builtin_modules
    assert "PIL._imaging" in runtime.builtin_modules
    assert all(port["name"] != "zlib" for port in runtime.manifest["native_ports"])


def test_compression_and_png_round_trip_in_vfs(native_runtime):
    runtime, environment = native_runtime
    result = runtime.run(
        environment,
        [
            "-c",
            """
import zlib
from PIL import Image
payload = bytes(range(256)) * 32
compressed = zlib.compress(payload)
assert zlib.decompress(compressed) == payload
assert zlib.ZLIB_RUNTIME_VERSION == '1.3.1'
image = Image.new('RGB', (4, 3))
pixels = [(x * 15, x * 7, 255 - x * 10) for x in range(12)]
image.putdata(pixels)
image.save('/tmp/shared-zlib.png')
with Image.open('/tmp/shared-zlib.png') as restored:
    assert restored.format == 'PNG'
    assert restored.size == (4, 3)
    assert restored.mode == 'RGB'
    assert restored.tobytes() == image.tobytes()
print('shared zlib compression and PNG passed')
""",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"shared zlib compression and PNG passed\n"
    assert environment.read_file("/tmp/shared-zlib.png").startswith(b"\x89PNG\r\n\x1a\n")
    assert result.usage.cpu_used > 0
    assert result.usage.memory_peak > 0


def test_invalid_stream_and_disabled_codecs(native_runtime):
    runtime, environment = native_runtime
    result = runtime.run(
        environment,
        [
            "-c",
            """
import zlib
from PIL import Image, features, UnidentifiedImageError
from io import BytesIO
try:
    zlib.decompress(b'invalid compressed data')
except zlib.error:
    pass
else:
    raise AssertionError('invalid stream accepted')
try:
    Image.open(BytesIO(b'invalid image'))
except UnidentifiedImageError:
    pass
else:
    raise AssertionError('invalid image accepted')
assert features.check_codec('zlib')
assert not features.check_codec('jpg')
assert not features.check_codec('jpg_2000')
assert not features.check_codec('libtiff')
try:
    Image.new('RGB', (2, 2)).save('/tmp/disabled.jpg')
except OSError:
    pass
else:
    raise AssertionError('disabled JPEG encoder accepted')
print('invalid data and optional codec frontiers passed')
""",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"invalid data and optional codec frontiers passed\n"


def test_native_compression_obeys_guest_cpu_budget(native_runtime):
    runtime, environment = native_runtime
    result = runtime.run(
        environment,
        [
            "-c",
            """
import zlib
print('compression loop entered', flush=True)
payload = bytes(range(256)) * 256
while True:
    zlib.compress(payload)
""",
        ],
    )
    assert result.stdout == b"compression loop entered\n"
    assert result.returncode != 0
    assert result.stop_reason == "cpu_exhausted"
