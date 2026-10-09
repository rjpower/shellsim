import zlib

from PIL import Image

payload = bytes(range(256)) * 32
compressed = zlib.compress(payload)
assert zlib.decompress(compressed) == payload
assert zlib.ZLIB_RUNTIME_VERSION == "1.3.1"
image = Image.new("RGB", (4, 3))
pixels = [(x * 15, x * 7, 255 - x * 10) for x in range(12)]
image.putdata(pixels)
image.save("/tmp/shared-zlib.png")
with Image.open("/tmp/shared-zlib.png") as restored:
    assert restored.format == "PNG"
    assert restored.size == (4, 3)
    assert restored.mode == "RGB"
    assert restored.tobytes() == image.tobytes()
print("shared zlib compression and PNG passed")
