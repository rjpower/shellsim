from io import BytesIO

from PIL import Image, ImageFont, features

assert features.check_module("freetype2")
assert not features.check_feature("raqm")
font = ImageFont.truetype("/font.ttf", 24)
mask = font.getmask("A")
assert mask.size[0] > 0 and mask.size[1] > 0 and sum(mask) > 0
try:
    ImageFont.truetype(BytesIO(b"invalid font"), 24)
except OSError:
    pass
else:
    raise AssertionError("invalid font accepted")
image = Image.new("RGB", (8, 8), (240, 20, 30))
image.save("/tmp/roundtrip.jpg", quality=95)
encoded = open("/tmp/roundtrip.jpg", "rb").read()
for invalid in (encoded[: len(encoded) // 2], encoded[:-20]):
    try:
        with Image.open(BytesIO(invalid)) as restored:
            restored.load()
    except OSError:
        pass
    else:
        raise AssertionError("truncated JPEG accepted")
with Image.open("/tmp/roundtrip.jpg") as restored:
    assert restored.size == (8, 8) and restored.mode == "RGB"
    assert all(abs(a - b) <= 3 for a, b in zip(restored.getpixel((0, 0)), (240, 20, 30)))
print("JPEG recovery and FreeType rasterization passed")
