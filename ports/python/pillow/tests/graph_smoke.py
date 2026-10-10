"""Exercise Pillow codecs and its bundled scalable default font in guest CPython."""

from io import BytesIO

import PIL
from PIL import Image, ImageFont, ImageMath, ImageMorph, features

assert PIL.__version__ == "12.3.0"
assert features.check_module("freetype2")
assert features.check_codec("jpg") and features.check_codec("zlib")
assert not features.check_feature("raqm")
image = Image.new("RGB", (8, 8), (240, 20, 30))
image.save("/tmp/roundtrip.png")
with Image.open("/tmp/roundtrip.png") as restored:
    assert restored.tobytes() == image.tobytes()
image.save("/tmp/roundtrip.jpg", quality=95)
encoded = open("/tmp/roundtrip.jpg", "rb").read()
for truncated in (encoded[: len(encoded) // 2], encoded[:-20]):
    try:
        with Image.open(BytesIO(truncated)) as restored:
            restored.load()
    except OSError:
        pass
    else:
        raise AssertionError("truncated JPEG accepted")
with Image.open(BytesIO(encoded)) as restored:
    assert restored.size == (8, 8) and restored.mode == "RGB"
    assert all(abs(a - b) <= 3 for a, b in zip(restored.getpixel((0, 0)), (240, 20, 30)))
font = ImageFont.load_default(size=24)
mask = font.getmask("A")
assert mask.size[0] > 0 and mask.size[1] > 0 and sum(mask) > 0
assert isinstance(font, ImageFont.FreeTypeFont)
try:
    ImageFont.truetype(BytesIO(b"invalid font"), 24)
except OSError:
    pass
else:
    raise AssertionError("invalid font accepted")
a, b = Image.new("I", (2, 2), 2), Image.new("I", (2, 2), 5)
computed = ImageMath.lambda_eval(lambda args: args["a"] + args["b"], a=a, b=b)
assert computed.getpixel((0, 0)) == 7
binary = Image.new("L", (5, 5), 0)
binary.putpixel((2, 2), 255)
count, dilated = ImageMorph.MorphOp(op_name="dilation8").apply(binary)
assert count == 8 and dilated.size == (5, 5) and dilated.getpixel((1, 1)) > 0
print("Pillow PNG/JPEG recovery, font, math and morphology passed")
