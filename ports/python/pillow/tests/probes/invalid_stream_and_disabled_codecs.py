import zlib
from io import BytesIO

from PIL import Image, UnidentifiedImageError, features

try:
    zlib.decompress(b"invalid compressed data")
except zlib.error:
    pass
else:
    raise AssertionError("invalid stream accepted")
try:
    Image.open(BytesIO(b"invalid image"))
except UnidentifiedImageError:
    pass
else:
    raise AssertionError("invalid image accepted")
assert features.check_codec("zlib")
assert features.check_codec("jpg")
assert not features.check_codec("jpg_2000")
assert not features.check_codec("libtiff")
try:
    Image.new("RGB", (2, 2)).save("/tmp/disabled.jp2")
except OSError:
    pass
else:
    raise AssertionError("disabled JPEG2000 encoder accepted")
print("invalid data and optional codec frontiers passed")
