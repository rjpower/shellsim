"""Use explicit Pillow plugins to keep ImageIO probes local and deterministic."""

from io import BytesIO

import imageio.v3 as iio
import numpy as np

pixels = np.zeros((8, 8, 3), dtype=np.uint8)
pixels[:, :] = [240, 20, 30]
encoded = iio.imwrite("<bytes>", pixels, extension=".png", plugin="pillow")
restored = iio.imread(encoded, extension=".png", plugin="pillow")
assert restored.dtype == np.uint8 and restored.shape == pixels.shape
assert np.array_equal(restored, pixels)
encoded = iio.imwrite("<bytes>", pixels, extension=".jpg", plugin="pillow", quality=95)
restored = iio.imread(encoded, extension=".jpg", plugin="pillow")
assert restored.dtype == np.uint8 and restored.shape == pixels.shape
assert np.max(np.abs(restored.astype(np.int16) - pixels)) <= 3
try:
    iio.imread(BytesIO(b"invalid image"), extension=".png", plugin="pillow")
except OSError:
    pass
else:
    raise AssertionError("invalid image accepted")
assert np.array_equal(
    iio.imread(iio.imwrite("<bytes>", pixels, extension=".png", plugin="pillow"), extension=".png", plugin="pillow"),
    pixels,
)
print("ImageIO PNG/JPEG arrays and malformed input recovery passed")
