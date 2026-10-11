"""Check actual upstream image operations on guest NumPy arrays."""

import cv2
import numpy as np

image = np.arange(9, dtype=np.uint8).reshape(3, 3)
resized = cv2.resize(image, (6, 6), interpolation=cv2.INTER_NEAREST)
assert resized.shape == (6, 6) and resized.dtype == np.uint8
assert int(resized[0, 0]) == 0 and int(resized[5, 5]) == 8
assert np.array_equal(cv2.flip(image, 1), image[:, ::-1])
threshold, mask = cv2.threshold(image, 4, 255, cv2.THRESH_BINARY)
assert threshold == 4 and np.array_equal(mask, (image > 4).astype(np.uint8) * 255)
constant = np.full((5, 5), 42, dtype=np.uint8)
assert np.array_equal(cv2.GaussianBlur(constant, (3, 3), 0), constant)
assert cv2.mean(constant)[0] == 42
try:
    cv2.resize(image, (0, 0))
except cv2.error:
    pass
else:
    raise AssertionError("invalid image size accepted")
print("opencv: native resize, flip, threshold, Gaussian blur and errors passed")
