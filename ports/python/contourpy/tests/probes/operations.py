"""Exercise serial contours, a real native extension, and invalid grids."""

import contourpy
import numpy as np

z = np.array([[0.0, 1.0], [0.0, 1.0]])
generator = contourpy.contour_generator(z=z, name="serial")
lines = generator.lines(0.5)
assert len(lines) == 1
assert np.allclose(lines[0][:, 0], 0.5)
assert sorted(lines[0][:, 1].tolist()) == [0.0, 1.0]
try:
    contourpy.contour_generator(z=np.zeros((1, 1)))
except (TypeError, ValueError):
    pass
else:
    raise AssertionError("undersized contour grid accepted")
print("contourpy: serial contours and invalid grid passed")
