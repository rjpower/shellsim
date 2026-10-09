"""Exercise upstream ImageIO against real guest NumPy, Pillow and ctypes."""

import ctypes
import json
import os
import sys

import imageio
import imageio.v3 as iio
import numpy as np
import PIL
from imageio.core.findlib import load_lib

assert imageio.__version__ == "2.37.0"
assert np.__version__ == "2.3.5"
assert PIL.__version__ == "12.3.0"
pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
iio.imwrite("/tmp/graph.png", pixels, plugin="pillow")
restored = iio.imread("/tmp/graph.png", plugin="pillow")
assert restored.dtype == np.uint8
assert restored.shape == (3, 4, 3)
assert np.array_equal(restored, pixels)

library, selected_path = load_lib(["/lib/libz.so"], ["libz"])
assert selected_path == "/lib/libz.so"
library.zlibVersion.restype = ctypes.c_char_p
zlib_version = library.zlibVersion().decode()
assert zlib_version.startswith("1.3.")
assert ctypes.pythonapi.Py_IsInitialized() == 1

host_library = sys.argv[1]
assert host_library.startswith("/")
assert not os.path.exists(host_library)
try:
    ctypes.CDLL(host_library)
except OSError:
    pass
else:
    raise AssertionError("host Linux library escaped into the guest")
try:
    iio.imread(b"not an image", plugin="pillow")
except OSError:
    pass
else:
    raise AssertionError("invalid image accepted")

print(
    json.dumps(
        {
            "imageio": imageio.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "shape": restored.shape,
            "dtype": str(restored.dtype),
            "pixel_sum": int(restored.sum()),
            "zlib": zlib_version,
            "host_library_rejected": True,
            "invalid_image_rejected": True,
        },
        sort_keys=True,
    )
)
