import json
import os
import platform
import sys

import imageio
import imageio.v3 as iio
import numpy as np
import PIL
from imageio.core.findlib import load_lib

markers = {
    "implementation_name": sys.implementation.name,
    "implementation_version": ".".join(str(v) for v in sys.implementation.version[:3]),
    "os_name": os.name,
    "platform_machine": platform.machine(),
    "platform_python_implementation": platform.python_implementation(),
    "platform_release": platform.release(),
    "platform_system": platform.system(),
    "platform_version": platform.version(),
    "python_full_version": platform.python_version(),
    "python_version": ".".join(platform.python_version_tuple()[:2]),
    "sys_platform": sys.platform,
}

assert imageio.__version__ == "2.37.0"
assert np.__version__ == "2.3.5" and PIL.__version__ == "12.3.0"
pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
iio.imwrite("/tmp/graph.png", pixels, plugin="pillow")
restored = iio.imread("/tmp/graph.png", plugin="pillow")
assert restored.shape == (3, 4, 3) and restored.dtype == np.uint8
assert np.array_equal(restored, pixels)

try:
    load_lib(["forbidden"], ["forbidden"])
except NotImplementedError:
    pass
else:
    raise AssertionError("dynamic library loading accepted")
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
            "shape": list(restored.shape),
            "dtype": str(restored.dtype),
            "pixels": restored.tolist(),
            "invalid_rejected": True,
            "guest_markers": markers,
        },
        sort_keys=True,
    )
)
