import json
import os
import platform
import sys

import magiccube
import numpy as np
import numpy._core._multiarray_umath as core

markers = {
    "implementation_name": sys.implementation.name,
    "implementation_version": ".".join(str(value) for value in sys.implementation.version[:3]),
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

assert np.__version__ == "2.3.5"
a = np.arange(12, dtype=np.int64).reshape(3, 4)
assert np.array_equal(a.sum(axis=0), [12, 15, 18, 21])
cube = magiccube.Cube(3)
assert cube.is_done()
cube.rotate("R U F")
assert not cube.is_done()
cube.rotate("F' U' R'")
assert cube.is_done()
assert cube.cube.shape == (3, 3, 3)
assert cube.cube.dtype == np.dtype(object)
print(
    json.dumps(
        {
            "guest_markers": markers,
            "numpy": np.__version__,
            "native_origin": core.__spec__.origin,
            "sum": a.sum(axis=0).tolist(),
            "cube_shape": list(cube.cube.shape),
            "cube_restored": cube.is_done(),
        },
        sort_keys=True,
    )
)
