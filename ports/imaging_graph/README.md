# Basic imageio graph

`ports/spike_images.py` resolves imageio 2.37.0's real pure wheel against the
source metadata of the static NumPy 2.3.5 and Pillow 12.3.0 providers. Pillow
consumes the native zlib 1.3.1, libjpeg-turbo 2.1.5.1 and FreeType 2.13.3 artifacts
under the SDK 34 v2 profile. The resolver has two local indexes and
runs offline with the guest's WASI markers; optional imageio plugins are outside
this graph.

Build the combined image, then execute the graph with an installed current
Shellsim Python adapter:

```sh
uv run --no-project --python 3.13 ports/cpython/build.py \
  --work-dir /tmp/shellsim-imaging-graph --with-numpy --with-pillow
uv run --no-project --python 3.13 ports/spike_images.py \
  --bundle /tmp/shellsim-imaging-graph
```

The metadata wheels representing native providers are resolution inputs only.
Execution verifies the bundle recipes, mounts its checked image, and installs
a derived imageio pure wheel with explicit import adaptations: dynamic library
loading raises `NotImplementedError` on WASI before discovery, and `ctypes` is
imported only by that loader. The original wheel, patch and installed wheel
identities are recorded. The guest writes a NumPy uint8 RGB array as
PNG through Pillow in Shellsim's VFS, reloads it, checks exact pixels, and rejects
invalid image bytes. `result.json` records the source/wheel pins, resolved lock,
runtime manifest identity, native artifact closure, guest output and resource
usage. It is written only after successful guest execution.

This is a static graph proof. It does not add native libraries to the public wheel
installer or implement dynamic native dependency resolution. JPEG, GIF and font
rendering are outside this PNG graph. The imageio Pillow plugin imports its GIF
implementation only inside the GIF branch; GIF requires Pillow extensions absent
from this profile and is not claimed as supported.

The imageio wheel pin is published on [PyPI](https://pypi.org/project/ImageIO/2.37.0/).
