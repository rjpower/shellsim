# ImageIO 2.37.0

`build.py --wheel <upstream-wheel> --output <directory>` verifies the pinned
upstream pure wheel, its metadata, dependencies and member safety, then copies
it unchanged. The output manifest records the wheel hash. Add that wheel to an
ABI-matched package catalog alongside native NumPy 2.3.5 and Pillow 12.3.0;
`env.install_pypi("imageio==2.37.0")` resolves the graph through the public
CPython installer.

The port-local guest probe writes a NumPy RGB array through the upstream
ImageIO Pillow plugin to a PNG in the virtual filesystem, reads identical
pixels back, and checks invalid image handling. It also loads guest
`/lib/libz.so` through ImageIO's unmodified `findlib.load_lib`, calls the
guest CPython API through `ctypes.pythonapi`, and confirms that the installed
host shellsim native library cannot be opened from the guest. These checks
require the assembled dynamic CPython bundle with `_ctypes`, zlib and Pillow's
native providers.
Other ImageIO plugins and codecs need their own native package closures.
