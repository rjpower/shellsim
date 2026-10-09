# Imageio 2.37.0

`build.py --wheel <upstream-wheel> --output <directory>` verifies the pinned pure
wheel and builds its reviewed WASI adaptation. It preserves upstream package
metadata and licenses, updates RECORD, and records source, patch and output
hashes. Add the output wheel to the ordinary curated package catalog. The command
builds a wheel; runtime selection and dependency resolution use the public
`CPythonRuntime` package installer.

The adaptation rejects Imageio dynamic library loading with
`NotImplementedError` on WASI before library discovery and imports ctypes lazily.
It also imports Pillow's GIF implementation only inside the GIF branch.
The checked-in guest probe writes a NumPy uint8 RGB array through Pillow to PNG,
reads identical pixels back, and rejects invalid input and dynamic loading.

The original static imaging profile supplies NumPy 2.3.5 and Pillow 12.3.0 with
native zlib 1.3.1, libjpeg-turbo 2.1.5.1 and FreeType 2.13.3. That profile lacks the
Pillow extensions required for GIF; this PNG test does not establish GIF support.
An independent dynamic Pillow wheel remains unfinished and is not substituted by
this build command.
