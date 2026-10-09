# Independent Pillow 12.3.0 extensions

`dynamic.py` builds the upstream `_imaging`, `_imagingft`, `_imagingmath`, and
`_imagingmorph` extensions as separate Wasm shared modules. They import the fixed
CPython runtime and retain their original Python initializer names. The wheel
preserves upstream Python files, metadata and licenses. It includes codec/font
licenses, per-extension hashes, dependency edges, and source/toolchain provenance.

The shared native products are zlib 1.3.1 (`libz.so`), libjpeg-turbo 2.1.5.1
(`libjpeg.so`) and FreeType 2.13.3 (`libfreetype.so`). FreeType imports the same
`libz.so` used by Pillow. Each provider is independently sealed; shared admission
checks target, ABI, linkage, version, toolchain, exact transitive artifact
identities and emitted dependencies. Static products remain separate recipes.
Dependency admission follows the full declared provider graph, including shared
providers reached through more than one dependency path.

Build with the pinned source archives, SDK34, and the recipe's reviewed CPython
source headers and target pyconfig:

```sh
uv run --no-project --python 3.13 ports/python/pillow/dynamic.py \
  --downloads /path/to/verified-downloads \
  --cpython-source /path/to/Python-3.13.7 \
  --cpython-build /path/to/wasi-build \
  --sdk /path/to/wasi-sdk-34.0-x86_64-linux \
  --llvm /path/to/verified-llvm-prefix \
  --runtime /path/to/process-runtime \
  --output /path/to/pillow-output
```

The output includes the wheel, `manifest.json`, native artifacts, and the existing
package catalog at `universe/catalog.json`. Install `pillow==12.3.0` through
`CPythonRuntime.install_pypi` with that catalog and patched uv. The interpreter is
an input; package installation leaves it unchanged. The main must own the SDK
setjmp functions and `__c_longjmp` tag needed for JPEG error recovery.

The LLVM prefix must match the pinned generic linker recipe and its hashed
binary and license. FreeType uses the explicit `shellsim.deferred-init` version 1
protocol so its own function GOT slots are bound before global relocation.
The loader must support that protocol; earlier runtimes reject its metadata.
This v2 profile rejects deferred modules with self-referential data GOT imports.
The zlib provider must match the runtime's existing shared zlib bytes.

```python
from shellsim import CPythonRuntime, Environment

runtime = CPythonRuntime(
    "/path/to/process-ctypes-zlib-runtime",
    universe="/path/to/pillow-output/universe",
    uv="/path/to/patched-uv",
)
env = Environment(cpu=10_000_000_000, memory=1024**3, disk=128 * 1024**2)
runtime.mount(env)
runtime.install_pypi(env, "pillow==12.3.0")
result = runtime.run(env, ["-c", "from PIL import Image; print(Image.new('RGB', (8, 8)).size)"])
assert result.returncode == 0
```

The port-local acceptance test checks PNG pixels, JPEG encoding and decoding,
truncated JPEG errors followed by successful decoding, real font rasterization,
invalid fonts, image arithmetic and morphology. Enable it with
`SHELLSIM_PILLOW_BUNDLE`, `SHELLSIM_PILLOW_UNIVERSE`, and `SHELLSIM_PATCHED_UV`.

JPEG2000, TIFF, WebP, AVIF, LCMS, Raqm, imagequant, XCB and GIF remain outside this
profile. These recipes do not establish support for those optional extensions.
