# Pillow imaging profiles

[The independent extension build](DYNAMIC.md) produces an installable wheel
with shared zlib, JPEG and FreeType providers, plus imaging math and morphology.
It requires the declared LLVM deferred-initialization toolchain and a compatible
fixed CPython runtime. The static recipe below remains a separate product.

## Static imaging profile

This recipe builds upstream Pillow 12.3.0 for CPython 3.13.7 and the SDK 34
`wasi-cpython-v2` profile. `PIL._imaging` and `PIL._imagingft` are qualified static
builtins. The image installs upstream Python modules, distribution metadata and
Pillow, zlib, libjpeg-turbo and FreeType license notices.

The declared native closure enables zlib 1.3.1, scalar libjpeg-turbo 2.1.5.1 and
FreeType 2.13.3. JPEG 2000, libtiff, WebP, AVIF, LittleCMS, Raqm, libimagequant and
XCB are disabled. The imaging math and morphology extensions are not built.
Pillow's own PNG codecs consume zlib without a libpng provider. Other formats
implemented within the core remain upstream code.

The trusted builder reads source lists from the pinned `setup.py` as literal
syntax, including the internal `pil_imaging_mode` library. It does not execute
setuptools or host-library searches. Target CPython headers and `pyconfig.h`
come from the guest build. The extension uses `HAVE_LIBZ` and `HAVE_LIBJPEG`,
and uses the profile's actual standard Wasm setjmp/longjmp support for JPEG
error recovery. Font shaping through Raqm and HarfBuzz is outside this profile.

Build and verify with [the native dependency guide](../../native/README.md).
The guest checks cover PNG and JPEG VFS round trips, truncated JPEG rejection
followed by successful decoding in the same interpreter, invalid font rejection,
and scalable TrueType glyph rasterization using the licensed pinned test font.
These measured operations do not establish complete Pillow compatibility.
