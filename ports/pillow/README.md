# Pillow PNG profile

This recipe builds upstream Pillow 12.3.0 for CPython 3.13.7 and WASI SDK 24.
`PIL._imaging` is a qualified static builtin. Upstream Python modules and exact
distribution metadata, including optional-extra requirements, are installed in
the guest image. The image preserves Pillow and zlib license notices.

The approved external-library profile enables zlib 1.3.1 and tests PNG encoding
and decoding. JPEG, JPEG 2000, libtiff, WebP, AVIF, FreeType, LittleCMS, Raqm,
libimagequant, and XCB are disabled. The separate imaging math and morphology
extensions are not built. Formats implemented within Pillow's core remain
upstream code; this profile does not claim that PNG is its only possible format.

The trusted builder reads `_IMAGING`, `_LIB_IMAGING`, and the internal
`pil_imaging_mode` library from the pinned source's `setup.py` as literal syntax.
The latter supplies `src/libImaging/Mode.c` in Pillow 12.3.0. It does not execute
setuptools or its host-library searches. All these sources enter one archive;
`HAVE_LIBZ` and `PILLOW_VERSION` match upstream's zlib feature configuration.
Optional external-library macros remain undefined. The manifest records the
exact upstream source list and its source/header content identity.

The extension consumes target CPython headers and `pyconfig.h` separately from
the native build helper. The same verified zlib artifact supplies CPython's
stdlib module and Pillow's core, through an explicit dependency prefix. PNG uses
Pillow's own codecs and zlib; no libpng provider is needed.

Build and verify with the commands in [the native dependency guide](../native/README.md).
The guest proof covers an RGB PNG round trip, zlib round trips, invalid input,
optional codec frontiers, and resource exhaustion. Other image modes, animation,
metadata, fonts, external decoders, and the full Pillow test suite are outside
this measured foundation profile.
