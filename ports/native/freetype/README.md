# FreeType build candidate

This candidate pins FreeType 2.13.3 and selects TrueType/CFF fonts, autohinting,
grayscale and monochrome rasterization. It consumes the declared native zlib
1.3.1 artifact and invokes the pinned SDK compiler directly. Optional libpng,
bzip2, Brotli, HarfBuzz, SVG, SDF and LZW support are disabled.

The source SHA256 comes from the [upstream release announcement](https://lists.gnu.org/archive/html/freetype-announce/2024-08/msg00000.html).

The build is blocked by WASI SDK 24's lack of setjmp/longjmp in the existing
`wasi-cpython-v1` profile. Compiling `src/base/ftbase.c` includes
`include/freetype/config/ftstdlib.h`, which includes the SDK's `setjmp.h` and
fails. The upstream TrueType cmap validator, general validator and grayscale
rasterizer all use non-local jumps. Removing this support would change error
handling and rasterization behavior.

SDK 24 recommends `-mllvm -wasm-enable-sjlj`, which requires WebAssembly exception
handling support. This candidate does not enable that flag or claim a verified
artifact. Resolving the runtime/toolchain profile boundary is required before
building the library, linking Pillow `_imagingft`, or running a font probe.

When enabled, consumers use `-I<prefix>/include/freetype2` and link
`lib/libfreetype.a`, followed by `lib/libz.a` and `-lm`. The build API accepts
verified dependency providers and emits the shared immutable artifact contract.
