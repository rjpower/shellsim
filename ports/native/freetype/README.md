# FreeType static target library

FreeType 2.13.3 selects TrueType/CFF fonts, autohinting, grayscale and monochrome
rasterization. It consumes the declared zlib 1.3.1 artifact under the SDK 34
`wasi-cpython-v2` profile. Optional libpng, bzip2, Brotli, HarfBuzz, SVG, SDF and
LZW support are disabled. No host-library discovery is performed.

The source SHA256 comes from the
[upstream release announcement](https://lists.gnu.org/archive/html/freetype-announce/2024-08/msg00000.html).
The real cmap validators and rasterizer retain their upstream setjmp/longjmp
paths. The shared profile supplies standard Wasm exception instructions and
`libsetjmp`, and consumers link FreeType before zlib and `-lm`.

`--with-pillow` builds this artifact and the `_imagingft` consumer. A separate
probe loads a scalable TrueType font through the VFS, rasterizes `A`, rejects
malformed font bytes, and exercises the guest CPU limit:

```sh
PYTHONPATH=. uv run --no-project python ports/native/freetype/verify.py \
  --bundle /tmp/shellsim-native --work-dir /tmp/shellsim-freetype
SHELLSIM_FREETYPE_ARTIFACTS=/tmp/shellsim-freetype \
  cargo test --test wasm_freetype -- --include-ignored
```

The font and license in `tests/fixtures/fonts` come from the pinned Pillow source
archive. That directory records their hashes and redistribution terms. This
subset does not claim arbitrary font format or shaping compatibility.
