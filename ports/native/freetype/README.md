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
PYTHONPATH=. uv run --no-project python ports/native/freetype/tests/verify.py \
  --bundle /tmp/shellsim-native --work-dir /tmp/shellsim-freetype
SHELLSIM_FREETYPE_ARTIFACTS=/tmp/shellsim-freetype \
  cargo test --test wasm_freetype -- --include-ignored
```

The font and license in `tests/fixtures/fonts` come from the pinned Pillow source
archive. That directory records their hashes and redistribution terms. This
subset does not claim arbitrary font format or shaping compatibility.

## pkg-config version

The native artifact renders the pinned upstream `builds/unix/freetype2.in`
template. FreeType 2.13.3 defines `version_info='26:2:20'` in `configure.raw`
and converts it to `26.2.20` for pkg-config. This is the upstream libtool
version. The source recipe and `FT_Library_Version` report the semantic release
`2.13.3`.

Both metadata source files are hashed in artifact inputs. The template preserves
upstream description and URL; relocatable paths and private dependencies describe
the selected static TrueType/CFF profile with zlib and libm. A consumer can use
the upstream version checks without changing package release metadata.

Set `SHELLSIM_FREETYPE_SOURCE`, `SHELLSIM_FREETYPE_PREFIX` and
`SHELLSIM_FREETYPE_ZLIB_PREFIX` to run the pinned-source and actual pkg-config
checks in `ports/native/freetype/tests/test_port.py`. This metadata correction does
not enable additional font formats, HarfBuzz or Matplotlib providers.
